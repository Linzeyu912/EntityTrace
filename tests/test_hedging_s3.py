from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from entitytrace.config import Service
from entitytrace.models import Models, RequestBudget
from entitytrace.pipeline import metrics
from entitytrace.storage import Store
from entitytrace.util import dumps


def _chat_response(marker: str) -> dict:
    return {
        "model": "fake-discovery-v1",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps({"marker": marker})}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _invalid_response() -> bytes:
    return b'{"model": "fake-discovery-v1", "choices": [{"finish_reason": "stop", "message": {"content": "not-json{{{"}]}'


class CompositeOpener:
    """按序分配子 opener（每项可以是 callable）。"""

    def __init__(self, handlers: list):
        self.handlers = list(handlers)
        self.lock = threading.Lock()
        self.calls = 0

    def __call__(self, request, timeout=None):
        with self.lock:
            idx = self.calls
            self.calls += 1
        handler = self.handlers[idx] if idx < len(self.handlers) else self.handlers[-1]
        return handler(request, timeout=timeout)


def _delayed(delay: float, body: bytes):
    def handler(request, timeout=None):
        if delay:
            time.sleep(delay)
        return io.BytesIO(body)
    return handler


def _ok(delay: float, marker: str):
    return _delayed(delay, dumps(_chat_response(marker)).encode("utf-8"))


def _invalid(delay: float):
    return _delayed(delay, _invalid_response())


def _validate(parsed):
    if not isinstance(parsed, dict) or "marker" not in parsed:
        raise ValueError("bad shape")


class HedgingS3Tests(unittest.TestCase):
    def _make_models(self, tmp: Path, opener, **kwargs) -> Models:
        store = Store(tmp / "test.sqlite3")
        self.addCleanup(store.db.close)
        services = {
            "discovery": Service(name="discovery", url="https://api.fake.local", model="fake-discovery-v1", key="k"),
            "decision": Service(name="decision", url="https://api.fake.local", model="fake-decision-v1", key="k"),
        }
        params = dict(attempts=3, timeout=5, opener=opener,
                      hedge_min_seconds=0.05, hedge_max_ratio=1.0)
        params.update(kwargs)
        return Models(store, "run-1", "invoke-1", services, **params)
    def test_hedge_wins_over_slow_primary(self):
        # 首发 0.6s，对冲 0.01s：采用对冲结果，总耗时远小于 0.6s。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            opener = CompositeOpener([_ok(0.6, "primary"), _ok(0.01, "hedge")])
            models = self._make_models(tmp, opener)
            start = time.monotonic()
            parsed, request_id = models.generate("sys", {"x": 1}, "discovery", _validate, temperature=0.2)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.4)
            self.assertEqual(parsed["marker"], "hedge")
            rows = models.store.rows("SELECT status, detail FROM calls WHERE run_id=? ORDER BY id", ("run-1",))
            self.assertEqual(len(rows), 2)
            statuses = [r["status"] for r in rows]
            self.assertIn("success", statuses)
            self.assertIn("discarded", statuses)
            details = [json.loads(r["detail"]) for r in rows]
            self.assertTrue(any(d.get("hedge") for d in details))
            m = metrics(models.store, "run-1")
            self.assertEqual(m["hedged_requests"], 1)
            self.assertEqual(m["failed_attempts"], 0)
            self.assertEqual(m["successful_requests"], 1)
            self.assertEqual(m["network_attempts"], 2)
            cache = models.store.cache(request_id)
            self.assertIsNotNone(cache)
            time.sleep(0.7)  # 败者随后返回，不得覆写缓存
            cache2 = models.store.cache(request_id)
            marker = json.loads(cache2["choices"][0]["message"]["content"])["marker"]
            self.assertEqual(marker, "hedge")

    def test_primary_wins_hedge_discarded(self):
        # 对冲已发出但首发先返回：采用首发，对冲记 discarded。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            opener = CompositeOpener([_ok(0.10, "primary"), _ok(0.30, "hedge")])
            models = self._make_models(tmp, opener)  # delay=0.05 < 0.10，对冲会发出
            parsed, request_id = models.generate("sys", {"x": 1}, "discovery", _validate)
            self.assertEqual(parsed["marker"], "primary")
            rows = models.store.rows("SELECT status, detail FROM calls WHERE run_id=? ORDER BY id", ("run-1",))
            self.assertEqual(len(rows), 2)
            discarded = [r for r in rows if r["status"] == "discarded"]
            self.assertEqual(len(discarded), 1)
            self.assertTrue(json.loads(discarded[0]["detail"])["hedge"])
            m = metrics(models.store, "run-1")
            self.assertEqual(m["hedged_requests"], 1)
            self.assertEqual(m["failed_attempts"], 0)
            self.assertEqual(m["successful_requests"], 1)
            time.sleep(0.35)
            marker = json.loads(models.store.cache(request_id)["choices"][0]["message"]["content"])["marker"]
            self.assertEqual(marker, "primary")

    def test_primary_invalid_hedge_succeeds_same_attempt(self):
        # 首发校验失败先到、对冲成功：同一 attempt 采用对冲结果，不重试。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            opener = CompositeOpener([_invalid(0.10), _ok(0.20, "hedge")])
            models = self._make_models(tmp, opener)
            parsed, _ = models.generate("sys", {"x": 1}, "discovery", _validate)
            self.assertEqual(parsed["marker"], "hedge")
            self.assertEqual(opener.calls, 2)  # 无重试
            rows = models.store.rows("SELECT status FROM calls WHERE run_id=? ORDER BY id", ("run-1",))
            self.assertEqual([r["status"] for r in rows], ["failed", "success"])

    def test_exhausted_budget_blocks_hedge(self):
        # max_requests=1：预算恰好用尽时不发对冲，首发正常完成。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            budget = RequestBudget(max_requests=1, opener=_ok(0.20, "primary"))
            models = self._make_models(tmp, None, budget=budget)
            parsed, _ = models.generate("sys", {"x": 1}, "discovery", _validate)
            self.assertEqual(parsed["marker"], "primary")
            self.assertEqual(budget.count, 1)
            rows = models.store.rows("SELECT status FROM calls WHERE run_id=?", ("run-1",))
            self.assertEqual([r["status"] for r in rows], ["success"])

    def test_hedge_ratio_cap(self):
        # 默认 5% 上限：20 次请求后最多 1 次对冲；样本不足时第 1 次慢请求不发对冲。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            calls = {"n": 0}
            lock = threading.Lock()

            def opener(request, timeout=None):
                with lock:
                    calls["n"] += 1
                    n = calls["n"]
                # 第 21 次起变慢，观察是否触发对冲
                delay = 0.30 if n >= 21 else 0.0
                if delay:
                    time.sleep(delay)
                return io.BytesIO(dumps(_chat_response(f"r{n}")).encode("utf-8"))

            models = self._make_models(tmp, opener, hedge_max_ratio=0.05)
            for i in range(20):
                models.generate("sys", {"i": i}, "discovery", _validate)
            self.assertEqual(calls["n"], 20)
            # 第 21 次慢请求：允许 1 次对冲（1 <= 21*0.05? 1 <= 1.05 允许）。
            models.generate("sys", {"i": 20}, "discovery", _validate)
            self.assertEqual(calls["n"], 22)
            hedged_rows = models.store.rows(
                "SELECT status FROM calls WHERE run_id=? AND detail LIKE '%\"hedge\":true%'", ("run-1",))
            self.assertGreaterEqual(len(hedged_rows), 1)
            # 第 2 次慢请求：2 <= 22*0.05=1.1 不成立 → 拦截。
            models.generate("sys", {"i": 21}, "discovery", _validate)
            self.assertEqual(calls["n"], 23)

        # 样本不足：只发 1 次慢请求时 _sent_count*0.05 < 1 → 无对冲。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            opener = CompositeOpener([_ok(0.15, "primary")])
            models = self._make_models(tmp, opener, hedge_max_ratio=0.05)
            parsed, _ = models.generate("sys", {"x": 1}, "discovery", _validate)
            self.assertEqual(parsed["marker"], "primary")
            self.assertEqual(opener.calls, 1)

    def test_halted_blocks_hedge(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            models = self._make_models(tmp, _ok(0.0, "x"))
            models._sent_count = 100
            models.halted["discovery"] = ("halted", 500, "quota")
            self.assertFalse(models._try_reserve_hedge("discovery"))
            # 冷却中也拦截
            del models.halted["discovery"]
            models.cooldown["discovery"] = time.monotonic() + 60
            self.assertFalse(models._try_reserve_hedge("discovery"))

    def test_dynamic_threshold_and_decision_not_hedged(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            models = self._make_models(tmp, _ok(0.0, "x"))
            # 样本不足 8 → 返回 hedge_min_seconds
            self.assertAlmostEqual(models._hedge_delay("s"), 0.05)
            with models.lock:
                models._stage_latencies["s"] = [0.04] * 8
            # max(0.05, 3*0.04) = max(0.05, 0.12) = 0.12
            self.assertAlmostEqual(models._hedge_delay("s"), 0.12)
            with models.lock:
                models._stage_latencies["s"] = [0.001] * 8
            self.assertAlmostEqual(models._hedge_delay("s"), 0.05)
            # decision 服务不启用对冲
            models._sent_count = 100
            self.assertFalse(models._try_reserve_hedge("decision"))

    def test_single_slot_concurrency(self):
        # 单次请求触发对冲只占一个并发槽位。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            opener = CompositeOpener([_ok(0.6, "primary"), _ok(0.01, "hedge")])
            models = self._make_models(tmp, opener)
            models.generate("sys", {"x": 1}, "discovery", _validate)
            self.assertEqual(models.peaks["discovery"], 1)
            self.assertEqual(models.active["discovery"], 0)

    def test_both_fail_retries(self):
        # 首发与对冲都失败 → 正常进入重试；格式错误丢弃 temperature。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            tmp = Path(d)
            seen = []

            def opener(request, timeout=None):
                payload = json.loads(request.data.decode("utf-8"))
                seen.append("temperature" in payload)
                time.sleep(0.10)
                return io.BytesIO(_invalid_response())

            models = self._make_models(tmp, opener, attempts=1)
            with self.assertRaises(Exception):
                models.generate("sys", {"x": 1}, "discovery", _validate, temperature=0.5)
            rows = models.store.rows("SELECT status FROM calls WHERE run_id=?", ("run-1",))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["status"] == "failed" for r in rows))


if __name__ == "__main__":
    unittest.main()
