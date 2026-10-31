import inspect
import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import tools.bounded_identity_run as bounded_tool
import tools.identity_calibration as calibration_tool
from entitytrace.cli import main as cli_main
from entitytrace.models import BudgetError, ExperimentBudget, Models, RequestBudget
from entitytrace.pipeline import run_pipeline
from entitytrace.util import dumps


def _make_fake_http_opener(fail_network=False):
    lock = threading.Lock()
    seen_payloads = []

    def opener(request, timeout=None):
        if fail_network:
            raise ConnectionResetError("模拟传输中断")
        payload = json.loads(request.data.decode("utf-8"))
        with lock:
            seen_payloads.append(dumps(payload))
        if request.full_url.endswith("/systemone"):
            questions = payload["questions"]
            state = payload["state"]
            answers = {}
            for key, question in questions.items():
                criteria = question["criteria"]
                if "same" in criteria:
                    choice = "same"
                elif "person" in criteria:
                    choice = "person"
                elif "n0" in criteria:
                    choice = "n0"
                elif "accept" in criteria:
                    choice = "accept"
                else:
                    unit = next(u for u in state["units"] if u["id"] == key)
                    choice = "chapter" if unit["text"].startswith("第") else "body"
                answers[key] = {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 1.0,
                    "probabilities": {k: float(k == choice) for k in criteria},
                }
            body = dumps({"model": payload["model"], "answers": answers}).encode("utf-8")
            return io.BytesIO(body)
        # /chat/completions
        messages = payload["messages"]
        system_prompt = messages[0]["content"]
        user_state = json.loads(messages[1]["content"])
        if "章节内实体称呼归并器" in system_prompt:
            surfaces = [row["surface"] for row in user_state["surfaces"]]
            ev_ref = user_state["units"][0]["id"]
            entities = []
            if surfaces:
                entities.append({
                    "local_id": "e1",
                    "name": surfaces[0],
                    "kind": "person",
                    "surfaces": [{"surface": s, "surface_type": "proper_name"} for s in surfaces],
                    "evidence_ref": ev_ref,
                })
            content = {"entities": entities, "assignments": [], "not_entity": []}
        elif "补合并提议器" in system_prompt:
            content = {"proposals": []}
        else:
            refs = [u["id"] for u in user_state["units"] if "林舟" in u["text"]]
            mentions = [{"text": "林舟", "kind": "person", "refs": refs}] if refs else []
            content = {"mentions": mentions, "links": []}
        raw = {
            "choices": [{"finish_reason": "stop", "message": {"content": dumps(content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return io.BytesIO(dumps(raw).encode("utf-8"))

    return opener, seen_payloads


class BudgetTests(unittest.TestCase):
    def test_compatibility_reexports_have_single_implementation(self):
        self.assertIs(ExperimentBudget, BudgetError)
        self.assertIs(calibration_tool.ExperimentBudget, BudgetError)
        self.assertIs(bounded_tool.RequestBudget, RequestBudget)
        self.assertNotIn("class ExperimentBudget", inspect.getsource(calibration_tool))
        self.assertNotIn("class RequestBudget", inspect.getsource(bounded_tool))

    def test_parallel_workers_share_one_request_cap(self):
        network = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(b'ok'))
        budget = RequestBudget(2, 60, network)
        def worker(_):
            try:
                with budget(None, 120) as response:
                    return response.read().decode()
            except ExperimentBudget:
                return 'stopped'
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(worker, range(8)))
        self.assertEqual(results.count('ok'), 2)
        self.assertEqual((budget.count, network.call_count), (2, 2))
        self.assertEqual(budget.stop_reason, 'request_budget')
        self.assertTrue(all(call.kwargs['timeout'] <= 60 for call in network.call_args_list))

    def test_time_budget_stops_network_and_does_not_consume_attempt(self):
        network = Mock()
        with patch('entitytrace.models.time.perf_counter', side_effect=[0, 61]):
            budget = RequestBudget(2, 60, network)
            with self.assertRaises(ExperimentBudget):
                budget(None, 120)
        self.assertEqual((budget.count, network.call_count), (0, 0))
        self.assertEqual(budget.stop_reason, 'time_budget')

    def test_repeated_transport_failure_stops_without_extra_attempts(self):
        network = Mock(side_effect=ConnectionResetError('测试连接中断'))
        budget = RequestBudget(10, 60, network, max_network_failures=2)
        for _ in range(2):
            with self.assertRaises(ConnectionResetError):
                budget(None, 30)
        with self.assertRaises(ExperimentBudget) as ctx:
            budget(None, 30)
        self.assertEqual(ctx.exception.category, 'network')
        self.assertEqual((budget.count, network.call_count), (2, 2))
        self.assertEqual(budget.stop_reason, 'network_outage')

    def test_success_resets_consecutive_failures(self):
        network = Mock(side_effect=[ConnectionResetError(), io.BytesIO(b'ok'), ConnectionResetError()])
        budget = RequestBudget(10, 60, network, max_network_failures=2)
        with self.assertRaises(ConnectionResetError):
            budget(None, 30)
        self.assertEqual(budget(None, 30).read(), b'ok')
        with self.assertRaises(ConnectionResetError):
            budget(None, 30)
        self.assertEqual(budget.consecutive_network_failures, 1)
        self.assertIsNone(budget.stop_reason)

    def test_budget_stop_and_expand_budget_resumes_same_run_without_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "novel.txt"
            text.write_text("第一章 初遇\n林舟在江边。\n第二章 重逢\n林舟回到城里。\n", encoding="utf-8")
            cfg = root / ".env"
            cfg.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            db_path = root / "db.sqlite3"
            out_dir = root / "out"

            opener, seen_payloads = _make_fake_http_opener()

            class FakeHttpModels(Models):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs, opener=opener)

            with patch("entitytrace.pipeline.Models", FakeHttpModels):
                # 第一次：请求预算设为 2，在完成前触发预算停止，退出码为 2
                code1 = cli_main([
                    "run", str(text), "--db", str(db_path), "--output", str(out_dir),
                    "--config", str(cfg), "--max-requests", "2", "--max-seconds", "300",
                ])
                self.assertEqual(code1, 2)
                self.assertEqual(len(seen_payloads), 2)

                subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
                self.assertEqual(len(subdirs), 1)
                first_output = subdirs[0]
                manifest1 = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
                result1 = json.loads((first_output / "result.json").read_text(encoding="utf-8"))
                run_id_1 = manifest1["run_id"]
                self.assertEqual(result1["status"], "incomplete")
                self.assertNotIn("max_requests", manifest1["config"])
                self.assertNotIn("max_seconds", manifest1["config"])
                self.assertIsInstance(manifest1["state"]["last_error"], dict)
                self.assertEqual(manifest1["state"]["last_error"]["category"], "budget")
                self.assertEqual(manifest1["state"]["last_error"]["stop_reason"], "request_budget")

                # 第二次：扩大预算（默认不限）续跑同一命令，复用同一 run_id 且已完成请求零重复
                code2 = cli_main([
                    "run", str(text), "--db", str(db_path), "--output", str(out_dir),
                    "--config", str(cfg), "--max-requests", "100", "--max-seconds", "600",
                ])
                self.assertEqual(code2, 0)
                manifest2 = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
                result2 = json.loads((first_output / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest2["run_id"], run_id_1)
                self.assertIn(result2["status"], {"completed", "completed_with_review"})
                self.assertIsNone(manifest2["state"]["last_error"])
                # 验证每次实际发往 opener 的 HTTP 请求载荷互不重复
                self.assertEqual(len(seen_payloads), len(set(seen_payloads)))

                # 第三次：不带预算参数再次调用，run_id 仍完全一致且零新增请求
                before_total = len(seen_payloads)
                rid3, out3, complete3 = run_pipeline(text, db_path, out_dir, cfg, log=lambda _: None)
                self.assertEqual(rid3, run_id_1)
                self.assertEqual(out3, first_output)
                self.assertTrue(complete3)
                self.assertEqual(len(seen_payloads), before_total)

    def test_network_outage_stops_pipeline_and_records_network_category(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # 4 章产生 4 个发现任务；章节扫描成功后，发现阶段连续 3 次网络失败即熔断停止后续任务
            chapters_text = "\n".join(f"第{i}章 标题\n林舟在第{i}章出场。" for i in range(1, 5)) + "\n"
            text = root / "novel.txt"
            text.write_text(chapters_text, encoding="utf-8")
            cfg = root / ".env"
            cfg.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            db_path = root / "db.sqlite3"
            out_dir = root / "out"

            base_opener, _ = _make_fake_http_opener()
            discovery_attempts = 0
            lock = threading.Lock()

            def discovery_outage_opener(request, timeout=None):
                nonlocal discovery_attempts
                if request.full_url.endswith("/chat/completions"):
                    with lock:
                        discovery_attempts += 1
                    raise ConnectionResetError("模拟发现阶段断网")
                return base_opener(request, timeout=timeout)

            class OutageModels(Models):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs, attempts=2, opener=discovery_outage_opener)

            with patch("entitytrace.pipeline.Models", OutageModels), patch("entitytrace.models.time.sleep"):
                rid, output, complete = run_pipeline(
                    text, db_path, out_dir, cfg,
                    discovery_concurrency=1, max_network_failures=3, log=lambda _: None,
                )
            self.assertFalse(complete)
            # 4 个任务 × 2 次重试原本会发 8 次请求，连续 3 次断网后立即熔断停止
            self.assertEqual(discovery_attempts, 3)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], rid)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(manifest["state"]["last_error"]["category"], "network")
            self.assertEqual(manifest["state"]["last_error"]["stop_reason"], "network_outage")


if __name__ == '__main__':
    unittest.main()
