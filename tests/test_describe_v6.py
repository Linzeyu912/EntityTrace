from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.cli import main as cli_main
from entitytrace.config import Service
from entitytrace.describe import (
    D1_PROMPT,
    DESCRIBE_VERSION,
    EVENT_TYPE_DEFINITIONS,
    EVENT_VERIFY_PROMPT,
    EVENT_VERIFY_PROMPT_VERSION,
    FACETS,
    VERIFY_PROMPT,
    VERIFY_PROMPT_VERSION,
    build_entity_stages,
    build_event_verify_tasks,
    check_event_verify_output,
    decide_event_verdict,
    export_describe_run,
    run_describe,
)
from entitytrace.models import ModelError, Models
from entitytrace.roster import VERSION as ROSTER_VERSION
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import dumps
from tests.test_describe import FakeModels, FixtureCase, SERVICES, _find_all, _span


def _event_echo(items: list[dict], overrides: dict[str, dict] | None = None) -> dict:
    overrides = overrides or {}
    verdicts = []
    for item in items:
        short_id = item["id"]
        spec = overrides.get(short_id, {})
        verdict = spec.get("verdict", "keep")
        verdicts.append({
            "id": short_id,
            "value": spec.get("value", item["value"]),
            "verdict": verdict,
            "reason": spec.get("reason", None if verdict == "keep" else "not_state_event"),
            "stage_split": spec.get("stage_split", False),
        })
    return {"verdicts": verdicts}


class _FakeHttpResponse:
    def __init__(self, body: dict):
        self._bytes = dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _assert_stage_invariants(testcase: unittest.TestCase, entity: dict,
                             range_bounds: tuple[int, int]) -> None:
    stages = entity["stages"]
    facts = entity["facts"]
    kind = entity["kind"]
    testcase.assertGreaterEqual(len(stages), 1)
    testcase.assertEqual([st["stage_index"] for st in stages], list(range(1, len(stages) + 1)))
    testcase.assertEqual(stages[0]["start"], range_bounds[0])
    testcase.assertEqual(stages[-1]["end"], max(range_bounds[1], stages[-1]["start"],
                                                max((f["end"] for f in facts), default=0)))
    for prev, curr in zip(stages, stages[1:]):
        testcase.assertEqual(prev["end"], curr["start"])
        testcase.assertLessEqual(prev["start"], prev["end"])
    if facts:
        for st in stages:
            testcase.assertGreaterEqual(len(st["facts"]), 1)
            for f in st["facts"]:
                testcase.assertGreaterEqual(f["start"], st["start"])
                testcase.assertLess(f["start"], st["end"])
            covered = {f["facet"] for f in st["facts"]}
            testcase.assertEqual(st["gaps"], [fc for fc in FACETS[kind] if fc not in covered])
        flat_ids = [f["id"] for st in stages for f in st["facts"]]
        testcase.assertEqual(flat_ids, [f["id"] for f in facts])
    else:
        testcase.assertEqual(len(stages), 1)
        testcase.assertEqual(stages[0]["facts"], [])
        testcase.assertEqual(stages[0]["gaps"], list(FACETS[kind]))


class EventVerifyBatchAndValidationTests(unittest.TestCase):
    """T1 & T2: 事件复核分批、第 0 遍字节等价与回抄防错位校验。"""

    def _make_events(self, count: int) -> tuple[str, list[dict], list[dict], dict[str, str]]:
        body = "".join(f"沈墨第{i:02d}段换衣受创。" for i in range(count))
        text = "第一章 试剑\n" + body
        chapters = [{
            "id": "ch1", "title": "第一章 试剑", "kind": "chapter",
            "start": 0, "end": len(text), "body_start": 7, "chapter_set_version": "chs-1",
        }]
        events = []
        for i in range(count):
            piece = f"沈墨第{i:02d}段换衣受创。"
            pos = text.index(piece)
            events.append({
                "id": f"secret-ev-{i:02d}",
                "entity_id": "e_shen",
                "event_type": "costume_change" if i % 2 == 0 else "injury",
                "splits_entity": i % 3 == 0,
                "value": f"换衣{i:02d}",
                "quote": piece,
                "start": pos,
                "end": pos + len(piece),
                "chapter_id": "ch1",
                "support": 2,
            })
        return text, chapters, list(reversed(events)), {"e_shen": "person"}

    def test_t1_staggered_batches_and_pass0_byte_equivalence(self):
        text, chapters, events, kinds = self._make_events(25)
        names = {"e_shen": ["沈墨", "沈道人"]}
        tasks_1 = build_event_verify_tasks(
            "did-one", events, kinds, names, text, chapters, passes=1,
        )
        tasks_3 = build_event_verify_tasks(
            "did-three", events, kinds, names, text, chapters, passes=3,
        )
        pass0 = [t for t in tasks_3 if t["pass"] == 0]
        pass1 = [t for t in tasks_3 if t["pass"] == 1]
        pass2 = [t for t in tasks_3 if t["pass"] == 2]

        self.assertEqual([len(t["event_ids"]) for t in pass0], [12, 12, 1])
        self.assertEqual([len(t["event_ids"]) for t in pass1], [8, 12, 5])
        self.assertEqual([len(t["event_ids"]) for t in pass2], [4, 12, 9])
        self.assertEqual(len(tasks_1), len(pass0))

        for t1, t3 in zip(tasks_1, pass0):
            self.assertEqual(t1["prompt"], EVENT_VERIFY_PROMPT)
            self.assertEqual(t3["prompt"], EVENT_VERIFY_PROMPT)
            self.assertEqual(dumps(t1["state"]), dumps(t3["state"]))
            self.assertEqual(t1["event_ids"], t3["event_ids"])

        for task in tasks_3:
            raw_state = dumps(task["state"])
            self.assertNotIn("did-three", raw_state)
            self.assertNotIn("pass", raw_state)
            for eid in task["event_ids"]:
                self.assertNotIn(eid, raw_state)
            for item in task["state"]["items"]:
                self.assertEqual(
                    item["event_type_definition"],
                    EVENT_TYPE_DEFINITIONS[item["event_type"]],
                )

    def test_t2_check_event_verify_output_echo_and_boolean_guards(self):
        items = [
            {"id": "E1", "value": "改穿白袍"},
            {"id": "E2", "value": "左肩中剑"},
        ]
        expected = {"E1": "改穿白袍", "E2": "左肩中剑"}
        check_event_verify_output({
            "verdicts": [
                {"id": "E2", "value": "左肩中剑", "verdict": "keep",
                 "reason": None, "stage_split": False},
                {"id": "E1", "value": "改穿白袍", "verdict": "keep",
                 "reason": None, "stage_split": True},
            ]
        }, expected)

        with self.assertRaises(ModelError) as ctx:
            check_event_verify_output(_event_echo(items, {"E1": {"value": "改穿青袍"}}), expected)
        self.assertIn("value-echo-mismatch", str(ctx.exception))

        # drop 时 stage_split 必须为 False
        with self.assertRaises(ModelError):
            check_event_verify_output(_event_echo(items, {
                "E1": {"verdict": "drop", "reason": "not_state_event", "stage_split": True},
            }), expected)

        # stage_split 非布尔值（0 / 1 / 字符串 / None）一律拦截
        for bad_bool in (0, 1, "false", "true", None):
            with self.assertRaises(ModelError):
                check_event_verify_output(_event_echo(items, {
                    "E1": {"verdict": "keep", "reason": None, "stage_split": bad_bool},
                }), expected)

        # keep 不得带 reason，drop 必须带合法 reason
        with self.assertRaises(ModelError):
            check_event_verify_output(_event_echo(items, {
                "E1": {"verdict": "keep", "reason": "not_state_event", "stage_split": False},
            }), expected)
        with self.assertRaises(ModelError):
            check_event_verify_output(_event_echo(items, {
                "E1": {"verdict": "drop", "reason": "wrong_facet", "stage_split": False},
            }), expected)


class EventVerdictVotingTests(unittest.TestCase):
    """T3: decide_event_verdict 单遍透传与三遍表决规则。"""

    def test_t3_single_pass_and_three_pass_voting(self):
        self.assertEqual(
            decide_event_verdict([(0, "keep", None, True)], 1),
            ("keep", None, True),
        )
        self.assertEqual(
            decide_event_verdict([(0, "drop", "wrong_entity", False)], 1),
            ("drop", "wrong_entity", False),
        )
        self.assertEqual(decide_event_verdict([], 1), ("unverified", "invalid-output", False))

        # 三遍 drop >= 2：多数原因胜出
        self.assertEqual(
            decide_event_verdict([
                (0, "drop", "wrong_event_type", False),
                (1, "drop", "wrong_event_type", False),
                (2, "keep", None, True),
            ], 3),
            ("drop", "wrong_event_type", False),
        )

        # 三遍 drop 原因平票：按 not_state_event -> wrong_entity -> wrong_event_type -> unsupported_value
        self.assertEqual(
            decide_event_verdict([
                (0, "drop", "unsupported_value", False),
                (1, "drop", "wrong_entity", False),
                (2, "keep", None, True),
            ], 3),
            ("drop", "wrong_entity", False),
        )
        self.assertEqual(
            decide_event_verdict([
                (0, "drop", "wrong_entity", False),
                (1, "drop", "not_state_event", False),
            ], 3),
            ("drop", "not_state_event", False),
        )

        # 3 keep：stage_split 两票过半
        self.assertEqual(
            decide_event_verdict([
                (0, "keep", None, True),
                (1, "keep", None, False),
                (2, "keep", None, True),
            ], 3),
            ("keep", None, True),
        )
        self.assertEqual(
            decide_event_verdict([
                (0, "keep", None, True),
                (1, "keep", None, False),
                (2, "keep", None, False),
            ], 3),
            ("keep", None, False),
        )

        # 2 keep + 1 drop：仅当两张 keep 都投 stage_split=True 才切分
        self.assertEqual(
            decide_event_verdict([
                (0, "keep", None, True),
                (1, "keep", None, False),
                (2, "drop", "not_state_event", False),
            ], 3),
            ("keep", None, False),
        )
        self.assertEqual(
            decide_event_verdict([
                (0, "keep", None, True),
                (1, "keep", None, True),
                (2, "drop", "not_state_event", False),
            ], 3),
            ("keep", None, True),
        )

        # 有效票 < 2 -> unverified
        self.assertEqual(
            decide_event_verdict([(0, "keep", None, True)], 3),
            ("unverified", "invalid-output", False),
        )


class EntityStageSplittingTests(unittest.TestCase):
    """T4: build_entity_stages 五项核心不变量。"""

    def setUp(self):
        self.chapters = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 100},
            {"id": "ch2", "kind": "chapter", "start": 100, "end": 200},
            {"id": "ch3", "kind": "chapter", "start": 200, "end": 300},
        ]
        self.bounds = (0, 300)

    def _fact(self, fid: str, facet: str, value: str, start: int, ch: str) -> dict:
        return {
            "id": fid, "facet": facet, "value": value, "quote": value,
            "start": start, "end": start + len(value), "chapter_id": ch,
            "source": "narration", "holder_entity_id": None, "support": 2,
        }

    def _event(self, eid: str, etype: str, value: str, start: int, ch: str, split: bool) -> dict:
        return {
            "id": eid, "event_type": etype, "splits_entity": split, "stage_split": split,
            "value": value, "quote": value, "start": start, "end": start + len(value),
            "chapter_id": ch, "support": 2,
        }

    def test_t4_1_no_split_events_keeps_single_stage(self):
        facts = [
            self._fact("f1", "build", "身长八尺", 10, "ch1"),
            self._fact("f2", "costume", "青衫", 120, "ch2"),
        ]
        events = [self._event("ev1", "injury", "左臂轻伤", 50, "ch1", False)]
        stages = build_entity_stages("person", facts, events, self.chapters, self.bounds)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["from_chapter"], 1)
        self.assertEqual(stages[0]["to_chapter"], 2)
        self.assertEqual(stages[0]["trigger_events"], [])
        _assert_stage_invariants(self, {"kind": "person", "facts": facts, "stages": stages},
                                 self.bounds)

    def test_t4_2_cross_chapter_and_intra_chapter_owner_change(self):
        # 同一章（ch2: 100..200）前半归甲（pos 110），pos 150 易主归乙，后半归乙（pos 170）
        facts = [
            self._fact("f1", "shape", "金砖", 20, "ch1"),
            self._fact("f2", "holder", "太乙真人", 110, "ch2"),
            self._fact("f3", "holder", "哪吒", 170, "ch2"),
        ]
        events = [self._event("ev1", "owner_change", "赐予哪吒", 150, "ch2", True)]
        stages = build_entity_stages("object", facts, events, self.chapters, self.bounds)
        self.assertEqual(len(stages), 2)
        self.assertEqual([f["id"] for f in stages[0]["facts"]], ["f1", "f2"])
        self.assertEqual([f["id"] for f in stages[1]["facts"]], ["f3"])
        self.assertEqual(stages[0]["from_chapter"], 1)
        self.assertEqual(stages[0]["to_chapter"], 2)
        self.assertEqual(stages[1]["from_chapter"], 2)
        self.assertEqual(stages[1]["to_chapter"], 2)
        self.assertIn("shape", stages[1]["gaps"])
        self.assertNotIn("shape", stages[0]["gaps"])
        _assert_stage_invariants(self, {"kind": "object", "facts": facts, "stages": stages},
                                 self.bounds)

    def test_t4_3_same_chapter_consecutive_splits_without_facts_coalesce(self):
        facts = [
            self._fact("f1", "face", "粉面红唇", 20, "ch1"),
            self._fact("f2", "feature", "莲花化身", 220, "ch3"),
        ]
        events = [
            self._event("ev1", "death", "割肉剔骨", 120, "ch2", True),
            self._event("ev2", "transform", "莲花化身重塑", 160, "ch2", True),
        ]
        stages = build_entity_stages("person", facts, events, self.chapters, self.bounds)
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[1]["start"], 120)
        self.assertEqual(len(stages[1]["trigger_events"]), 2)
        self.assertEqual([e["value"] for e in stages[1]["trigger_events"]],
                         ["割肉剔骨", "莲花化身重塑"])
        _assert_stage_invariants(self, {"kind": "person", "facts": facts, "stages": stages},
                                 self.bounds)

    def test_t4_4_non_empty_guard_drops_leading_trailing_and_empty_gap_cuts(self):
        facts = [
            self._fact("f1", "costume", "黄金甲", 50, "ch1"),
            self._fact("f2", "costume", "白袍", 250, "ch3"),
        ]
        events = [
            self._event("ev0", "costume_change", "初赐战袍", 10, "ch1", True),
            self._event("ev1", "costume_change", "卸甲换衣", 90, "ch1", True),
            self._event("ev2", "transform", "现出法相", 150, "ch2", True),
            self._event("ev3", "death", "阵亡上榜", 280, "ch3", True),
        ]
        stages = build_entity_stages("person", facts, events, self.chapters, self.bounds)
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["facts"][0]["id"], "f1")
        self.assertEqual(stages[1]["facts"][0]["id"], "f2")
        self.assertEqual(stages[1]["start"], 90)
        _assert_stage_invariants(self, {"kind": "person", "facts": facts, "stages": stages},
                                 self.bounds)

    def test_t4_5_zero_fact_entity_emits_single_stage_with_full_gaps(self):
        events = [self._event("ev1", "death", "陨落", 140, "ch2", True)]
        stages = build_entity_stages("ability", [], events, self.chapters, self.bounds)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["facts"], [])
        self.assertEqual(stages[0]["gaps"], list(FACETS["ability"]))
        self.assertIsNone(stages[0]["first_fact_chapter_id"])
        self.assertIsNone(stages[0]["last_fact_chapter_id"])
        _assert_stage_invariants(self, {"kind": "ability", "facts": [], "stages": stages},
                                 self.bounds)


class CacheUpgradeAndEchoRetryTests(FixtureCase):
    """T5 & T2 集成：已有 D1 与事实复核缓存升级至 describe-v6 时仅事件复核发请求。"""

    def test_t5_cached_d1_and_fact_verify_make_zero_network_calls_on_v6(self):
        service = Service(
            name="discovery", url="http://127.0.0.1:9/v1", model="example/model",
            key="example", provider="openai",
        )
        services = {"discovery": service, "decision": service}
        transport_stages: list[str] = []

        def fake_opener(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            sys_prompt = payload["messages"][0]["content"]
            user_state = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                stage = "describe-d1"
                text = user_state.get("text", "")
                facts, events = [], []
                if "穿着一领青衫" in text:
                    facts.append({
                        "entity_id": "e_shen", "facet": "costume", "value": "青衫",
                        "quote": "穿着一领青衫", "source": "narration",
                    })
                if "换了白衣" in text:
                    facts.append({
                        "entity_id": "e_shen", "facet": "costume", "value": "白衣",
                        "quote": "换了白衣", "source": "narration",
                    })
                    events.append({
                        "entity_id": "e_shen", "event_type": "costume_change",
                        "splits_entity": True, "value": "换了白衣", "quote": "换了白衣",
                    })
                content = dumps({"facts": facts, "events": events})
            elif sys_prompt == VERIFY_PROMPT:
                stage = "describe-verify"
                content = dumps({
                    "verdicts": [
                        {"id": it["id"], "value": it["value"],
                         "verdict": "keep", "reason": None}
                        for it in user_state["items"]
                    ]
                })
            elif sys_prompt == EVENT_VERIFY_PROMPT:
                stage = "describe-event-verify"
                content = dumps(_event_echo(
                    user_state["items"],
                    {"E1": {"verdict": "keep", "reason": None, "stage_split": True}},
                ))
            else:
                raise AssertionError("未知 prompt")
            transport_stages.append(stage)
            return _FakeHttpResponse({
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        store = Store(self.db_path)
        try:
            # 1. 模拟旧版本已把 D1 与 describe-verify 写入 request_cache
            with patch("entitytrace.describe.DESCRIBE_VERSION", "describe-v5"):
                m_old = Models(store, self.run_id, "inv-old", services, opener=fake_opener)
                run_describe(
                    store, self.run_id, services=services, models=m_old,
                    top_k=10, kinds="person", views=3, entities_per_request=4, verify=3,
                )
            for row in store.rows("SELECT id, request FROM request_cache"):
                req = json.loads(row["request"])
                if req["messages"][0]["content"] == EVENT_VERIFY_PROMPT:
                    store.execute("DELETE FROM request_cache WHERE id=?",
                                  (row["id"],))

            transport_stages.clear()

            # 2. 正式运行 describe-v6 (--verify 3)：D1 与 describe-verify 必须 0 网络请求
            m_v6_c = Models(store, self.run_id, "inv-v6-c", services, opener=fake_opener)
            res_c = run_describe(
                store, self.run_id, services=services, models=m_v6_c,
                top_k=10, kinds="person", views=3, entities_per_request=4, verify=3,
            )
            self.assertEqual(transport_stages.count("describe-d1"), 0)
            self.assertEqual(transport_stages.count("describe-verify"), 0)
            self.assertEqual(transport_stages.count("describe-event-verify"), 3)
            self.assertEqual(res_c["counts"]["multi_stage_entities"], 1)
            self.assertEqual(res_c["counts"]["stages"], 2)

            # 3. 同库重跑 --verify 1 与 --verify 0：全部 0 网络请求
            transport_stages.clear()
            m_v6_b = Models(store, self.run_id, "inv-v6-b", services, opener=fake_opener)
            res_b = run_describe(
                store, self.run_id, services=services, models=m_v6_b,
                top_k=10, kinds="person", views=3, entities_per_request=4, verify=1,
            )
            self.assertEqual(transport_stages, [])
            self.assertEqual(res_b["counts"]["stages"], 2)

            m_v6_v0 = Models(store, self.run_id, "inv-v6-v0", services, opener=fake_opener)
            res_v0 = run_describe(
                store, self.run_id, services=services, models=m_v6_v0,
                top_k=10, kinds="person", views=3, entities_per_request=4, verify=0,
            )
            self.assertEqual(transport_stages, [])
            self.assertEqual(res_v0["document"]["event_verification"], {"enabled": False})
        finally:
            store.close()

    def test_t2_event_echo_mismatch_retry_is_counted_in_summary(self):
        service = Service(
            name="discovery", url="http://127.0.0.1:9/v1", model="example/model",
            key="example", provider="openai",
        )
        services = {"discovery": service, "decision": service}
        attempts = {"event": 0}

        def fake_opener(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            sys_prompt = payload["messages"][0]["content"]
            user_state = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                text = user_state.get("text", "")
                events = []
                if "换了白衣" in text:
                    events.append({
                        "entity_id": "e_shen", "event_type": "costume_change",
                        "splits_entity": True, "value": "换了白衣", "quote": "换了白衣",
                    })
                content = dumps({"facts": [], "events": events})
            elif sys_prompt == VERIFY_PROMPT:
                content = dumps({"verdicts": []})
            elif sys_prompt == EVENT_VERIFY_PROMPT:
                attempts["event"] += 1
                if attempts["event"] == 1:
                    content = dumps(_event_echo(
                        user_state["items"], {"E1": {"value": "错抄内容"}},
                    ))
                else:
                    content = dumps(_event_echo(
                        user_state["items"], {"E1": {"stage_split": False}},
                    ))
            else:
                raise AssertionError(sys_prompt)
            return _FakeHttpResponse({
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            })

        store = Store(self.db_path)
        try:
            with patch("entitytrace.models.urllib.request.urlopen", side_effect=fake_opener), \
                 patch("entitytrace.models.time.sleep", return_value=None):
                res = run_describe(
                    store, self.run_id, services=services, models=None,
                    top_k=10, kinds="person", views=1, entities_per_request=0, verify=1,
                )
            ev_sum = res["document"]["event_verification"]
            self.assertEqual(ev_sum["echo_mismatch_retries"], 1)
            self.assertEqual(ev_sum["kept"], 1)
        finally:
            store.close()


class ChapterRangeAndLegacyCompatibilityTests(unittest.TestCase):
    """T6, T7 & T8: 章节范围筛选、CLI 同义参数、旧库兼容重导出与禁词检查。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "range.sqlite3"
        self.config_path = self.root / "config.env"
        self.config_path.write_text("DEEPSEEK_API_KEY=test_key\n", encoding="utf-8")
        self.run_id = "run-range"
        self._seed_three_chapter_book()

    def tearDown(self):
        self.temp.cleanup()

    def _seed_three_chapter_book(self):
        self.text = (
            "引子 楔子\n沈墨本名初见于楔子。\n"
            "第一章 初出茅庐\n沈道人身长八尺，持铁剑下山。\n"
            "第二章 荒原试炼\n林川面如冠玉，在荒原独行。\n"
            "第三章 绝顶重逢\n林川换穿紫衣，林川再登绝顶。\n"
        )
        txt_path = self.root / "book.txt"
        txt_path.write_bytes(self.text.encode("utf-8"))
        store = Store(self.db_path)
        try:
            source = import_text(store, txt_path)
            _, chapters = build_chapters(source, [
                _span(self.text, "引子 楔子", "front_matter"),
                _span(self.text, "第一章 初出茅庐", "chapter"),
                _span(self.text, "第二章 荒原试炼", "chapter"),
                _span(self.text, "第三章 绝顶重逢", "chapter"),
            ])
            config = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {
                    "complete": True,
                    "chapter_set_version": chapters[0]["chapter_set_version"],
                },
                "discovery": {"complete": True},
            }
            store.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                (self.run_id, source["id"], dumps(config), dumps(state), "now", "now"),
            )
            for chapter in chapters:
                store.put(self.run_id, "chapter", chapter)

            roster_spec = [
                ("e_shen", "沈墨", "person", ["沈墨", "沈道人"]),
                ("e_lin", "林川", "person", ["林川"]),
            ]
            entities, rows = [], []
            for eid, canon_name, kind, surfaces in roster_spec:
                mentions = []
                ch_stats: dict[str, dict] = {}
                surface_counts: dict[str, int] = {}
                for sf in surfaces:
                    for pos in _find_all(self.text, sf):
                        ch = next(c for c in chapters if c["start"] <= pos < c["end"])
                        mentions.append({
                            "id": f"m_{eid}_{pos}", "chapter_id": ch["id"],
                            "start": pos, "end": pos + len(sf),
                            "text": sf, "link_method": "chapter_local",
                        })
                        st = ch_stats.setdefault(ch["id"], {"count": 0, "min_start": pos})
                        st["count"] += 1
                        st["min_start"] = min(st["min_start"], pos)
                        surface_counts[sf] = surface_counts.get(sf, 0) + 1
                mentions.sort(key=lambda m: m["start"])
                entities.append({
                    "id": eid, "name": canon_name, "kind": kind,
                    "identity_status": "linked_by_roster",
                    "mention_count": len(mentions),
                    "first_chapter_id": mentions[0]["chapter_id"],
                    "surface_counts": surface_counts,
                    "chapter_stats": ch_stats,
                })
                header = {
                    "id": eid, "name": canon_name, "names": surfaces, "aliases": [],
                    "kind": kind, "identity_status": "linked_by_roster",
                    "mention_count": len(mentions),
                    "first_chapter_id": mentions[0]["chapter_id"],
                    "link_methods": {"chapter_local": len(mentions)},
                }
                rows.append((eid, dumps(header), dumps(mentions)))
            meta = {
                "result_revision": "rev-r1", "status": "completed",
                "source_id": source["id"], "chapters": chapters,
                "counts": {}, "review_counts": {}, "snapshot_version": 2,
            }
            store.save_query_snapshot(
                self.run_id, store.query_fingerprint(self.run_id), meta, entities, [], rows,
            )
        finally:
            store.close()

    def test_t6_chapter_range_synonyms_ranking_and_full_book_names(self):
        captured_d1_states: list[dict] = []

        class RangeFakeModels:
            def __init__(self, store, run_id, invocation, services, *args, **kwargs):
                self.services = services

            def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
                if stage == "describe-d1":
                    captured_d1_states.append(state)
                    text = state.get("text", "")
                    facts = []
                    if "身长八尺" in text:
                        facts.append({
                            "entity_id": "e_shen", "facet": "build",
                            "value": "身长八尺", "quote": "身长八尺", "source": "narration",
                        })
                    if "面如冠玉" in text:
                        facts.append({
                            "entity_id": "e_lin", "facet": "face",
                            "value": "面如冠玉", "quote": "面如冠玉", "source": "narration",
                        })
                    val = {"facts": facts, "events": []}
                else:
                    val = {"verdicts": []}
                validate(val)
                return val, "req-range"

        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith(("DEEPSEEK_", "TYPESAFE_", "ENTITYTRACE_"))
                     and k != "GENERATOR_PROVIDER"}

        out1 = self.root / "r1.json"
        out2 = self.root / "r2.json"
        buf1, buf2 = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("entitytrace.describe.Models", RangeFakeModels), \
             contextlib.redirect_stdout(buf1), contextlib.redirect_stderr(io.StringIO()):
            rc1 = cli_main([
                "describe", self.run_id, "--db", str(self.db_path),
                "--from-chapter", "2", "--to-chapter", "3",
                "--top-k", "1", "--kinds", "person",
                "--views", "1", "--entities-per-request", "0", "--verify", "0",
                "--config", str(self.config_path), "--out", str(out1),
            ])
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("entitytrace.describe.Models", RangeFakeModels), \
             contextlib.redirect_stdout(buf2), contextlib.redirect_stderr(io.StringIO()):
            rc2 = cli_main([
                "describe", self.run_id, "--db", str(self.db_path),
                "--start-chapter", "2", "--end-chapter", "3",
                "--top-k", "1", "--kinds", "person",
                "--views", "1", "--entities-per-request", "0", "--verify", "0",
                "--config", str(self.config_path), "--out", str(out2),
            ])
        self.assertEqual(rc1, 0)
        self.assertEqual(rc2, 0)
        rec1 = json.loads(buf1.getvalue().strip())
        rec2 = json.loads(buf2.getvalue().strip())
        self.assertEqual(rec1["describe_id"], rec2["describe_id"])
        self.assertEqual(rec1["chapter_range"], {
            "from_chapter": 2, "to_chapter": 3, "chapter_count": 2, "filtered": True,
        })
        doc1 = json.loads(out1.read_text(encoding="utf-8"))
        self.assertEqual([e["entity_id"] for e in doc1["entities"]], ["e_lin"])

        # 只扫第 1 章时，e_shen 在楔子才出现的本名“沈墨”仍保留在复核 names 中
        captured_d1_states.clear()
        captured_verify_states: list[dict] = []

        def ch1_handler(stg, st, idx):
            if stg == "describe-d1":
                captured_d1_states.append(st)
                return {
                    "facts": [{
                        "entity_id": "e_shen", "facet": "build",
                        "value": "身长八尺", "quote": "身长八尺", "source": "narration",
                    }],
                    "events": [],
                }
            if stg == "describe-verify":
                captured_verify_states.append(st)
                return {
                    "verdicts": [{
                        "id": it["id"], "value": it["value"],
                        "verdict": "keep", "reason": None,
                    } for it in st["items"]],
                }
            return {"verdicts": []}

        store = Store(self.db_path)
        try:
            res_ch1 = run_describe(
                store, self.run_id, services=SERVICES,
                models=FakeModels(SERVICES, ch1_handler),
                top_k=1, kinds="person", views=1, entities_per_request=0, verify=1,
                from_chapter=1, to_chapter=1,
            )
            self.assertEqual(res_ch1["chapter_range"], {
                "from_chapter": 1, "to_chapter": 1, "chapter_count": 1, "filtered": True,
            })
            self.assertEqual(len(captured_d1_states), 1)
            self.assertNotIn("引子", captured_d1_states[0]["text"])
            self.assertEqual(len(captured_verify_states), 1)
            self.assertIn("沈墨", captured_verify_states[0]["items"][0]["names"])

            res_from_only = run_describe(
                store, self.run_id, services=SERVICES,
                models=FakeModels(SERVICES, lambda *_: {"facts": [], "events": []}),
                top_k=2, kinds="person", views=1, entities_per_request=0, verify=0,
                from_chapter=2,
            )
            self.assertEqual(res_from_only["chapter_range"], {
                "from_chapter": 2, "to_chapter": 3, "chapter_count": 2, "filtered": True,
            })

            with self.assertRaises(ValueError):
                run_describe(
                    store, self.run_id, services=SERVICES, models=None,
                    from_chapter=True,
                )
        finally:
            store.close()

        for bad_flags in (
            ["--from-chapter", "0"],
            ["--to-chapter", "4"],
            ["--start-chapter", "3", "--end-chapter", "2"],
        ):
            ebuf = io.StringIO()
            with patch.dict(os.environ, clean_env, clear=True), \
                 patch("entitytrace.describe.Models", RangeFakeModels), \
                 contextlib.redirect_stdout(ebuf), contextlib.redirect_stderr(io.StringIO()):
                rc_bad = cli_main([
                    "describe", self.run_id, "--db", str(self.db_path),
                    "--config", str(self.config_path),
                ] + bad_flags)
            self.assertEqual(rc_bad, 1)
            err_payload = json.loads(ebuf.getvalue().strip().splitlines()[-1])
            self.assertEqual(err_payload["error"]["category"], "input")

    def test_t7_legacy_export_describe_run_without_event_verify_prompt_version(self):
        legacy_db = self.root / "legacy_v5.sqlite3"
        store = Store(legacy_db)
        try:
            store.execute(
                "INSERT INTO describe_runs VALUES(?,?,?,?,?,?)",
                ("did-v5", "run-v5", dumps({
                    "describe_version": "describe-v5", "top_k": 20,
                    "kinds": ["person"], "entity_ids": ["e1"], "run_id": "run-v5",
                    "views": 3, "entities_per_request": 4, "verify": 1,
                    "verify_prompt_version": VERIFY_PROMPT_VERSION, "verify_batch": 12,
                    "generator": {"provider": "example", "model": "example/model"},
                }), dumps({"status": "completed"}), "t0", "t0"),
            )
            for idx in range(2):
                tid = f"t-{idx}"
                store.execute(
                    "INSERT INTO describe_tasks VALUES(?,?,?,?,?,?,?,?)",
                    ("did-v5", tid, None, 0, 20, "completed",
                     dumps({"view": idx + 1}), None),
                )
                store.execute(
                    "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("did-v5", f"f-{idx}", "e1", "build", "高大", "身长八尺",
                     2, 6, "ch1", "narration", None, tid),
                )
                store.execute(
                    "INSERT INTO state_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    ("did-v5", f"ev-{idx}", "e1", "costume_change", 1, "换白衣",
                     "换了白衣", 10, 14, "ch1", tid),
                )
            store.execute(
                "INSERT INTO fact_verdicts VALUES(?,?,?,?,?)",
                ("did-v5", "f-0", "keep", None, "t1"),
            )
            doc = export_describe_run(store, "did-v5")
        finally:
            store.close()

        self.assertEqual(doc["event_verification"], {"enabled": False})
        self.assertEqual(doc["verification"]["enabled"], True)
        ent = doc["entities"][0]
        self.assertEqual(len(ent["facts"]), 1)
        self.assertEqual(len(ent["state_events"]), 1)
        self.assertEqual(len(ent["stages"]), 1)

    def test_t8_tracked_files_have_zero_forbidden_terms(self):
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True,
            encoding="utf-8", check=True,
        )
        terms = [
            "Agent" + "Collab",
            "\u534f\u4f5c\u673a\u5236",
            "\u4e09\u7ea7\u8eab\u4efd",
            r"\bL" + r"1\b",
            r"\bL" + r"2\b",
            r"\bL" + r"3\b",
            "\u5927\u8111",
            "\u6267\u884c\u603b\u7ba1",
            "\u4e3b\u63a7\u677f",
            "Mag" + "pie",
            "Work" + "Buddy",
            "De" + "vin",
            "swe" + "-2",
            r"\bd" + r"sh\b",
        ]
        pattern = re.compile("|".join(terms))
        tracked = set(proc.stdout.splitlines()) | {"tests/test_describe_v6.py"}
        hits = []
        for rel in sorted(tracked):
            if rel.startswith("test/text-library/"):
                continue
            p = repo_root / rel
            if not p.is_file() or p.suffix.lower() in {".png", ".jpg", ".sqlite3", ".pack"}:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            m = pattern.search(text)
            if m:
                hits.append(f"{rel}: {m.group(0)}")
        self.assertEqual(hits, [], f"已跟踪文件命中禁词: {hits}")
        self.assertEqual(DESCRIBE_VERSION, "describe-v6")
        self.assertEqual(EVENT_VERIFY_PROMPT_VERSION, "event-verify-v1")


if __name__ == "__main__":
    unittest.main()
