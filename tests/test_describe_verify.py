from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.cli import main as cli_main
from entitytrace.describe import (
    D1_PROMPT,
    DESCRIBE_VERSION,
    FACET_DEFINITIONS,
    VERIFY_BATCH,
    VERIFY_PROMPT,
    VERIFY_PROMPT_VERSION,
    build_d1_tasks,
    build_describe_id,
    build_name_roster,
    build_verify_tasks,
    check_verify_output,
    decide_verdict,
    export_describe_run,
    extend_holder_quote,
    ground_output,
    resolve_value_entity,
    run_describe,
)
from entitytrace.models import ModelError
from entitytrace.storage import Store
from entitytrace.util import digest, dumps
from tests.test_describe import SERVICES, FixtureCase

D1_TEXT = "第一章 山门\n沈墨身长八尺。"
D1_CHAPTERS = [{
    "id": "ch1", "title": "第一章 山门", "kind": "chapter",
    "start": 0, "end": len(D1_TEXT), "body_start": 7, "chapter_set_version": "chs-1",
}]
D1_SOURCE = {"id": "src-d1", "text": D1_TEXT}
D1_ENTITIES = [{"entity_id": "e1", "kind": "person", "score": 1.0}]
D1_MENTIONS = {"e1": [{"start": 8, "end": 10, "text": "沈墨", "chapter_id": "ch1"}]}


def _d1_tasks() -> list[dict]:
    return build_d1_tasks(
        D1_SOURCE, D1_CHAPTERS, D1_ENTITIES, D1_MENTIONS, "did-freeze",
        task_chars=9000, views=1, entities_per_request=0,
    )


def _sha256(value) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_entity(store: Store, run_id: str, entity_id: str, name: str, kind: str,
                 surfaces: list[tuple[str, int]]) -> None:
    mentions = [
        {"start": start, "end": start + len(surface), "text": surface}
        for surface, start in surfaces
    ]
    header = {"id": entity_id, "name": name, "kind": kind, "names": [], "aliases": []}
    store.execute(
        "INSERT INTO query_entity_mentions VALUES(?,?,?,?)",
        (run_id, entity_id, dumps(header), dumps(mentions)),
    )


class NameRosterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "roster.sqlite3")
        self.run_id = "run-roster"

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_names_are_ordered_filtered_and_deterministic(self):
        _seed_entity(self.store, self.run_id, "e2", "沈墨", "person",
                     [("沈真人", 40), ("沈墨", 5), ("沈墨", 5), ("沈", 30)])
        _seed_entity(self.store, self.run_id, "e1", "铁剑", "object",
                     [("铁剑", 8), ("古剑", 3), ("剑", 1)])
        first = build_name_roster(self.store, self.run_id)
        second = build_name_roster(self.store, self.run_id)
        self.assertEqual(first, second)
        self.assertEqual(list(first), ["e1", "e2"])
        self.assertEqual(first["e2"], ["沈墨", "沈真人"])
        self.assertEqual(first["e1"], ["古剑", "铁剑"])

    def test_single_char_names_fall_back_to_snapshot_name_only_when_long_enough(self):
        _seed_entity(self.store, self.run_id, "e3", "甲", "person", [("甲", 1)])
        roster = build_name_roster(self.store, self.run_id)
        self.assertEqual(roster["e3"], [])

    def test_resolve_picks_longest_matching_person_name(self):
        roster = {
            "e1": ["沈墨", "沈墨真人"],
            "e2": ["真人"],
            "e3": ["道人"],
        }
        self.assertEqual(resolve_value_entity("沈墨真人道长", roster, {"e1", "e2"}), "e1")
        self.assertEqual(resolve_value_entity("真人道长", roster, {"e1", "e2"}), "e2")
        self.assertIsNone(resolve_value_entity("真人道人", roster, {"e1", "e2", "e3"}))
        self.assertIsNone(resolve_value_entity("无名氏", roster, {"e1", "e2"}))

    def test_resolve_tie_between_entities_returns_none(self):
        roster = {"e1": ["甲乙"], "e2": ["甲乙"]}
        self.assertIsNone(resolve_value_entity("甲乙丙丁", roster, {"e1", "e2"}))
        self.assertEqual(resolve_value_entity("甲乙丙丁", roster, {"e1"}), "e1")

    def test_resolve_ignores_non_person_entities(self):
        roster = {"e1": ["沈墨"], "e2": ["铁剑"]}
        self.assertIsNone(resolve_value_entity("沈墨持剑", roster, {"e2"}))
        self.assertEqual(resolve_value_entity("沈墨持剑", roster, {"e1", "e2"}), "e1")


class ExtendHolderQuoteTests(unittest.TestCase):
    def test_name_already_in_quote_keeps_span(self):
        text = "沈墨手执铁剑。"
        self.assertEqual(extend_holder_quote(text, 0, 0, 4, ["沈墨"]), (0, 4))
        start = text.index("铁剑")
        self.assertEqual(extend_holder_quote(text, 0, start, len(text) - 1, ["沈墨"]),
                         (0, len(text) - 1))

    def test_name_before_quote_moves_start_and_keeps_length_bound(self):
        text = "话说那沈墨，掌中铁剑寒光四射。"
        start, end = text.index("寒光四射"), text.index("寒光四射") + 4
        extended = extend_holder_quote(text, 0, start, end, ["沈墨"])
        self.assertEqual(extended, (text.index("沈墨"), end))
        self.assertLessEqual(extended[1] - extended[0], 120)
        self.assertEqual(text[extended[0]:extended[1]], "沈墨，掌中铁剑寒光四射")

    def test_name_outside_window_returns_none(self):
        text = "沈墨" + "甲" * 200 + "寒光四射"
        start = text.index("寒光四射")
        self.assertIsNone(extend_holder_quote(text, 0, start, start + 4, ["沈墨"]))

    def test_name_before_segment_start_returns_none(self):
        text = "沈墨甲乙寒光四射"
        start = text.index("寒光四射")
        self.assertIsNone(extend_holder_quote(text, 2, start, start + 4, ["沈墨"]))

    def test_multiple_names_pick_largest_position(self):
        text = "老者沈墨与少年沈墨同行，寒光四射。"
        start = text.index("寒光四射")
        extended = extend_holder_quote(text, 0, start, start + 4, ["少年沈墨", "沈墨"])
        self.assertEqual(extended, (text.rindex("沈墨", 0, start), start + 4))

    def test_empty_names_or_blank_returns_none(self):
        text = "沈墨持剑。"
        self.assertIsNone(extend_holder_quote(text, 0, 3, 4, []))
        self.assertIsNone(extend_holder_quote(text, 0, 3, 4, ["", "  ", None]))


class HolderGroundingTests(unittest.TestCase):
    TEXT = "话说那沈墨，掌中铁剑寒光四射，威势惊人。"

    def _task(self, entity_kinds: dict, mentions: dict) -> dict:
        return {
            "id": "task-holder",
            "stage": "describe-d1",
            "entity_id": None,
            "start": 0,
            "end": len(self.TEXT),
            "segments": [{"start": 0, "end": len(self.TEXT)}],
            "entity_kinds": entity_kinds,
            "mentions_by_entity": mentions,
            "default_entity_id": None,
            "text": self.TEXT,
        }

    def _chapters(self) -> list[dict]:
        return [{"id": "ch1", "start": 0, "end": len(self.TEXT)}]

    def test_roster_name_is_used_when_holder_is_not_in_request(self):
        task = self._task({"e_jian": "object"}, {})
        value = {"facts": [{
            "entity_id": "e_jian", "facet": "holder", "value": "沈墨",
            "quote": "寒光四射", "source": "narration", "holder_entity_id": "e_shen",
        }], "events": []}
        result = ground_output(
            "did", task, value, self._chapters(), {"e_shen", "e_jian"},
            name_roster={"e_shen": ["沈墨"], "e_jian": ["铁剑"]},
            person_ids={"e_shen"},
        )
        self.assertEqual(result["reviews"], [])
        fact = result["facts"][0]
        self.assertEqual(fact["start"], 3)
        self.assertEqual(fact["quote"], "沈墨，掌中铁剑寒光四射")
        self.assertEqual(self.TEXT[fact["start"]:fact["end"]], fact["quote"])
        self.assertEqual(fact["id"], digest([
            "describe-fact-v1", "did", task["id"], 0, "e_jian", "holder",
            fact["start"], fact["end"], fact["quote"],
        ]))

    def test_value_with_suffix_resolves_to_person_and_extends_by_short_name(self):
        task = self._task({"e_jian": "object"}, {})
        value = {"facts": [{
            "entity_id": "e_jian", "facet": "holder", "value": "沈墨真人",
            "quote": "寒光四射", "source": "narration",
        }], "events": []}
        result = ground_output(
            "did", task, value, self._chapters(), {"e_jian"},
            name_roster={"e_shen": ["沈墨", "沈墨真人"], "e_jian": ["铁剑"]},
            person_ids={"e_shen"},
        )
        self.assertEqual(result["reviews"], [])
        fact = result["facts"][0]
        self.assertEqual(fact["start"], self.TEXT.index("沈墨"))
        self.assertEqual(fact["quote"], "沈墨，掌中铁剑寒光四射")
        self.assertIsNone(fact["holder_entity_id"])

    def test_holder_without_name_in_quote_goes_to_review(self):
        text = "铁剑寒光四射"
        task = {
            "id": "task-holder", "stage": "describe-d1", "entity_id": None,
            "start": 0, "end": len(text), "segments": [{"start": 0, "end": len(text)}],
            "entity_kinds": {"e_jian": "object"}, "mentions_by_entity": {},
            "default_entity_id": None, "text": text,
        }
        value = {"facts": [{
            "entity_id": "e_jian", "facet": "holder", "value": "沈墨",
            "quote": "寒光四射", "source": "narration", "holder_entity_id": "e_shen",
        }], "events": []}
        result = ground_output(
            "did", task, value, [{"id": "ch1", "start": 0, "end": len(text)}],
            {"e_jian", "e_shen"},
            name_roster={"e_shen": ["沈墨"]}, person_ids={"e_shen"},
        )
        self.assertEqual(result["facts"], [])
        self.assertEqual([review["reason"] for review in result["reviews"]],
                         ["holder-not-in-quote"])

    def test_non_holder_facet_is_unchanged(self):
        task = self._task({"e_shen": "person"}, {})
        value = {"facts": [{
            "entity_id": "e_shen", "facet": "face", "value": "沈墨",
            "quote": "威势惊人", "source": "narration",
        }], "events": []}
        result = ground_output(
            "did", task, value, self._chapters(), {"e_shen"},
            name_roster={"e_shen": ["沈墨"]}, person_ids={"e_shen"},
        )
        fact = result["facts"][0]
        self.assertEqual((fact["start"], fact["end"]), (self.TEXT.index("威势惊人"),
                                                       self.TEXT.index("威势惊人") + 4))
        self.assertEqual(fact["quote"], "威势惊人")


class D1PayloadFreezeTests(unittest.TestCase):
    # 冻结值：D1 提示词与任务 state 的任何改动都会让摘要变化。
    D1_FINGERPRINT = "16efd6ccbaa3e6ff70cf4277bb825fbaa3229ba6dc0cada58076e4c98ceb1e24"
    VERIFY_FINGERPRINT = "eec15626b79737a5e916e7b446304abbae54842325bcbfac1fb210e2bec9e68b"

    def test_d1_tasks_keep_prompt_and_state(self):
        tasks = _d1_tasks()
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["prompt"], D1_PROMPT)
        self.assertEqual(set(task["state"]), {"view", "chunk_start", "chunk_end", "text", "entities"})
        self.assertEqual(set(task["state"]["entities"][0]),
                         {"entity_id", "kind", "surfaces_in_chunk", "allowed_facets"})
        fingerprint = _sha256(dumps({"prompt": task["prompt"], "state": task["state"]}))
        self.assertEqual(fingerprint, self.D1_FINGERPRINT)

    def test_verify_prompt_is_frozen_v2(self):
        self.assertEqual(VERIFY_PROMPT_VERSION, "verify-v2")
        self.assertEqual(_sha256(VERIFY_PROMPT), self.VERIFY_FINGERPRINT)
        for code in ("not_description", "wrong_entity", "wrong_facet", "unsupported_value"):
            self.assertIn(code, VERIFY_PROMPT)
        self.assertIn("value 必须原样抄写", VERIFY_PROMPT)
        self.assertEqual(FACET_DEFINITIONS["object"]["holder"], "持有者或乘骑者")


class VerifyTaskTests(unittest.TestCase):
    TEXT = ("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
            "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏")
    CHAPTERS = [
        {"id": "ch1", "start": 0, "end": 20},
        {"id": "ch2", "start": 20, "end": 56},
    ]

    def _facts(self, count: int = 26) -> list[dict]:
        facts = []
        for index in range(count):
            start = index
            end = index + 2
            entity_id = "e1" if index % 2 == 0 else "e2"
            facts.append({
                "id": f"f{index:02d}", "entity_id": entity_id, "facet": "build",
                "value": f"值{index}", "quote": self.TEXT[start:end],
                "start": start, "end": end,
                "chapter_id": "ch1" if start < 20 else "ch2",
            })
        return facts

    def _build(self, facts: list[dict], passes: int = 1) -> list[dict]:
        return build_verify_tasks(
            "did", facts, {"e1": "person", "e2": "person"},
            {"e1": ["甲", "甲乙"], "e2": ["丙"]}, self.TEXT, self.CHAPTERS,
            passes=passes,
        )

    def test_batches_are_sorted_and_bounded(self):
        facts = self._facts()
        shuffled = list(reversed(facts))
        first = self._build(shuffled)
        second = self._build(list(shuffled))
        self.assertEqual(dumps(first), dumps(second))
        self.assertEqual([len(task["fact_ids"]) for task in first], [12, 12, 2])
        self.assertLessEqual(max(len(task["fact_ids"]) for task in first), VERIFY_BATCH)
        ordered_ids = [fid for task in first for fid in task["fact_ids"]]
        expected = [fact["id"] for fact in sorted(
            facts, key=lambda f: (f["entity_id"], f["start"], f["end"],
                                  f["facet"], f["value"], f["id"]))]
        self.assertEqual(ordered_ids, expected)
        self.assertEqual([list(task["short_to_fact"]) for task in first],
                         [["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
                           "F11", "F12"]] * 2 + [["F1", "F2"]])
        self.assertNotEqual(first[0]["id"], first[1]["id"])
        self.assertEqual([task["pass"] for task in first], [0, 0, 0])

    def test_three_pass_first_batch_sizes(self):
        tasks = self._build(self._facts(30), passes=3)
        by_pass = {0: [], 1: [], 2: []}
        for task in tasks:
            by_pass[task["pass"]].append(len(task["fact_ids"]))
        self.assertEqual(by_pass[0], [12, 12, 6])
        self.assertEqual(by_pass[1], [8, 12, 10])
        self.assertEqual(by_pass[2], [4, 12, 12, 2])
        self.assertEqual([len(by_pass[k]) for k in (0, 1, 2)], [3, 3, 4])
        self.assertEqual(
            [task["pass"] for task in tasks],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
        )

    def test_pass_zero_matches_single_pass_and_task_ids_differ(self):
        facts = self._facts(30)
        single = self._build(facts, passes=1)
        triple = self._build(facts, passes=3)
        pass_zero = [task for task in triple if task["pass"] == 0]
        self.assertEqual([task["id"] for task in pass_zero],
                         [task["id"] for task in single])
        self.assertEqual(dumps([task["state"] for task in pass_zero]),
                         dumps([task["state"] for task in single]))
        pass_one_ids = [task["id"] for task in triple if task["pass"] == 1]
        self.assertTrue(set(pass_one_ids).isdisjoint({task["id"] for task in single}))
        for task in triple:
            self.assertEqual(task["id"], digest([
                "describe-verify-task-v2", "did", task["pass"], task["fact_ids"],
            ]))

    def test_payload_has_no_describe_id_pass_or_fact_ids(self):
        facts = [
            {"id": "f0", "entity_id": "e1", "facet": "build", "value": "值0",
             "quote": self.TEXT[0:2], "start": 0, "end": 2, "chapter_id": "ch1"},
            {"id": "f1", "entity_id": "e2", "facet": "build", "value": "值1",
             "quote": self.TEXT[22:24], "start": 22, "end": 24, "chapter_id": "ch2"},
        ]
        tasks = self._build(facts, passes=3)
        for task in tasks:
            self.assertEqual(set(task["state"]), {"items"})
            self.assertNotIn("pass", task["state"])
        serialized = dumps([task["state"] for task in tasks])
        self.assertNotIn("did", serialized)
        self.assertNotIn('"pass"', serialized)
        for fact in facts:
            self.assertNotIn(fact["id"], serialized)
        first = tasks[0]["state"]["items"][0]
        self.assertEqual(set(first), {"id", "kind", "names", "facet", "facet_definition",
                                      "value", "quote", "before", "after"})
        self.assertEqual(first["names"], ["甲", "甲乙"])
        self.assertEqual(first["facet_definition"], FACET_DEFINITIONS["person"]["build"])
        self.assertEqual(first["before"], "")
        self.assertEqual(first["after"], self.TEXT[2:20])
        chapter_two = tasks[0]["state"]["items"][1]
        self.assertEqual(chapter_two["before"], self.TEXT[20:22])
        self.assertEqual(chapter_two["after"], self.TEXT[24:56])

    def test_invalid_passes_rejected(self):
        for bad in (0, 2, True, "3", 1.5):
            with self.assertRaises(ValueError):
                self._build(self._facts(2), passes=bad)


class VerifyOutputTests(unittest.TestCase):
    EXPECTED = {"F1": "值一", "F2": "值二"}

    def test_valid_output_passes(self):
        check_verify_output({"verdicts": [
            {"id": "F1", "value": "值一", "verdict": "keep", "reason": None},
            {"id": "F2", "value": "值二", "verdict": "drop", "reason": "wrong_entity"},
        ]}, self.EXPECTED)

    def test_invalid_outputs_are_rejected(self):
        cases = [
            {},
            {"verdicts": "x"},
            {"verdicts": [{"id": "F1", "value": "值一", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": "值一", "verdict": "keep", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None},
                          {"id": "F3", "value": "值三", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": "值一", "verdict": "keep", "reason": None},
                          {"id": "F1", "value": "值一", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": "值一", "verdict": "maybe", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": "值一", "verdict": "drop", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": "值一", "verdict": "drop", "reason": "other"},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": "值一", "verdict": "keep", "reason": "ok"},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": 1, "value": "值一", "verdict": "keep", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            # 缺 value、value 不一致、value 非字符串都被拒绝。
            {"verdicts": [{"id": "F1", "verdict": "keep", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": "值二", "verdict": "keep", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": "F1", "value": 1, "verdict": "keep", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            # 未知 id 与缺 id 的替换项。
            {"verdicts": [{"id": "F9", "value": "值一", "verdict": "keep", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
            {"verdicts": [{"id": None, "value": "值一", "verdict": "keep", "reason": None},
                          {"id": "F2", "value": "值二", "verdict": "keep", "reason": None}]},
        ]
        for value in cases:
            with self.assertRaises(ModelError):
                check_verify_output(value, self.EXPECTED)

    def test_value_echo_mismatch_uses_marker(self):
        with self.assertRaises(ModelError) as caught:
            check_verify_output({"verdicts": [
                {"id": "F1", "value": "改写", "verdict": "keep", "reason": None},
                {"id": "F2", "value": "值二", "verdict": "keep", "reason": None},
            ]}, self.EXPECTED)
        self.assertEqual(str(caught.exception), "value-echo-mismatch")


class DecideVerdictTests(unittest.TestCase):
    def test_single_pass_uses_pass_zero(self):
        self.assertEqual(decide_verdict([(0, "keep", None)], 1), ("keep", None))
        self.assertEqual(decide_verdict([(0, "drop", "wrong_facet")], 1), ("drop", "wrong_facet"))
        self.assertEqual(decide_verdict([], 1), ("unverified", "invalid-output"))
        self.assertEqual(
            decide_verdict([(0, "unverified", "invalid-output")], 1),
            ("unverified", "invalid-output"),
        )

    def test_two_drop_votes_drop_with_majority_reason(self):
        votes = [(0, "drop", "wrong_entity"), (1, "drop", "wrong_entity"),
                 (2, "keep", None)]
        self.assertEqual(decide_verdict(votes, 3), ("drop", "wrong_entity"))

    def test_drop_reason_tie_follows_fixed_order(self):
        votes = [(0, "keep", None), (1, "drop", "wrong_entity"),
                 (2, "drop", "not_description")]
        self.assertEqual(decide_verdict(votes, 3), ("drop", "not_description"))
        votes = [(0, "keep", None), (1, "drop", "unsupported_value"),
                 (2, "drop", "wrong_facet")]
        self.assertEqual(decide_verdict(votes, 3), ("drop", "wrong_facet"))

    def test_one_drop_vote_keeps(self):
        votes = [(0, "drop", "not_description"), (1, "keep", None), (2, "keep", None)]
        self.assertEqual(decide_verdict(votes, 3), ("keep", None))
        votes = [(0, "drop", "not_description"), (1, "keep", None),
                 (2, "unverified", "invalid-output")]
        self.assertEqual(decide_verdict(votes, 3), ("keep", None))

    def test_two_or_more_unverified_is_unverified(self):
        votes = [(0, "drop", "not_description"),
                 (1, "unverified", "invalid-output"),
                 (2, "unverified", "invalid-output")]
        self.assertEqual(decide_verdict(votes, 3), ("unverified", "invalid-output"))
        votes = [(0, "unverified", "invalid-output"), (1, "unverified", "invalid-output"),
                 (2, "unverified", "invalid-output")]
        self.assertEqual(decide_verdict(votes, 3), ("unverified", "invalid-output"))

    def test_empty_three_pass_votes_is_unverified(self):
        self.assertEqual(decide_verdict([], 3), ("unverified", "invalid-output"))

    def test_invalid_passes_rejected(self):
        for bad in (0, 2, True, "3"):
            with self.assertRaises(ValueError):
                decide_verdict([], bad)


class RecordingModels:
    """按阶段记录调用并返回预置输出，零网络请求。"""

    def __init__(self, services, handler):
        self.services = services
        self.handler = handler
        self.calls: list[dict] = []

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        count = sum(1 for call in self.calls if call["stage"] == stage) + 1
        self.calls.append({"stage": stage, "state": state, "count": count,
                           "max_tokens": max_tokens, "temperature": temperature})
        value = self.handler(stage, state, count)
        validate(value)
        return value, f"req_{len(self.calls)}"

    def stage_calls(self, stage: str) -> list[dict]:
        return [call for call in self.calls if call["stage"] == stage]


def _echo(items, verdicts: dict[str, tuple[str, str | None]]) -> dict:
    output = []
    for item in items:
        verdict, reason = verdicts.get(item["value"], ("keep", None))
        output.append({"id": item["id"], "value": item["value"],
                       "verdict": verdict, "reason": reason})
    return {"verdicts": output}


def _two_fact_handler(stage, state, count):
    if stage == "describe-verify":
        pass_index = count - 1
        plan = {
            0: {"身长八尺": ("keep", None), "面如冠玉": ("drop", "wrong_entity")},
            1: {"身长八尺": ("drop", "not_description"), "面如冠玉": ("keep", None)},
            2: {"身长八尺": ("drop", "not_description"), "面如冠玉": ("keep", None)},
        }[pass_index if pass_index < 3 else 0]
        return _echo(state["items"], plan)
    chunk = state["text"]
    facts = []
    if "身长八尺" in chunk:
        facts.append({"entity_id": "e_shen", "facet": "build", "value": "身长八尺",
                      "quote": "身长八尺", "source": "narration"})
    if "面如冠玉" in chunk:
        facts.append({"entity_id": "e_shen", "facet": "face", "value": "面如冠玉",
                      "quote": "面如冠玉", "source": "narration"})
    return {"facts": facts, "events": []}


class VerifyRunTests(FixtureCase):
    def test_verify_one_filters_export_and_counts(self):
        models = RecordingModels(SERVICES, _two_fact_handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=1, views=1,
                                  verify=1, out=self.out_path)
            document = json.loads(self.out_path.read_text(encoding="utf-8"))
            votes = store.rows(
                "SELECT fact_id, pass, verdict FROM fact_verify_votes WHERE describe_id=?",
                (result["describe_id"],),
            )
        finally:
            store.close()
        self.assertEqual(result["verify"], 1)
        self.assertEqual(result["views"], 1)
        self.assertEqual(result["counts"]["tasks"], len(models.calls))
        self.assertEqual(len(models.stage_calls("describe-verify")), 1)
        self.assertEqual(document["verification"], {
            "enabled": True, "passes": 1, "checked": 2, "kept": 1,
            "dropped": {"not_description": 0, "wrong_entity": 1,
                        "wrong_facet": 0, "unsupported_value": 0},
            "unverified": 0, "echo_mismatch_retries": 0,
        })
        self.assertEqual(len(votes), 2)
        self.assertEqual({row["pass"] for row in votes}, {0})
        self.assertNotIn("verify", document)
        entity = document["entities"][0]
        self.assertEqual([fact["facet"] for fact in entity["facts"]], ["build"])
        self.assertNotIn("build", entity["gaps"])
        self.assertIn("face", entity["gaps"])

    def test_verify_three_votes_and_exports_final_keep(self):
        models = RecordingModels(SERVICES, _two_fact_handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=1, verify=3,
                                  out=self.out_path)
            document = json.loads(self.out_path.read_text(encoding="utf-8"))
            votes = store.rows(
                "SELECT fact_id, pass, verdict, reason FROM fact_verify_votes "
                "WHERE describe_id=? ORDER BY pass, fact_id", (result["describe_id"],),
            )
            verdicts = store.rows(
                "SELECT fact_id, verdict, reason FROM fact_verdicts WHERE describe_id=?",
                (result["describe_id"],),
            )
        finally:
            store.close()
        self.assertEqual(len(models.stage_calls("describe-verify")), 3)
        self.assertEqual(len(votes), 6)
        self.assertEqual({row["pass"] for row in votes}, {0, 1, 2})
        self.assertEqual(document["verification"], {
            "enabled": True, "passes": 3, "checked": 2, "kept": 1,
            "dropped": {"not_description": 1, "wrong_entity": 0,
                        "wrong_facet": 0, "unsupported_value": 0},
            "unverified": 0, "echo_mismatch_retries": 0,
        })
        self.assertEqual(len(verdicts), 2)
        by_reason = {row["verdict"]: row["reason"] for row in verdicts}
        self.assertEqual(by_reason["drop"], "not_description")
        self.assertIsNone(by_reason["keep"])
        entity = document["entities"][0]
        self.assertEqual([fact["facet"] for fact in entity["facts"]], ["face"])

    def test_invalid_verify_output_becomes_unverified_without_export(self):
        def handler(stage, state, count):
            if stage == "describe-verify":
                return {"verdicts": []}
            return _two_fact_handler(stage, state, count)

        models = RecordingModels(SERVICES, handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=1, verify=3)
            document = result["document"]
            votes = store.rows(
                "SELECT fact_id, pass, verdict, reason FROM fact_verify_votes "
                "WHERE describe_id=?", (result["describe_id"],),
            )
        finally:
            store.close()
        self.assertEqual(len(models.stage_calls("describe-verify")), 3)
        self.assertEqual(document["verification"]["checked"], 2)
        self.assertEqual(document["verification"]["kept"], 0)
        self.assertEqual(document["verification"]["unverified"], 2)
        self.assertEqual(document["entities"][0]["facts"], [])
        self.assertEqual(len(votes), 6)
        self.assertEqual({row["verdict"] for row in votes}, {"unverified"})
        self.assertEqual({row["reason"] for row in votes}, {"invalid-output"})

    def test_network_error_stops_and_resume_skips_completed_batches(self):
        def handler(stage, state, count):
            if stage == "describe-verify":
                if count == 2:
                    raise ModelError("复核网络失败", status=503)
                return _echo(state["items"], {})
            chunk = state["text"]
            facts = []
            for offset in range(0, len(chunk) - 1, 2):
                facts.append({
                    "entity_id": "e_shen",
                    "facet": ("age", "gender", "build", "face",
                              "hair", "feature", "costume", "voice")[(offset // 2) % 8],
                    "value": f"值{offset}",
                    "quote": chunk[offset:offset + 2],
                    "source": "narration",
                })
            return {"facts": facts, "events": []}

        models = RecordingModels(SERVICES, handler)
        store = Store(self.db_path)
        try:
            with self.assertRaises(ModelError):
                run_describe(store, self.run_id, services=SERVICES, models=models,
                             top_k=10, kinds="person", concurrency=1, verify=3)
            first_run = list(models.stage_calls("describe-verify"))
            self.assertEqual(len(first_run), 2)
            failed_state = store.rows("SELECT state FROM describe_runs")[0]["state"]
            self.assertEqual(json.loads(failed_state)["status"], "failed")
            stored_first = {
                (row["fact_id"], row["pass"]) for row in store.rows(
                    "SELECT fact_id, pass FROM fact_verify_votes", ())
            }
            resumed = run_describe(store, self.run_id, services=SERVICES, models=models,
                                   top_k=10, kinds="person", concurrency=1, verify=3)
            second_run = models.stage_calls("describe-verify")[2:]
            resumed_ids = {
                row["fact_id"] for row in store.rows(
                    "SELECT fact_id FROM fact_verdicts", ())
            }
        finally:
            store.close()
        self.assertGreater(len(stored_first), 0)
        self.assertTrue({fact_id for fact_id, _ in stored_first} < resumed_ids)
        self.assertEqual(len(second_run), resumed["new_verify_tasks"])
        self.assertEqual(len(resumed_ids), resumed["document"]["verification"]["checked"])
        self.assertEqual(resumed["document"]["verification"]["kept"], len(resumed_ids))
        resent_payloads = {dumps(call["state"]) for call in second_run}
        self.assertNotIn(dumps(first_run[0]["state"]), resent_payloads)
        self.assertIn(dumps(first_run[1]["state"]), resent_payloads)
        checked = resumed["document"]["verification"]["checked"]
        expected_tasks = 0
        for pass_index in (0, 1, 2):
            first = VERIFY_BATCH - 4 * pass_index
            expected_tasks += 1 + max(0, -(-(checked - first) // VERIFY_BATCH))
        self.assertEqual(resumed["verify_tasks"], expected_tasks)

    def test_echo_mismatch_retries_are_counted_in_export(self):
        models = RecordingModels(SERVICES, _two_fact_handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=1, verify=1)
            store.execute(
                "INSERT INTO calls(run_id,invocation,stage,request_id,service,attempt,"
                "status,seconds,detail,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, result["describe_id"], "describe-verify", "req_echo", "example",
                 1, "failed", 0.1, dumps({"validation_error": "value-echo-mismatch"}), "t0"),
            )
            document = export_describe_run(store, result["describe_id"])
        finally:
            store.close()
        self.assertEqual(document["verification"]["echo_mismatch_retries"], 1)
        self.assertEqual(document["verification"]["passes"], 1)

    def test_verify_zero_exports_voted_facts(self):
        models = RecordingModels(SERVICES, _two_fact_handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=1, verify=0,
                                  out=self.out_path)
            document = json.loads(self.out_path.read_text(encoding="utf-8"))
        finally:
            store.close()
        self.assertEqual(result["verify"], 0)
        self.assertEqual(document["verification"], {"enabled": False})
        entity = document["entities"][0]
        self.assertEqual({fact["facet"] for fact in entity["facts"]}, {"build", "face"})
        self.assertEqual(len(models.stage_calls("describe-verify")), 0)
        self.assertNotIn("passes", document["verification"])

    def test_describe_id_depends_on_verify_and_prompt_version(self):
        base = build_describe_id(20, ["person"], ["e1"], "run", "example", "example/model")
        one = build_describe_id(20, ["person"], ["e1"], "run", "example",
                                "example/model", verify=1)
        three = build_describe_id(20, ["person"], ["e1"], "run", "example",
                                  "example/model", verify=3)
        self.assertNotEqual(base, one)
        self.assertNotEqual(base, three)
        self.assertNotEqual(one, three)
        self.assertEqual(three, build_describe_id(20, ["person"], ["e1"], "run", "example",
                                                  "example/model", verify=3))
        with patch("entitytrace.describe.VERIFY_PROMPT_VERSION", "verify-v9"):
            other = build_describe_id(20, ["person"], ["e1"], "run", "example",
                                      "example/model", verify=3)
        self.assertNotEqual(three, other)
        with patch("entitytrace.describe.VERIFY_BATCH", 8):
            other_batch = build_describe_id(20, ["person"], ["e1"], "run", "example",
                                            "example/model", verify=3)
        self.assertNotEqual(three, other_batch)
        self.assertEqual(DESCRIBE_VERSION, "describe-v5")
        for bad in (-1, 2, 4, True, 1.5, "1"):
            with self.assertRaises(ValueError):
                build_describe_id(20, ["person"], ["e1"], "run", "example",
                                  "example/model", verify=bad)
        store = Store(self.db_path)
        try:
            with self.assertRaises(ValueError):
                run_describe(store, self.run_id, services=SERVICES, models=None,
                             top_k=10, kinds="person", verify=2)
        finally:
            store.close()


class _CliFailingVerifyModels:
    calls: list[str] = []

    def __init__(self, store, run_id, invocation, services, *args, **kwargs):
        self.services = services

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        type(self).calls.append(stage)
        if stage == "describe-verify":
            raise ModelError("复核网络失败", status=503)
        value = {"facts": [], "events": []}
        if "身长八尺" in state.get("text", ""):
            value["facts"].append({
                "entity_id": "e_shen", "facet": "build", "value": "身长八尺",
                "quote": "身长八尺", "source": "narration",
            })
        validate(value)
        return value, "req_cli"


class _CliVerifyingModels(_CliFailingVerifyModels):
    calls: list[str] = []

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        type(self).calls.append(stage)
        if stage == "describe-verify":
            value = _echo(state["items"], {})
        else:
            value = {"facts": [], "events": []}
            if "身长八尺" in state.get("text", ""):
                value["facts"].append({
                    "entity_id": "e_shen", "facet": "build", "value": "身长八尺",
                    "quote": "身长八尺", "source": "narration",
                })
        validate(value)
        return value, "req_cli"


class VerifyCliTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.config_path = self.root / "config.env"
        self.config_path.write_text("DEEPSEEK_API_KEY=test_key\n", encoding="utf-8")

    def _call(self, fake, arguments: list[str]) -> tuple[int, str, str]:
        out_buf, err_buf = io.StringIO(), io.StringIO()
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith(("DEEPSEEK_", "TYPESAFE_", "ENTITYTRACE_"))
                     and k != "GENERATOR_PROVIDER"}
        fake.calls = []
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("entitytrace.describe.Models", fake), \
             contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = cli_main(["describe", self.run_id, "--db", str(self.db_path),
                           "--top-k", "10", "--kinds", "person", "--concurrency", "1",
                           "--config", str(self.config_path)] + arguments)
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_network_error_on_verify_exits_two(self):
        rc, out, err = self._call(_CliFailingVerifyModels, ["--verify", "1"])
        self.assertEqual(rc, 2, out + err)
        receipt = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(receipt["error"]["category"], "network")
        self.assertEqual(_CliFailingVerifyModels.calls[-1], "describe-verify")

    def test_success_receipt_carries_verify_and_export_counts(self):
        rc, out, err = self._call(_CliVerifyingModels,
                                  ["--verify", "1", "--out", str(self.out_path)])
        self.assertEqual(rc, 0, out + err)
        receipt = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(receipt["verify"], 1)
        document = json.loads(self.out_path.read_text(encoding="utf-8"))
        self.assertEqual(document["verification"]["enabled"], True)
        self.assertEqual(document["verification"]["passes"], 1)
        self.assertEqual(document["verification"]["kept"], 1)

    def test_verify_three_runs_three_passes(self):
        rc, out, err = self._call(_CliVerifyingModels,
                                  ["--verify", "3", "--out", str(self.out_path)])
        self.assertEqual(rc, 0, out + err)
        receipt = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(receipt["verify"], 3)
        self.assertEqual(_CliVerifyingModels.calls.count("describe-verify"), 3)
        document = json.loads(self.out_path.read_text(encoding="utf-8"))
        self.assertEqual(document["verification"]["passes"], 3)
        self.assertEqual(document["verification"]["kept"], 1)

    def test_invalid_verify_value_exits_nonzero_before_any_request(self):
        for bad in ("2", "-1", "4"):
            rc, out, err = self._call(_CliVerifyingModels, ["--verify", bad])
            self.assertNotEqual(rc, 0)
            self.assertEqual(_CliVerifyingModels.calls, [])
            receipt = json.loads(out.strip().splitlines()[-1])
            self.assertEqual(receipt["error"]["category"], "input")


class LegacyVerifyExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "legacy.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_old_run_without_verify_exports_disabled(self):
        store = Store(self.db_path)
        try:
            store.execute("INSERT INTO describe_runs VALUES(?,?,?,?,?,?)",
                          ("did-v3", "run-v3", dumps({
                              "describe_version": "describe-v3", "top_k": 20,
                              "kinds": ["person"], "entity_ids": ["e1"], "run_id": "run-v3",
                              "views": 1, "entities_per_request": 0,
                              "generator": {"provider": "example", "model": "example/model"},
                          }), dumps({"status": "completed"}), "t0", "t0"))
            store.execute("INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                          ("did-v3", "f1", "e1", "build", "高大", "身长八尺", 0, 4,
                           "ch1", "narration", None, "t1"))
        finally:
            store.close()
        store = Store(self.db_path)
        try:
            document = export_describe_run(store, "did-v3")
        finally:
            store.close()
        self.assertEqual(document["verification"], {"enabled": False})
        self.assertEqual(document["views"], 1)

    def test_v4_verify_one_library_still_exports(self):
        store = Store(self.db_path)
        try:
            store.execute("INSERT INTO describe_runs VALUES(?,?,?,?,?,?)",
                          ("did-v4", "run-v4", dumps({
                              "describe_version": "describe-v4", "top_k": 20,
                              "kinds": ["person"], "entity_ids": ["e1"], "run_id": "run-v4",
                              "views": 1, "entities_per_request": 0, "verify": 1,
                              "verify_batch": 12,
                              "generator": {"provider": "example", "model": "example/model"},
                          }), dumps({"status": "completed"}), "t0", "t0"))
            for fact in (
                ("f1", "e1", "build", "高大", "身长八尺", 0, 4, "ch1", "narration", None, "t1"),
                ("f2", "e1", "face", "冠玉", "面如冠玉", 5, 9, "ch1", "narration", None, "t1"),
            ):
                store.execute("INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                              ("did-v4", *fact))
            store.execute("INSERT INTO fact_verdicts VALUES(?,?,?,?,?)",
                          ("did-v4", "f1", "keep", None, "t1"))
            store.execute("INSERT INTO fact_verdicts VALUES(?,?,?,?,?)",
                          ("did-v4", "f2", "drop", "wrong_entity", "t1"))
        finally:
            store.close()
        store = Store(self.db_path)
        try:
            document = export_describe_run(store, "did-v4")
        finally:
            store.close()
        self.assertEqual(document["verification"], {
            "enabled": True, "passes": 1, "checked": 2, "kept": 1,
            "dropped": {"not_description": 0, "wrong_entity": 1,
                        "wrong_facet": 0, "unsupported_value": 0},
            "unverified": 0, "echo_mismatch_retries": 0,
        })
        self.assertEqual([fact["id"] for fact in document["entities"][0]["facts"]], ["f1"])


if __name__ == "__main__":
    unittest.main()
