from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.models import ModelError
from entitytrace.roster import (
    _check_c2_review,
    _normalize_c2_review,
    resolve_roster,
)
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import dumps


def _person_group(group_id: str, name: str, row_ids: list[str], member_ids: list[str]) -> dict:
    return {
        "group_id": group_id,
        "name": name,
        "kind": "person",
        "row_ids": row_ids,
        "member_ids": member_ids,
        "reason": "同名候选",
    }


class C2NormalizeUnitTests(unittest.TestCase):
    def test_same_group_duplicates_dedup_members_and_rows_keep_first_order(self):
        value = {
            "groups": [
                _person_group("g1", "林舟", ["r2", "r1", "r2"], ["r1_m2", "r1_m1", "r1_m2", "r1_m1"]),
            ],
            "unresolved_row_ids": [],
            "unresolved_member_ids": [],
        }
        original = copy.deepcopy(value)
        normalized, notes = _normalize_c2_review(value)
        self.assertEqual(normalized["groups"][0]["member_ids"], ["r1_m2", "r1_m1"])
        self.assertEqual(normalized["groups"][0]["row_ids"], ["r2", "r1"])
        self.assertEqual(notes, {"dedup_in_group": 3, "cross_group_members": []})
        _check_c2_review(normalized, {"r1", "r2"}, {"r1_m1", "r1_m2"})
        # 入参不被修改
        self.assertEqual(value, original)

    def test_cross_group_member_moves_to_unresolved_once(self):
        value = {
            "groups": [
                _person_group("g1", "林舟", ["r1"], ["r1_m1", "r2_m1"]),
                _person_group("g2", "林公子", ["r2"], ["r2_m1"]),
            ],
            "unresolved_row_ids": [],
            "unresolved_member_ids": ["r9_m1", "r2_m1"],
        }
        original = copy.deepcopy(value)
        normalized, notes = _normalize_c2_review(value)
        self.assertEqual(normalized["groups"][0]["member_ids"], ["r1_m1"])
        self.assertEqual(normalized["groups"][1]["member_ids"], [])
        self.assertEqual(normalized["unresolved_member_ids"], ["r9_m1", "r2_m1"])
        self.assertEqual(notes, {"dedup_in_group": 0, "cross_group_members": ["r2_m1"]})
        _check_c2_review(normalized, {"r1", "r2"}, {"r1_m1", "r2_m1", "r9_m1"})
        # 跨组成员已不在任何组内，交由 unresolved 路径独立成条目
        self.assertFalse(any("r2_m1" in g["member_ids"] for g in normalized["groups"]))
        self.assertEqual(value, original)

    def test_cross_group_member_appended_in_first_appearance_order(self):
        value = {
            "groups": [
                _person_group("g1", "甲", ["r1"], ["r1_m2", "r1_m1"]),
                _person_group("g2", "乙", ["r2"], ["r1_m1", "r1_m2"]),
            ],
            "unresolved_row_ids": [],
            "unresolved_member_ids": [],
        }
        normalized, notes = _normalize_c2_review(value)
        self.assertEqual(notes["cross_group_members"], ["r1_m2", "r1_m1"])
        self.assertEqual(normalized["unresolved_member_ids"], ["r1_m2", "r1_m1"])
        self.assertEqual([g["member_ids"] for g in normalized["groups"]], [[], []])

    def test_cross_group_row_id_still_raises_duplicate(self):
        value = {
            "groups": [
                _person_group("g1", "甲", ["r1"], ["r1_m1"]),
                _person_group("g2", "乙", ["r1"], ["r2_m1"]),
            ],
            "unresolved_row_ids": [],
            "unresolved_member_ids": [],
        }
        normalized, notes = _normalize_c2_review(value)
        self.assertEqual(notes, {"dedup_in_group": 0, "cross_group_members": []})
        # 跨组重复的 row_id 原样保留，由原校验报错
        self.assertEqual([g["row_ids"] for g in normalized["groups"]], [["r1"], ["r1"]])
        with self.assertRaises(ModelError) as ctx:
            _check_c2_review(normalized, {"r1", "r2"}, {"r1_m1", "r2_m1"})
        self.assertIn("C2 重复分配 row_id", str(ctx.exception))

    def test_no_duplicates_byte_identical_and_input_untouched(self):
        value = {
            "groups": [
                _person_group("g1", "林舟", ["r1"], ["r1_m1", "r1_m2"]),
                _person_group("g2", "林公子", ["r2"], ["r2_m1"]),
            ],
            "unresolved_row_ids": ["r3"],
            "unresolved_member_ids": [],
            "extra_key": {"keep": ["as-is", 1]},
        }
        original = copy.deepcopy(value)
        normalized, notes = _normalize_c2_review(value)
        self.assertEqual(dumps(normalized), dumps(value))
        self.assertEqual(notes, {"dedup_in_group": 0, "cross_group_members": []})
        self.assertEqual(value, original)
        self.assertIsNot(normalized, value)
        self.assertIsNot(normalized["groups"][0]["member_ids"], value["groups"][0]["member_ids"])
        # 缺失 unresolved_member_ids 时也不凭空添加字段
        sparse = {"groups": [_person_group("g1", "甲", ["r1"], ["r1_m1"])], "unresolved_row_ids": []}
        sparse_norm, _ = _normalize_c2_review(sparse)
        self.assertNotIn("unresolved_member_ids", sparse_norm)
        self.assertEqual(dumps(sparse_norm), dumps(sparse))

    def test_repeated_calls_are_deterministic(self):
        value = {
            "groups": [
                _person_group("g1", "甲", ["r1", "r1"], ["r1_m1", "r1_m1", "r2_m1"]),
                _person_group("g2", "乙", ["r2"], ["r2_m1", "r2_m2"]),
            ],
            "unresolved_row_ids": [],
            "unresolved_member_ids": ["r3_m1"],
        }
        original = copy.deepcopy(value)
        first_norm, first_notes = _normalize_c2_review(value)
        second_norm, second_notes = _normalize_c2_review(value)
        self.assertEqual(dumps(first_norm), dumps(second_norm))
        self.assertEqual(first_notes, second_notes)
        self.assertEqual(value, original)


class _ScriptedModels:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def generate(self, system, state, stage, validate, *args, **kwargs):
        self.calls.append((stage, state))
        value, req_id = self.handlers[stage](state, len(self.calls))
        validate(value)
        return value, req_id


class C2NormalizePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "test.sqlite3")
        self.text = "第一章\n林舟在渡口等船，阿舟背着行囊。\n第二章\n林公子在渡口等船，阿舟赶来相送。\n"

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def _prepare(self, run_id: str):
        path = self.root / f"{run_id}.txt"
        path.write_bytes(self.text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = self.text.index("第一章"), self.text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        discoveries = [
            {"mentions": [{"text": "林舟", "kind": "person", "refs": [u1]}, {"text": "阿舟", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "林公子", "kind": "person", "refs": [u2]}, {"text": "阿舟", "kind": "person", "refs": [u2]}], "links": []},
        ]
        for task, disc in zip(tasks, discoveries):
            grounded = ground_result(source, task, disc, "req_" + task["id"])
            records = [
                (kind, rec)
                for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                for rec in grounded[key]
            ]
            self.store.finish_task(run_id, task, {"request_id": "req_" + task["id"]}, records)
        return source

    def _c1_handler(self, state, n):
        uid = state["units"][0]["id"]
        if state["chapter_title"] == "第一章":
            return {
                "entities": [
                    {"local_id": "e1", "name": "林舟", "kind": "person", "surfaces": ["林舟"], "evidence_ref": uid},
                    {"local_id": "e2", "name": "阿舟", "kind": "person", "surfaces": ["阿舟"], "evidence_ref": uid},
                ],
                "assignments": [],
                "not_entity": [],
            }, f"c1_{n}"
        return {
            "entities": [
                {"local_id": "e1", "name": "林公子", "kind": "person", "surfaces": ["林公子", "阿舟"], "evidence_ref": uid},
            ],
            "assignments": [],
            "not_entity": [],
        }, f"c1_{n}"

    def _c2_review_handler(self, state, n, duplicate: bool):
        rows = state["rows"]
        row_by_name = {r["name"]: r["row_id"] for r in rows}
        member_by_row = {r["row_id"]: [m["member_id"] for m in r["members"]] for r in rows}
        r_azhou = row_by_name["阿舟"]
        r_ling = row_by_name["林公子"]
        shared = member_by_row[r_ling][0]
        if duplicate:
            groups = [
                _person_group("g1", "阿舟", [r_azhou], [shared]),
                _person_group("g2", "林公子", [r_ling], [shared]),
            ]
        else:
            groups = [
                _person_group("g1", "阿舟", [r_azhou], []),
                _person_group("g2", "林公子", [r_ling], []),
            ]
        return {"groups": groups, "unresolved_row_ids": [], "unresolved_member_ids": []}, f"c2_rev_{n}"

    def _run_variant(self, run_id: str, duplicate: bool, proposal: bool):
        source = self._prepare(run_id)
        propose_entries = []
        verify_calls = []

        def handle_c2_review(state, n):
            return self._c2_review_handler(state, n, duplicate)

        def handle_propose(state, n):
            entries = state["entries"]
            propose_entries.append(copy.deepcopy(entries))
            if not proposal or len(entries) < 2:
                return {"proposals": []}, f"c2_prop_{n}"
            azhou = next(e for e in entries if e["name"] == "阿舟")
            ling = next(e for e in entries if e["name"] == "林公子")
            return {"proposals": [{"left": azhou["entry_id"], "right": ling["entry_id"], "reason": "同指一人"}]}, f"c2_prop_{n}"

        def handle_verify(state, n):
            verify_calls.append(state)
            return {"verdict": "same", "reason": "确为同人"}, f"c2_ver_{n}"

        models = _ScriptedModels(
            {
                "roster-c1": self._c1_handler,
                "roster-c2-review": handle_c2_review,
                "roster-c2-propose": handle_propose,
                "roster-c2-verify": handle_verify,
            }
        )
        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"], outcome)
        return propose_entries, verify_calls

    def test_cross_group_member_becomes_independent_entry_without_cannot_merge(self):
        run_id = "run_cross_group_member"
        propose_entries, verify_calls = self._run_variant(run_id, duplicate=True, proposal=True)
        out = self.store.records(run_id, "roster_output")[0]

        # 校验前已完成规范化，任务结果记录 notes
        task_rows = self.store.rows(
            "SELECT result FROM tasks WHERE run_id=? AND stage='roster-c2-review'", (run_id,)
        )
        self.assertEqual(len(task_rows), 1)
        task_result = json.loads(task_rows[0]["result"])
        self.assertEqual(task_result["normalization"], {"dedup_in_group": 0, "cross_group_members": ["r2_m1"]})
        self.assertNotIn("r2_m1", task_result["value"]["groups"][1]["member_ids"])

        # 该成员经 unresolved 路径独立成条目，不并入任何组
        self.assertEqual(len(propose_entries), 1)
        entries = propose_entries[0]
        azhou = next(e for e in entries if e["name"] == "阿舟")
        self.assertEqual(azhou["mention_count"], 1)
        unresolved_reviews = [r for r in out["reviews"] if r["stage"] == "roster-c2-review"]
        self.assertEqual(len(unresolved_reviews), 1)
        self.assertEqual(unresolved_reviews[0]["impact"], 2)

        # 它不在任何组中，因此合并提议不被 cannot_merge 约束拦截：核验被调用且最终合并
        self.assertEqual(len(verify_calls), 1)
        self.assertEqual(sorted(e["mention_count"] for e in out["entities"]), [1, 3])

    def test_without_duplicates_task_result_has_no_normalization_field(self):
        run_id = "run_without_duplicates"
        self._run_variant(run_id, duplicate=False, proposal=False)
        task_rows = self.store.rows(
            "SELECT result FROM tasks WHERE run_id=? AND stage='roster-c2-review'", (run_id,)
        )
        self.assertEqual(len(task_rows), 1)
        task_result = json.loads(task_rows[0]["result"])
        self.assertNotIn("normalization", task_result)


if __name__ == "__main__":
    unittest.main()
