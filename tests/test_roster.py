from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.models import ModelError
from entitytrace.roster import (
    VERSION,
    apply_c1_task_result,
    normalize_surface,
    resolve_roster,
    resolve_surface_direct,
    validate_c1_structure,
)
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import dumps

ROOT = Path(__file__).resolve().parents[1]


class ScriptedRosterModels:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []
        self.peaks = {}

    def generate(self, system, state, stage, validate, *args, **kwargs):
        self.calls.append((stage, state))
        handler = self.handlers[stage]
        value, req_id = handler(state, len(self.calls))
        validate(value)
        return value, req_id


class RosterAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "test.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def _prepare_two_chapter_run(self, text: str, ch1_discovery: dict, ch2_discovery: dict) -> tuple[dict, str, list[dict]]:
        path = self.root / "book.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1 = text.index("第一章")
        idx2 = text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        run_id = "run_test"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        for task, disc in zip(tasks, [ch1_discovery, ch2_discovery]):
            grounded = ground_result(source, task, disc, "req_" + task["id"])
            records = [
                (kind, rec)
                for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                for rec in grounded[key]
            ]
            self.store.finish_task(run_id, task, {"request_id": "req_" + task["id"]}, records)
        return source, run_id, tasks

    def test_normalize_surface(self):
        self.assertEqual(normalize_surface("  Mr.   Darcy’s "), "mr. darcy")
        self.assertEqual(normalize_surface("Elizabeth's"), "elizabeth")
        self.assertEqual(normalize_surface("Ladies’"), "ladies")
        self.assertEqual(normalize_surface("“梁天来”"), '"梁天来"')
        self.assertEqual(normalize_surface("Miss Bennet"), "miss bennet")

    def test_c1_validation_and_missing_surface_repair(self):
        text = "第一章\n林舟见到阿舟与陈叔。\n第二章\n林舟离开。\n"
        path = self.root / "c1.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        _, chapters = build_chapters(source, [{"start": 0, "end": 3, "title": "第一章", "kind": "chapter"}])
        task = plan_tasks(source, chapters, 9000)[0]
        uid = task["units"][0]["id"]
        grounded = ground_result(
            source,
            task,
            {
                "mentions": [
                    {"text": "林舟", "kind": "person", "refs": [uid]},
                    {"text": "阿舟", "kind": "person", "refs": [uid]},
                    {"text": "陈叔", "kind": "person", "refs": [uid]},
                ],
                "links": [],
            },
            "req1",
        )
        # 非法 local_id 引用应抛出 ModelError
        with self.assertRaises(ModelError):
            validate_c1_structure(
                {
                    "entities": [{"local_id": "e1", "name": "林舟", "kind": "person", "surfaces": ["林舟"], "evidence_ref": uid}],
                    "assignments": [{"surface": "阿舟", "ref": uid, "local_id": "e999", "reason": "bad"}],
                    "not_entity": [],
                },
                {"林舟", "阿舟", "陈叔"},
                {uid},
                Strict=True,
            )

        # 漏掉“陈叔”时应自动按自成对象修补并产生待审
        raw = {
            "entities": [{"local_id": "e1", "name": "林舟", "kind": "person", "surfaces": ["林舟", "阿舟"], "evidence_ref": uid}],
            "assignments": [],
            "not_entity": [],
        }
        res = apply_c1_task_result(source, task, grounded["mentions"], raw, "req_c1")
        self.assertIn("陈叔", res["repaired_surfaces"])
        self.assertEqual(len(res["reviews"]), 1)
        self.assertEqual(len(res["entities"]), 2)

    def test_weak_surface_does_not_merge_across_chapters_and_full_short_name_merges(self):
        # 构造“同一称谓在两章分别指两人”及“章内全名与简称归并”
        text = "第一章\n李府的老夫人疼爱林舟，大家都叫林舟为阿舟。张府的老夫人也来拜访。\n第二章\n王府的老夫人赏赐了阿舟。\n"
        path = self.root / "weak.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        u1 = tasks[0]["units"][0]["id"]
        u2 = tasks[1]["units"][0]["id"]

        source, run_id, tasks = self._prepare_two_chapter_run(
            text,
            {
                "mentions": [
                    {"text": "老夫人", "kind": "person", "refs": [u1]},
                    {"text": "林舟", "kind": "person", "refs": [u1]},
                    {"text": "阿舟", "kind": "person", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "老夫人", "kind": "person", "refs": [u2]},
                    {"text": "阿舟", "kind": "person", "refs": [u2]},
                ],
                "links": [],
            },
        )

        def handle_c1(state, call_no):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "林舟", "kind": "person", "surfaces": ["林舟", "阿舟"], "evidence_ref": uid},
                        {"local_id": "e2", "name": "老夫人", "kind": "person", "surfaces": [], "evidence_ref": uid},
                        {"local_id": "e3", "name": "老夫人", "kind": "person", "surfaces": [], "evidence_ref": uid},
                    ],
                    "assignments": [
                        {"surface": "老夫人", "ref": uid, "local_id": "e2", "reason": "李府的老夫人"},
                        {"surface": "老夫人", "ref": uid, "local_id": "e3", "reason": "张府的老夫人"},
                    ],
                    "not_entity": [],
                }, f"req_c1_{call_no}"
            return {
                "entities": [
                    {"local_id": "e1", "name": "阿舟", "kind": "person", "surfaces": ["阿舟"], "evidence_ref": uid},
                    {"local_id": "e2", "name": "老夫人", "kind": "person", "surfaces": ["老夫人"], "evidence_ref": uid},
                ],
                "assignments": [],
                "not_entity": [],
            }, f"req_c1_{call_no}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-review": lambda state, n: (
                {
                    "groups": [{"group_id": "g1", "name": "林舟", "kind": "person", "row_ids": [r["row_id"] for r in state["rows"]], "reason": "同指林舟"}],
                    "unresolved_row_ids": [],
                },
                f"req_c2_rev_{n}",
            ),
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"req_c2_prop_{n}"),
        })

        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"])
        out = self.store.records(run_id, "roster_output")[0]
        entities = out["entities"]
        # 林舟（2处）与阿舟（2处）归为同一实体（共4处提及），而三位“老夫人”各自独立，不因“老夫人”跨章合并
        lin = next(e for e in entities if "林舟" in e["names"])
        self.assertEqual(lin["mention_count"], 4)
        laofuren_groups = [e for e in entities if e["names"] == ["老夫人"]]
        self.assertEqual(len(laofuren_groups), 3)

    def test_c2_review_splits_different_objects_and_verify_insufficient_does_not_merge(self):
        text = "第一章\n李老夫人（人称老夫人）住在东街。苏沛之来访。\n第二章\n张老夫人（人称老夫人）住在西街。沛之兄告辞。\n"
        path = self.root / "split.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]

        source, run_id, tasks = self._prepare_two_chapter_run(
            text,
            {
                "mentions": [
                    {"text": "李老夫人", "kind": "person", "refs": [u1]},
                    {"text": "老夫人", "kind": "person", "refs": [u1]},
                    {"text": "苏沛之", "kind": "person", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "张老夫人", "kind": "person", "refs": [u2]},
                    {"text": "老夫人", "kind": "person", "refs": [u2]},
                    {"text": "沛之兄", "kind": "person", "refs": [u2]},
                ],
                "links": [],
            },
        )

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "李老夫人", "kind": "person", "surfaces": ["李老夫人", "老夫人"], "evidence_ref": uid},
                        {"local_id": "e2", "name": "苏沛之", "kind": "person", "surfaces": ["苏沛之"], "evidence_ref": uid},
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_{n}"
            return {
                "entities": [
                    {"local_id": "e1", "name": "张老夫人", "kind": "person", "surfaces": ["张老夫人", "老夫人"], "evidence_ref": uid},
                    {"local_id": "e2", "name": "沛之兄", "kind": "person", "surfaces": ["沛之兄"], "evidence_ref": uid},
                ],
                "assignments": [],
                "not_entity": [],
            }, f"c1_{n}"

        def handle_c2_review(state, n):
            # 把 r1 (李老夫人) 与 r2 (张老夫人) 拆成两个对象
            rows = state["rows"]
            return {
                "groups": [
                    {"group_id": "g1", "name": rows[0]["name"], "kind": "person", "row_ids": [rows[0]["row_id"]], "reason": "东街李老夫人"},
                    {"group_id": "g2", "name": rows[1]["name"], "kind": "person", "row_ids": [rows[1]["row_id"]], "reason": "西街张老夫人"},
                ],
                "unresolved_row_ids": [],
            }, f"c2_rev_{n}"

        def handle_c2_propose(state, n):
            entries = state["entries"]
            su = next(e["entry_id"] for e in entries if e["name"] == "苏沛之")
            pei = next(e["entry_id"] for e in entries if e["name"] == "沛之兄")
            return {"proposals": [{"left": su, "right": pei, "reason": "姓名与字号相似"}]}, f"c2_prop_{n}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-review": handle_c2_review,
            "roster-c2-propose": handle_c2_propose,
            "roster-c2-verify": lambda state, n: ({"verdict": "insufficient_evidence", "reason": "证据不足"}, f"c2_ver_{n}"),
        })

        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"])
        out = self.store.records(run_id, "roster_output")[0]
        # 李老夫人与张老夫人被拆开，苏沛之与沛之兄因 insufficient_evidence 不合并且生成待审
        self.assertEqual(len(out["entities"]), 4)
        self.assertTrue(any(r["stage"] == "roster-c2-verify" for r in out["reviews"]))

        # 验证断点恢复零新增调用
        calls_before = len(models.calls)
        again = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(again["complete"])
        self.assertEqual(len(models.calls), calls_before)

    def test_model_failure_is_not_treated_as_identity_conclusion(self):
        text = "第一章\n林舟来了。\n第二章\n林舟走了。\n"
        path = self.root / "fail.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        source, run_id, _ = self._prepare_two_chapter_run(
            text,
            {"mentions": [{"text": "林舟", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "林舟", "kind": "person", "refs": [u2]}], "links": []},
        )

        state_tracker = {"fail_first": True}

        def flaky_c1(state, n):
            if state_tracker["fail_first"]:
                state_tracker["fail_first"] = False
                raise ModelError("模拟网络超时")
            uid = state["units"][0]["id"]
            return {
                "entities": [{"local_id": "e1", "name": "林舟", "kind": "person", "surfaces": ["林舟"], "evidence_ref": uid}],
                "assignments": [],
                "not_entity": [],
            }, f"c1_{n}"

        models = ScriptedRosterModels({
            "roster-c1": flaky_c1,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_{n}"),
        })

        first = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertFalse(first["complete"])
        self.assertEqual(self.store.records(run_id, "roster_output"), [])

        second = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(second["complete"])
        # 仅失败的那 1 个任务重跑（首轮 2 次尝试：1 败 1 成；次轮仅重跑失败的 1 个 C1 任务 + 0 个 propose）
        self.assertEqual(len(self.store.records(run_id, "roster_output")[0]["entities"]), 1)

    def test_r1_same_unit_repeated_surface_assigned_by_mention_id(self):
        text = "第一章\n林舟对陈朗说，那个知县昨日见了另一个知县。\n第二章\n林舟远行。\n"
        path = self.root / "r1.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        _, chapters = build_chapters(source, [{"start": 0, "end": 3, "title": "第一章", "kind": "chapter"}])
        task = plan_tasks(source, chapters, 9000)[0]
        uid = task["units"][0]["id"]
        grounded = ground_result(
            source,
            task,
            {
                "mentions": [
                    {"text": "林舟", "kind": "person", "refs": [uid]},
                    {"text": "陈朗", "kind": "person", "refs": [uid]},
                    {"text": "知县", "kind": "person", "refs": [uid]},
                ],
                "links": [],
            },
            "req_r1",
        )
        zhixian_ms = [m for m in sorted(grounded["mentions"], key=lambda x: x["start"]) if m["text"] == "知县"]
        self.assertEqual(len(zhixian_ms), 2)
        raw = {
            "entities": [
                {
                    "local_id": "e1",
                    "name": "林舟",
                    "kind": "person",
                    "surfaces": [{"surface": "林舟", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e2",
                    "name": "陈朗",
                    "kind": "person",
                    "surfaces": [{"surface": "陈朗", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                },
            ],
            "assignments": [
                {"mention_id": zhixian_ms[0]["id"], "surface": "知县", "ref": uid, "local_id": "e1", "reason": "前一位知县指林舟"},
                {"mention_id": zhixian_ms[1]["id"], "surface": "知县", "ref": uid, "local_id": "e2", "reason": "后一位知县指陈朗"},
            ],
            "not_entity": [],
        }
        res = apply_c1_task_result(source, task, grounded["mentions"], raw, "req_c1_r1", strict=True, require_surface_type=True)
        outcomes = {o["mention_id"]: o for o in res["mention_outcomes"]}
        self.assertEqual(outcomes[zhixian_ms[0]["id"]]["local_id"], "e1")
        self.assertEqual(outcomes[zhixian_ms[1]["id"]]["local_id"], "e2")
        self.assertIn("知县", res["weak_norms"])

    def test_r2_surface_type_and_c3_candidate_linking_and_member_split(self):
        text = (
            "第一章\n苏沛之在省城见到林小姐。王员外也在座。\n"
            "第二章\n林小姐（闺名林婉）在花园读书。王员外再次出现。\n"
        )
        path = self.root / "r2.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        source, run_id, tasks = self._prepare_two_chapter_run(
            text,
            {
                "mentions": [
                    {"text": "苏沛之", "kind": "person", "refs": [u1]},
                    {"text": "林小姐", "kind": "person", "refs": [u1]},
                    {"text": "王员外", "kind": "person", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "林小姐", "kind": "person", "refs": [u2]},
                    {"text": "林婉", "kind": "person", "refs": [u2]},
                    {"text": "王员外", "kind": "person", "refs": [u2]},
                ],
                "links": [],
            },
        )

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                # 第一章中“林小姐”只含 title_or_kinship 非专名称呼，按 R2.3 不在 C2 自动合并，而送 C3 链接
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": "苏沛之",
                            "kind": "person",
                            "surfaces": [{"surface": "苏沛之", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                        {
                            "local_id": "e2",
                            "name": "林小姐",
                            "kind": "person",
                            "surfaces": [{"surface": "林小姐", "surface_type": "title_or_kinship"}],
                            "evidence_ref": uid,
                        },
                        {
                            "local_id": "e3",
                            "name": "王员外",
                            "kind": "person",
                            "surfaces": [{"surface": "王员外", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_{n}"
            return {
                "entities": [
                    {
                        "local_id": "e1",
                        "name": "林婉",
                        "kind": "person",
                        "surfaces": [
                            {"surface": "林婉", "surface_type": "proper_name"},
                            {"surface": "林小姐", "surface_type": "title_or_kinship"},
                        ],
                        "evidence_ref": uid,
                    },
                    {
                        "local_id": "e2",
                        "name": "王员外",
                        "kind": "person",
                        "surfaces": [{"surface": "王员外", "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    },
                ],
                "assignments": [],
                "not_entity": [],
            }, f"c1_{n}"

        def handle_c3(state, n):
            # 验证第一章只含 title_or_kinship 的“林小姐”进入 C3，并链接到第二章建立的“林婉”名册条目
            wan_cand = next(c["entity_id"] for c in state["candidates"] if "林婉" in c["strong_surfaces"] or c["name"] == "林婉")
            return {
                "links": [
                    {"mention_id": it["mention_id"], "selected_entity_id": wan_cand, "reason": "同指林婉"}
                    for it in state["items"]
                ]
            }, f"c3_{n}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_{n}"),
            "roster-c3-link": handle_c3,
        })

        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"])
        out = self.store.records(run_id, "roster_output")[0]
        wan = next(e for e in out["entities"] if "林婉" in e["names"])
        self.assertEqual(wan["mention_count"], 3)
        self.assertEqual(wan["link_methods"].get("c3_model"), 1)

    def test_roster_module_does_not_import_gold(self):
        source = (ROOT / "entitytrace/roster.py").read_text(encoding="utf-8")
        self.assertNotIn("entity-gold", source)
        self.assertNotIn("evaluate_identity_gold", source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith("tools"))


if __name__ == "__main__":
    unittest.main()
