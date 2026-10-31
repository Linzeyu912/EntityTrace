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

    def test_d1_c3_proper_name_consistency_and_post_model_rejection(self):
        # 合成文本（不取自 4 本测试书）：测试 D1 含专名对象在 C3 的专名一致性硬约束（D1.1 无同名候选不调 C3、D1.2 模型越界返回异名被拦截为 c3_rejected_name_mismatch）及纯非专名对象保持不变
        text = (
            "第一章\n周牧野与沈知远在驿站饮茶，众人尊称周牧野为督办大人。\n"
            "第二章\n韩子敬单独拜访督办大人，沈知远也在场。\n"
        )
        path = self.root / "d1.txt"
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
                    {"text": "周牧野", "kind": "person", "refs": [u1]},
                    {"text": "沈知远", "kind": "person", "refs": [u1]},
                    {"text": "督办大人", "kind": "person", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "韩子敬", "kind": "person", "refs": [u2]},
                    {"text": "督办大人", "kind": "person", "refs": [u2]},
                    {"text": "沈知远", "kind": "person", "refs": [u2]},
                ],
                "links": [],
            },
        )

        # 场景 A（D1.1）：第二章中“韩子敬”被 C1 标为 proper_name 但误挂入非专名对象 e2，由于名册中无“韩子敬”条目，
        # e2 的候选池被 D1.1 过滤为空，不调用 C3 模型，直接留为未决/单列，绝不误链到同章的“沈知远”
        def handle_c1_no_cand(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": "周牧野",
                            "kind": "person",
                            "surfaces": [
                                {"surface": "周牧野", "surface_type": "proper_name"},
                                {"surface": "督办大人", "surface_type": "title_or_kinship"},
                            ],
                            "evidence_ref": uid,
                        },
                        {
                            "local_id": "e2",
                            "name": "沈知远",
                            "kind": "person",
                            "surfaces": [{"surface": "沈知远", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_{n}"
            m_han = next(r["mentions"][0]["mention_id"] for r in state["surfaces"] if r["surface"] == "韩子敬")
            m_duban = next(r["mentions"][0]["mention_id"] for r in state["surfaces"] if r["surface"] == "督办大人")
            return {
                "entities": [
                    {
                        "local_id": "e1",
                        "name": "韩子敬",
                        "kind": "person",
                        "surfaces": [{"surface": "韩子敬", "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    },
                    {
                        "local_id": "e2",
                        "name": "督办大人",
                        "kind": "person",
                        "surfaces": [{"surface": "督办大人", "surface_type": "title_or_kinship"}],
                        "evidence_ref": uid,
                    },
                    {
                        "local_id": "e3",
                        "name": "沈知远",
                        "kind": "person",
                        "surfaces": [{"surface": "沈知远", "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    },
                ],
                "assignments": [
                    {"mention_id": m_han, "surface": "韩子敬", "ref": uid, "local_id": "e2", "reason": "误挂到非专名对象"},
                    {"mention_id": m_duban, "surface": "督办大人", "ref": uid, "local_id": "e2", "reason": "称呼督办"},
                ],
                "not_entity": [],
            }, f"c1_{n}"

        c3_called = {"count": 0}

        def handle_c3_should_not_be_called(state, n):
            c3_called["count"] += 1
            return {"links": []}, f"c3_{n}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1_no_cand,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_{n}"),
            "roster-c3-link": handle_c3_should_not_be_called,
        })

        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"])
        self.assertEqual(c3_called["count"], 0)
        out = self.store.records(run_id, "roster_output")[0]
        shen = next(e for e in out["entities"] if "沈知远" in e["names"])
        self.assertNotIn("韩子敬", shen["names"])
        self.assertEqual(shen["mention_count"], 2)

        # 场景 B（D1.2 + 纯非专名对象不变）：当含专名未决项在名册中有同名候选从而进入 C3，但 C3 模型故意返回同任务另一异名候选时，
        # 后验校验将其作废并记录 link_method="c3_rejected_name_mismatch"，而纯非专名对象“督办大人”仍正常 c3_model 链接
        self.store.db.execute("DELETE FROM tasks WHERE run_id=? AND stage LIKE 'roster%'", (run_id,))
        self.store.db.execute("DELETE FROM records WHERE run_id=? AND kind='roster_output'", (run_id,))
        self.store.db.commit()

        def handle_c1_for_post_check(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": "周牧野",
                            "kind": "person",
                            "surfaces": [{"surface": "周牧野", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                        {
                            "local_id": "e2",
                            "name": "沈知远",
                            "kind": "person",
                            "surfaces": [{"surface": "沈知远", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                        {
                            "local_id": "e3",
                            "name": "督办大人",
                            "kind": "person",
                            "surfaces": [{"surface": "督办大人", "surface_type": "title_or_kinship"}],
                            "evidence_ref": uid,
                        },
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_b_{n}"
            # 第二章：让“沈知远”作为未归属残余提及进入 C3（通过 assignments local_id=None），同时让“督办大人”作为纯非专名对象进入 C3
            m_shen = next(r["mentions"][0]["mention_id"] for r in state["surfaces"] if r["surface"] == "沈知远")
            return {
                "entities": [
                    {
                        "local_id": "e1",
                        "name": "韩子敬",
                        "kind": "person",
                        "surfaces": [{"surface": "韩子敬", "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    },
                    {
                        "local_id": "e2",
                        "name": "督办大人",
                        "kind": "person",
                        "surfaces": [{"surface": "督办大人", "surface_type": "title_or_kinship"}],
                        "evidence_ref": uid,
                    },
                ],
                "assignments": [
                    {"mention_id": m_shen, "surface": "沈知远", "ref": uid, "local_id": None, "reason": "模拟未决专名提及"},
                ],
                "not_entity": [],
            }, f"c1_b_{n}"

        def handle_c3_hijack(state, n):
            # 将所有进入 C3 的项都故意链向异名条目“韩子敬”（若有）或第一个非沈知远条目
            wrong_id = next(
                c["entity_id"] for c in state["candidates"] if c["name"] != "沈知远"
            )
            return {
                "links": [
                    {"mention_id": it["mention_id"], "selected_entity_id": wrong_id, "reason": "故意返回异名候选测试后验拦截"}
                    for it in state["items"]
                ]
            }, f"c3_b_{n}"

        models2 = ScriptedRosterModels({
            "roster-c1": handle_c1_for_post_check,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_b_{n}"),
            "roster-c3-link": handle_c3_hijack,
        })
        outcome2 = resolve_roster(self.store, models2, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome2["complete"])
        out2 = self.store.records(run_id, "roster_output")[0]
        self.assertGreaterEqual(out2["roster_summary"].get("c3_rejected_name_mismatch", 0), 1)
        rejected_ents = [
            e for e in out2["entities"] if e.get("link_methods", {}).get("c3_rejected_name_mismatch", 0) > 0
        ]
        self.assertTrue(rejected_ents)
        # 同时验证场景 B 中纯非专名对象“督办大人”正常通过 c3_model 链接
        han_b = next(e for e in out2["entities"] if "韩子敬" in e["names"])
        self.assertGreaterEqual(han_b["link_methods"].get("c3_model", 0), 1)

    def test_d2_metonymy_place_and_person_assignments_and_mention_surface_validation(self):
        # 合成文本（不取自 4 本测试书）：同一任务内“云州县”既指辖区又借代该县知县，且验证 assignments 中 mention_id 与 surface 不一致时不污染专名提及
        text = "第一章\n钦使到了云州县。总督只请臬司与云州县进内室议事，赵迁之叔正是那位钦使。\n第二章\n云州县城门紧闭。\n"
        path = self.root / "d2.txt"
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
                    {"text": "钦使", "kind": "person", "refs": [uid]},
                    {"text": "云州县", "kind": "place", "refs": [uid]},
                    {"text": "总督", "kind": "person", "refs": [uid]},
                    {"text": "臬司", "kind": "person", "refs": [uid]},
                    {"text": "赵迁", "kind": "person", "refs": [uid]},
                ],
                "links": [],
            },
            "req_d2",
        )
        yun_ms = [m for m in sorted(grounded["mentions"], key=lambda x: x["start"]) if m["text"] == "云州县"]
        zhao_m = next(m for m in grounded["mentions"] if m["text"] == "赵迁")
        self.assertEqual(len(yun_ms), 2)

        raw = {
            "entities": [
                {
                    "local_id": "e_place",
                    "name": "云州县",
                    "kind": "place",
                    "surfaces": [],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e_magistrate",
                    "name": "云州县",
                    "kind": "person",
                    "surfaces": [],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e_envoy",
                    "name": "钦使",
                    "kind": "person",
                    "surfaces": [{"surface": "钦使", "surface_type": "title_or_kinship"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e_zhao",
                    "name": "赵迁",
                    "kind": "person",
                    "surfaces": [{"surface": "赵迁", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e_gov",
                    "name": "总督",
                    "kind": "person",
                    "surfaces": [{"surface": "总督", "surface_type": "title_or_kinship"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e_judge",
                    "name": "臬司",
                    "kind": "person",
                    "surfaces": [{"surface": "臬司", "surface_type": "title_or_kinship"}],
                    "evidence_ref": uid,
                },
            ],
            "assignments": [
                {"mention_id": yun_ms[0]["id"], "surface": "云州县", "ref": uid, "local_id": "e_place", "reason": "到了云州县指辖区地点"},
                {"mention_id": yun_ms[1]["id"], "surface": "云州县", "ref": uid, "local_id": "e_magistrate", "reason": "与臬司并列进内室议事，借代云州县主官"},
                # 模拟模型误将赵迁的 mention_id 填给上下文词“之叔”并指向 e_envoy
                {"mention_id": zhao_m["id"], "surface": "之叔", "ref": uid, "local_id": "e_envoy", "reason": "赵迁之叔是钦使"},
            ],
            "not_entity": [],
        }

        res = apply_c1_task_result(source, task, grounded["mentions"], raw, "req_c1_d2", strict=True, require_surface_type=True)
        outcomes = {o["mention_id"]: o for o in res["mention_outcomes"]}
        # 第一个“云州县”归地点 e_place，第二个“云州县”归人物 e_magistrate
        self.assertEqual(outcomes[yun_ms[0]["id"]]["local_id"], "e_place")
        self.assertEqual(outcomes[yun_ms[1]["id"]]["local_id"], "e_magistrate")
        # 赵迁不被错误挂入 e_envoy，而是归入 e_zhao
        self.assertEqual(outcomes[zhao_m["id"]]["local_id"], "e_zhao")

        by_local = {e["local_id"]: e for e in res["entities"]}
        # 地点对象的“云州县”保持为专名，人物借代对象的“云州县”不能作为人物 proper_surfaces
        self.assertIn("云州县", by_local["e_place"]["proper_surfaces"])
        self.assertEqual(by_local["e_magistrate"]["proper_surfaces"], [])
        self.assertTrue(by_local["e_magistrate"]["only_non_proper"])

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
