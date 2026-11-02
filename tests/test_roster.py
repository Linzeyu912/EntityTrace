from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.models import ModelError
from entitytrace.roster import (
    C1_MAX_TOKENS,
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
        self.assertTrue(outcome["complete"], outcome)
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
        self.assertTrue(outcome["complete"], outcome)
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
        self.assertTrue(outcome["complete"], outcome)
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
        self.assertTrue(outcome["complete"], outcome)
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

    def test_a1_explicit_mention_assignment_beats_c2_0_separation_and_split_preserves_c1_kind(self):
        # 合成文本（不取自 4 本测试书）：
        # 1) 第一章“严知县”与“青州县”（地点）分属不同对象（产生 c_sep[("严知县", "青州县")] >= 1）；
        #    第二章同一片段既有地点“青州县”，又有借代主官的“青州县”（C1 通过 assignments 按 mention_id 显式指派给 e1“严知县”），
        #    验证 A1：C2.0 不会将该显式指派提及从 e1 剥离，最终该处借代提及留在“严知县”名册组内（chapter_local）。
        # 2) 同时在第二章 e1 中放入一个发现阶段 provisional_kinds 误标为 "place" 的 description 称呼“那位县尊”，
        #    验证 _split_mentions_to_new_local_obj 拆分出该 description 对象时保留 C1 的 kind="person"，不被 provisional_kinds("place") 覆写。
        text = (
            "第一章\n严知县抵达青州县履新。\n"
            "第二章\n公文送至青州县。督宪传见严知县，只请青州县入内，那位县尊当即起身。\n"
        )
        path = self.root / "a1.txt"
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
                    {"text": "严知县", "kind": "person", "refs": [u1]},
                    {"text": "青州县", "kind": "place", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "青州县", "kind": "place", "refs": [u2]},
                    {"text": "严知县", "kind": "person", "refs": [u2]},
                    # 故意将发现阶段 provisional_kinds 标为 place，验证拆分时不覆写 C1 的 person
                    {"text": "那位县尊", "kind": "place", "refs": [u2]},
                ],
                "links": [],
            },
        )

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": "严知县",
                            "kind": "person",
                            "surfaces": [{"surface": "严知县", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                        {
                            "local_id": "e2",
                            "name": "青州县",
                            "kind": "place",
                            "surfaces": [{"surface": "青州县", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_a1_{n}"

            qing_mentions = next(r["mentions"] for r in state["surfaces"] if r["surface"] == "青州县")
            self.assertEqual(len(qing_mentions), 2)
            return {
                "entities": [
                    {
                        "local_id": "e1",
                        "name": "严知县",
                        "kind": "person",
                        "surfaces": [
                            {"surface": "严知县", "surface_type": "proper_name"},
                            {"surface": "那位县尊", "surface_type": "description"},
                        ],
                        "evidence_ref": uid,
                    },
                    {
                        "local_id": "e2",
                        "name": "青州县",
                        "kind": "place",
                        "surfaces": [{"surface": "青州县", "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    },
                ],
                "assignments": [
                    {"mention_id": qing_mentions[0]["mention_id"], "surface": "青州县", "ref": uid, "local_id": "e2", "reason": "公文送至青州县指地点"},
                    {"mention_id": qing_mentions[1]["mention_id"], "surface": "青州县", "ref": uid, "local_id": "e1", "reason": "请青州县入内借指严知县"},
                ],
                "not_entity": [],
            }, f"c1_a1_{n}"

        c3_seen_candidates = {}

        def handle_c3(state, n):
            # 被 _split_mentions_to_new_local_obj 拆出的 description 对象“那位县尊”因保留了 kind="person"，
            # 召回的候选应当是 person（严知县），而非 place（青州县）
            for it in state["items"]:
                c3_seen_candidates[it["surface"]] = [
                    next(c["kind"] for c in state["candidates"] if c["entity_id"] == cid)
                    for cid in it["candidate_entity_ids"]
                ]
            yan_id = next(c["entity_id"] for c in state["candidates"] if c["name"] == "严知县")
            return {
                "links": [
                    {"mention_id": it["mention_id"], "selected_entity_id": yan_id, "reason": "指代严知县"}
                    for it in state["items"]
                ]
            }, f"c3_a1_{n}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-review": lambda state, n: (
                {
                    "groups": [
                        {
                            "group_id": "g1",
                            "name": state["rows"][0]["name"],
                            "kind": state["rows"][0]["kind"],
                            "row_ids": [r["row_id"] for r in state["rows"]],
                            "reason": "同指青州县地点",
                        }
                    ],
                    "unresolved_row_ids": [],
                },
                f"rev_a1_{n}",
            ),
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_a1_{n}"),
            "roster-c3-link": handle_c3,
        })

        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"], outcome)
        out = self.store.records(run_id, "roster_output")[0]
        yan_ent = next(e for e in out["entities"] if e["name"] == "严知县")
        qing_ent = next(e for e in out["entities"] if e["name"] == "青州县")
        # 严知县：第一章1处 + 第二章“严知县”1处 + 第二章借代“青州县”1处（未被 C2.0 剥离） + “那位县尊”1处 = 4处
        self.assertEqual(yan_ent["kind"], "person")
        self.assertEqual(yan_ent["mention_count"], 4)
        self.assertEqual(qing_ent["kind"], "place")
        self.assertEqual(qing_ent["mention_count"], 2)
        # 验证“那位县尊”被拆分后保持 person 粗类，仅召回 person 候选
        self.assertEqual(c3_seen_candidates.get("那位县尊"), ["person"])

    def test_c1_max_tokens_16000_and_other_stages_unchanged(self):
        import io
        import json
        from entitytrace.chapters import detect_chapters
        from entitytrace.config import Service
        from entitytrace.discovery import discover
        from entitytrace.models import Models

        self.assertEqual(C1_MAX_TOKENS, 16000)
        text = "第一章 启程\n沈墨在渡口遇见那位旅人。\n第二章 夜宿\n沈墨即沈墨子，与墨先生在客栈饮茶。\n"
        path = self.root / "max_tok.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        run_id = "run_max_tokens_check"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))

        stage_max_tokens: dict[str, list[int]] = {}

        def opener(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            sys_prompt = payload["messages"][0]["content"]
            user_state = json.loads(payload["messages"][1]["content"])
            mt = payload["max_tokens"]
            if "items" in user_state and "candidates" in user_state:
                stage_max_tokens.setdefault("roster-c3-link", []).append(mt)
                content = {
                    "links": [
                        {"mention_id": it["mention_id"], "selected_entity_id": "uncertain", "reason": "待审"}
                        for it in user_state["items"]
                    ]
                }
            elif "candidates" in user_state:
                stage_max_tokens.setdefault("chapter-label", []).append(mt)
                content = {
                    "headings": [
                        {"id": c["id"], "kind": "chapter"}
                        for c in user_state["candidates"]
                        if c["text"].startswith("第")
                    ]
                }
            elif "surfaces" in user_state:
                stage_max_tokens.setdefault("roster-c1", []).append(mt)
                surfs = {r["surface"]: r for r in user_state["surfaces"]}
                ents = []
                assigns = []
                if "沈墨" in surfs:
                    s_items = [{"surface": "沈墨", "surface_type": "proper_name"}]
                    if "沈墨子" in surfs:
                        s_items.append({"surface": "沈墨子", "surface_type": "proper_name"})
                    ents.append({
                        "local_id": "e1",
                        "name": "沈墨子" if "沈墨子" in surfs else "沈墨",
                        "kind": "person",
                        "surfaces": s_items,
                        "evidence_ref": surfs["沈墨"]["refs"][0],
                    })
                if "墨先生" in surfs:
                    ents.append({
                        "local_id": "e2",
                        "name": "墨先生",
                        "kind": "person",
                        "surfaces": [{"surface": "墨先生", "surface_type": "proper_name"}],
                        "evidence_ref": surfs["墨先生"]["refs"][0],
                    })
                if "那位旅人" in surfs:
                    m_id = surfs["那位旅人"]["mentions"][0]["mention_id"]
                    assigns.append({
                        "mention_id": m_id,
                        "surface": "那位旅人",
                        "ref": surfs["那位旅人"]["refs"][0],
                        "local_id": "null",
                        "reason": "身份未明",
                    })
                content = {"entities": ents, "assignments": assigns, "not_entity": []}
            elif "rows" in user_state:
                stage_max_tokens.setdefault("roster-c2-review", []).append(mt)
                content = {
                    "groups": [{
                        "group_id": "g1",
                        "name": user_state["rows"][0]["name"],
                        "kind": "person",
                        "row_ids": [r["row_id"] for r in user_state["rows"]],
                        "member_ids": [],
                        "reason": "同指",
                    }],
                    "unresolved_row_ids": [],
                    "unresolved_member_ids": [],
                }
            elif "entries" in user_state:
                stage_max_tokens.setdefault("roster-c2-propose", []).append(mt)
                ents = user_state["entries"]
                props = (
                    [{"left": ents[0]["entry_id"], "right": ents[1]["entry_id"], "reason": "同指候选"}]
                    if len(ents) >= 2
                    else []
                )
                content = {"proposals": props}
            elif "left" in user_state and "right" in user_state:
                stage_max_tokens.setdefault("roster-c2-verify", []).append(mt)
                content = {"verdict": "same", "reason": "同为沈墨"}
            elif "items" in user_state and "candidates" in user_state:
                stage_max_tokens.setdefault("roster-c3-link", []).append(mt)
                content = {
                    "links": [
                        {"mention_id": it["mention_id"], "selected_entity_id": "uncertain", "reason": "待审"}
                        for it in user_state["items"]
                    ]
                }
            else:
                stage_max_tokens.setdefault("discovery", []).append(mt)
                ms = []
                for u in user_state["units"]:
                    for s in ("沈墨子", "沈墨", "那位旅人", "墨先生"):
                        if s in u["text"]:
                            ms.append({"text": s, "kind": "person", "refs": [u["id"]]})
                content = {"mentions": ms, "links": []}
            body = {"choices": [{"finish_reason": "stop", "message": {"content": dumps(content)}}]}
            return io.BytesIO(dumps(body).encode("utf-8"))

        svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")
        models = Models(self.store, run_id, "inv1", {"discovery": svc}, discovery_concurrency=1, opener=opener)
        ch_out = detect_chapters(self.store, models, source, run_id, lambda _: None)
        self.assertTrue(ch_out["complete"])
        disc_out = discover(
            self.store, models, source, run_id, self.store.records(run_id, "chapter"), 9000, 1, lambda _: None
        )
        self.assertTrue(disc_out["complete"])
        ros_out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(ros_out["complete"])

        self.assertEqual(set(stage_max_tokens["roster-c1"]), {16000})
        self.assertEqual(set(stage_max_tokens["chapter-label"]), {4000})
        self.assertEqual(set(stage_max_tokens["discovery"]), {8000})
        self.assertEqual(set(stage_max_tokens["roster-c2-review"]), {8000})
        self.assertEqual(set(stage_max_tokens["roster-c2-propose"]), {8000})
        self.assertEqual(set(stage_max_tokens["roster-c2-verify"]), {8000})
        self.assertEqual(set(stage_max_tokens["roster-c3-link"]), {8000})

    def test_c1_assignment_string_null_tolerance_and_defined_null_and_invalid_local_id(self):
        text = "第一章\n林掌柜向知县禀报，差役在堂下候命。\n"
        path = self.root / "null_tol.txt"
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
                    {"text": "林掌柜", "kind": "person", "refs": [uid]},
                    {"text": "知县", "kind": "person", "refs": [uid]},
                    {"text": "差役", "kind": "person", "refs": [uid]},
                ],
                "links": [],
            },
            "req_null",
        )
        mentions = grounded["mentions"]
        m_zhixian = next(m for m in mentions if m["text"] == "知县")
        m_chaiyi = next(m for m in mentions if m["text"] == "差役")
        allowed_surfs = {"林掌柜", "知县", "差役"}
        allowed_refs = {uid}
        allowed_mids = {m["id"] for m in mentions} | {"m1", "m2", "m3"}

        raw_json_null = {
            "entities": [
                {
                    "local_id": "e1",
                    "name": "林掌柜",
                    "kind": "person",
                    "surfaces": [{"surface": "林掌柜", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                }
            ],
            "assignments": [
                {"mention_id": m_zhixian["id"], "surface": "知县", "ref": uid, "local_id": None, "reason": "未决"},
                {"mention_id": m_chaiyi["id"], "surface": "差役", "ref": uid, "local_id": "not_entity", "reason": "泛称"},
            ],
            "not_entity": [],
        }
        raw_str_null = {
            "entities": raw_json_null["entities"],
            "assignments": [
                {"mention_id": m_zhixian["id"], "surface": "知县", "ref": uid, "local_id": "null", "reason": "未决"},
                {"mention_id": m_chaiyi["id"], "surface": "差役", "ref": uid, "local_id": "not_entity", "reason": "泛称"},
            ],
            "not_entity": [],
        }

        validate_c1_structure(raw_str_null, allowed_surfs, allowed_refs, Strict=True, allowed_mention_ids=allowed_mids)
        out_json_null = apply_c1_task_result(source, task, mentions, raw_json_null, "req_1", strict=True)
        out_str_null = apply_c1_task_result(source, task, mentions, raw_str_null, "req_1", strict=True)
        self.assertEqual(out_str_null, out_json_null)
        zhixian_outcome = next(o for o in out_str_null["mention_outcomes"] if o["mention_id"] == m_zhixian["id"])
        self.assertEqual(zhixian_outcome["status"], "unresolved")
        self.assertIsNone(zhixian_outcome["local_id"])

        # 当 "null" 恰为本次返回中已定义的 local_id 时，按已定义对象处理
        raw_defined_null = {
            "entities": [
                {
                    "local_id": "e1",
                    "name": "林掌柜",
                    "kind": "person",
                    "surfaces": [{"surface": "林掌柜", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "null",
                    "name": "知县",
                    "kind": "person",
                    "surfaces": [],
                    "evidence_ref": uid,
                },
            ],
            "assignments": [
                {"mention_id": m_zhixian["id"], "surface": "知县", "ref": uid, "local_id": "null", "reason": "指向已定义对象 null"},
                {"mention_id": m_chaiyi["id"], "surface": "差役", "ref": uid, "local_id": "not_entity", "reason": "泛称"},
            ],
            "not_entity": [],
        }
        validate_c1_structure(raw_defined_null, allowed_surfs, allowed_refs, Strict=True, allowed_mention_ids=allowed_mids)
        out_defined_null = apply_c1_task_result(source, task, mentions, raw_defined_null, "req_2", strict=True)
        zhixian_defined = next(o for o in out_defined_null["mention_outcomes"] if o["mention_id"] == m_zhixian["id"])
        self.assertEqual(zhixian_defined["status"], "assigned")
        self.assertEqual(zhixian_defined["local_id"], "null")

        # 其他未定义字符串 local_id 照旧报错
        for bad_lid in ("e999", "none", "unknown", "NULL"):
            raw_bad = {
                "entities": raw_json_null["entities"],
                "assignments": [
                    {"mention_id": m_zhixian["id"], "surface": "知县", "ref": uid, "local_id": bad_lid, "reason": "非法"},
                ],
                "not_entity": [],
            }
            with self.assertRaises(ModelError):
                validate_c1_structure(raw_bad, allowed_surfs, allowed_refs, Strict=True, allowed_mention_ids=allowed_mids)

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
