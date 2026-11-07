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
    C2_ROW_MEMBER_CAP,
    VERSION,
    _check_c3_links,
    apply_c1_task_result,
    compute_prelim_entry_ids,
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

    def test_cannot_merge_adjacency_index_matches_pairwise_scan(self):
        import collections
        import random

        rng = random.Random(20261103)
        obj_pool = [(f"t{i // 5}", f"e{i % 5}") for i in range(60)]
        cannot_merge_objs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
        for _ in range(250):
            a, b = rng.sample(obj_pool, 2)
            cannot_merge_objs.add((min(a, b), max(a, b)))

        cannot_merge_by_obj: dict[tuple[str, str], set[tuple[str, str]]] = collections.defaultdict(set)
        for ka, kb in cannot_merge_objs:
            cannot_merge_by_obj[ka].add(kb)
            cannot_merge_by_obj[kb].add(ka)

        for k in obj_pool:
            old_forb = {kb for (ka, kb) in cannot_merge_objs if ka == k} | {ka for (ka, kb) in cannot_merge_objs if kb == k}
            new_forb = cannot_merge_by_obj.get(k, set())
            self.assertEqual(new_forb, old_forb)

    def test_c2_cross_batch_continuation_and_row_member_cap(self):
        self.assertEqual(C2_ROW_MEMBER_CAP, 100)
        # 构造 130 个不同折叠行的同一候选簇（分 3 批：60 + 60 + 10）
        lines = []
        for idx in range(1, 131):
            alias = f"云山叟{idx:03d}"
            lines.append(f"第{idx}章\n沈青云（号{alias}）在此处驻足。\n")
        text = "".join(lines)
        path = self.root / "c2_batches.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        headings = []
        pos = 0
        for idx in range(1, 131):
            title = f"第{idx}章"
            start = text.index(title, pos)
            headings.append({"start": start, "end": start + len(title), "title": title, "kind": "chapter"})
            pos = start + len(title)
        _, chapters = build_chapters(source, headings)
        tasks = plan_tasks(source, chapters, 9000)

        def _setup_run(run_id: str) -> None:
            self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
            for ch in chapters:
                self.store.put(run_id, "chapter", ch)
            for idx, t in enumerate(tasks, start=1):
                alias = f"云山叟{idx:03d}"
                u = t["units"][0]
                disc = {
                    "mentions": [
                        {"text": "沈青云", "kind": "person", "refs": [u["id"]]},
                        {"text": alias, "kind": "person", "refs": [u["id"]]},
                    ],
                    "links": [],
                }
                grounded = ground_result(source, t, disc, f"req_disc_{run_id}_{idx}")
                records = [
                    (kind, rec)
                    for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                    for rec in grounded[key]
                ]
                self.store.finish_task(run_id, t, {"request_id": f"req_disc_{run_id}_{idx}"}, records)

        def handle_c1_130(state, n):
            u = state["units"][0]
            alias = next(r["surface"] for r in state["surfaces"] if r["surface"] != "沈青云")
            return {
                "entities": [{
                    "local_id": "e1",
                    "name": alias,
                    "kind": "person",
                    "surfaces": [
                        {"surface": "沈青云", "surface_type": "proper_name"},
                        {"surface": alias, "surface_type": "proper_name"},
                    ],
                    "evidence_ref": u["id"],
                }],
                "assignments": [],
                "not_entity": [],
            }, f"c1_{n}"

        # 1) 3 批同名续接成 1 组，且跨批无约束
        _setup_run("run_3batch_merge")
        c2_rev_calls_1 = []

        def handle_c2_rev_same_name(state, n):
            c2_rev_calls_1.append(state)
            return {
                "groups": [{
                    "group_id": "g1",
                    "name": "沈青云",
                    "kind": "person",
                    "row_ids": [r["row_id"] for r in state["rows"]],
                    "member_ids": [],
                    "reason": "同指沈青云",
                }],
                "unresolved_row_ids": [],
                "unresolved_member_ids": [],
            }, f"c2_rev_{n}"

        models_1 = ScriptedRosterModels({
            "roster-c1": handle_c1_130,
            "roster-c2-review": handle_c2_rev_same_name,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_{n}"),
        })
        out1 = resolve_roster(self.store, models_1, source, "run_3batch_merge", concurrency=1, log=lambda _: None)
        self.assertTrue(out1["complete"])
        self.assertEqual(len(c2_rev_calls_1), 3)
        roster_out_1 = self.store.records("run_3batch_merge", "roster_output")[0]
        self.assertEqual(len(roster_out_1["entities"]), 1)
        self.assertEqual(roster_out_1["entities"][0]["mention_count"], 260)

        # 2) 先前同名组 >=2 个时不续接；本批同名组 >=2 个时不续接；粗类不兼容（person 对 place）时不续接；
        #    同批拆开的两组之间有约束（C2 补合并被拦截），跨批且未同批出现的组之间无约束（可经 C2 补合并合拢）
        _setup_run("run_guard_and_constraints")
        rev_batch_seq = {"idx": 0}

        def handle_c2_rev_complex(state, n):
            b = rev_batch_seq["idx"]
            rev_batch_seq["idx"] += 1
            rids = [r["row_id"] for r in state["rows"]]
            if b == 0:
                # 第 1 批：输出两个同名组 "同名甲"（使得先前同名组 >= 2），以及 1 个 "沈青云" (person) 和 1 个被同批拆开的 "沈青河" (person)
                return {
                    "groups": [
                        {"group_id": "g1", "name": "同名甲", "kind": "person", "row_ids": [rids[0]], "member_ids": [], "reason": "甲1"},
                        {"group_id": "g2", "name": "同名甲", "kind": "person", "row_ids": [rids[1]], "member_ids": [], "reason": "甲2"},
                        {"group_id": "g3", "name": "沈青云", "kind": "person", "row_ids": rids[2:30], "member_ids": [], "reason": "青云"},
                        {"group_id": "g4", "name": "沈青河", "kind": "person", "row_ids": rids[30:], "member_ids": [], "reason": "同批拆开"},
                    ],
                    "unresolved_row_ids": [],
                    "unresolved_member_ids": [],
                }, f"c2_rev_c_{n}"
            if b == 1:
                # 第 2 批：
                # - "同名甲" 因先前同名组有 2 个，不续接；
                # - "本批双名" 在本批出现 2 次，不续接；
                # - "沈青云" 但 kind="place"（与先前 person 不兼容），不续接；
                # - "跨批别称"（与第 1 批从未同批出现，无跨批约束，后续可由 C2 补合并合入 "沈青云"）
                return {
                    "groups": [
                        {"group_id": "g1", "name": "同名甲", "kind": "person", "row_ids": [rids[0]], "member_ids": [], "reason": "先前有2个同名"},
                        {"group_id": "g2", "name": "本批双名", "kind": "person", "row_ids": [rids[1]], "member_ids": [], "reason": "本批重复1"},
                        {"group_id": "g3", "name": "本批双名", "kind": "person", "row_ids": [rids[2]], "member_ids": [], "reason": "本批重复2"},
                        {"group_id": "g4", "name": "沈青云", "kind": "place", "row_ids": [rids[3]], "member_ids": [], "reason": "粗类不兼容"},
                        {"group_id": "g5", "name": "跨批别称", "kind": "person", "row_ids": rids[4:], "member_ids": [], "reason": "跨批独立组"},
                    ],
                    "unresolved_row_ids": [],
                    "unresolved_member_ids": [],
                }, f"c2_rev_c_{n}"
            # 第 3 批：输出唯一的 "沈青云" (person)，由于此时先前名为 "沈青云" 的组有 2 个（b0 的 person 与 b1 的 place），len(prior_matches)==2 不续接
            return {
                "groups": [
                    {"group_id": "g1", "name": "沈青云", "kind": "person", "row_ids": rids, "member_ids": [], "reason": "第三批"},
                ],
                "unresolved_row_ids": [],
                "unresolved_member_ids": [],
            }, f"c2_rev_c_{n}"

        propose_seen = []

        def handle_c2_propose_check_constraints(state, n):
            if state["kind"] != "person":
                return {"proposals": []}, f"prop_c_{n}"
            entries = state["entries"]
            propose_seen.extend(entries)
            # 找出第 1 批的 "沈青云"（28行=56提及）、第 1 批同批拆开的 "沈青河"（30行=60提及）、第 2 批跨批的 "跨批别称"（56行=112提及）
            e_qingyun = next(e["entry_id"] for e in entries if e["mention_count"] == 56)
            e_qinghe = next(e["entry_id"] for e in entries if e["mention_count"] == 60)
            e_cross = next(e["entry_id"] for e in entries if e["mention_count"] == 112)
            # 同时提议合并 (e_qingyun, e_qinghe) [同批拆开，应被不可合并约束拦截] 与 (e_qingyun, e_cross) [跨批未同批出现，无约束应放行]
            return {
                "proposals": [
                    {"left": e_qingyun, "right": e_qinghe, "reason": "同批拆开测试拦截"},
                    {"left": e_qingyun, "right": e_cross, "reason": "跨批无约束测试放行"},
                ]
            }, f"prop_c_{n}"

        verify_calls = []

        def handle_c2_verify_check(state, n):
            verify_calls.append(state)
            return {"verdict": "same", "reason": "确认为同一人"}, f"ver_c_{n}"

        models_2 = ScriptedRosterModels({
            "roster-c1": handle_c1_130,
            "roster-c2-review": handle_c2_rev_complex,
            "roster-c2-propose": handle_c2_propose_check_constraints,
            "roster-c2-verify": handle_c2_verify_check,
        })
        out2 = resolve_roster(self.store, models_2, source, "run_guard_and_constraints", concurrency=1, log=lambda _: None)
        self.assertTrue(out2["complete"])
        # 同批拆开的 (e_qingyun, e_qinghe) 在生成 verify 任务前已被 cannot_merge_objs 过滤，只有跨批的 (e_qingyun, e_cross) 进入 verify 并合并為 56+112=168 提及
        self.assertEqual(len(verify_calls), 1)
        roster_out_2 = self.store.records("run_guard_and_constraints", "roster_output")[0]
        counts_2 = sorted(e["mention_count"] for e in roster_out_2["entities"])
        # 10个组的提及数：b0甲1(2), b0甲2(2), b1甲(2), b1双1(2), b1双2(2), b1place(2), b2青云(20), b0青河(60), 以及合并后的 b0青云+b1跨批(56+112=168)
        self.assertEqual(counts_2, [2, 2, 2, 2, 2, 2, 20, 60, 168])

        # 3) 150 个成员的行只展示 100 个，取样确定且含首尾，带 members_total；未展示的成员随整行归属；已展示的成员仍可经 member_ids 单独拆出
        lines_cap = []
        for idx in range(1, 152):
            if idx == 151:
                lines_cap.append(f"第{idx}章\n陆云川与云川君在此练剑。\n")
            else:
                lines_cap.append(f"第{idx}章\n陆云川在此练剑。\n")
        text_cap = "".join(lines_cap)
        path_cap = self.root / "cap150.txt"
        path_cap.write_bytes(text_cap.encode("utf-8"))
        source_cap = import_text(self.store, path_cap)
        headings_cap = []
        pos = 0
        for idx in range(1, 152):
            title = f"第{idx}章"
            start = text_cap.index(title, pos)
            headings_cap.append({"start": start, "end": start + len(title), "title": title, "kind": "chapter"})
            pos = start + len(title)
        _, chs_cap = build_chapters(source_cap, headings_cap)
        run_cap = "run_cap_150"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_cap, source_cap["id"], "{}", "{}", "now", "now"))
        for ch in chs_cap:
            self.store.put(run_cap, "chapter", ch)
        tasks_cap = plan_tasks(source_cap, chs_cap, 9000)
        for idx, t in enumerate(tasks_cap, start=1):
            u = t["units"][0]
            ms = [{"text": "陆云川", "kind": "person", "refs": [u["id"]]}]
            if idx == 151:
                ms.append({"text": "云川君", "kind": "person", "refs": [u["id"]]})
            disc = {"mentions": ms, "links": []}
            grounded = ground_result(source_cap, t, disc, "req_" + t["id"])
            recs = [
                (kind, rec)
                for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                for rec in grounded[key]
            ]
            self.store.finish_task(run_cap, t, {"request_id": "req_" + t["id"]}, recs)

        def handle_c1_cap(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第151章":
                # 第 1..150 章各产生 1 个只含“陆云川”的对象（折叠为 r1 的 150 个成员），第 151 章产生 1 个含“陆云川”与“云川君”的对象（r2，1 个成员）
                return {
                    "entities": [{
                        "local_id": "e1",
                        "name": "云川君",
                        "kind": "person",
                        "surfaces": [
                            {"surface": "陆云川", "surface_type": "proper_name"},
                            {"surface": "云川君", "surface_type": "proper_name"},
                        ],
                        "evidence_ref": uid,
                    }],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_cap_{n}"
            return {
                "entities": [{
                    "local_id": "e1",
                    "name": "陆云川",
                    "kind": "person",
                    "surfaces": [{"surface": "陆云川", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                }],
                "assignments": [],
                "not_entity": [],
            }, f"c1_cap_{n}"

        seen_c2_rows = []

        def handle_c2_rev_cap(state, n):
            rows = state["rows"]
            seen_c2_rows.extend(rows)
            r1 = next(r for r in rows if r.get("members_total") == 150)
            r2 = next(r for r in rows if r["row_id"] != r1["row_id"])
            # 把 r1 中最后 1 个展示的成员经 member_ids 单独拆到 g2，其余未展示与已展示成员随 r1 归入 g1
            last_mid = r1["members"][-1]["member_id"]
            return {
                "groups": [
                    {"group_id": "g1", "name": "陆云川", "kind": "person", "row_ids": [r1["row_id"]], "member_ids": [], "reason": "主群"},
                    {"group_id": "g2", "name": "云川君", "kind": "person", "row_ids": [r2["row_id"]], "member_ids": [last_mid], "reason": "拆出1个展示成员"},
                ],
                "unresolved_row_ids": [],
                "unresolved_member_ids": [],
            }, f"c2_rev_cap_{n}"

        models_cap = ScriptedRosterModels({
            "roster-c1": handle_c1_cap,
            "roster-c2-review": handle_c2_rev_cap,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_cap_{n}"),
        })
        out_cap = resolve_roster(self.store, models_cap, source_cap, run_cap, concurrency=1, log=lambda _: None)
        self.assertTrue(out_cap["complete"])
        r1_row = next(r for r in seen_c2_rows if r.get("members_total") == 150)
        r2_row = next(r for r in seen_c2_rows if r["row_id"] != r1_row["row_id"])
        self.assertEqual(r1_row["members_total"], 150)
        self.assertNotIn("members_total", r2_row)
        self.assertEqual(len(r1_row["members"]), 100)
        rid_prefix = r1_row["row_id"]
        # 验证首尾成员均被包含且顺序与原 member_id 编号递增一致
        mids_num = [int(m["member_id"].split("_m")[1]) for m in r1_row["members"]]
        self.assertEqual(mids_num, sorted(mids_num))
        self.assertEqual(len(set(mids_num)), 100)
        ch_id_to_ord = {c["id"]: c["ordinal"] for c in chs_cap}
        member_ch_ords = [ch_id_to_ord[m["chapter_id"]] for m in r1_row["members"]]
        self.assertIn(1, member_ch_ords)
        self.assertIn(150, member_ch_ords)
        roster_cap = self.store.records(run_cap, "roster_output")[0]
        cap_counts = sorted(e["mention_count"] for e in roster_cap["entities"])
        # 150 个 r1 成员中 1 个被拆入 g2（连同 r2 的 2 处提及共 3 处提及），其余 149 个（含 50 个未展示成员）全部随整行归属 g1
        self.assertEqual(cap_counts, [3, 149])

    def test_c3_invalid_output_turns_into_review_without_blocking_and_http_error_still_blocks(self):
        from entitytrace.pipeline import infer_incomplete_stage_error

        # 1) 验证 _check_c3_links 报错附上非法值前 80 字符
        bad_mid_str = "obj:" + ("a" * 100) + ":e3"
        with self.assertRaises(ModelError) as ctx_mid:
            _check_c3_links({"links": [{"mention_id": bad_mid_str, "selected_entity_id": "none"}]}, {"m1"}, {"none", "uncertain"})
        self.assertIn("C3 link mention_id 非法：", str(ctx_mid.exception))
        self.assertIn("obj:aaaa", str(ctx_mid.exception))
        self.assertLessEqual(len(str(ctx_mid.exception)), len("C3 link mention_id 非法：") + 80)

        bad_sel_str = "cand_" + ("b" * 100)
        with self.assertRaises(ModelError) as ctx_sel:
            _check_c3_links({"links": [{"mention_id": "m1", "selected_entity_id": bad_sel_str}]}, {"m1"}, {"none", "uncertain"})
        self.assertIn("C3 selected_entity_id 非法：", str(ctx_sel.exception))
        self.assertIn("cand_bbbb", str(ctx_sel.exception))

        # 2) 构造两章运行：第一章同时含一个非专名对象（林小姐）和一个未决单提及（那位公子）进入 C3
        text = (
            "第一章\n苏沛之在省城见到林小姐与那位公子。\n"
            "第二章\n林小姐（闺名林婉）与苏沛之饮茶。\n"
        )
        path = self.root / "c3_invalid.txt"
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

        def _prep(run_id: str) -> None:
            self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
            for ch in chapters:
                self.store.put(run_id, "chapter", ch)
            for t, disc in zip(
                tasks,
                [
                    {
                        "mentions": [
                            {"text": "苏沛之", "kind": "person", "refs": [u1]},
                            {"text": "林小姐", "kind": "person", "refs": [u1]},
                            {"text": "那位公子", "kind": "person", "refs": [u1]},
                        ],
                        "links": [],
                    },
                    {
                        "mentions": [
                            {"text": "林小姐", "kind": "person", "refs": [u2]},
                            {"text": "林婉", "kind": "person", "refs": [u2]},
                            {"text": "苏沛之", "kind": "person", "refs": [u2]},
                        ],
                        "links": [],
                    },
                ],
            ):
                grounded = ground_result(source, t, disc, f"req_disc_{run_id}_{t['id']}")
                recs = [
                    (kind, rec)
                    for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                    for rec in grounded[key]
                ]
                self.store.finish_task(run_id, t, {"request_id": f"req_disc_{run_id}_{t['id']}"}, recs)

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                m_gongzi = next(r["mentions"][0]["mention_id"] for r in state["surfaces"] if r["surface"] == "那位公子")
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
                    ],
                    "assignments": [
                        {"mention_id": m_gongzi, "surface": "那位公子", "ref": uid, "local_id": None, "reason": "身份未决"},
                    ],
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
                        "name": "苏沛之",
                        "kind": "person",
                        "surfaces": [{"surface": "苏沛之", "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    },
                ],
                "assignments": [],
                "not_entity": [],
            }, f"c1_{n}"

        # 场景 A：伪模型对 C3 任务始终返回非法 mention_id（ModelError 且 status=None, code=None）
        # -> 运行完成，该任务各项（林小姐对象与那位公子提及）进入待审且原因固定为“C3 链接请求连续返回非法输出，未能链接，转待审”
        _prep("run_c3_bad_mid")
        models_bad_mid = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_{n}"),
            "roster-c3-link": lambda state, n: (
                {"links": [{"mention_id": "obj:wrong_hash:e2", "selected_entity_id": "none", "reason": "抄错ID"}]},
                f"c3_bad_{n}",
            ),
        })
        out_bad = resolve_roster(self.store, models_bad_mid, source, "run_c3_bad_mid", concurrency=1, log=lambda _: None)
        self.assertTrue(out_bad["complete"], out_bad)
        roster_bad = self.store.records("run_c3_bad_mid", "roster_output")[0]
        c3_revs = [r for r in roster_bad["reviews"] if r["stage"] == "roster-c3-link"]
        self.assertEqual(len(c3_revs), 2)
        for r in c3_revs:
            self.assertEqual(r["reason"], "C3 链接请求连续返回非法输出，未能链接，转待审")

        # 补充验证经 CLI run --json 调用时退出码为 0 且 status 为 completed_with_review
        import contextlib
        import io
        import json
        import os
        from unittest.mock import patch
        from entitytrace.cli import main as cli_main
        from tests.test_cli_contract import CliContractStageFTests, ContractFakeModels

        class BadC3ContractModels(ContractFakeModels):
            def generate(self, system, state, stage, validate, *args, **kwargs):
                if stage == "roster-c3-link":
                    val = {"links": [{"mention_id": "obj:invalid_mid:e1", "selected_entity_id": "none", "reason": "bad"}]}
                    validate(val)
                return super().generate(system, state, stage, validate, *args, **kwargs)

        cli_book = self.root / "cli_bad_c3_book.txt"
        cli_book.write_text(CliContractStageFTests.NOVEL_TEXT, encoding="utf-8")
        cli_db = self.root / "cli_bad_c3.sqlite3"
        cli_out = self.root / "cli_bad_c3_out"
        cli_cfg = self.root / "cli.env"
        cli_cfg.write_text("DEEPSEEK_API_KEY=test_key\n", encoding="utf-8")
        out_buf, err_buf = io.StringIO(), io.StringIO()
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("DEEPSEEK_") and not k.startswith("TYPESAFE_")}
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("entitytrace.pipeline.Models", BadC3ContractModels), \
             contextlib.redirect_stdout(out_buf), \
             contextlib.redirect_stderr(err_buf):
            rc_cli = cli_main([
                "run", str(cli_book),
                "--db", str(cli_db),
                "--output", str(cli_out),
                "--config", str(cli_cfg),
                "--json",
            ])
        self.assertEqual(rc_cli, 0, err_buf.getvalue())
        receipt_cli = json.loads(out_buf.getvalue().strip())
        self.assertEqual(receipt_cli["status"], "completed_with_review")
        self.assertIsNone(receipt_cli["error"])
        self.assertEqual(receipt_cli["review_top"][0]["reason"], "C3 链接请求连续返回非法输出，未能链接，转待审")

        # 场景 B：伪网络失败（code="network_error"）或 HTTP 500（status=500） -> 仍未完成，retryable=True
        for idx_err, sim_err in enumerate([
            ModelError("连接超时：网络失败或超时", code="network_error"),
            ModelError("服务端错误：HTTP 500", status=500),
        ]):
            rid_err = f"run_c3_net_{idx_err}"
            _prep(rid_err)

            def raise_err(state, n, exc=sim_err):
                raise exc

            models_err = ScriptedRosterModels({
                "roster-c1": handle_c1,
                "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_{n}"),
                "roster-c3-link": raise_err,
            })
            out_err = resolve_roster(self.store, models_err, source, rid_err, concurrency=1, log=lambda _: None)
            self.assertFalse(out_err["complete"])
            self.assertEqual(out_err["stage"], "roster-c3-link")
            self.assertEqual(len(out_err["failures"]), 1)
            inferred = infer_incomplete_stage_error({"identity": out_err})
            self.assertIsNotNone(inferred)
            self.assertTrue(inferred["retryable"])

    def test_c2_prelim_entry_id_content_digest_stability_and_collision_lengthening(self):
        # 1. 人工构造 8 位前缀冲突与 12 位前缀冲突时自动加长，不产生重复 entry_id
        clusters_collision = [
            {"member_keys": [("t1", "e1")]},
            {"member_keys": [("t2", "e2")]},
            {"member_keys": [("t3", "e3")]},
            {"member_keys": [("t4", "e4")]},
        ]
        fake_digests = {
            (("t1", "e1"),): "abcd1234ef01aaaa" + "0" * 48,
            (("t2", "e2"),): "abcd12349988bbbb" + "0" * 48,  # 与 t1 在 8 位冲突，12 位不同
            (("t3", "e3"),): "777788889999cccc" + "0" * 48,
            (("t4", "e4"),): "777788889999dddd" + "0" * 48,  # 与 t3 在 12 位仍冲突，16 位不同
        }
        eids = compute_prelim_entry_ids(clusters_collision, digest_fn=lambda mks: fake_digests[tuple(mks)])
        self.assertEqual(len(set(eids)), 4)
        self.assertEqual(eids[0], "pabcd1234ef01")
        self.assertEqual(eids[1], "pabcd12349988")
        self.assertEqual(eids[2], "p777788889999cccc")
        self.assertEqual(eids[3], "p777788889999dddd")

        # 无冲突条目保持 p + 8 位
        clusters_mixed = [
            {"member_keys": [("t0", "e0")]},
            {"member_keys": [("t1", "e1")]},
            {"member_keys": [("t2", "e2")]},
        ]
        fake_mixed = {
            (("t0", "e0"),): "1111222233334444" + "0" * 48,
            (("t1", "e1"),): "abcd1234ef01aaaa" + "0" * 48,
            (("t2", "e2"),): "abcd12349988bbbb" + "0" * 48,
        }
        eids_mixed = compute_prelim_entry_ids(clusters_mixed, digest_fn=lambda mks: fake_mixed[tuple(mks)])
        self.assertEqual(eids_mixed, ["p11112222", "pabcd1234ef01", "pabcd12349988"])

        # 2. 在合成数据上，增删一个无关簇时，其余簇的 entry_id 不变，未受影响的 C2.4 核验对 request_id 与任务 id 保持不变
        from entitytrace.util import digest as real_digest

        text = "第一章\n路人甲经过。林舟与阿舟先后进城。\n第二章\n林舟再次出现。\n"
        path = self.root / "c2_prelim_stability.txt"
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

        def run_variant(run_id: str, include_extra_cluster: bool):
            self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
            for ch in chapters:
                self.store.put(run_id, "chapter", ch)
            ch1_mentions = [
                {"text": "林舟", "kind": "person", "refs": [u1]},
                {"text": "阿舟", "kind": "person", "refs": [u1]},
            ]
            if include_extra_cluster:
                ch1_mentions.insert(0, {"text": "路人甲", "kind": "person", "refs": [u1]})
            ch2_mentions = [{"text": "林舟", "kind": "person", "refs": [u2]}]
            for task, disc in zip(tasks, [{"mentions": ch1_mentions, "links": []}, {"mentions": ch2_mentions, "links": []}]):
                grounded = ground_result(source, task, disc, "req_" + task["id"])
                records = [
                    (kind, rec)
                    for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                    for rec in grounded[key]
                ]
                self.store.finish_task(run_id, task, {"request_id": "req_" + task["id"]}, records)

            propose_entries_seen = []
            verify_states_seen = []

            def handle_c1(state, n):
                uid = state["units"][0]["id"]
                if state["chapter_title"] == "第一章":
                    ents = [
                        {"local_id": "e_lin", "name": "林舟", "kind": "person", "surfaces": [{"surface": "林舟", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e_azhou", "name": "阿舟", "kind": "person", "surfaces": [{"surface": "阿舟", "surface_type": "proper_name"}], "evidence_ref": uid},
                    ]
                    if include_extra_cluster:
                        ents.insert(0, {"local_id": "e_000_extra", "name": "路人甲", "kind": "person", "surfaces": [{"surface": "路人甲", "surface_type": "proper_name"}], "evidence_ref": uid})
                    return {"entities": ents, "assignments": [], "not_entity": []}, f"req_c1_{n}"
                return {
                    "entities": [
                        {"local_id": "e_lin", "name": "林舟", "kind": "person", "surfaces": [{"surface": "林舟", "surface_type": "proper_name"}], "evidence_ref": uid},
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"req_c1_{n}"

            def handle_propose(state, n):
                by_name = {e["name"]: e["entry_id"] for e in state["entries"]}
                propose_entries_seen.append(by_name)
                return {
                    "proposals": [
                        {"left": by_name["林舟"], "right": by_name["阿舟"], "reason": "林舟与阿舟疑似同人"}
                    ]
                }, "req_propose_fixed"

            def handle_verify(state, n):
                req_id = real_digest(["verify_req", state])
                verify_states_seen.append((state, req_id))
                return {"verdict": "same", "reason": "确为同人"}, req_id

            models = ScriptedRosterModels({
                "roster-c1": handle_c1,
                "roster-c2-propose": handle_propose,
                "roster-c2-verify": handle_verify,
            })
            out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda *_: None)
            self.assertTrue(out["complete"])
            verify_task_rows = self.store.rows("SELECT id FROM tasks WHERE run_id=? AND stage='roster-c2-verify'", (run_id,))
            self.assertEqual(len(verify_task_rows), 1)
            return propose_entries_seen[0], verify_task_rows[0]["id"], verify_states_seen[0][1]

        entries_without, v_tid_without, v_req_without = run_variant("run_without_extra", False)
        entries_with, v_tid_with, v_req_with = run_variant("run_with_extra", True)

        self.assertTrue(entries_without["林舟"].startswith("p"))
        self.assertEqual(len(entries_without["林舟"]), 9)
        self.assertEqual(entries_without["林舟"], entries_with["林舟"])
        self.assertEqual(entries_without["阿舟"], entries_with["阿舟"])
        self.assertEqual(v_tid_without, v_tid_with)
        self.assertEqual(v_req_without, v_req_with)

    def test_c1_not_entity_echo_does_not_mark_claimed_proper_as_weak_norm(self):
        # 0.14e 规格一单测：
        # ① 单一对象认领的专名同时出现在 not_entity 中，不进 weak_norms、提及仍归该对象、规则 T 可补全；
        # ② 两个对象认领同一称呼，仍进 weak_norms；
        # ③ 只出现在 not_entity、未被认领的称呼，提及仍为 not_entity；
        # ④ 无回显时 apply_c1_task_result 输出与修改前逐字节相同。
        text = "第一章\n沈砚手持青霜剑，侍卫甲与侍卫乙同称铁卫，路边有碎石。稍后沈砚又握紧青霜剑。\n"
        path = self.root / "c1_not_entity_echo.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1 = text.index("第一章")
        _, chapters = build_chapters(source, [{"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"}])
        tasks = plan_tasks(source, chapters, 9000)
        task = tasks[0]
        uid = task["units"][0]["id"]

        # 注意：发现阶段只报出第一处“青霜剑”，第二处“青霜剑”留给规则 T 补全
        first_sword_start = text.index("青霜剑")
        mentions = [
            {
                "id": "m_shen_1",
                "text": "沈砚",
                "start": text.index("沈砚"),
                "end": text.index("沈砚") + 2,
                "address": f"{source['id']}:{text.index('沈砚')}..{text.index('沈砚') + 2}",
                "chapter_id": task["chapter_id"],
                "task_id": task["id"],
                "provisional_kinds": ["person"],
                "observation_ids": [],
                "evidence": [f"{source['id']}:{task['units'][0]['start']}..{task['units'][0]['end']}"],
            },
            {
                "id": "m_sword_1",
                "text": "青霜剑",
                "start": first_sword_start,
                "end": first_sword_start + 3,
                "address": f"{source['id']}:{first_sword_start}..{first_sword_start + 3}",
                "chapter_id": task["chapter_id"],
                "task_id": task["id"],
                "provisional_kinds": ["object"],
                "observation_ids": [],
                "evidence": [f"{source['id']}:{task['units'][0]['start']}..{task['units'][0]['end']}"],
            },
            {
                "id": "m_tie_1",
                "text": "铁卫",
                "start": text.index("铁卫"),
                "end": text.index("铁卫") + 2,
                "address": f"{source['id']}:{text.index('铁卫')}..{text.index('铁卫') + 2}",
                "chapter_id": task["chapter_id"],
                "task_id": task["id"],
                "provisional_kinds": ["person"],
                "observation_ids": [],
                "evidence": [f"{source['id']}:{task['units'][0]['start']}..{task['units'][0]['end']}"],
            },
            {
                "id": "m_stone_1",
                "text": "碎石",
                "start": text.index("碎石"),
                "end": text.index("碎石") + 2,
                "address": f"{source['id']}:{text.index('碎石')}..{text.index('碎石') + 2}",
                "chapter_id": task["chapter_id"],
                "task_id": task["id"],
                "provisional_kinds": ["object"],
                "observation_ids": [],
                "evidence": [f"{source['id']}:{task['units'][0]['start']}..{task['units'][0]['end']}"],
            },
        ]

        raw_no_echo = {
            "entities": [
                {
                    "local_id": "e1",
                    "name": "沈砚",
                    "kind": "person",
                    "surfaces": [{"surface": "沈砚", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e2",
                    "name": "青霜剑",
                    "kind": "object",
                    "surfaces": [{"surface": "青霜剑", "surface_type": "proper_name"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e3",
                    "name": "侍卫甲",
                    "kind": "person",
                    "surfaces": [{"surface": "铁卫", "surface_type": "title_or_kinship"}],
                    "evidence_ref": uid,
                },
                {
                    "local_id": "e4",
                    "name": "侍卫乙",
                    "kind": "person",
                    "surfaces": [{"surface": "铁卫", "surface_type": "title_or_kinship"}],
                    "evidence_ref": uid,
                },
            ],
            "assignments": [],
            "not_entity": ["碎石"],
        }
        raw_with_echo = {
            **raw_no_echo,
            # 模型在 not_entity 中回显了已被单一对象认领的“沈砚”“青霜剑”
            "not_entity": ["沈砚", "青霜剑", "铁卫", "碎石"],
        }

        res_no_echo = apply_c1_task_result(source, task, mentions, raw_no_echo, "req_c1_echo")
        res_with_echo = apply_c1_task_result(source, task, mentions, raw_with_echo, "req_c1_echo")

        # ④ 无回显与有回显（对已认领对象）输出逐字节相同；单一认领专名不进 weak_norms，多重认领仍进 weak_norms
        self.assertEqual(dumps(res_no_echo), dumps(res_with_echo))
        self.assertNotIn("沈砚", res_with_echo["weak_norms"])
        self.assertNotIn("青霜剑", res_with_echo["weak_norms"])
        self.assertIn("铁卫", res_with_echo["weak_norms"])  # ② 两个对象认领同一称呼，仍进 weak_norms

        outcomes = {o["mention_id"]: o for o in res_with_echo["mention_outcomes"]}
        self.assertEqual(outcomes["m_shen_1"]["status"], "assigned")
        self.assertEqual(outcomes["m_shen_1"]["local_id"], "e1")
        self.assertEqual(outcomes["m_sword_1"]["status"], "assigned")
        self.assertEqual(outcomes["m_sword_1"]["local_id"], "e2")
        self.assertEqual(outcomes["m_stone_1"]["status"], "not_entity")  # ③ 未被认领的 not_entity 提及仍为 not_entity

        # ① 验证完整 resolve_roster 下，规则 T 能补全第二处“沈砚”与“青霜剑”
        run_id = "run_c1_echo_prop"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        self.store.finish_task(
            run_id,
            task,
            {"request_id": "req_disc"},
            [("mention", m) for m in mentions],
        )
        models = ScriptedRosterModels({
            "roster-c1": lambda state, n: (raw_with_echo, "req_c1_echo"),
            "roster-c2-propose": lambda state, n: ({"proposals": []}, "req_prop"),
        })
        out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda *_: None)
        self.assertTrue(out["complete"])
        roster_out = self.store.records(run_id, "roster_output")[0]
        sword_ent = next(e for e in roster_out["entities"] if e["name"] == "青霜剑")
        shen_ent = next(e for e in roster_out["entities"] if e["name"] == "沈砚")
        self.assertEqual(sword_ent["mention_count"], 2)
        self.assertEqual(sword_ent["link_methods"].get("same_task_proper"), 1)
        self.assertEqual(shen_ent["mention_count"], 2)
        self.assertEqual(shen_ent["link_methods"].get("same_task_proper"), 1)

    def test_a4a_prime_unstable_proper_downgrade(self):
        """0.14e 规格三（A4-a′）：验证比例计算、人物豁免、降级后不同名并查、C3 不用降级键、规则 T 跳过降级称呼、无降级时逐字节不变。"""
        from unittest.mock import patch
        from entitytrace.roster import compute_unstable_proper_norms
        from entitytrace.util import dumps

        # 构造 3 章合成文本（无任何四本测试书或封神/凡人原文）
        ch1 = "第一章\n沈砚带黑木匣进玄铁堡，黑木匣紧锁。赵校尉守卫玄铁堡。\n"
        ch2 = "第二章\n林婉在青石谷打开黑木匣，赵校尉亦至青石谷。\n"
        ch3 = "第三章\n苏沛之在白鹤渡见黑木匣，赵校尉随后赶到白鹤渡。\n"
        text = ch1 + ch2 + ch3
        path = self.root / "a4a_prime.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)

        c1_end = len(ch1)
        c2_end = c1_end + len(ch2)
        _, chapters = build_chapters(
            source,
            [
                {"start": 0, "end": 3, "title": "第一章", "kind": "chapter"},
                {"start": c1_end, "end": c1_end + 3, "title": "第二章", "kind": "chapter"},
                {"start": c2_end, "end": c2_end + 3, "title": "第三章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        self.assertEqual(len(tasks), 3)

        def _make_mentions():
            res = []
            # 第一章：第一处“黑木匣”被发现，第二处“黑木匣”留给规则 T 检验是否跳过
            for idx, (tid, cid, base, sub, s_list) in enumerate([
                (tasks[0]["id"], chapters[0]["id"], 0, ch1, ["沈砚", "黑木匣", "玄铁堡", "赵校尉"]),
                (tasks[1]["id"], chapters[1]["id"], c1_end, ch2, ["林婉", "黑木匣", "青石谷", "赵校尉"]),
                (tasks[2]["id"], chapters[2]["id"], c2_end, ch3, ["苏沛之", "黑木匣", "白鹤渡", "赵校尉"]),
            ]):
                u_id = tasks[idx]["units"][0]["id"]
                u_span = f"{source['id']}:{tasks[idx]['units'][0]['start']}..{tasks[idx]['units'][0]['end']}"
                for surf in s_list:
                    pos = base + sub.index(surf)
                    kind = "object" if surf == "黑木匣" else ("place" if surf in {"玄铁堡", "青石谷", "白鹤渡"} else "person")
                    res.append({
                        "id": f"m_{cid}_{surf}",
                        "text": surf,
                        "start": pos,
                        "end": pos + len(surf),
                        "address": f"{source['id']}:{pos}..{pos + len(surf)}",
                        "chapter_id": cid,
                        "task_id": tid,
                        "provisional_kinds": [kind],
                        "observation_ids": [f"o_{cid}_{surf}"],
                        "evidence": [u_span],
                        "unit_id": u_id,
                    })
            return res

        mentions = _make_mentions()

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            title = state["chapter_title"]
            if title == "第一章":
                # 第一章：黑木匣被判为 proper_name（且与玄铁堡同任务出现）；赵校尉判为 proper_name (person)
                return {
                    "entities": [
                        {"local_id": "e1", "name": "沈砚", "kind": "person", "surfaces": [{"surface": "沈砚", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e2", "name": "玄铁堡", "kind": "place", "surfaces": [{"surface": "玄铁堡", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e3", "name": "黑木匣", "kind": "object", "surfaces": [{"surface": "黑木匣", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e4", "name": "赵校尉", "kind": "person", "surfaces": [{"surface": "赵校尉", "surface_type": "proper_name"}], "evidence_ref": uid},
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"req_c1_{n}"
            if title == "第二章":
                # 第二章：黑木匣被判为 description（挂在专名器物“青玉匣”或单独对象上）；赵校尉判为 title_or_kinship (person)
                return {
                    "entities": [
                        {"local_id": "e1", "name": "林婉", "kind": "person", "surfaces": [{"surface": "林婉", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e2", "name": "青石谷", "kind": "place", "surfaces": [{"surface": "青石谷", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e3", "name": "黑木匣", "kind": "object", "surfaces": [{"surface": "黑木匣", "surface_type": "description"}], "evidence_ref": uid},
                        {"local_id": "e4", "name": "赵校尉", "kind": "person", "surfaces": [{"surface": "赵校尉", "surface_type": "title_or_kinship"}], "evidence_ref": uid},
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"req_c1_{n}"
            # 第三章：黑木匣被判为 description；赵校尉判为 title_or_kinship (person)
            return {
                "entities": [
                    {"local_id": "e1", "name": "苏沛之", "kind": "person", "surfaces": [{"surface": "苏沛之", "surface_type": "proper_name"}], "evidence_ref": uid},
                    {"local_id": "e2", "name": "白鹤渡", "kind": "place", "surfaces": [{"surface": "白鹤渡", "surface_type": "proper_name"}], "evidence_ref": uid},
                    {"local_id": "e3", "name": "黑木匣", "kind": "object", "surfaces": [{"surface": "黑木匣", "surface_type": "description"}], "evidence_ref": uid},
                    {"local_id": "e4", "name": "赵校尉", "kind": "person", "surfaces": [{"surface": "赵校尉", "surface_type": "title_or_kinship"}], "evidence_ref": uid},
                ],
                "assignments": [],
                "not_entity": [],
            }, f"req_c1_{n}"

        def _prep_run(rid):
            self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (rid, source["id"], "{}", "{}", "now", "now"))
            for ch in chapters:
                self.store.put(rid, "chapter", ch)
            for t in tasks:
                t_ms = [m for m in mentions if m["task_id"] == t["id"]]
                self.store.finish_task(rid, t, {"request_id": f"disc_{rid}_{t['id']}"}, [("mention", m) for m in t_ms])

        # 1) 验证开启 A4-a′ (theta=0.5)：
        # - “黑木匣” P=1, D=2, r=1/3 < 0.5 且非人物 -> 降级
        # - “赵校尉” P=1, D=2, r=1/3 < 0.5 但 kind=='person' -> 人物豁免，不降级
        _prep_run("run_with_a4a")
        c3_states_a4a = []

        def handle_c3(state, n):
            c3_states_a4a.append(state)
            return {
                "links": [
                    {"mention_id": it["mention_id"], "selected_entity_id": it["candidate_entity_ids"][0], "reason": "默认选首候选"}
                    for it in state["items"]
                ]
            }, f"req_c3_{n}"

        models_a4a = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_{n}"),
            "roster-c2-review": lambda state, n: (
                {"groups": [{"group_id": "g1", "name": "赵校尉", "kind": "person", "row_ids": [r["row_id"] for r in state["rows"]], "member_ids": [], "reason": "同人"}], "unresolved_row_ids": [], "unresolved_member_ids": []},
                f"rev_{n}",
            ),
            "roster-c3-link": handle_c3,
        })
        with patch("entitytrace.roster.UNSTABLE_PROPER_THETA", 0.5):
            out_a4a = resolve_roster(self.store, models_a4a, source, "run_with_a4a", concurrency=1, log=lambda *_: None)
        self.assertTrue(out_a4a["complete"])

        # 直接检验 compute_unstable_proper_norms：黑木匣在降级集合中，赵校尉因人物豁免不在其中
        c1_res_map = {t["id"]: self.store.task("run_with_a4a", self.store.rows("SELECT id FROM tasks WHERE run_id=? AND stage='roster-c1'", ("run_with_a4a",))[i]["id"])["result"] for i, t in enumerate(tasks)}
        # 按 task_id 映射
        c1_by_tid = {}
        for row in self.store.rows("SELECT id FROM tasks WHERE run_id=? AND stage='roster-c1'", ("run_with_a4a",)):
            tr = self.store.task("run_with_a4a", row["id"])["result"]
            c1_by_tid[tr["task_id"]] = tr
        downgraded = compute_unstable_proper_norms(tasks, c1_by_tid, mentions, theta=0.5)
        self.assertEqual(downgraded, {"黑木匣"})

        roster_a4a = self.store.records("run_with_a4a", "roster_output")[0]
        box_ents = [e for e in roster_a4a["entities"] if e["name"] == "黑木匣"]
        # 三章的“黑木匣”降级后均为只含非专名对象，且各章内无其他 object 候选，C3 不召回跨章同名“黑木匣” -> 保持 3 个独立章内对象，且第一章第二处“黑木匣”不被规则 T 补全
        self.assertEqual(len(box_ents), 3)
        for be in box_ents:
            self.assertEqual(be["mention_count"], 1)
            self.assertEqual(be["link_methods"].get("same_task_proper", 0), 0)
        # C3 请求中不应出现任何“黑木匣”待链项（因无同章 object 候选且跨章精确/词干键已被禁用）
        for st in c3_states_a4a:
            for it in st["items"]:
                self.assertNotEqual(it["surface"], "黑木匣")

        # 2) 验证无降级时（所设阈值下無词满足降级，如 theta=0.2 vs theta=0.0），导出与请求负载逐字节不变
        _prep_run("run_no_downgrade_02")
        _prep_run("run_no_downgrade_00")
        calls_02, calls_00 = [], []

        def make_recording_models(log_list):
            return ScriptedRosterModels({
                "roster-c1": lambda state, n: (log_list.append(("c1", dumps(state))) or handle_c1(state, n)),
                "roster-c2-propose": lambda state, n: (log_list.append(("prop", dumps(state))) or ({"proposals": []}, f"prop_{n}")),
                "roster-c2-review": lambda state, n: (
                    log_list.append(("rev", dumps(state)))
                    or ({"groups": [{"group_id": "g1", "name": "赵校尉", "kind": "person", "row_ids": [r["row_id"] for r in state["rows"]], "member_ids": [], "reason": "同人"}], "unresolved_row_ids": [], "unresolved_member_ids": []}, f"rev_{n}")
                ),
                "roster-c3-link": lambda state, n: (
                    log_list.append(("c3", dumps(state)))
                    or ({"links": [{"mention_id": it["mention_id"], "selected_entity_id": "none", "reason": "无"} for it in state["items"]]}, f"c3_{n}")
                ),
            })

        with patch("entitytrace.roster.UNSTABLE_PROPER_THETA", 0.2):
            resolve_roster(self.store, make_recording_models(calls_02), source, "run_no_downgrade_02", concurrency=1, log=lambda *_: None)
        with patch("entitytrace.roster.UNSTABLE_PROPER_THETA", 0.0):
            resolve_roster(self.store, make_recording_models(calls_00), source, "run_no_downgrade_00", concurrency=1, log=lambda *_: None)

        self.assertEqual(calls_02, calls_00)
        out_02 = self.store.records("run_no_downgrade_02", "roster_output")[0]
        out_00 = self.store.records("run_no_downgrade_00", "roster_output")[0]
        self.assertEqual(dumps(out_02), dumps(out_00))

    def test_a2_c3_short_ids_mapping_validation_and_task_digest(self):
        """0.15b 规格一（A2）：验证 C3 短编号确定性、存储完整 ID、未知短编号与完整长 ID 被拒、
        跨项候选转 c3_rejected_name_mismatch、映射不同任务键不同、无 C3 项时导出逐字节不变、C3_LINK_PROMPT SHA-256 不变。"""
        import hashlib
        from entitytrace.roster import C3_LINK_PROMPT
        from entitytrace.util import dumps

        # (7) C3_LINK_PROMPT SHA-256 与 da392d5 逐字节不变
        self.assertEqual(
            hashlib.sha256(C3_LINK_PROMPT.encode("utf-8")).hexdigest(),
            "04729a1f14afb9b3738418d4851d113fc40164faca8b8c0cdd4d91ea9f88b9b4",
        )

        # 构造两章合成文本（无任何测试书原文）：
        # 第一章含人物专名“沈砚”、地点专名“玄铁堡”，以及非专名人物对象“那位剑客”和非专名地点对象“那座古堡”；
        # 第二章含人物专名“沈砚”与“林婉”、地点专名“玄铁堡”。
        text = (
            "第一章\n沈砚与那位剑客来到玄铁堡，望见那座古堡。\n"
            "第二章\n沈砚与林婉重返玄铁堡。\n"
        )
        path = self.root / "a2_short_ids.txt"
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
                    {"text": "沈砚", "kind": "person", "refs": [u1]},
                    {"text": "那位剑客", "kind": "person", "refs": [u1]},
                    {"text": "玄铁堡", "kind": "place", "refs": [u1]},
                    {"text": "那座古堡", "kind": "place", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "沈砚", "kind": "person", "refs": [u2]},
                    {"text": "林婉", "kind": "person", "refs": [u2]},
                    {"text": "玄铁堡", "kind": "place", "refs": [u2]},
                ],
                "links": [],
            },
        )

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "沈砚", "kind": "person", "surfaces": [{"surface": "沈砚", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e2", "name": "那位剑客", "kind": "person", "surfaces": [{"surface": "那位剑客", "surface_type": "description"}], "evidence_ref": uid},
                        {"local_id": "e3", "name": "玄铁堡", "kind": "place", "surfaces": [{"surface": "玄铁堡", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e4", "name": "那座古堡", "kind": "place", "surfaces": [{"surface": "那座古堡", "surface_type": "description"}], "evidence_ref": uid},
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_a2_{n}"
            return {
                "entities": [
                    {"local_id": "e1", "name": "沈砚", "kind": "person", "surfaces": [{"surface": "沈砚", "surface_type": "proper_name"}], "evidence_ref": uid},
                    {"local_id": "e2", "name": "林婉", "kind": "person", "surfaces": [{"surface": "林婉", "surface_type": "proper_name"}], "evidence_ref": uid},
                    {"local_id": "e3", "name": "玄铁堡", "kind": "place", "surfaces": [{"surface": "玄铁堡", "surface_type": "proper_name"}], "evidence_ref": uid},
                ],
                "assignments": [],
                "not_entity": [],
            }, f"c1_a2_{n}"

        seen_c3_states = []

        def handle_c3_check_and_mismatch(state, n):
            seen_c3_states.append(state)
            # (1) 编号确定性：items 按顺序为 m1, m2；candidates 按 sorted(all_cand_ids) 编为 e1, e2
            self.assertEqual([it["mention_id"] for it in state["items"]], ["m1", "m2"])
            self.assertEqual([c["entity_id"] for c in state["candidates"]], ["e1", "e2"])
            for it in state["items"]:
                self.assertTrue(all(cid in {"e1", "e2"} for cid in it["candidate_entity_ids"]))

            # 找出“那位剑客”(person) 与“那座古堡”(place) 各自的短编号与候选
            swordsman_item = next(it for it in state["items"] if it["surface"] == "那位剑客")
            castle_item = next(it for it in state["items"] if it["surface"] == "那座古堡")
            shen_short = next(c["entity_id"] for c in state["candidates"] if c["name"] == "沈砚")
            fort_short = next(c["entity_id"] for c in state["candidates"] if c["name"] == "玄铁堡")
            self.assertEqual(swordsman_item["candidate_entity_ids"], [shen_short])
            self.assertEqual(castle_item["candidate_entity_ids"], [fort_short])

            # (4) 让“那位剑客”正常选 shen_short，让“那座古堡”故意选本请求内存在但不属于其候选的 shen_short
            # -> 通过 _check_c3_links，但在回写后触发 sel not in set(it["candidate_ids"]) 转 c3_rejected_name_mismatch
            return {
                "links": [
                    {"mention_id": swordsman_item["mention_id"], "selected_entity_id": shen_short, "reason": "同指沈砚"},
                    {"mention_id": castle_item["mention_id"], "selected_entity_id": shen_short, "reason": "故意选非本项候选"},
                ]
            }, f"c3_a2_{n}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_a2_{n}"),
            "roster-c3-link": handle_c3_check_and_mismatch,
        })
        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"])
        self.assertEqual(len(seen_c3_states), 1)

        # (2) 存储结果为完整 ID（非 m1/m2、非 e1/e2）
        c3_task_rows = self.store.rows("SELECT id FROM tasks WHERE run_id=? AND stage='roster-c3-link'", (run_id,))
        self.assertEqual(len(c3_task_rows), 1)
        c3_tid_1 = c3_task_rows[0]["id"]
        stored_c3 = self.store.task(run_id, c3_tid_1)["result"]
        for row in stored_c3["links"]:
            self.assertNotIn(row["mention_id"], {"m1", "m2"})
            self.assertTrue(row["mention_id"].startswith("obj:"))
            self.assertNotIn(row["selected_entity_id"], {"e1", "e2"})

        out = self.store.records(run_id, "roster_output")[0]
        shen_ent = next(e for e in out["entities"] if "沈砚" in e["names"])
        self.assertEqual(shen_ent["link_methods"].get("c3_model"), 1)
        castle_ent = next(e for e in out["entities"] if e["name"] == "那座古堡")
        self.assertEqual(castle_ent["link_methods"].get("c3_rejected_name_mismatch"), 1)

        # (3) 未知短编号（m99 / e99）与完整长 ID 均被 _check_c3_links 拒绝
        full_mid = stored_c3["links"][0]["mention_id"]
        full_eid = stored_c3["links"][0]["selected_entity_id"]
        for bad_payload in [
            {"links": [{"mention_id": "m99", "selected_entity_id": "e1"}]},
            {"links": [{"mention_id": full_mid, "selected_entity_id": "e1"}]},
            {"links": [{"mention_id": "m1", "selected_entity_id": "e99"}]},
            {"links": [{"mention_id": "m1", "selected_entity_id": full_eid}]},
        ]:
            with self.assertRaises(ModelError):
                _check_c3_links(bad_payload, {"m1", "m2"}, {"e1", "e2", "none", "uncertain"})

        # (5) 映射不同则任务键 c3_tid 不同：当 local_id 从 e2/e4 变为 e20/e40 时，state 内容相同但 short_to_full_mid 映射不同，c3_tid 必须不同
        import copy
        run_id_alt = "run_a2_alt_map"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id_alt, source["id"], "{}", "{}", "now", "now"))
        for ch in self.store.records(run_id, "chapter"):
            self.store.put(run_id_alt, "chapter", ch)
        for m in self.store.records(run_id, "mention"):
            if m.get("link_method") != "same_task_proper":
                self.store.put(run_id_alt, "mention", m)

        def handle_c1_alt(state, n):
            res, req = handle_c1(state, n)
            res = copy.deepcopy(res)
            if state["chapter_title"] == "第一章":
                res["entities"][1]["local_id"] = "e20"
                res["entities"][3]["local_id"] = "e40"
            return res, req

        models_alt = ScriptedRosterModels({
            "roster-c1": handle_c1_alt,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_alt_{n}"),
            "roster-c3-link": handle_c3_check_and_mismatch,
        })
        out_alt = resolve_roster(self.store, models_alt, source, run_id_alt, concurrency=1, log=lambda _: None)
        self.assertTrue(out_alt["complete"], out_alt)
        c3_tid_2 = self.store.rows("SELECT id FROM tasks WHERE run_id=? AND stage='roster-c3-link'", (run_id_alt,))[0]["id"]
        self.assertNotEqual(c3_tid_1, c3_tid_2)
        # 且请求 state 逐字节相同（都用 m1/m2, e1/e2）
        self.assertEqual(dumps(seen_c3_states[0]), dumps(seen_c3_states[1]))

        # (6) 无 C3 项时导出逐字节不变
        run_no_c3 = "run_a2_no_c3"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_no_c3, source["id"], "{}", "{}", "now", "now"))
        for ch in self.store.records(run_id, "chapter"):
            self.store.put(run_no_c3, "chapter", ch)
        for m in self.store.records(run_id, "mention"):
            if m["text"] in {"沈砚", "林婉", "玄铁堡"}:
                self.store.put(run_no_c3, "mention", m)
        models_no_c3 = ScriptedRosterModels({
            "roster-c1": lambda state, n: (
                {
                    "entities": [
                        {"local_id": "e1", "name": "沈砚", "kind": "person", "surfaces": [{"surface": "沈砚", "surface_type": "proper_name"}], "evidence_ref": state["units"][0]["id"]},
                        {"local_id": "e3", "name": "玄铁堡", "kind": "place", "surfaces": [{"surface": "玄铁堡", "surface_type": "proper_name"}], "evidence_ref": state["units"][0]["id"]},
                    ] + ([{"local_id": "e2", "name": "林婉", "kind": "person", "surfaces": [{"surface": "林婉", "surface_type": "proper_name"}], "evidence_ref": state["units"][0]["id"]}] if state["chapter_title"] == "第二章" else []),
                    "assignments": [],
                    "not_entity": [],
                },
                f"c1_noc3_{n}",
            ),
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_noc3_{n}"),
        })
        out_noc3 = resolve_roster(self.store, models_no_c3, source, run_no_c3, concurrency=1, log=lambda _: None)
        self.assertTrue(out_noc3["complete"])
        self.assertEqual(len(self.store.rows("SELECT id FROM tasks WHERE run_id=? AND stage='roster-c3-link'", (run_no_c3,))), 0)

    def _prepare_s3_propose_batch_order_fixture(self):
        """构造同时产生 place 批次（下标 0）与 cross_kind 批次（下标 1）的合成双章输入。
        两条目均有 place 与 organization 双 provisional_kinds，故既属于 place 批，又属于 cross_kind 批。"""
        text = (
            "第一章\n青云书院坐落于北麓，世人皆敬仰青云书院。\n"
            "第二章\n青云院藏书万卷，学子齐聚青云院。\n"
        )
        path = self.root / "s3_propose_order.txt"
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
        source, base_run_id, tasks = self._prepare_two_chapter_run(
            text,
            {
                "mentions": [
                    {"text": "青云书院", "kind": "place", "refs": [u1]},
                    {"text": "青云书院", "kind": "organization", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "青云院", "kind": "place", "refs": [u2]},
                    {"text": "青云院", "kind": "organization", "refs": [u2]},
                ],
                "links": [],
            },
        )

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            surf = "青云书院" if state["chapter_title"] == "第一章" else "青云院"
            return {
                "entities": [
                    {
                        "local_id": "e1",
                        "name": surf,
                        "kind": "place",
                        "surfaces": [{"surface": surf, "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    }
                ],
                "assignments": [],
                "not_entity": [],
            }, f"c1_s3_{state['chapter_title']}"

        return source, base_run_id, handle_c1

    def test_s3_c2_propose_dedup_order_invariant_under_reverse_threads_and_partial_resume(self):
        """0.15b.1 规格 S3 单测 ①：让补合并提议的工作线程完成顺序与批次顺序相反，
        或一批来自已完成任务、一批来自新请求，所得 proposals（含 reason、propose_request_id）与按批次顺序完成时逐项相同。"""
        import threading

        source, base_run_id, handle_c1 = self._prepare_s3_propose_batch_order_fixture()

        def clone_run(new_run_id: str) -> str:
            self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (new_run_id, source["id"], "{}", "{}", "now", "now"))
            for ch in self.store.records(base_run_id, "chapter"):
                self.store.put(new_run_id, "chapter", ch)
            for m in self.store.records(base_run_id, "mention"):
                self.store.put(new_run_id, "mention", m)
            return new_run_id

        # 情形 A：按批次顺序串行完成（concurrency=1，place 批先完成，cross_kind 批后完成）
        run_seq = clone_run("run_s3_seq")
        seq_verify_states = []

        def propose_seq(state, n):
            ids = sorted(e["entry_id"] for e in state["entries"])
            k = state["kind"]
            return (
                {"proposals": [{"left": ids[0], "right": ids[1], "reason": f"reason_from_{k}"}]},
                f"req_propose_{k}",
            )

        def verify_capture_seq(state, n):
            seq_verify_states.append(state)
            return {"verdict": "same", "reason": "同指青云书院"}, "req_verify_s3"

        models_seq = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": propose_seq,
            "roster-c2-verify": verify_capture_seq,
        })
        out_seq = resolve_roster(self.store, models_seq, source, run_seq, concurrency=1, log=lambda _: None)
        self.assertTrue(out_seq["complete"])
        self.assertEqual(len(seq_verify_states), 1)
        seq_roster_out = self.store.records(run_seq, "roster_output")[0]

        # 情形 B：双线程并发（concurrency=2），强制下标 1 的 cross_kind 批先返回、下标 0 的 place 批等待事件后才返回
        run_rev = clone_run("run_s3_rev_threads")
        rev_verify_states = []
        cross_kind_done = threading.Event()
        completion_order = []

        def propose_reverse_threads(state, n):
            ids = sorted(e["entry_id"] for e in state["entries"])
            k = state["kind"]
            if k == "place":
                # 等待 cross_kind 线程先完成返回
                cross_kind_done.wait(timeout=5.0)
                completion_order.append("place")
            else:
                completion_order.append("cross_kind")
                cross_kind_done.set()
            return (
                {"proposals": [{"left": ids[0], "right": ids[1], "reason": f"reason_from_{k}"}]},
                f"req_propose_{k}",
            )

        def verify_capture_rev(state, n):
            rev_verify_states.append(state)
            return {"verdict": "same", "reason": "同指青云书院"}, "req_verify_s3"

        models_rev = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": propose_reverse_threads,
            "roster-c2-verify": verify_capture_rev,
        })
        out_rev = resolve_roster(self.store, models_rev, source, run_rev, concurrency=2, log=lambda _: None)
        self.assertTrue(out_rev["complete"])
        self.assertEqual(completion_order, ["cross_kind", "place"])
        self.assertEqual(len(rev_verify_states), 1)
        rev_roster_out = self.store.records(run_rev, "roster_output")[0]

        # 情形 C：下标 1 的 cross_kind 批来自 store.task 已完成任务，下标 0 的 place 批来自本次新请求
        run_mixed = clone_run("run_s3_mixed_cache")
        for r in self.store.rows("SELECT id, stage, result FROM tasks WHERE run_id=? AND stage='roster-c2-propose'", (run_seq,)):
            import json
            res_obj = json.loads(r["result"])
            if res_obj["request_id"] == "req_propose_cross_kind":
                self.store.finish_task(run_mixed, {"id": r["id"], "stage": "roster-c2-propose"}, res_obj)

        mixed_verify_states = []
        mixed_propose_called_kinds = []

        def propose_mixed(state, n):
            mixed_propose_called_kinds.append(state["kind"])
            return propose_seq(state, n)

        def verify_capture_mixed(state, n):
            mixed_verify_states.append(state)
            return {"verdict": "same", "reason": "同指青云书院"}, "req_verify_s3"

        models_mixed = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": propose_mixed,
            "roster-c2-verify": verify_capture_mixed,
        })
        out_mixed = resolve_roster(self.store, models_mixed, source, run_mixed, concurrency=2, log=lambda _: None)
        self.assertTrue(out_mixed["complete"])
        self.assertEqual(mixed_propose_called_kinds, ["place"])
        self.assertEqual(len(mixed_verify_states), 1)
        mixed_roster_out = self.store.records(run_mixed, "roster_output")[0]

        # 断言情形 B、情形 C 与情形 A 的核验输入（含 proposal_reason）及最终导出（含 link_refs 中的 propose_request_id）逐项相同
        self.assertEqual(dumps(rev_verify_states), dumps(seq_verify_states))
        self.assertEqual(dumps(mixed_verify_states), dumps(seq_verify_states))
        self.assertEqual(dumps(rev_roster_out), dumps(seq_roster_out))
        self.assertEqual(dumps(mixed_roster_out), dumps(seq_roster_out))

    def test_s3_c2_propose_duplicate_pair_keeps_smaller_batch_index_reason(self):
        """0.15b.1 规格 S3 单测 ②：两批提议同一对、理由不同时，保留下标较小批次（place，pb_idx=0）的理由与 request_id。"""
        source, base_run_id, handle_c1 = self._prepare_s3_propose_batch_order_fixture()
        seen_propose_kinds = []
        verify_states = []

        def handle_propose(state, n):
            k = state["kind"]
            seen_propose_kinds.append(k)
            ids = sorted(e["entry_id"] for e in state["entries"])
            reason = "批0（place）全称简称理由" if k == "place" else "批1（cross_kind）跨粗类理由"
            return (
                {"proposals": [{"left": ids[0], "right": ids[1], "reason": reason}]},
                f"req_prop_{k}",
            )

        def handle_verify(state, n):
            verify_states.append(state)
            return {"verdict": "same", "reason": "确系同一书院"}, f"req_ver_{n}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": handle_propose,
            "roster-c2-verify": handle_verify,
        })
        out = resolve_roster(self.store, models, source, base_run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(out["complete"])
        # 确认两批顺序为下标 0=place, 下标 1=cross_kind
        self.assertEqual(seen_propose_kinds, ["place", "cross_kind"])
        # 同一对只触发 1 次核验，且保留下标较小批次（place，pb_idx=0）的理由与 propose_request_id
        self.assertEqual(len(verify_states), 1)
        self.assertEqual(verify_states[0]["proposal_reason"], "批0（place）全称简称理由")
        roster_out = self.store.records(base_run_id, "roster_output")[0]
        merged_refs = {ref for m in roster_out["mentions"] for ref in m.get("link_refs", [])}
        self.assertIn("req_prop_place", merged_refs)
        self.assertNotIn("req_prop_cross_kind", merged_refs)

    def test_c3v_cross_item_candidate_verification(self):
        """0.15b.2 规格 C3-V 单测：
        1. 本项 candidate_ids 之外、同请求内的同粗类条目：通过 D1 与 cannot_merge_by_obj，按 c3_model 归入；
        2. 本项 candidate_ids 之外、同请求内的条目违反 D1 req_pnorms <= sel_pnorms：转 c3_rejected_name_mismatch 并生成待审；
        3. 本项 candidate_ids 之外、同请求内的条目违反 cannot_merge_by_obj（对象项）：转 c3_rejected_name_mismatch 并生成待审；
        4. 本项 candidate_ids 之外、同请求内的跨粗类条目：person vs place 拒绝；person vs organization 按当前臂规则判定；
        5. 同一合成输入下，修改前后的 C3 请求 state、id_map 与 c3_tid 逐字节相同。"""
        from entitytrace.roster import C3_LINK_PROMPT, VERSION
        from entitytrace.util import digest, dumps

        text = (
            "第一章\n"
            "褚六与沈甲师爷同行。沈甲剑客、沈甲使者、那座玄铁堡、玄铁盟分舵、温七随从、林昭先后登场。\n"
            "第二章\n"
            "沈甲一、沈甲二、沈甲三、沈甲四、沈甲五与温七、林昭、褚六、沈甲师爷在玄铁堡会合玄铁盟。\n"
            "第三章\n"
            "另一位林昭独自远行。\n"
        )
        path = self.root / "c3v_synthetic.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2, idx3 = text.index("第一章"), text.index("第二章"), text.index("第三章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
                {"start": idx3, "end": idx3 + 3, "title": "第三章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2, u3 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"], tasks[2]["units"][0]["id"]

        run_id = "run_c3v_test"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)

        disc_payloads = [
            {
                "mentions": [
                    {"text": "褚六", "kind": "person", "refs": [u1]},
                    {"text": "沈甲师爷", "kind": "person", "refs": [u1]},
                    {"text": "沈甲剑客", "kind": "person", "refs": [u1]},
                    {"text": "沈甲使者", "kind": "person", "refs": [u1]},
                    {"text": "那座玄铁堡", "kind": "place", "refs": [u1]},
                    {"text": "玄铁盟分舵", "kind": "organization", "refs": [u1]},
                    {"text": "温七随从", "kind": "person", "refs": [u1]},
                    {"text": "林昭", "kind": "person", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "沈甲一", "kind": "person", "refs": [u2]},
                    {"text": "沈甲二", "kind": "person", "refs": [u2]},
                    {"text": "沈甲三", "kind": "person", "refs": [u2]},
                    {"text": "沈甲四", "kind": "person", "refs": [u2]},
                    {"text": "沈甲五", "kind": "person", "refs": [u2]},
                    {"text": "温七", "kind": "person", "refs": [u2]},
                    {"text": "林昭", "kind": "person", "refs": [u2]},
                    {"text": "褚六", "kind": "person", "refs": [u2]},
                    {"text": "沈甲师爷", "kind": "person", "refs": [u2]},
                    {"text": "玄铁堡", "kind": "place", "refs": [u2]},
                    {"text": "玄铁盟", "kind": "organization", "refs": [u2]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "林昭", "kind": "person", "refs": [u3]},
                ],
                "links": [],
            },
        ]
        for t, disc in zip(tasks, disc_payloads):
            grounded = ground_result(source, t, disc, f"req_disc_{t['id']}")
            recs = [
                (kind, rec)
                for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                for rec in grounded[key]
            ]
            self.store.finish_task(run_id, t, {"request_id": f"req_disc_{t['id']}"}, recs)

        def handle_c1(state, n):
            uid = state["units"][0]["id"]
            title = state["chapter_title"]
            if title == "第一章":
                m_wen_guest = next(r["mentions"][0]["mention_id"] for r in state["surfaces"] if r["surface"] == "温七随从")
                m_lin_ch1 = next(r["mentions"][0]["mention_id"] for r in state["surfaces"] if r["surface"] == "林昭")
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": "褚六",
                            "kind": "person",
                            "surfaces": [
                                {"surface": "褚六", "surface_type": "proper_name"},
                                {"surface": "沈甲师爷", "surface_type": "title_or_kinship"},
                            ],
                            "evidence_ref": uid,
                        },
                        {"local_id": "e2", "name": "沈甲剑客", "kind": "person", "surfaces": [{"surface": "沈甲剑客", "surface_type": "description"}], "evidence_ref": uid},
                        {"local_id": "e4", "name": "沈甲使者", "kind": "person", "surfaces": [{"surface": "沈甲使者", "surface_type": "description"}], "evidence_ref": uid},
                        {"local_id": "e5", "name": "那座玄铁堡", "kind": "place", "surfaces": [{"surface": "那座玄铁堡", "surface_type": "description"}], "evidence_ref": uid},
                        {"local_id": "e6", "name": "玄铁盟分舵", "kind": "organization", "surfaces": [{"surface": "玄铁盟分舵", "surface_type": "description"}], "evidence_ref": uid},
                    ],
                    "assignments": [
                        {"mention_id": m_wen_guest, "surface": "温七随从", "ref": uid, "local_id": None, "reason": "未决"},
                        {"mention_id": m_lin_ch1, "surface": "林昭", "ref": uid, "local_id": None, "reason": "同名待决"},
                    ],
                    "not_entity": [],
                }, f"c1_c3v_{n}"
            if title == "第二章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "沈甲一", "kind": "person", "surfaces": [{"surface": "沈甲一", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e2", "name": "沈甲二", "kind": "person", "surfaces": [{"surface": "沈甲二", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e3", "name": "沈甲三", "kind": "person", "surfaces": [{"surface": "沈甲三", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e4", "name": "沈甲四", "kind": "person", "surfaces": [{"surface": "沈甲四", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e5", "name": "沈甲五", "kind": "person", "surfaces": [{"surface": "沈甲五", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e6", "name": "温七", "kind": "person", "surfaces": [{"surface": "温七", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e7", "name": "林昭", "kind": "person", "surfaces": [{"surface": "林昭", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e8", "name": "玄铁堡", "kind": "place", "surfaces": [{"surface": "玄铁堡", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e9", "name": "玄铁盟", "kind": "organization", "surfaces": [{"surface": "玄铁盟", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e10", "name": "褚六", "kind": "person", "surfaces": [{"surface": "褚六", "surface_type": "proper_name"}], "evidence_ref": uid},
                        {"local_id": "e11", "name": "沈甲师爷", "kind": "person", "surfaces": [{"surface": "沈甲师爷", "surface_type": "title_or_kinship"}], "evidence_ref": uid},
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, f"c1_c3v_{n}"
            return {
                "entities": [
                    {"local_id": "e1", "name": "林昭", "kind": "person", "surfaces": [{"surface": "林昭", "surface_type": "proper_name"}], "evidence_ref": uid},
                ],
                "assignments": [],
                "not_entity": [],
            }, f"c1_c3v_{n}"

        captured_c3_state = []

        def handle_c3(state, n):
            cands_by_name = {c["name"]: c["entity_id"] for c in state["candidates"]}
            items_by_surf = {it["surface"]: it for it in state["items"]}

            if "沈甲剑客" not in items_by_surf:
                # 第二章 C3（沈甲师爷）
                return {
                    "links": [
                        {"mention_id": it["mention_id"], "selected_entity_id": "none", "reason": "无"}
                        for it in state["items"]
                    ]
                }, f"c3_c3v_{n}"

            captured_c3_state.append(state)

            wen7_eid = cands_by_name["温七"]
            chu6_eid = cands_by_name["褚六"]
            fort_eid = cands_by_name["玄铁堡"]
            guild_eid = cands_by_name["玄铁盟"]

            # 确认各项选择的目标均不在其自身 candidate_entity_ids 中，但在本请求 candidates 中
            self.assertNotIn(wen7_eid, items_by_surf["沈甲剑客"]["candidate_entity_ids"])
            self.assertNotIn(wen7_eid, items_by_surf["林昭"]["candidate_entity_ids"])
            self.assertNotIn(chu6_eid, items_by_surf["沈甲师爷"]["candidate_entity_ids"])
            self.assertNotIn(fort_eid, items_by_surf["温七随从"]["candidate_entity_ids"])
            self.assertNotIn(guild_eid, items_by_surf["沈甲使者"]["candidate_entity_ids"])

            links_dict = {
                # 1. 同粗类越项选择（person -> 温七），通过 D1 与 cannot_merge_by_obj -> c3_model
                items_by_surf["沈甲剑客"]["mention_id"]: {"selected_entity_id": wen7_eid, "reason": "越项同粗类合法"},
                # 2. 越项选择违反 D1（专名提及“林昭” -> 温七） -> c3_rejected_name_mismatch
                items_by_surf["林昭"]["mention_id"]: {"selected_entity_id": wen7_eid, "reason": "越项违反D1"},
                # 3. 对象项越项选择违反 cannot_merge_by_obj（第一章被 C2.0 剥离的“沈甲师爷” -> 原父对象“褚六”） -> c3_rejected_name_mismatch
                items_by_surf["沈甲师爷"]["mention_id"]: {"selected_entity_id": chu6_eid, "reason": "越项违反同章不可合并"},
                # 4a. 跨粗类越项选择（person“温七随从” -> place“玄铁堡”） -> c3_rejected_name_mismatch
                items_by_surf["温七随从"]["mention_id"]: {"selected_entity_id": fort_eid, "reason": "跨粗类person-place"},
                # 4b. 跨粗类越项选择（person“沈甲使者” -> organization“玄铁盟”）
                items_by_surf["沈甲使者"]["mention_id"]: {"selected_entity_id": guild_eid, "reason": "跨粗类person-org"},
                # 其余项正常处理
                items_by_surf["那座玄铁堡"]["mention_id"]: {"selected_entity_id": fort_eid, "reason": "本项候选"},
                items_by_surf["玄铁盟分舵"]["mention_id"]: {"selected_entity_id": guild_eid, "reason": "本项候选"},
            }
            return {
                "links": [
                    {"mention_id": it["mention_id"], **links_dict[it["mention_id"]]}
                    for it in state["items"]
                ]
            }, f"c3_c3v_{n}"

        models = ScriptedRosterModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"prop_c3v_{n}"),
            "roster-c2-review": lambda state, n: (
                {
                    "groups": [
                        {"group_id": f"g{idx}", "name": r["name"], "kind": r["kind"], "row_ids": [r["row_id"]], "member_ids": [], "reason": "分立"}
                        for idx, r in enumerate(state["rows"], start=1)
                    ],
                    "unresolved_row_ids": [],
                    "unresolved_member_ids": [],
                },
                f"rev_c3v_{n}",
            ),
            "roster-c3-link": handle_c3,
        })
        out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(out["complete"], out)
        self.assertGreaterEqual(len(captured_c3_state), 1)

        # (5) 验证第一章 C3 请求 state 不含 item_kind，且 c3_tid 与按公开字段计算的摘要逐字节一致
        c3_state = captured_c3_state[0]
        for it in c3_state["items"]:
            self.assertEqual(set(it.keys()), {"mention_id", "surface", "context", "candidate_entity_ids"})
        c3_task_rows = self.store.rows("SELECT id, result FROM tasks WHERE run_id=? AND stage='roster-c3-link'", (run_id,))
        import json
        ch1_c3_row = next(r for r in c3_task_rows if len(json.loads(r["result"])["links"]) == 7)
        stored_c3_tid = ch1_c3_row["id"]
        stored_c3_links = json.loads(ch1_c3_row["result"])["links"]
        roster_out = self.store.records(run_id, "roster_output")[0]
        by_surf_m = {m["text"]: m for m in roster_out["mentions"] if m["chapter_id"] == chapters[0]["id"]}
        short_to_full_mid = {f"m{idx}": row["mention_id"] for idx, row in enumerate(stored_c3_links, start=1)}
        cand_full_ids = sorted(
            e["id"]
            for e in roster_out["entities"]
            if set(e["aliases"]) & {"沈甲一", "沈甲二", "沈甲三", "沈甲四", "沈甲五", "褚六", "玄铁堡", "温七", "林昭", "玄铁盟"}
        )
        short_to_full_eid = {f"e{idx}": rid for idx, rid in enumerate(cand_full_ids, start=1)}
        expected_c3_tid = digest([VERSION, "c3-link", tasks[0]["id"], c3_state, {"items": short_to_full_mid, "candidates": short_to_full_eid}, C3_LINK_PROMPT])
        self.assertEqual(stored_c3_tid, expected_c3_tid)

        # 验证 (1) 沈甲剑客 -> 温七 按 c3_model 成功归入，不产生待审
        wen7_ent = next(e for e in roster_out["entities"] if "温七" in e["aliases"])
        self.assertEqual(by_surf_m["沈甲剑客"]["entity_id"], wen7_ent["id"])
        self.assertEqual(by_surf_m["沈甲剑客"]["link_method"], "c3_model")

        # 验证 (2) 林昭（第一章未决专名提及） -> 温七 因违反 D1 转 c3_rejected_name_mismatch 并生成待审
        self.assertEqual(by_surf_m["林昭"]["link_method"], "c3_rejected_name_mismatch")
        # 验证 (3) 沈甲师爷 -> 褚六 因违反 cannot_merge_by_obj 转 c3_rejected_name_mismatch 并生成待审
        self.assertEqual(by_surf_m["沈甲师爷"]["link_method"], "c3_rejected_name_mismatch")
        # 验证 (4a) 温七随从(person) -> 玄铁堡(place) 因跨粗类转 c3_rejected_name_mismatch 并生成待审
        self.assertEqual(by_surf_m["温七随从"]["link_method"], "c3_rejected_name_mismatch")
        # 验证 (4b) 沈甲使者(person) -> 玄铁盟(organization)：V2 在通过 D1 与 cannot_merge_by_obj 时按 c3_model 采纳
        guild_ent = next(e for e in roster_out["entities"] if "玄铁盟" in e["aliases"])
        self.assertEqual(by_surf_m["沈甲使者"]["entity_id"], guild_ent["id"])
        self.assertEqual(by_surf_m["沈甲使者"]["link_method"], "c3_model")

        c3_mismatch_surfaces = {
            r["surface"]
            for r in roster_out["reviews"]
            if r["stage"] == "roster-c3-link" and r["reason"] == "C3 选中条目违反专名一致性硬约束，已作废转待审"
        }
        self.assertEqual(c3_mismatch_surfaces, {"林昭", "沈甲师爷", "温七随从"})

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
