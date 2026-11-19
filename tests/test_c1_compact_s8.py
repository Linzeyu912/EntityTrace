# -*- coding: utf-8 -*-
"""S8 C1 紧凑短编号输出的单元测试（纯本地，零网络、零生产代码改动）。

覆盖：p/t/d 短码还原、s/u 短编号还原与发给模型的 state 结构、D2.2 辖区借代双对象
不变量、二元数组 assign、按 (surface, ref) 单元指派、Strict 严格/宽松行为、旧版
冗长格式向后兼容、C1_PROMPT 静态断言、以及 resolve_roster 两章紧凑输出端到端回归。
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.models import ModelError
from entitytrace.roster import (
    C1_PROMPT,
    C1_SURFACE_TYPE_CODES,
    apply_c1_task_result,
    normalize_c1_raw_output,
    normalize_surface,
    resolve_roster,
    validate_c1_structure,
)
from entitytrace.storage import Store
from entitytrace.text import address, import_text
from entitytrace.util import digest

FORBIDDEN_C1_PROMPT_TERMS = (
    "Bennet",
    "Gardiner",
    "Lucases",
    "Bingleys",
    "Lady Catherine",
    "Sir William",
    "黄知县",
    "鲍师爷",
    "刘太守",
    "孔制台",
    "凌氏",
    "郑氏",
    "叶氏",
    "陈氏",
    "我今天来",
    "大有关系",
    "老表台",
    "侄老爹",
    "两位钦差",
)

REQUIRED_C1_PROMPT_TERMS = ("u1..uM", "s1..sK", '"assign"')


class ScriptedModels:
    """最小模型桩：按 stage 返回脚本值，并记录每次 generate 的 state。"""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def generate(self, system, state, stage, validate, *args, **kwargs):
        self.calls.append((stage, copy.deepcopy(state)))
        handler = self.handlers[stage]
        value, req_id = handler(state, len(self.calls))
        validate(value)
        return value, req_id


class C1CompactS8Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "test.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    # ------------------------------------------------------------------
    # 夹具
    # ------------------------------------------------------------------
    def _manual_mentions(self, source, task, text: str, specs: list[dict]) -> list[dict]:
        """按“同一称呼第 n 次出现”手工构造提及，避免 discovery 把同形词合并到同一位置。"""
        occurrences: dict[str, int] = {}
        mentions = []
        for spec in specs:
            surf = spec["text"]
            nth = occurrences.get(surf, 0)
            occurrences[surf] = nth + 1
            cursor, start = task["start"], -1
            for _ in range(nth + 1):
                start = text.find(surf, cursor, task["end"])
                if start < 0:
                    raise AssertionError(f"原文缺少第 {nth + 1} 次 {surf!r}")
                cursor = start + 1
            end = start + len(surf)
            mentions.append({
                "id": digest([source["id"], start, end, surf]),
                "text": surf,
                "start": start,
                "end": end,
                "address": address(source["id"], start, end),
                "chapter_id": task["chapter_id"],
                "task_id": task["id"],
                "provisional_kinds": [spec.get("kind", "unknown")],
                "observation_ids": [],
                "evidence": [],
            })
        return sorted(mentions, key=lambda m: (m["start"], m["end"], m["id"]))

    @staticmethod
    def _short_maps(surfaces: list[str], units: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
        short_to_surf = {f"s{i}": s for i, s in enumerate(sorted(set(surfaces)), start=1)}
        short_to_uid = {f"u{i}": u["id"] for i, u in enumerate(units, start=1)}
        return short_to_surf, short_to_uid

    def _two_chapter_run(self, store: Store, legacy: bool) -> dict:
        """两章小书：沈砚/沈公子同指、陆珩跨章同指、青州城为地点、众人为 not_entity。"""
        text = (
            "第一章\n沈砚在青州城读书，陆珩来访，众人在门外散去。\n"
            "第二章\n沈砚赴京，陆珩相送，沈公子辞行。\n"
        )
        path = self.root / "two_chapters.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        run_id = "run_two_chapters"
        store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        self.assertEqual(len(tasks), 2)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        discoveries = [
            {
                "mentions": [
                    {"text": "沈砚", "kind": "person", "refs": [u1]},
                    {"text": "青州城", "kind": "place", "refs": [u1]},
                    {"text": "陆珩", "kind": "person", "refs": [u1]},
                    {"text": "众人", "kind": "person", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "沈砚", "kind": "person", "refs": [u2]},
                    {"text": "陆珩", "kind": "person", "refs": [u2]},
                    {"text": "沈公子", "kind": "person", "refs": [u2]},
                ],
                "links": [],
            },
        ]
        for task, disc in zip(tasks, discoveries):
            grounded = ground_result(source, task, disc, "req_" + task["id"])
            records = [
                (kind, rec)
                for kind, key in [
                    ("mention", "mentions"),
                    ("observation", "observations"),
                    ("review", "reviews"),
                    ("link", "links"),
                ]
                for rec in grounded[key]
            ]
            store.finish_task(run_id, task, {"request_id": "req_" + task["id"]}, records)

        def handler(state, call_no):
            rows = {row["surface"]: row["id"] for row in state["surfaces"]}
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                groups = [
                    ("person", [("沈砚", "p")]),
                    ("place", [("青州城", "p")]),
                    ("person", [("陆珩", "p")]),
                ]
                not_entity = ["众人"]
            else:
                groups = [
                    ("person", [("沈砚", "p"), ("沈公子", "t")]),
                    ("person", [("陆珩", "p")]),
                ]
                not_entity = []
            if legacy:
                return {
                    "entities": [
                        {
                            "local_id": f"e{i}",
                            "name": members[0][0],
                            "kind": kind,
                            "surfaces": [
                                {
                                    "surface": surf,
                                    "surface_type": C1_SURFACE_TYPE_CODES[stype],
                                }
                                for surf, stype in members
                            ],
                            "evidence_ref": uid,
                        }
                        for i, (kind, members) in enumerate(groups, start=1)
                    ],
                    "assignments": [],
                    "not_entity": not_entity,
                }, f"req_c1_{call_no}"
            return {
                "entities": [
                    {
                        "id": f"e{i}",
                        "name": rows[members[0][0]],
                        "kind": kind,
                        "s": [[rows[surf], stype] for surf, stype in members],
                        "ref": uid,
                    }
                    for i, (kind, members) in enumerate(groups, start=1)
                ],
                "assign": [],
                "not_entity": [rows[s] for s in not_entity],
            }, f"req_c1_{call_no}"

        models = ScriptedModels({
            "roster-c1": handler,
            "roster-c2-review": lambda state, n: (
                {
                    "groups": [{
                        "group_id": "g1",
                        "name": state["rows"][0]["name"],
                        "kind": state["rows"][0]["kind"],
                        "row_ids": [r["row_id"] for r in state["rows"]],
                        "reason": "合成样本同指",
                    }],
                    "unresolved_row_ids": [],
                },
                f"req_c2_review_{n}",
            ),
            "roster-c2-propose": lambda state, n: ({"proposals": []}, f"req_c2_propose_{n}"),
        })
        outcome = resolve_roster(store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"], outcome)
        return store.records(run_id, "roster_output")[0]

    # ------------------------------------------------------------------
    # 1. p/t/d 短码还原与 declared/proper 精确一致
    # ------------------------------------------------------------------
    def test_surface_type_short_codes_restored_and_declared_proper_matches(self):
        text = "第一章\n沈砚拜见沈大人。他父亲早年曾任知县。\n"
        surfaces = ["沈砚", "沈大人", "他父亲"]
        path = self.root / "type_codes.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        head = text.index("第一章")
        _, chapters = build_chapters(
            source, [{"start": head, "end": head + 3, "title": "第一章", "kind": "chapter"}]
        )
        task = plan_tasks(source, chapters, 9000)[0]
        uid = task["units"][0]["id"]
        grounded = ground_result(
            source, task,
            {"mentions": [{"text": s, "kind": "person", "refs": [uid]} for s in surfaces], "links": []},
            "req_type",
        )
        short_to_surf, short_to_uid = self._short_maps(surfaces, task["units"])
        surf_sid = {v: k for k, v in short_to_surf.items()}

        def short_sid(surface: str) -> str:
            return surf_sid[surface]

        self.assertEqual(C1_SURFACE_TYPE_CODES["p"], "proper_name")
        self.assertEqual(C1_SURFACE_TYPE_CODES["t"], "title_or_kinship")
        self.assertEqual(C1_SURFACE_TYPE_CODES["d"], "description")
        raw = {
            "entities": [{
                "id": "e1",
                "name": short_sid("沈砚"),
                "kind": "person",
                "s": [[short_sid("沈砚"), "p"], [short_sid("沈大人"), "t"], [short_sid("他父亲"), "d"]],
                "ref": "u1",
            }],
            "assign": [],
            "not_entity": [],
        }
        norm = normalize_c1_raw_output(
            raw, short_to_surf, short_to_uid,
            {m["id"]: m for m in grounded["mentions"]},
            {m["id"]: uid for m in grounded["mentions"]},
        )
        entity = norm["entities"][0]
        self.assertEqual(entity["name"], "沈砚")
        self.assertEqual(entity["evidence_ref"], uid)
        self.assertEqual(
            entity["surfaces"],
            [
                {"surface": "沈砚", "surface_type": "proper_name"},
                {"surface": "沈大人", "surface_type": "title_or_kinship"},
                {"surface": "他父亲", "surface_type": "description"},
            ],
        )
        res = apply_c1_task_result(
            source, task, grounded["mentions"], raw, "req_apply",
            strict=True, require_surface_type=True, short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        self.assertEqual(len(res["entities"]), 1)
        applied = res["entities"][0]
        self.assertEqual(applied["surface_types"]["沈砚"], "proper_name")
        self.assertEqual(applied["surface_types"]["沈大人"], "title_or_kinship")
        self.assertEqual(applied["surface_types"]["他父亲"], "description")
        self.assertEqual(applied["proper_surfaces"], ["沈砚"])
        self.assertEqual(res["declared_proper_norms"], [normalize_surface("沈砚")])
        self.assertEqual(
            sorted(res["declared_proper_norms"]),
            sorted(normalize_surface(s) for s in applied["proper_surfaces"]),
        )
        self.assertTrue(all(o["status"] == "assigned" and o["local_id"] == "e1"
                            for o in res["mention_outcomes"]))

    # ------------------------------------------------------------------
    # 2. s1..sK / u1..uM 短编号还原与 c1_worker 发出的 state 结构
    # ------------------------------------------------------------------
    def test_short_surface_and_unit_ids_restore_and_state_shape(self):
        text = "第一章\n沈砚在城东读书，子墨作陪。\n沈砚又去城西。\n"
        path = self.root / "short_ids.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        head = text.index("第一章")
        _, chapters = build_chapters(
            source, [{"start": head, "end": head + 3, "title": "第一章", "kind": "chapter"}]
        )
        tasks = plan_tasks(source, chapters, 9000)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(len(task["units"]), 2)
        u1, u2 = task["units"][0]["id"], task["units"][1]["id"]
        grounded = ground_result(
            source, task,
            {"mentions": [
                {"text": "沈砚", "kind": "person", "refs": [u1, u2]},
                {"text": "子墨", "kind": "person", "refs": [u1]},
            ], "links": []},
            "req_short",
        )
        self.assertEqual(len([m for m in grounded["mentions"] if m["text"] == "沈砚"]), 2)
        captured: dict = {}

        def handler(state, call_no):
            captured.update(copy.deepcopy(state))
            rows = {row["surface"]: row["id"] for row in state["surfaces"]}
            return {
                "entities": [{
                    "id": "e1",
                    "name": rows["沈砚"],
                    "kind": "person",
                    "s": [[rows["沈砚"], "p"], [rows["子墨"], "p"]],
                    "ref": state["units"][0]["id"],
                }],
                "assign": [],
                "not_entity": [],
            }, "req_c1_short"

        run_id = "run_short"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        self.store.put(run_id, "chapter", chapters[0])
        self.store.finish_task(run_id, task, {"request_id": "req_short"}, [
            ("mention", rec) for rec in grounded["mentions"]
        ] + [("observation", rec) for rec in grounded["observations"]]
          + [("review", rec) for rec in grounded["reviews"]]
          + [("link", rec) for rec in grounded["links"]])
        models = ScriptedModels({"roster-c1": handler})
        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"], outcome)
        self.assertEqual([c[0] for c in models.calls], ["roster-c1"])

        self.assertEqual([u["id"] for u in captured["units"]], ["u1", "u2"])
        self.assertEqual(captured["units"][0]["text"], task["units"][0]["text"])
        self.assertEqual(captured["units"][1]["text"], task["units"][1]["text"])
        self.assertEqual(sorted(row["id"] for row in captured["surfaces"]), ["s1", "s2"])
        row_by_surface = {row["surface"]: row for row in captured["surfaces"]}
        self.assertEqual(row_by_surface["沈砚"]["id"], "s2")
        self.assertEqual(row_by_surface["子墨"]["id"], "s1")
        self.assertEqual(row_by_surface["沈砚"]["refs"], ["u1", "u2"])
        self.assertEqual(row_by_surface["子墨"]["refs"], ["u1"])
        by_ref = {m["ref"] for m in row_by_surface["沈砚"]["mentions"]}
        self.assertEqual(by_ref, {"u1", "u2"})
        self.assertTrue(all(m["mention_id"].startswith("m") for row in captured["surfaces"]
                            for m in row.get("mentions", [])))
        self.assertTrue(all(m["ref"] in {"u1", "u2"} for row in captured["surfaces"]
                            for m in row.get("mentions", [])))
        # 真实 unit id（形如 u12_34）不得泄漏到发给模型的短编号 state
        self.assertNotIn(u1, [u["id"] for u in captured["units"]])
        self.assertNotIn(u2, [u["id"] for u in captured["units"]])

    # ------------------------------------------------------------------
    # 3. D2.2 辖区借代双对象不变量
    # ------------------------------------------------------------------
    def test_d2_metonymy_dual_object_does_not_leak_shared_surface(self):
        text = "第一章\n青州县坐落在城东。青州县升堂问案。众人称青州知县为父母官。\n"
        path = self.root / "d2_dual.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        head = text.index("第一章")
        _, chapters = build_chapters(
            source, [{"start": head, "end": head + 3, "title": "第一章", "kind": "chapter"}]
        )
        task = plan_tasks(source, chapters, 9000)[0]
        mentions = self._manual_mentions(source, task, text, [
            {"text": "青州县", "kind": "place"},
            {"text": "青州县", "kind": "person"},
            {"text": "青州知县", "kind": "person"},
        ])
        self.assertEqual([m["provisional_kinds"] for m in mentions],
                         [["place"], ["person"], ["person"]])
        short_to_surf, short_to_uid = self._short_maps([m["text"] for m in mentions], task["units"])
        self.assertEqual(short_to_surf, {"s1": "青州县", "s2": "青州知县"})
        raw = {
            "entities": [
                {"id": "e1", "name": "s1", "kind": "place", "s": [], "ref": "u1"},
                {"id": "e2", "name": "s2", "kind": "person", "s": [["s2", "p"]], "ref": "u1"},
            ],
            "assign": [{"m": "m1", "to": "e1"}, {"m": "m2", "to": "e2"}],
            "not_entity": [],
        }
        mention_by_key = {m["id"]: m for m in mentions}
        mention_by_key.update({f"m{i}": m for i, m in enumerate(mentions, start=1)})
        norm = normalize_c1_raw_output(
            raw, short_to_surf, short_to_uid, mention_by_key,
            {m["id"]: task["units"][0]["id"] for m in mentions},
        )
        self.assertEqual([a["surface"] for a in norm["assignments"]], ["青州县", "青州县"])
        self.assertEqual([a["ref"] for a in norm["assignments"]],
                         [task["units"][0]["id"], task["units"][0]["id"]])

        res = apply_c1_task_result(
            source, task, mentions, raw, "req_d2",
            strict=True, require_surface_type=True,
            short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        outcomes = {o["mention_id"]: o for o in res["mention_outcomes"]}
        self.assertEqual(outcomes[mentions[0]["id"]]["local_id"], "e1")
        self.assertEqual(outcomes[mentions[1]["id"]]["local_id"], "e2")
        self.assertEqual(outcomes[mentions[2]["id"]]["local_id"], "e2")
        by_local = {ent["local_id"]: ent for ent in res["entities"]}
        place_ent, person_ent = by_local["e1"], by_local["e2"]
        self.assertEqual(place_ent["kind"], "place")
        self.assertEqual(place_ent["surfaces"], ["青州县"])
        self.assertEqual(place_ent["proper_surfaces"], ["青州县"])
        # 地点 e1 合法保留辖区专名；被禁止的是把共享的同形称呼塞回“借代主官” e2 ——
        # e2 只能保留自己独立的称呼（青州知县），绝不能把辖区名当作人物 proper 称呼。
        self.assertNotIn("青州县", person_ent["surfaces"])
        self.assertNotIn("青州县", person_ent["proper_surfaces"])
        self.assertEqual(person_ent["surfaces"], ["青州知县"])
        self.assertEqual(person_ent["proper_surfaces"], ["青州知县"])
        self.assertIn(normalize_surface("青州县"), res["weak_norms"])
        covered = sorted(mid for ent in res["entities"] for mid in ent["mention_ids"])
        self.assertEqual(covered, sorted(m["id"] for m in mentions))

    # ------------------------------------------------------------------
    # 4. 二元数组 assign 兼容
    # ------------------------------------------------------------------
    def test_binary_assign_array_not_entity_null_and_entity(self):
        text = "第一章\n沈园花开。闲人散去。沈砚来访。\n"
        path = self.root / "binary_assign.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        head = text.index("第一章")
        _, chapters = build_chapters(
            source, [{"start": head, "end": head + 3, "title": "第一章", "kind": "chapter"}]
        )
        task = plan_tasks(source, chapters, 9000)[0]
        uid = task["units"][0]["id"]
        grounded = ground_result(
            source, task,
            {"mentions": [
                {"text": "沈园", "kind": "place", "refs": [uid]},
                {"text": "闲人", "kind": "person", "refs": [uid]},
                {"text": "沈砚", "kind": "person", "refs": [uid]},
            ], "links": []},
            "req_binary",
        )
        mentions = sorted(grounded["mentions"], key=lambda m: (m["start"], m["end"], m["id"]))
        self.assertEqual([m["text"] for m in mentions], ["沈园", "闲人", "沈砚"])
        short_to_surf, short_to_uid = self._short_maps([m["text"] for m in mentions], task["units"])
        self.assertEqual(short_to_surf, {"s1": "沈园", "s2": "沈砚", "s3": "闲人"})
        raw = {
            "entities": [{"id": "e1", "name": "s2", "kind": "person", "s": [["s2", "p"]], "ref": "u1"}],
            "assign": [["m1", "not_entity"], ["m2", "null"], ["m3", "e1"]],
            "not_entity": [],
        }
        res = apply_c1_task_result(
            source, task, mentions, raw, "req_binary_apply",
            strict=True, require_surface_type=True,
            short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        outcomes = {o["mention_id"]: o for o in res["mention_outcomes"]}
        self.assertEqual(outcomes[mentions[0]["id"]]["status"], "not_entity")
        self.assertIsNone(outcomes[mentions[0]["id"]]["local_id"])
        self.assertEqual(outcomes[mentions[1]["id"]]["status"], "unresolved")
        self.assertIsNone(outcomes[mentions[1]["id"]]["local_id"])
        self.assertEqual(outcomes[mentions[2]["id"]]["status"], "assigned")
        self.assertEqual(outcomes[mentions[2]["id"]]["local_id"], "e1")

    # ------------------------------------------------------------------
    # 5. 高频称呼按 (surface, ref) 单元指派
    # ------------------------------------------------------------------
    def test_high_frequency_surface_assigned_by_surface_and_unit(self):
        text = "第一章\n沈砚在城东读书。\n沈砚在城西下棋，陆珩旁观。\n"
        path = self.root / "surf_ref.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        head = text.index("第一章")
        _, chapters = build_chapters(
            source, [{"start": head, "end": head + 3, "title": "第一章", "kind": "chapter"}]
        )
        task = plan_tasks(source, chapters, 9000)[0]
        self.assertEqual(len(task["units"]), 2)
        u1, u2 = task["units"][0]["id"], task["units"][1]["id"]
        grounded = ground_result(
            source, task,
            {"mentions": [
                {"text": "沈砚", "kind": "person", "refs": [u1, u2]},
                {"text": "陆珩", "kind": "person", "refs": [u2]},
            ], "links": []},
            "req_surf_ref",
        )
        mentions = sorted(grounded["mentions"], key=lambda m: (m["start"], m["end"], m["id"]))
        self.assertEqual([(m["text"], m["start"] >= task["units"][1]["start"]) for m in mentions],
                         [("沈砚", False), ("沈砚", True), ("陆珩", True)])
        short_to_surf, short_to_uid = self._short_maps([m["text"] for m in mentions], task["units"])
        self.assertEqual(short_to_surf, {"s1": "沈砚", "s2": "陆珩"})
        raw = {
            "entities": [
                {"id": "e1", "name": "s1", "kind": "person", "s": [["s1", "p"]], "ref": "u1"},
                {"id": "e2", "name": "s2", "kind": "person", "s": [["s2", "p"]], "ref": "u2"},
            ],
            "assign": [{"s": "s1", "ref": "u2", "to": "e2"}],
            "not_entity": [],
        }
        res = apply_c1_task_result(
            source, task, mentions, raw, "req_surf_ref_apply",
            strict=True, require_surface_type=True,
            short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        outcomes = {o["mention_id"]: o for o in res["mention_outcomes"]}
        self.assertEqual(outcomes[mentions[0]["id"]]["local_id"], "e1")
        self.assertEqual(outcomes[mentions[1]["id"]]["local_id"], "e2")
        self.assertEqual(outcomes[mentions[2]["id"]]["local_id"], "e2")
        self.assertEqual(outcomes[mentions[0]["id"]]["status"], "assigned")

    # ------------------------------------------------------------------
    # 6. 非法短码与缺失字段：Strict=True/False 行为
    # ------------------------------------------------------------------
    def _strict_fixture(self):
        text = "第一章\n沈园花开。闲人散去。沈砚来访。\n"
        path = self.root / "strict.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        head = text.index("第一章")
        _, chapters = build_chapters(
            source, [{"start": head, "end": head + 3, "title": "第一章", "kind": "chapter"}]
        )
        task = plan_tasks(source, chapters, 9000)[0]
        uid = task["units"][0]["id"]
        grounded = ground_result(
            source, task,
            {"mentions": [
                {"text": "沈园", "kind": "place", "refs": [uid]},
                {"text": "闲人", "kind": "person", "refs": [uid]},
                {"text": "沈砚", "kind": "person", "refs": [uid]},
            ], "links": []},
            "req_strict",
        )
        short_to_surf, short_to_uid = self._short_maps([m["text"] for m in grounded["mentions"]], task["units"])
        base = {
            "entities": [{
                "id": "e1", "name": "s2", "kind": "person",
                "s": [["s2", "p"], ["s1", "p"], ["s3", "p"]], "ref": "u1",
            }],
            "assign": [], "not_entity": [],
        }
        return source, task, grounded["mentions"], short_to_surf, short_to_uid, base

    def test_missing_assign_and_non_dict_output_raise_model_error(self):
        source, task, mentions, _, _, base = self._strict_fixture()
        missing = copy.deepcopy(base)
        missing.pop("assign")
        with self.assertRaises(ModelError):
            apply_c1_task_result(source, task, mentions, missing, "req_missing", strict=False)
        with self.assertRaises(ModelError):
            validate_c1_structure(
                {"entities": base["entities"], "not_entity": []}, {"沈园", "闲人", "沈砚"},
                {task["units"][0]["id"]}, Strict=False,
            )
        with self.assertRaises(ModelError):
            apply_c1_task_result(source, task, mentions, ["not", "a", "dict"], "req_list")
        with self.assertRaises(ModelError):
            validate_c1_structure(["not", "a", "dict"], set(), set())

    def test_illegal_short_codes_strict_raises_and_lenient_degrades(self):
        source, task, mentions, short_to_surf, short_to_uid, base = self._strict_fixture()
        uid = task["units"][0]["id"]
        cases = {
            "unknown_surface": {**copy.deepcopy(base), "entities": [
                {"id": "e1", "name": "s2", "kind": "person",
                 "s": [["s1", "p"], ["s2", "p"], ["s3", "p"], ["s99", "p"]], "ref": "u1"}]},
            "unknown_ref": {**copy.deepcopy(base), "entities": [
                {"id": "e1", "name": "s2", "kind": "person",
                 "s": [["s1", "p"], ["s2", "p"], ["s3", "p"]], "ref": "u99"}]},
            "unknown_not_entity": {**copy.deepcopy(base), "not_entity": ["s99"]},
        }
        for name, raw in cases.items():
            with self.subTest(case=name, strict=True):
                with self.assertRaises(ModelError):
                    apply_c1_task_result(
                        source, task, mentions, raw, "req_" + name,
                        strict=True, short_to_surf=short_to_surf, short_to_uid=short_to_uid,
                    )
            with self.subTest(case=name, strict=False):
                res = apply_c1_task_result(
                    source, task, mentions, raw, "req_" + name,
                    strict=False, short_to_surf=short_to_surf, short_to_uid=short_to_uid,
                )
                self.assertEqual(len(res["entities"]), 1)
                self.assertNotIn("s99", res["entities"][0]["surfaces"])
                self.assertNotEqual(res["entities"][0]["evidence_ref"], "u99")
                if name == "unknown_ref":
                    self.assertEqual(res["entities"][0]["evidence_ref"], uid)

    def test_unknown_bare_mention_id_rejected_and_invalid_surface_type_fallback(self):
        source, task, mentions, short_to_surf, short_to_uid, base = self._strict_fixture()
        bare = copy.deepcopy(base)
        bare["assign"] = [{"m": "m99", "to": "e1"}]
        with self.assertRaises(ModelError):
            apply_c1_task_result(
                source, task, mentions, bare, "req_bare",
                strict=True, short_to_surf=short_to_surf, short_to_uid=short_to_uid,
            )
        bare_lenient = apply_c1_task_result(
            source, task, mentions, bare, "req_bare",
            strict=False, short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        self.assertEqual(len(bare_lenient["entities"]), 1)
        self.assertEqual(len(bare_lenient["mention_outcomes"]), len(mentions))
        # 带有效 surface/ref 的未知 m 在宽松模式下降级：未知提及被忽略，不污染已有归属
        graceful = copy.deepcopy(base)
        graceful["assign"] = [{"m": "m99", "s": "s2", "ref": "u1", "to": "e1"}]
        res = apply_c1_task_result(
            source, task, mentions, graceful, "req_graceful",
            strict=False, short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        self.assertEqual(len(res["entities"]), 1)
        self.assertEqual(len(res["mention_outcomes"]), len(mentions))

        bad_type = copy.deepcopy(base)
        bad_type["entities"] = [{
            "id": "e1", "name": "s2", "kind": "person",
            "s": [["s2", "p"], ["s1", "x"]], "ref": "u1",
        }]
        with self.assertRaises(ModelError):
            apply_c1_task_result(
                source, task, mentions, bad_type, "req_bad_type",
                strict=True, require_surface_type=True,
                short_to_surf=short_to_surf, short_to_uid=short_to_uid,
            )
        lenient = apply_c1_task_result(
            source, task, mentions, bad_type, "req_bad_type",
            strict=False, require_surface_type=False,
            short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        fallback = [e for e in lenient["entities"] if "沈园" in e["surfaces"]]
        self.assertTrue(fallback)
        self.assertEqual(fallback[0]["surface_types"]["沈园"], "proper_name")

    # ------------------------------------------------------------------
    # 7. 旧版冗长格式 100% 向后兼容
    # ------------------------------------------------------------------
    def test_legacy_verbose_format_equals_compact_format(self):
        text = "第一章\n沈砚拜见沈大人。他父亲早年曾任知县。\n"
        path = self.root / "legacy.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        head = text.index("第一章")
        _, chapters = build_chapters(
            source, [{"start": head, "end": head + 3, "title": "第一章", "kind": "chapter"}]
        )
        task = plan_tasks(source, chapters, 9000)[0]
        uid = task["units"][0]["id"]
        grounded = ground_result(
            source, task,
            {"mentions": [
                {"text": "沈砚", "kind": "person", "refs": [uid]},
                {"text": "沈大人", "kind": "person", "refs": [uid]},
                {"text": "他父亲", "kind": "person", "refs": [uid]},
            ], "links": []},
            "req_legacy",
        )
        short_to_surf, short_to_uid = self._short_maps(["沈砚", "沈大人", "他父亲"], task["units"])
        surf_sid = {v: k for k, v in short_to_surf.items()}
        compact = {
            "entities": [{
                "id": "e1", "name": surf_sid["沈砚"], "kind": "person",
                "s": [
                    [surf_sid["沈砚"], "p"],
                    [surf_sid["沈大人"], "t"],
                    [surf_sid["他父亲"], "d"],
                ],
                "ref": "u1",
            }],
            "assign": [], "not_entity": [],
        }
        legacy = {
            "entities": [{
                "local_id": "e1",
                "name": "沈砚",
                "kind": "person",
                "surfaces": [
                    {"surface": "沈砚", "surface_type": "proper_name"},
                    {"surface": "沈大人", "surface_type": "title_or_kinship"},
                    {"surface": "他父亲", "surface_type": "description"},
                ],
                "evidence_ref": uid,
            }],
            "assignments": [],
            "not_entity": [],
        }
        compact_res = apply_c1_task_result(
            source, task, grounded["mentions"], compact, "req_same",
            strict=True, require_surface_type=True,
            short_to_surf=short_to_surf, short_to_uid=short_to_uid,
        )
        legacy_res = apply_c1_task_result(
            source, task, grounded["mentions"], legacy, "req_same",
            strict=True, require_surface_type=True,
        )
        self.assertEqual(legacy_res, compact_res)

    def test_legacy_verbose_format_resolve_roster_matches_compact(self):
        compact_store = Store(self.root / "compact.sqlite3")
        legacy_store = Store(self.root / "legacy.sqlite3")
        try:
            compact_out = self._two_chapter_run(compact_store, legacy=False)
            legacy_out = self._two_chapter_run(legacy_store, legacy=True)
        finally:
            compact_store.close()
            legacy_store.close()

        def shape(out: dict) -> dict:
            return {
                "entities": [
                    {
                        "name": e["name"],
                        "names": e["names"],
                        "aliases": e["aliases"],
                        "kind": e["kind"],
                        "mention_count": e["mention_count"],
                        "identity_status": e["identity_status"],
                    }
                    for e in out["entities"]
                ],
                "mention_entities": sorted(
                    (m["text"], m["start"], m["entity_id"], m["link_method"]) for m in out["mentions"]
                ),
                "roster_summary": out["roster_summary"],
            }

        self.assertEqual(shape(legacy_out), shape(compact_out))

    # ------------------------------------------------------------------
    # 8. C1_PROMPT 零旧开发书专名静态断言
    # ------------------------------------------------------------------
    def test_c1_prompt_has_no_old_development_book_terms_and_has_compact_template(self):
        for term in FORBIDDEN_C1_PROMPT_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, C1_PROMPT)
        for term in REQUIRED_C1_PROMPT_TERMS:
            with self.subTest(required=term):
                self.assertIn(term, C1_PROMPT)
        self.assertNotIn("local_id", C1_PROMPT)
        self.assertNotIn("evidence_ref", C1_PROMPT)
        self.assertIn('"id":"e1"', C1_PROMPT)
        self.assertIn('"ref":"u1"', C1_PROMPT)

    # ------------------------------------------------------------------
    # 9. resolve_roster 紧凑输出两章端到端回归
    # ------------------------------------------------------------------
    def test_compact_two_chapter_resolve_roster_end_to_end(self):
        out = self._two_chapter_run(self.store, legacy=False)
        entities = out["entities"]
        mentions = out["mentions"]
        by_name = {e["name"]: e for e in entities}
        self.assertEqual(set(by_name), {"沈砚", "陆珩", "青州城", "众人"})
        shen = by_name["沈砚"]
        self.assertEqual(shen["kind"], "person")
        self.assertEqual(shen["mention_count"], 3)
        self.assertEqual(shen["names"], ["沈公子", "沈砚"])
        self.assertEqual(shen["aliases"], ["沈砚"])
        self.assertEqual(shen["identity_status"], "linked_by_roster")
        lu = by_name["陆珩"]
        self.assertEqual(lu["mention_count"], 2)
        self.assertEqual(by_name["青州城"]["kind"], "place")
        self.assertEqual(by_name["青州城"]["mention_count"], 1)
        crowd = by_name["众人"]
        self.assertEqual(crowd["identity_status"], "not_entity_suggested")
        self.assertEqual(crowd["link_methods"], {"not_entity_suggested": 1})

        mention_entity = {m["id"]: m["entity_id"] for m in mentions}
        covered = [mid for e in entities for mid in e["mention_ids"]]
        self.assertEqual(sorted(covered), sorted(mention_entity))
        self.assertEqual(len(covered), len(set(covered)))
        for ent in entities:
            for mid in ent["mention_ids"]:
                self.assertEqual(mention_entity[mid], ent["id"])
        crowd_mention = next(m for m in mentions if m["text"] == "众人")
        self.assertEqual(crowd_mention["identity_status"], "not_entity_suggested")
        self.assertEqual(crowd_mention["link_method"], "not_entity_suggested")
        self.assertEqual(out["roster_summary"]["entity_count"], 4)
        self.assertEqual(out["roster_summary"]["active_entity_count"], 3)


if __name__ == "__main__":
    unittest.main()
