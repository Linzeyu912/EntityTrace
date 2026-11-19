from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.roster import resolve_c2_merge_components, resolve_roster
from entitytrace.storage import Store
from entitytrace.text import import_text


class ResolveC2MergeComponentsTests(unittest.TestCase):
    def test_clean_transitive_component_merges(self):
        props = [
            {"left": "a", "right": "b"},
            {"left": "b", "right": "c"},
        ]
        vres = {
            ("a", "b"): {"verdict": "same"},
            ("b", "c"): {"verdict": "same"},
        }
        merge, conflict = resolve_c2_merge_components(props, vres, [], set())
        self.assertEqual(merge, [["a", "b", "c"]])
        self.assertEqual(conflict, [])

    def test_conflict_component_blocks_whole_group_and_keeps_clean_group(self):
        props = [
            {"left": "a", "right": "b"},
            {"left": "b", "right": "c"},
            {"left": "a", "right": "c"},
            {"left": "d", "right": "e"},
            {"left": "e", "right": "f"},
        ]
        vres = {
            ("a", "b"): {"verdict": "same"},
            ("b", "c"): {"verdict": "same"},
            ("a", "c"): {"verdict": "different"},
            ("d", "e"): {"verdict": "same"},
            ("e", "f"): {"verdict": "same"},
        }
        merge, conflict = resolve_c2_merge_components(props, vres, [], {("a", "c")})
        self.assertEqual(merge, [["d", "e", "f"]])
        self.assertEqual(conflict, [["a", "b", "c"]])

    def test_existing_cannot_merge_entries_blocks_pair_and_multi_node_bridge(self):
        props = [
            {"left": "e1", "right": "e2"},
            {"left": "e2", "right": "e3"},
            {"left": "e3", "right": "e4"},
            {"left": "x1", "right": "x2"},
        ]
        vres = {
            ("e1", "e2"): {"verdict": "same"},
            ("e2", "e3"): {"verdict": "same"},
            ("e3", "e4"): {"verdict": "same"},
            ("x1", "x2"): {"verdict": "same"},
        }
        merge, conflict = resolve_c2_merge_components(props, vres, [], {("e1", "e4")})
        self.assertEqual(merge, [["x1", "x2"]])
        self.assertEqual(conflict, [["e1", "e2", "e3", "e4"]])

    def test_deterministic_under_shuffle(self):
        props = [
            {"left": "a", "right": "b"},
            {"left": "b", "right": "c"},
            {"left": "d", "right": "e"},
            {"left": "f", "right": "g"},
            {"left": "g", "right": "h"},
        ]
        vres = {
            ("a", "b"): {"verdict": "same"},
            ("b", "c"): {"verdict": "same"},
            ("d", "e"): {"verdict": "same"},
            ("f", "g"): {"verdict": "same"},
            ("g", "h"): {"verdict": "same"},
        }
        cannot = {("a", "c")}
        base = resolve_c2_merge_components(props, vres, [], cannot)
        self.assertEqual(base, ([["d", "e"], ["f", "g", "h"]], [["a", "b", "c"]]))
        rng = random.Random(20261119)
        for _ in range(10):
            shuffled = props[:]
            rng.shuffle(shuffled)
            self.assertEqual(resolve_c2_merge_components(shuffled, vres, [], cannot), base)


class ScriptedModels:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def generate(self, system, state, stage, validate, *args, **kwargs):
        self.calls.append((stage, state))
        value, req_id = self.handlers[stage](state)
        validate(value)
        return value, req_id


class C2ConflictComponentEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "test.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_resolve_roster_blocks_conflicting_bridge_and_merges_clean_pair(self):
        # 5 个跨章独立条目：林大人、赵大夫、孙掌柜（通过 C2 提议构成桥接冲突连通块），
        # 以及钱大人、周老爷（构成独立干净合并对）。
        ch1 = "林大人与钱大人在堂上议事。"
        ch2 = "赵大夫、孙掌柜与周老爷同往城南。"
        text = "第一章\n" + ch1 + "\n第二章\n" + ch2
        path = self.root / "book.txt"
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
        run_id = "run_c2_guard"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        self.assertEqual(len(tasks), 2)
        u1 = tasks[0]["units"][0]["id"]
        u2 = tasks[1]["units"][0]["id"]

        discoveries = [
            {
                "mentions": [
                    {"text": "林大人", "kind": "person", "refs": [u1]},
                    {"text": "钱大人", "kind": "person", "refs": [u1]},
                ],
                "links": [],
            },
            {
                "mentions": [
                    {"text": "赵大夫", "kind": "person", "refs": [u2]},
                    {"text": "孙掌柜", "kind": "person", "refs": [u2]},
                    {"text": "周老爷", "kind": "person", "refs": [u2]},
                ],
                "links": [],
            },
        ]
        for task, disc in zip(tasks, discoveries):
            grounded = ground_result(source, task, disc, "req_" + task["id"])
            records = [
                (kind, rec)
                for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                for rec in grounded[key]
            ]
            self.store.finish_task(run_id, task, {"request_id": "req_" + task["id"]}, records)

        def handle_c1(state):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {"local_id": "EA", "name": "林大人", "kind": "person", "surfaces": ["林大人"], "evidence_ref": uid},
                        {"local_id": "EF", "name": "钱大人", "kind": "person", "surfaces": ["钱大人"], "evidence_ref": uid},
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, "req_c1_1"
            return {
                "entities": [
                    {"local_id": "EB", "name": "赵大夫", "kind": "person", "surfaces": ["赵大夫"], "evidence_ref": uid},
                    {"local_id": "EC", "name": "孙掌柜", "kind": "person", "surfaces": ["孙掌柜"], "evidence_ref": uid},
                    {"local_id": "EG", "name": "周老爷", "kind": "person", "surfaces": ["周老爷"], "evidence_ref": uid},
                ],
                "assignments": [],
                "not_entity": [],
            }, "req_c1_2"

        def handle_propose(state):
            by_name = {e["name"]: e["entry_id"] for e in state["entries"]}
            proposals = []
            for left_n, right_n, reason in (
                ("林大人", "赵大夫", "疑似同一人甲乙"),
                ("林大人", "孙掌柜", "疑似同一人甲丙"),
                ("赵大夫", "孙掌柜", "疑似同一人乙丙"),
                ("钱大人", "周老爷", "疑似同一人丁戊"),
            ):
                if left_n in by_name and right_n in by_name:
                    a, b = sorted((by_name[left_n], by_name[right_n]))
                    proposals.append({"left": a, "right": b, "reason": reason})
            return {"proposals": proposals}, "req_prop"

        def handle_verify(state):
            pair_names = {state["left"]["name"], state["right"]["name"]}
            if pair_names == {"赵大夫", "孙掌柜"}:
                return {"verdict": "different", "reason": "实为二人"}, "req_ver_diff"
            return {"verdict": "same", "reason": "同指一人"}, f"req_ver_{'_'.join(sorted(pair_names))}"

        models = ScriptedModels({
            "roster-c1": handle_c1,
            "roster-c2-propose": handle_propose,
            "roster-c2-verify": handle_verify,
        })

        outcome = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(outcome["complete"], outcome)

        out = self.store.records(run_id, "roster_output")[0]
        entities = out["entities"]
        names = sorted(e["name"] for e in entities)
        # 桥接冲突块 {林大人, 赵大夫, 孙掌柜} 整组保持拆开(3个)；干净对 {钱大人, 周老爷} 正常合并(1个) -> 共 4 个实体
        self.assertEqual(len(entities), 4, names)
        merged = [e for e in entities if "钱大人" in e["aliases"] and "周老爷" in e["aliases"]]
        self.assertEqual(len(merged), 1, entities)
        for isolated_name in ("林大人", "赵大夫", "孙掌柜"):
            matches = [e for e in entities if isolated_name in e["aliases"]]
            self.assertEqual(len(matches), 1, entities)
            self.assertEqual(matches[0]["aliases"], [isolated_name], matches[0])

        reviews = out.get("reviews", [])
        conflict_reviews = [
            r for r in reviews
            if r.get("reason") == "C2 名册补合并连通块内存在冲突，整组暂不合并"
        ]
        self.assertEqual(len(conflict_reviews), 1, reviews)
        cr = conflict_reviews[0]
        self.assertEqual(cr["stage"], "roster-c2-verify")
        self.assertEqual(cr["status"], "pending")
        self.assertEqual(sorted(cr["member_names"]), ["孙掌柜", "林大人", "赵大夫"])
        self.assertEqual(cr["impact"], 3)
        self.assertEqual(cr["options"], ["same", "different", "uncertain"])
        self.assertEqual(len(cr["evidence"]), 3)


if __name__ == "__main__":
    unittest.main()
