from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from entitytrace.identity import group_entities, plan_pairs
from entitytrace.models import ModelError
from entitytrace.resolution import (VERSION, annotate_groups, ask_cases, candidate_edges,
    comparison_id, group_evidence, pending_reviews, plan_round, resolve_groups, resolve_task)
from entitytrace.storage import Store
from entitytrace.text import address, import_text
from entitytrace.util import digest


def response(choice, options, probability=1.0):
    return {"type": "choice", "choice": choice, "confidence": probability,
            "probabilities": {key: probability if key == choice else (1 - probability) / (len(options) - 1) for key in options}}


class Model:
    def __init__(self, choose=None):
        self.choose = choose
        self.calls = []
        self.fail = None

    def choices(self, state, questions, stage):
        if self.fail == stage:
            self.fail = None
            raise ModelError("模拟服务失败")
        self.calls.append((stage, state, questions))
        answers = {}
        for label, question in questions.items():
            options = question["criteria"]
            case = state["cases"][label]
            choice = self.choose(case, options, stage) if self.choose else None
            if choice is None:
                choice = "same" if "same" in options else "person" if "person" in options else "n0"
            answers[label] = response(choice, options)
        return answers, digest([stage, state, questions])


class GroupResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "db.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def setup_source(self, surfaces, kinds=None):
        text = "。\n".join(surfaces) + "。"
        path = self.root / "source.txt"
        path.write_bytes(text.encode("utf-8"))
        self.source = import_text(self.store, path)
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("r", self.source["id"], "{}", "{}", "now", "now"))
        start = 0
        self.mentions = {}
        for index, surface in enumerate(surfaces):
            mid = f"m{index}"
            m = {"id": mid, "text": surface, "start": start, "end": start + len(surface), "chapter_id": str(index),
                 "address": address(self.source["id"], start, start + len(surface)),
                 "provisional_kinds": [kinds[index] if kinds else "person"]}
            self.mentions[mid] = m
            self.store.put("r", "mention", m)
            start += len(surface) + 2
        self.groups = group_entities(list(self.mentions.values()), [])[0]
        return self.groups

    def manual_task(self, target=0, candidates=(1,)):
        groups = [self.groups[target], *(self.groups[i] for i in candidates)]
        edges = []
        for candidate in groups[1:]:
            a, b = groups[0], candidate
            left, right = a["mention_ids"][0], b["mention_ids"][0]
            pair = {"id": digest([left, right]), "left": left, "right": right, "start": self.mentions[left]["start"],
                    "reasons": ["同名相邻提及"]}
            edges.append({"id": comparison_id(a, b), "groups": [a["id"], b["id"]], "pair": pair})
        return {"id": "t", "stage": "identity-groups", "round": 1, "target": groups[0]["id"], "groups": groups, "edges": edges}

    def test_exact_target_is_not_other_occurrence_in_same_context(self):
        self.setup_source(["Alex", "Alex"])
        full, small = group_evidence(self.source, self.groups[1], self.mentions)
        target = small["representatives"][0]
        self.assertEqual(target["before"], "Alex。\n")
        self.assertEqual(target["target_mention"], "Alex")
        self.assertEqual(full["representatives"][0]["address"]["start"], 6)

    def test_type_is_soft_ranking_not_exclusion(self):
        self.setup_source(["Alex", "Alex", "Alex"], ["person", "object", "unknown"])
        pairs = plan_pairs(list(self.mentions.values()), [])
        edges = candidate_edges(self.groups, self.mentions, pairs, [], set())
        self.assertTrue(any(set(e["groups"]) == {self.groups[0]["id"], self.groups[1]["id"]} for e in edges))
        self.assertEqual(edges[0]["rank"][1], 0)

    def test_wrong_provisional_type_can_be_corrected_by_text(self):
        self.setup_source(["Ada", "Ada"], ["person", "object"])
        records = resolve_task(Model(), self.source, self.mentions, self.manual_task())
        relations = [r for k, r in records if k == "identity"]
        self.assertEqual(relations[0]["choice"], "same")
        self.assertTrue(any(k == "decision" and r["stage"] == "identity-type-check" for k, r in records))

    def test_unresolved_type_conflict_is_not_forced_into_one_group(self):
        self.setup_source(["Ada", "Ada’s letter"], ["person", "object"])
        def choose(case, options, stage):
            if stage == "identity-type-check":
                return "object" if "letter" in case["representatives"][0]["target_mention"] else "person"
        records = resolve_task(Model(choose), self.source, self.mentions, self.manual_task())
        self.assertEqual(next(r for k, r in records if k == "identity")["choice"], "uncertain")

    def test_competing_people_do_not_merge_through_ambiguous_name(self):
        self.setup_source(["Mr. Lee", "Alex Lee", "Ben Lee"])
        def choose(case, options, stage):
            if "same" in options:
                names = {case[s]["representatives"][0]["target_mention"] for s in ("A", "B")}
                return "different" if names == {"Alex Lee", "Ben Lee"} else "same"
        records = resolve_task(Model(choose), self.source, self.mentions, self.manual_task(candidates=(1, 2)))
        relations = [r for k, r in records if k == "identity"]
        self.assertFalse(any(p["choice"] == "same" for p in relations))
        self.assertEqual(sum(p["choice"] == "different" for p in relations), 1)
        self.assertEqual(len(group_entities(list(self.mentions.values()), relations)[0]), 3)

    def test_known_negative_constraint_overrides_new_positive_bridge(self):
        self.setup_source(["Lee", "Alex Lee", "Ben Lee"])
        task = self.manual_task(candidates=(1, 2))
        task["cannot_link"] = [[self.groups[1]["id"], self.groups[2]["id"]]]
        records = resolve_task(Model(), self.source, self.mentions, task)
        self.assertFalse(any(r["choice"] == "same" for k, r in records if k == "identity"))

    def test_competing_cross_check_preserves_saved_negative_constraint(self):
        self.setup_source(["Lee", "Alex Lee", "Ben Lee"])
        a, b = self.groups[1:]
        negative = {"id": comparison_id(a, b), "left": "m1", "right": "m2", "choice": "different",
                    "accepted": True, "decision_ids": ["prior-proof"], "group_versions": [a["id"], b["id"]]}
        self.store.put("r", "identity", negative)
        task = self.manual_task(candidates=(1, 2))
        task["cannot_link"] = [[a["id"], b["id"]]]
        records = resolve_task(Model(), self.source, self.mentions, task)
        self.store.finish_task("r", task, {}, records)
        saved = {p["id"]: p for p in self.store.records("r", "identity")}
        self.assertEqual(saved[negative["id"]], negative)
        self.assertEqual(len(group_entities(list(self.mentions.values()), list(saved.values()))[0]), 3)

    def test_cross_negative_without_confirmation_does_not_become_hard_constraint(self):
        self.setup_source(["Mr. Lee", "Alex Lee", "Ben Lee"])
        def choose(case, opts, stage):
            if "same" in opts:
                names = {case[s]["representatives"][0]["target_mention"] for s in ("A", "B")}
                if names == {"Alex Lee", "Ben Lee"}:
                    return "different" if stage == "identity-group-guard" else "uncertain"
        records = resolve_task(Model(choose), self.source, self.mentions, self.manual_task(candidates=(1, 2)))
        self.assertTrue(all(r["choice"] == "uncertain" for k, r in records if k == "identity"))

    def test_group_guard_can_reject_initial_high_score(self):
        self.setup_source(["Alex", "Al"])
        model = Model(lambda case, opts, stage: "different" if stage == "identity-group-guard" else None)
        records = resolve_task(model, self.source, self.mentions, self.manual_task())
        pair = next(r for k, r in records if k == "identity")
        self.assertEqual(pair["choice"], "uncertain")
        self.assertEqual(len(pair["decision_ids"]), 2)

    def test_negative_decision_also_requires_consistent_guard(self):
        self.setup_source(["Alex", "Al"])
        model = Model(lambda case, opts, stage: "different" if stage == "identity-group" else "same")
        records = resolve_task(model, self.source, self.mentions, self.manual_task())
        pair = next(r for k, r in records if k == "identity")
        self.assertEqual(pair["choice"], "uncertain")
        self.assertIn("禁止合并", pair["reason"])

    def test_round_neighbourhoods_are_disjoint_and_order_is_reproducible(self):
        self.setup_source(["Alex"] * 12)
        pairs = plan_pairs(list(self.mentions.values()), [])
        edges = candidate_edges(self.groups, self.mentions, pairs, [], set())
        tasks = plan_round(self.groups, edges, 1, 3)
        gids = [g["id"] for t in tasks for g in t["groups"]]
        self.assertEqual(len(gids), len(set(gids)))
        reversed_edges = candidate_edges(list(reversed(self.groups)), self.mentions, list(reversed(pairs)), [], set())
        self.assertEqual(tasks, plan_round(list(reversed(self.groups)), reversed_edges, 1, 3))

    def test_completed_round_snapshot_survives_failed_task(self):
        self.setup_source(["Alex"] * 8)
        model = Model()
        model.fail = "identity-group"
        failed = resolve_groups(self.store, model, self.source, "r", 1, lambda _: None, 4, 1)
        self.assertFalse(failed["complete"])
        original_plan = self.store.records("r", "resolution_plan")[0]
        completed = self.store.rows("SELECT id FROM tasks WHERE run_id='r' AND status='completed'")
        self.assertTrue(completed)
        succeeded = resolve_groups(self.store, model, self.source, "r", 1, lambda _: None, 4, 1)
        self.assertTrue(succeeded["complete"])
        self.assertEqual(original_plan["tasks"], self.store.records("r", "resolution_plan")[0]["tasks"])
        entities, conflicts = group_entities(list(self.mentions.values()), self.store.records("r", "identity"))
        self.assertEqual((len(entities), len(conflicts)), (1, 0))
        before = len(model.calls)
        resolve_groups(self.store, model, self.source, "r", 1, lambda _: None, 4, 1)
        self.assertEqual(len(model.calls), before)

    def test_round_budget_reports_uncompared_current_groups(self):
        self.setup_source(["Alex"] * 8)
        outcome = resolve_groups(self.store, Model(), self.source, "r", 1, lambda _: None, 1, 1)
        self.assertTrue(outcome["complete"])
        self.assertEqual(outcome["stop_reason"], "round_budget")
        self.assertGreater(outcome["uncompared_group_pairs"], 0)
        self.assertTrue(any("轮次预算" in r["reason"] for r in self.store.records("r", "review")))

    def test_type_and_display_annotation_cannot_change_membership(self):
        self.setup_source(["Alex", "Al"])
        task = self.manual_task()
        relations = [r for k, r in resolve_task(Model(), self.source, self.mentions, task) if k == "identity"]
        groups, _ = group_entities(list(self.mentions.values()), relations)
        annotation = [{"group_id": groups[0]["id"], "field": "name", "value": "Invented name", "status": "accepted", "decision_ids": ["d"]},
                      {"group_id": groups[0]["id"], "field": "kind", "value": "unknown", "status": "conflict"}]
        result = annotate_groups(groups, annotation)
        self.assertEqual(result[0]["mention_ids"], groups[0]["mention_ids"])
        self.assertIn(result[0]["name"], groups[0]["names"])
        self.assertEqual(result[0]["identity_status"], "needs_review")

    def test_reviews_are_collapsed_to_current_group_pairs(self):
        self.setup_source(["Alex", "Al"])
        relations = [{"id": str(i), "left": "m0", "right": "m1", "choice": "uncertain", "decision_ids": [str(i)], "round": i} for i in range(3)]
        reviews = pending_reviews(self.groups, relations, [], [])
        self.assertEqual(len(reviews), 1)
        self.assertEqual(len(reviews[0]["comparison_ids"]), 3)

    def test_batch_token_split_keeps_question_and_evidence_references(self):
        class Limited(Model):
            def choices(self, state, questions, stage):
                if len(questions) > 1:
                    raise ModelError("长输入", 400, "max_tokens_exceeded")
                return super().choices(state, questions, stage)
        cases = [{"id": str(i), "input": {"address": {"start": i}}, "state": {"target": str(i)},
                  "instructions": "判断", "options": {"same": "同一", "different": "不同"}} for i in range(3)]
        model = Limited()
        answers = ask_cases(model, cases, "test")
        self.assertEqual(set(answers), {"0", "1", "2"})
        for key, decision in answers.items():
            self.assertEqual(decision["input"]["address"]["start"], int(key))
            self.assertEqual(decision["question_id"], "c0")
        self.assertEqual(len(model.calls), 3)

    def test_identical_cached_payload_keeps_distinct_decision_provenance(self):
        model = Model()
        case = {"id": "first", "input": {"address": {"start": 0}}, "state": {"text": "same excerpt"},
                "instructions": "判断", "options": {"same": "同一", "different": "不同"}}
        other = {**case, "id": "second", "input": {"address": {"start": 100}}}
        first = ask_cases(model, [case], "identity-group")["first"]
        second = ask_cases(model, [other], "identity-group")["second"]
        guard = ask_cases(model, [case], "identity-group-guard")["first"]
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["id"], guard["id"])
        self.assertEqual(second["input"]["address"]["start"], 100)


if __name__ == "__main__":
    unittest.main()
