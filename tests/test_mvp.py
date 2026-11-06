from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import (
    build_chapters,
    check_chapter_numbering,
    collect_candidates,
    detect_chapters,
    filter_headings_with_body,
    parse_chapter_number,
    scan_window,
)
from entitytrace.config import Service, load_services
from entitytrace.discovery import ground_result
from entitytrace.identity import ask_identity_pairs, audit_aliases, group_entities, plan_pairs
from entitytrace.cli import main as cli_main
from entitytrace.models import ModelError, Models, bounded_map, check_choices, choice_question
from entitytrace.pipeline import export_run, run_pipeline
from entitytrace.storage import Store
from entitytrace.text import address, import_text, units, validate_span
from entitytrace.util import digest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "db.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def source(self, text="第一章\r\n林舟🙂来了。\r\n", bom=True):
        path = self.root / "input.txt"
        path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
        return import_text(self.store, path)

    def test_original_bytes_and_code_points(self):
        source = self.source()
        self.assertEqual(source["raw"], b"\xef\xbb\xbf" + source["text"].encode("utf-8"))
        at = source["text"].index("🙂")
        self.assertEqual(validate_span(source["text"], address(source["id"], at, at + 1), source["id"]), "🙂")
        self.assertIn("\r\n", source["text"])
        self.assertEqual(len("🙂".encode("utf-16-le")), 4)

    def test_changed_text_is_new_version(self):
        a, b = self.source("林舟。"), self.source("阿舟。")
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(self.store.source(a["id"])["text"], "林舟。")

    def test_invalid_addresses_are_rejected(self):
        source = self.source("甲乙丙")
        bad = [address(source["id"], -1, 2), address(source["id"], 0, 4), address(source["id"], 1, 1),
               address("wrong", 0, 1), {**address(source["id"], 0, 1), "offset_unit": "utf16"}]
        for span in bad:
            with self.assertRaises(ValueError):
                validate_span(source["text"], span, source["id"])

    def test_chapter_ranges_restore_all_characters(self):
        source = self.source("书名\r\n第一章\r\n林舟。\r\n第一章\r\n阿舟。")
        text = source["text"]
        starts = [i for i in range(len(text)) if text.startswith("第一章", i)]
        _, chapters = build_chapters(source, [{"start": a, "end": a + 3, "title": "第一章", "kind": "chapter"} for a in starts])
        self.assertEqual("".join(text[c["start"]:c["end"]] for c in chapters), text)
        self.assertNotEqual(chapters[1]["id"], chapters[2]["id"])

    def test_grounding_preserves_repeated_occurrences(self):
        source = self.source("林舟问林舟的朋友。", False)
        task = {"id": "task", "chapter_id": "c", "start": 0, "end": len(source["text"]), "units": units(source["text"])}
        value = {"mentions": [{"text": "林舟", "kind": "person", "refs": [task["units"][0]["id"]]}], "links": []}
        result = ground_result(source, task, value, "request")
        self.assertEqual([m["start"] for m in result["mentions"]], [0, 3])
        self.assertEqual(len(result["observations"]), 1)

    def test_unlocated_candidate_is_retained(self):
        source = self.source("Anne entered.", False)
        task = {"id": "t", "chapter_id": "c", "start": 0, "end": len(source["text"]), "units": units(source["text"])}
        value = {"mentions": [{"text": "Ann", "kind": "person", "refs": [task["units"][0]["id"], "unknown"]}], "links": []}
        result = ground_result(source, task, value, "r")
        self.assertFalse(result["mentions"])
        self.assertEqual(result["observations"][0]["status"], "unlocated")
        self.assertEqual(len(result["reviews"]), 2)

    def test_task_completion_is_atomic(self):
        source = self.source()
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("r", source["id"], "{}", "{}", "now", "now"))
        with self.assertRaises(TypeError):
            self.store.finish_task("r", {"id": "t", "stage": "discovery"}, {},
                                   [("mention", {"id": "m"}), ("mention", {"id": "bad", "x": {1, 2}})])
        self.assertIsNone(self.store.task("r", "t"))
        self.assertEqual(self.store.records("r", "mention"), [])


class IdentityTests(unittest.TestCase):
    def test_conflicting_transitivity_does_not_merge(self):
        mentions = [{"id": x, "start": i, "end": i + 1, "text": x} for i, x in enumerate("abc")]
        pairs = [{"id": "1", "left": "a", "right": "b", "choice": "same"},
                 {"id": "2", "left": "b", "right": "c", "choice": "same"},
                 {"id": "3", "left": "a", "right": "c", "choice": "different"}]
        groups, conflicts = group_entities(mentions, pairs)
        self.assertEqual(len(conflicts), 1)
        self.assertFalse(any({"a", "c"}.issubset(g["mention_ids"]) for g in groups))
        self.assertEqual(sorted(m for g in groups for m in g["mention_ids"]), list("abc"))

    def test_uncompared_is_not_different(self):
        groups, _ = group_entities([{"id": "a", "start": 0, "end": 2, "text": "林舟"}], [])
        self.assertEqual(groups[0]["identity_status"], "unresolved_singleton")

    def test_same_name_is_only_candidate(self):
        mentions = [{"id": x, "start": i * 100, "end": i * 100 + 2, "text": "老周", "chapter_id": str(i)} for i, x in enumerate("ab")]
        pairs = plan_pairs(mentions, [])
        self.assertEqual(len(pairs), 1)
        self.assertNotIn("choice", pairs[0])

    def test_self_alias_does_not_make_pair(self):
        m = {"id": "a", "start": 0, "end": 2, "text": "林舟", "chapter_id": "c"}
        pairs = plan_pairs([m], [{"id": "l", "left": "林舟", "right": "林舟", "evidence": [{"start": 0}]}])
        self.assertEqual(pairs, [])


class AliasAuditTests(unittest.TestCase):
    def audit_case(self, audit_answer, kinds):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.txt"
            text = "Ada owns Ada’s letter."
            path.write_text(text, encoding="utf-8")
            store = Store(root / "db.sqlite3")
            source = import_text(store, path)
            store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("r", source["id"], "{}", "{}", "now", "now"))
            for index, surface in enumerate(["Ada", "Ada’s letter"]):
                start = text.index(surface)
                store.put("r", "mention", {"id": str(index), "start": start, "end": start + len(surface), "text": surface,
                    "address": address(source["id"], start, start + len(surface)), "provisional_kinds": [kinds[index]]})
            store.put("r", "identity", {"id": "pair", "left": "0", "right": "1", "start": 0, "reasons": ["名称召回"],
                                      "choice": "same", "accepted": True, "decision_ids": ["initial"]})
            calls = []
            case = self
            class Model:
                def choices(self, state, questions, stage):
                    calls.append(state)
                    target = state["pairs"]["p0"]
                    case.assertEqual(target["A"]["target_mention"], "Ada")
                    case.assertEqual(target["B"]["target_mention"], "Ada’s letter")
                    case.assertEqual(target["B"]["before"], "Ada owns ")
                    return {"pair": {"choice": audit_answer, "confidence": 1,
                        "probabilities": {k: int(k == audit_answer) for k in ["same", "different", "uncertain"]}}}, "request"
            model = Model()
            outcome = audit_aliases(store, model, source, "r", 1, lambda _: None, .95, .8)
            self.assertTrue(outcome["complete"])
            pair = store.records("r", "identity")[0]
            self.assertEqual(pair["choice"], "uncertain")
            self.assertEqual(pair["pre_audit_choice"], "same")
            self.assertEqual(len(pair["decision_ids"]), 2)
            groups, _ = group_entities(store.records("r", "mention"), [pair])
            self.assertEqual(len(groups), 2)
            audit_aliases(store, model, source, "r", 1, lambda _: None, .95, .8)
            self.assertEqual(len(calls), 1)
            store.close()

    def test_audit_disagreement_suspends_merge_and_resumes(self):
        self.audit_case("different", ["person", "unknown"])

    def test_provisional_type_conflict_requires_review_even_with_high_score(self):
        self.audit_case("same", ["person", "object"])


class RequestTests(unittest.TestCase):
    def test_pool_refills_before_slowest_finishes(self):
        third_started = threading.Event()
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()
        def worker(i):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            if i == 0:
                if not third_started.wait(2):
                    raise AssertionError("未及时补充第三个任务")
            if i == 2:
                third_started.set()
            with lock:
                state["active"] -= 1
            return i
        values = list(bounded_map([0, 1, 2], worker, 2))
        self.assertTrue(all(error is None for _, _, error in values))
        self.assertEqual(state["peak"], 2)

    def test_probability_validation(self):
        q = {"q": choice_question("?", {"a": "a", "b": "b"})}
        for probs, choice in [({"a": .8, "b": .8}, "a"), ({"a": .2, "b": .8}, "a"), ({"a": 1}, "a")]:
            with self.assertRaises(ModelError):
                check_choices({"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "choice": choice, "confidence": .9, "probabilities": probs}}}, q, "jev-1.13.0")

    def test_network_outside_transaction_and_cache_without_key(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "db.sqlite3")
            seen = []
            raw = {"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "choice": "a", "confidence": 1, "probabilities": {"a": 1, "b": 0}}}}
            def opener(request, **kwargs):
                self.assertFalse(store.db.in_transaction)
                seen.append(request)
                return io.BytesIO(json.dumps(raw).encode())
            services = {"decision": Service("decision", "https://api.typesafe.ai/v1", "jev-1.13.0", "TEST_SECRET"),
                        "discovery": Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")}
            model = Models(store, "r", "i", services, opener=opener)
            q = {"q": choice_question("?", {"a": "a", "b": "b"})}
            model.choices({}, q, "test")
            model.choices({}, q, "test")
            self.assertEqual(len(seen), 1)
            self.assertNotIn("TEST_SECRET", str(store.rows("SELECT * FROM request_cache")))
            self.assertEqual([x["status"] for x in store.rows("SELECT * FROM calls")], ["success", "cached"])
            store.close()


class RequestBudgetTests(unittest.TestCase):
    def test_hard_quota_stops_queued_requests_but_preserves_cache_and_other_service(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "db.sqlite3")
            services = {"decision": Service("decision", "https://api.typesafe.ai/v1", "jev-1.13.0", "TEST_SECRET"),
                        "discovery": Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")}
            raw = {"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "choice": "a", "confidence": 1,
                                                                   "probabilities": {"a": 1, "b": 0}}}}
            opened, healthy = [], False
            def opener(request, **kwargs):
                opened.append(request.full_url)
                if "deepseek" in request.full_url:
                    return io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}')
                if not healthy and not json.loads(request.data)["state"].get("warm"):
                    raise urllib.error.HTTPError(request.full_url, 402, "Payment Required", {},
                        io.BytesIO(b'{"detail":{"error_type":"billing_error"}}'))
                return io.BytesIO(json.dumps(raw).encode())
            model = Models(store, "r", "i", services, decision_concurrency=1, opener=opener)
            questions = {"q": choice_question("?", {"a": "a", "b": "b"})}
            model.choices({"warm": True}, questions, "test")
            results = list(bounded_map(list(range(8)), lambda n: model.choices({"n": n}, questions, "test"), 4))
            self.assertEqual(len(opened), 2)
            self.assertTrue(all(isinstance(error, ModelError) and error.status == 402 for _, _, error in results))
            model.choices({"warm": True}, questions, "test")
            self.assertEqual(len(opened), 2)
            model.generate("test", {}, "test", lambda value: None)
            self.assertEqual(len(opened), 3)
            healthy = True
            restarted = Models(store, "r", "j", services, opener=opener)
            restarted.choices({"n": 9}, questions, "test")
            self.assertEqual(len(opened), 4)
            self.assertEqual(sum(r["status"] == "failed" for r in store.rows("SELECT * FROM calls")), 1)
            store.close()

    def test_provider_http_error_code_is_available_to_batch_splitter(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "db.sqlite3")
            def opener(request, **kwargs):
                raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                    io.BytesIO(b'{"detail":{"error_type":"max_tokens_exceeded"}}'))
            services = {"decision": Service("decision", "https://api.typesafe.ai/v1", "jev-1.13.0", "TEST_SECRET"),
                        "discovery": Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")}
            model = Models(store, "r", "i", services, opener=opener)
            with self.assertRaises(ModelError) as error:
                model.choices({}, {"q": choice_question("?", {"a": "a", "b": "b"})}, "identity")
            self.assertEqual((error.exception.status, error.exception.code), (400, "max_tokens_exceeded"))
            calls = store.rows("SELECT * FROM calls")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("TEST_SECRET", str(calls))
            store.close()

    def evidence(self, count, length):
        return {str(i): {"mentions": [{"text": "林舟", "context": "原" * length, "address": {"start": i}}],
                         "evidence": [{"text": "补充证据", "address": {"start": i}}]} for i in range(count)}

    def test_large_context_is_split_without_losing_evidence(self):
        seen = []
        class Model:
            def choices(self, state, questions, stage):
                seen.append(state["pairs"])
                return {key: {"choice": "uncertain"} for key in questions}, "request-" + str(len(seen))
        evidence = self.evidence(12, 4000)
        answers, request_ids = ask_identity_pairs(Model(), evidence, "identity-expanded")
        self.assertEqual(set(answers), set(evidence))
        self.assertEqual(set(request_ids), set(evidence))
        self.assertGreater(len(seen), 1)
        for pair in [pair for batch in seen for pair in batch]:
            self.assertEqual(pair["mentions"][0]["context"], "原" * 4000)
            self.assertEqual(pair["extra_evidence"], ["补充证据"])

    def test_provider_token_limit_splits_and_preserves_request_references(self):
        seen = []
        class Model:
            def choices(self, state, questions, stage):
                seen.append(list(questions))
                if len(questions) > 2:
                    raise ModelError("输入过长", 400, "max_tokens_exceeded")
                return {key: {"choice": "same"} for key in questions}, "request-" + str(len(seen))
        answers, request_ids = ask_identity_pairs(Model(), self.evidence(5, 100), "identity")
        self.assertEqual(set(answers), {"0", "1", "2", "3", "4"})
        for key, request_id in request_ids.items():
            self.assertIn(key, seen[int(request_id.split("-")[1]) - 1])

    def test_other_service_failure_is_not_silently_split_or_accepted(self):
        class Model:
            def choices(self, *args):
                raise ModelError("授权失败", 401)
        with self.assertRaisesRegex(ModelError, "授权失败"):
            ask_identity_pairs(Model(), self.evidence(4, 100), "identity")


class FakeModels:
    generated = 0
    chosen = 0
    fail_once = False
    def __init__(self, *args, **kwargs):
        self.peaks = {}

    def choices(self, state, questions, stage):
        type(self).chosen += 1
        answers = {}
        for key, question in questions.items():
            if "same" in question["criteria"]:
                choice = "same"
            elif "person" in question["criteria"]:
                choice = "person"
            elif "n0" in question["criteria"]:
                choice = "n0"
            elif "accept" in question["criteria"]:
                choice = "accept"
            else:
                raise AssertionError(f"choices 不应被章节阶段或未知阶段调用: {stage}")
            answers[key] = {"type": "choice", "choice": choice, "confidence": 1,
                            "probabilities": {k: int(k == choice) for k in question["criteria"]}}
        return answers, digest([stage, state, questions])

    def generate(self, system, state, stage, validate, *args, **kwargs):
        type(self).generated += 1
        if type(self).fail_once:
            type(self).fail_once = False
            raise ModelError("模拟单任务失败")
        if stage == "chapter-label":
            value = {
                "headings": [
                    {
                        "id": c["id"],
                        "kind": "volume" if "卷" in c["text"].lstrip().split()[0] else "chapter",
                    }
                    for c in state["candidates"]
                    if c["text"].lstrip().startswith("第")
                ]
            }
        elif stage == "roster-c1":
            surfaces = [row["surface"] for row in state["surfaces"]]
            ev_ref = state["units"][0]["id"]
            entities = [{
                "local_id": "e1",
                "name": surfaces[0],
                "kind": "person",
                "surfaces": [{"surface": s, "surface_type": "proper_name"} for s in surfaces],
                "evidence_ref": ev_ref,
            }] if surfaces else []
            value = {"entities": entities, "assignments": [], "not_entity": []}
        elif stage == "roster-c2-propose":
            value = {"proposals": []}
        elif stage == "roster-c2-review":
            value = {
                "groups": [{
                    "group_id": "g1",
                    "name": state["rows"][0]["name"],
                    "kind": "person",
                    "row_ids": [r["row_id"] for r in state["rows"]],
                    "member_ids": [],
                    "reason": "同指",
                }],
                "unresolved_row_ids": [],
                "unresolved_member_ids": [],
            }
        elif stage == "roster-c2-verify":
            value = {"verdict": "same", "reason": "同指"}
        elif stage == "roster-c3-link":
            value = {
                "links": [
                    {"mention_id": it["mention_id"], "selected_entity_id": it["candidates"][0]["entity_id"], "reason": "同指"}
                    for it in state["items"]
                ]
            }
        else:
            refs = [u["id"] for u in state["units"] if "林舟" in u["text"]]
            value = {"mentions": [{"text": "林舟", "kind": "person", "refs": refs}], "links": []}
        validate(value)
        return value, digest([stage, state])


class PipelineTests(unittest.TestCase):
    def run_case(self, fail=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "input.txt"
            text.write_text("第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n", encoding="utf-8")
            config = root / ".env"
            config.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            FakeModels.generated = 0
            FakeModels.chosen = 0
            FakeModels.fail_once = fail
            with patch("entitytrace.pipeline.Models", FakeModels):
                run_id, output, complete = run_pipeline(text, root / "db.sqlite3", root / "out", config, log=lambda _: None)
                self.assertEqual(complete, not fail)
                if fail:
                    self.assertEqual(json.loads((output / "result.json").read_text(encoding="utf-8"))["status"], "incomplete")
                before = FakeModels.generated
                choices_before = FakeModels.chosen
                revision_before = json.loads((output / "result.json").read_text(encoding="utf-8"))["result_revision"]
                same_id, same_output, complete = run_pipeline(text, root / "db.sqlite3", root / "out", config, log=lambda _: None)
                self.assertTrue(complete)
                self.assertEqual(run_id, same_id)
                self.assertEqual(output, same_output)
                if fail:
                    self.assertGreaterEqual(FakeModels.generated - before, 1)
                else:
                    self.assertEqual(FakeModels.generated - before, 0)
                result = json.loads((output / "result.json").read_text(encoding="utf-8"))
                if not fail:
                    self.assertEqual(FakeModels.chosen, choices_before)
                    self.assertEqual(result["result_revision"], revision_before)
                self.assertEqual(len(result["mentions"]), 2)
                self.assertEqual(len(result["entities"]), 1)
                self.assertEqual(result["schema_version"], 3)
                self.assertEqual(result["roster_summary"]["algorithm"], "entity-roster-v1")
                self.assertEqual(result["entities"][0]["kind"], "person")
                self.assertEqual(result["entities"][0]["aliases"], ["林舟"])
                self.assertEqual(result["entities"][0]["mention_count"], 2)
                self.assertTrue(all(m["link_method"] == "chapter_local" for m in result["mentions"]))
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["config"]["identity_algorithm"], "entity-roster-v1")
                latest = next(i for i in manifest["invocations"] if i["id"] == manifest["state"]["last_invocation"])
                self.assertGreaterEqual(latest["wall_seconds"], latest["seconds"])
                self.assertGreaterEqual(latest["export_seconds"], 0)
                self.assertEqual((output / "source.original").read_bytes(), text.read_bytes())

    def test_pipeline_and_resume(self):
        self.run_case()

    def test_failed_unit_only_is_retried(self):
        self.run_case(True)

    def test_groups_v2_fixture_export_unchanged_before_and_after_switch(self):
        from entitytrace.util import dumps

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "input.txt"
            text.write_text("第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n", encoding="utf-8")
            config = root / ".env"
            config.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            db_path = root / "db.sqlite3"
            expected_run_id = "92c0a63e5b29e4d9afebef39eeaaf635118b8bc08f376e1296532afa947adbb7"
            # 预置历史固定夹具的章节记录（含旧版 decision_id），验证 groups-v2 身份归组与 export_run 字节级 SHA-256 完全不变
            seed_store = Store(db_path)
            try:
                src = import_text(seed_store, text)
                fixture_cfg = {
                    "implementation": "fixed-v2-baseline",
                    "models": {
                        "decision": {"service": "decision", "url": "https://api.typesafe.ai/v1", "model": "jev-1.13.0"},
                        "discovery": {"service": "discovery", "url": "https://api.deepseek.com", "model": "deepseek-flash"},
                    },
                    "discovery_concurrency": 6,
                    "decision_concurrency": 3,
                    "task_chars": 9000,
                    "offset_unit": "unicode_code_point",
                    "scope": "entity-identity-v2",
                    "identity_algorithm": "entity-groups-v2",
                    "identity_rounds": 12,
                    "group_candidates": 3,
                }
                self.assertEqual(digest([src["id"], fixture_cfg]), expected_run_id)
                fixture_chapters_state = {
                    "chapter_set_version": "5657ba596cb7428bfcad86e96460083b65b1dad215e5f62d06291757df041f40",
                    "complete": True,
                    "issues": 0,
                    "scan_windows": 1,
                    "scanned_code_points": 30,
                }
                seed_store.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                    (expected_run_id, src["id"], dumps(fixture_cfg), dumps({"chapters": fixture_chapters_state}), "now", "now"),
                )
                for ch in [
                    {
                        "address": address(src["id"], 0, 15),
                        "body_start": 6,
                        "chapter_set_version": "5657ba596cb7428bfcad86e96460083b65b1dad215e5f62d06291757df041f40",
                        "decision_id": "e045d042a1c2a2c6635f5b5659fbbe1a7836a2bc44449ecb50593bcac646f050",
                        "end": 15,
                        "heading_end": 6,
                        "id": "c025b1b458473b38e19af83ec0c84c2c660c5fbf9aa5a6cf25592cd7d426f160",
                        "kind": "chapter",
                        "ordinal": 1,
                        "parent_volume_id": None,
                        "start": 0,
                        "title": "第一章 初见",
                    },
                    {
                        "address": address(src["id"], 15, 30),
                        "body_start": 21,
                        "chapter_set_version": "5657ba596cb7428bfcad86e96460083b65b1dad215e5f62d06291757df041f40",
                        "decision_id": "6e8283c61291d855150783bcee830147d825ea5992b7cfe90e26bce71e1df946",
                        "end": 30,
                        "heading_end": 21,
                        "id": "2c70ddf1f057d1848e4b45e024c95647ca573f2db570460494a68308ccdb3406",
                        "kind": "chapter",
                        "ordinal": 2,
                        "parent_volume_id": None,
                        "start": 15,
                        "title": "第二章 再见",
                    },
                ]:
                    seed_store.put(expected_run_id, "chapter", ch)
            finally:
                seed_store.close()

            FakeModels.generated = 0
            FakeModels.chosen = 0
            FakeModels.fail_once = False
            with patch("entitytrace.pipeline.Models", FakeModels), patch(
                "entitytrace.pipeline.implementation_hash", return_value="fixed-v2-baseline"
            ):
                run_id, output, complete = run_pipeline(
                    text, db_path, root / "out", config,
                    identity_algorithm="groups-v2", log=lambda _: None,
                )
            self.assertTrue(complete)
            raw_bytes = (output / "result.json").read_bytes()
            result = json.loads(raw_bytes)
            # 与本单修改前 groups-v2 夹具导出的 run_id、result_revision 与完整字节 SHA-256 逐位一致
            self.assertEqual(run_id, "92c0a63e5b29e4d9afebef39eeaaf635118b8bc08f376e1296532afa947adbb7")
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["result_revision"], "a6fe9ea010f92620e50f312457bb16c67e5f1fd3447c3cfb03bad45dc646aea8")
            self.assertEqual(
                hashlib.sha256(raw_bytes).hexdigest(),
                "b1d5ee3104a3a481eb4406a51658af9031cf020610eb39492a3c6ee06e77c741",
            )
            store = Store(db_path)
            try:
                reexported = export_run(store, run_id, root / "reexport")
            finally:
                store.close()
            self.assertEqual((reexported / "result.json").read_bytes(), raw_bytes)

            # 同时验证全新库下 groups-v2 配合新章节识别可端到端跑通且 chapter_set_version 不变
            fresh_db = root / "fresh_v2.sqlite3"
            with patch("entitytrace.pipeline.Models", FakeModels), patch(
                "entitytrace.pipeline.implementation_hash", return_value="fixed-v2-baseline"
            ):
                fresh_id, fresh_out, fresh_complete = run_pipeline(
                    text, fresh_db, root / "fresh_out", config,
                    identity_algorithm="groups-v2", log=lambda _: None,
                )
            self.assertTrue(fresh_complete)
            self.assertEqual(fresh_id, expected_run_id)
            fresh_res = json.loads((fresh_out / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(fresh_res["schema_version"], 2)
            self.assertEqual(
                fresh_res["chapters"][0]["chapter_set_version"],
                "5657ba596cb7428bfcad86e96460083b65b1dad215e5f62d06291757df041f40",
            )
            self.assertIn("request_id", fresh_res["chapters"][0])

    def test_cli_identity_algorithm_roster_default_and_groups_v2_option(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "input.txt"
            text.write_text("第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n", encoding="utf-8")
            config = root / ".env"
            config.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            db_path = root / "db.sqlite3"
            out_roster = root / "out_roster"
            out_v2 = root / "out_v2"
            FakeModels.fail_once = False
            with patch("entitytrace.pipeline.Models", FakeModels):
                self.assertEqual(
                    cli_main(["run", str(text), "--db", str(db_path), "--output", str(out_roster), "--config", str(config)]),
                    0,
                )
                self.assertEqual(
                    cli_main([
                        "run", str(text), "--db", str(db_path), "--output", str(out_v2),
                        "--config", str(config), "--identity-algorithm", "groups-v2",
                    ]),
                    0,
                )
            roster_dir = next(p for p in out_roster.iterdir() if p.is_dir())
            v2_dir = next(p for p in out_v2.iterdir() if p.is_dir())
            roster_res = json.loads((roster_dir / "result.json").read_text(encoding="utf-8"))
            v2_res = json.loads((v2_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(roster_res["schema_version"], 3)
            self.assertEqual(v2_res["schema_version"], 2)
            self.assertNotEqual(roster_res["run_id"], v2_res["run_id"])

    def test_roster_experiment_delegates_export_to_export_run(self):
        from tools.roster_experiment import export_experiment, read_source_run, seed_experiment_store
        from entitytrace.models import RequestBudget
        from entitytrace.roster import DIRECT_VERSION, resolve_surface_direct

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "input.txt"
            text.write_text("第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n", encoding="utf-8")
            config = root / ".env"
            config.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            src_db = root / "src.sqlite3"
            FakeModels.fail_once = False
            with patch("entitytrace.pipeline.Models", FakeModels):
                src_run_id, _, _ = run_pipeline(text, src_db, root / "src_out", config, log=lambda _: None)
            bundle = read_source_run(src_db, src_run_id)
            exp_db = root / "exp.sqlite3"
            exp_store = Store(exp_db)
            try:
                exp_cfg = {"identity_algorithm": DIRECT_VERSION, "source_run_id": src_run_id, "arm": "B"}
                exp_run_id = "exp-b-run-0001"
                seed_experiment_store(exp_store, bundle, exp_run_id, exp_cfg)
                roster_out = resolve_surface_direct(
                    bundle["source"], bundle["chapters"], bundle["mentions"], bundle["links"], bundle["reviews"]
                )
                exp_out = root / "exp_out"
                manifest = export_experiment(
                    exp_store, bundle, exp_run_id, exp_cfg, roster_out, exp_out,
                    True, 0.01, 0.02, RequestBudget(10, 60), "inv-1", None,
                )
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["source_run_id"], src_run_id)
                exp_result = json.loads((exp_out / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(exp_result["schema_version"], 3)
                self.assertEqual(exp_result["source_run_id"], src_run_id)
            finally:
                exp_store.close()

    def test_roster_experiment_stops_when_chapters_incomplete(self):
        import argparse
        from tools.roster_experiment import read_source_run, run_experiment, seed_experiment_store

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "input.txt"
            # 章号从第一章跳至第三章且区间内含未独立成题的“第二章”，触发章号连续性阻断（chapters.complete == False）
            text.write_text("第一章 初见\n林舟在第二章前停步。\n第三章 跳章\n林舟走了。\n", encoding="utf-8")
            config = root / ".env"
            config.write_text("DEEPSEEK_API_KEY=test\n", encoding="utf-8")
            src_db = root / "src.sqlite3"
            with patch("entitytrace.pipeline.Models", FakeModels):
                src_run_id, _, complete = run_pipeline(text, src_db, root / "src_out", config, log=lambda _: None)
            self.assertFalse(complete)
            bundle = read_source_run(src_db, src_run_id)
            self.assertFalse(bundle["source_state"]["chapters"]["complete"])

            exp_db = root / "exp.sqlite3"
            exp_store = Store(exp_db)
            try:
                seed_experiment_store(exp_store, bundle, "exp-incomplete-01", {"identity_algorithm": "entity-roster-v1"})
                seeded_state = exp_store.run("exp-incomplete-01")["state"]
                self.assertFalse(seeded_state["chapters"]["complete"])
            finally:
                exp_store.close()

            exp_out = root / "exp_out"
            args = argparse.Namespace(
                source_db=src_db,
                source_run=src_run_id,
                source_file=None,
                task_chars=9000,
                recompute_from_cache=False,
                arm="C",
                db=root / "exp_run.sqlite3",
                output=exp_out,
                config=config,
                concurrency=1,
                max_requests=100,
                max_seconds=60,
            )
            with patch("tools.roster_experiment.Models", FakeModels):
                code = run_experiment(args)
            self.assertEqual(code, 2)
            res_doc = json.loads((exp_out / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(res_doc["status"], "incomplete")


class ChapterCandidateAndStructureTests(unittest.TestCase):
    def test_overlong_and_incomplete_lines_excluded_from_candidates(self):
        exact_120 = "甲" * 120
        over_121 = "乙" * 121
        # 715 码点的单行被 units(limit=700) 拆成 700 + 15 两段，两段 complete_line 均为 False（含长度<=120的尾段）
        over_700 = "丙" * 715
        short_head = "第一章 启程"
        text = f"{exact_120}\n{over_121}\n\n{short_head}\n{over_700}\n"
        cands = collect_candidates(text)
        titles = [c["title"] for c in cands]
        self.assertEqual(titles, [exact_120, short_head])
        head_cand = cands[1]
        # 紧邻的非空物理行分别为 over_121 与 over_700，各截取到 80 码点
        self.assertEqual(head_cand["prev"], "乙" * 80)
        self.assertEqual(head_cand["next"], "丙" * 80)
        self.assertEqual(text[head_cand["start"]:head_cand["end"]], short_head)

    def test_ending_line_labeled_extra_is_dropped_and_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = Store(root / "db.sqlite3")
            try:
                path = root / "story.txt"
                path.write_text("第一章 夜渡\n江风吹动灯火。\n——尾记完——\n\n", encoding="utf-8")
                source = import_text(store, path)
                store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("r1", source["id"], "{}", "{}", "now", "now"))

                class EndingExtraModels:
                    def choices(s, *args, **kwargs):
                        raise AssertionError("章节阶段禁止调用 choices")

                    def generate(s, system, state, stage, validate, *args, **kwargs):
                        self.assertEqual(kwargs.get("temperature"), 0)
                        cands = state["candidates"]
                        val = {
                            "headings": [
                                {"id": cands[0]["id"], "kind": "chapter"},
                                {"id": cands[2]["id"], "kind": "extra"},
                            ]
                        }
                        validate(val)
                        return val, "req-label-ending"

                outcome = detect_chapters(store, EndingExtraModels(), source, "r1", lambda _: None)
                self.assertTrue(outcome["complete"])
                self.assertEqual(outcome["issues"], 0)
                self.assertEqual(outcome["path_counts"]["labeled"], 2)
                self.assertEqual(outcome["path_counts"]["adopted"], 1)
                self.assertEqual(outcome["path_counts"]["dropped_no_body"], 1)
                self.assertEqual([c["title"] for c in outcome["chapters"]], ["第一章 夜渡"])
                decisions = store.records("r1", "decision")
                self.assertEqual(len(decisions), 2)
                by_res = {rec["question_id"]: rec["result"] for rec in decisions}
                self.assertEqual(sorted(by_res.values()), ["adopted", "dropped_no_body"])
            finally:
                store.close()

    def test_toc_run_drops_all_before_real_heading_or_keeps_only_last_before_body(self):
        # 情形 A：目录串后紧跟正式章节标题（中间无正文）-> 目录串全部丢弃
        text_a = (
            "第一章 临江\n"
            "第二章 宿驿\n"
            "第三章 入城\n\n"
            "第一章 临江\n"
            "晨雾散尽，船靠北岸。\n"
            "第二章 宿驿\n"
            "灯下清点文书。\n"
            "第三章 入城\n"
            "城门刚刚开启。\n"
        )
        cands_a = collect_candidates(text_a)
        labeled_a = [
            {**c, "kind": "chapter", "request_id": "req-a"}
            for c in cands_a
            if c["title"].startswith("第")
        ]
        self.assertEqual(len(labeled_a), 6)
        kept_a, dropped_a = filter_headings_with_body(text_a, labeled_a)
        self.assertEqual(len(dropped_a), 3)
        self.assertEqual(len(kept_a), 3)
        self.assertTrue(all(h["start"] >= text_a.index("\n\n") for h in kept_a))

        # 情形 B：目录串后面没有其他被标出的标题，直接接正文 -> 目录串只保留最后一条
        text_b = "第一章 临江\n第二章 宿驿\n第三章 入城\n正文中没有再重复章题。\n"
        cands_b = collect_candidates(text_b)
        labeled_b = [
            {**c, "kind": "chapter", "request_id": "req-b"}
            for c in cands_b
            if c["title"].startswith("第")
        ]
        self.assertEqual(len(labeled_b), 3)
        kept_b, dropped_b = filter_headings_with_body(text_b, labeled_b)
        self.assertEqual(len(dropped_b), 2)
        self.assertEqual([h["title"] for h in kept_b], ["第三章 入城"])

    def test_volume_immediately_followed_by_chapter_is_kept_but_empty_volume_dropped(self):
        text = (
            "空卷甲\n"
            "第一卷 朔风\n"
            "第一章 启行\n"
            "车马出关。\n"
            "末尾空卷\n"
        )
        cands = collect_candidates(text)
        kind_map = {
            "空卷甲": "volume",
            "第一卷 朔风": "volume",
            "第一章 启行": "chapter",
            "末尾空卷": "volume",
        }
        labeled = [
            {**c, "kind": kind_map[c["title"]], "request_id": "req-vol"}
            for c in cands
            if c["title"] in kind_map
        ]
        kept, dropped = filter_headings_with_body(text, labeled)
        self.assertEqual([h["title"] for h in kept], ["第一卷 朔风", "第一章 启行"])
        self.assertEqual([h["title"] for h in dropped], ["空卷甲", "末尾空卷"])

    def test_chapter_number_parsing_chinese_roman_and_arabic_three_forms(self):
        cases = [
            ("第一章 初雪", 1),
            ("第十回 夜审", 10),
            ("第十一节 转折", 11),
            ("第二十回 渡口", 20),
            ("第三十六回 终局", 36),
            ("第一百零八回 聚散", 108),
            ("第一百〇九回 远帆", 109),
            ("第一百十回 归途", 110),
            ("第一百二十回 尾声", 120),
            ("第两千章 纪年", 2000),
            ("第15章 驿站", 15),
            ("Chapter 1", 1),
            ("CHAPTER XLIV.", 44),
            ("part iii", 3),
            ("PART 12", 12),
            ("1. Dawn", 1),
            ("135. Horizon", 135),
            ("I. Opening", 1),
            ("XIV. Crossing", 14),
            ("Chapter One", None),
            ("楔子", None),
            ("3.14 Ratio", None),
        ]
        for title, expected in cases:
            with self.subTest(title=title):
                self.assertEqual(parse_chapter_number(title), expected)

    def test_chapter_numbering_gap_or_duplicate_blocks_run(self):
        # 区间内无缺失章号：非阻断提示（blocking=False）；区间内含缺失章号：阻断待审（blocking=True）
        gap_adopted = [
            {"start": 0, "end": 3, "title": "第一章", "kind": "chapter"},
            {"start": 10, "end": 13, "title": "第三章", "kind": "chapter"},
        ]
        issues_gap_clean, unnum_gap = check_chapter_numbering("src-1", gap_adopted, "第一章\n正文平稳。\n第三章\n")
        self.assertEqual(unnum_gap, 0)
        self.assertEqual(len(issues_gap_clean), 1)
        self.assertFalse(issues_gap_clean[0]["blocking"])
        self.assertEqual(issues_gap_clean[0]["reason"], "原文章号跳号：区间内未找到缺失章号")
        self.assertEqual(issues_gap_clean[0]["missing_count"], 1)
        self.assertEqual(len(issues_gap_clean[0]["evidence"]), 2)

        issues_gap_block, _ = check_chapter_numbering("src-1b", gap_adopted, "第一章\n正文第二章\n第三章\n")
        self.assertEqual(len(issues_gap_block), 1)
        self.assertTrue(issues_gap_block[0]["blocking"])
        self.assertEqual(issues_gap_block[0]["reason"], "章号序列不连续：区间内存在缺失章号的疑似标题")
        self.assertEqual(len(issues_gap_block[0]["evidence"]), 3)

        roman_dot_gap = [
            {"start": 0, "end": 7, "title": "I. East", "kind": "chapter"},
            {"start": 20, "end": 29, "title": "III. West", "kind": "chapter"},
        ]
        issues_roman, _ = check_chapter_numbering("src-2", roman_dot_gap, "I. East\nII. Mid\n    III. West")
        self.assertEqual(len(issues_roman), 1)
        self.assertTrue(issues_roman[0]["blocking"])
        self.assertEqual(issues_roman[0]["reason"], "章号序列不连续：区间内存在缺失章号的疑似标题")

        dup_adopted = [
            {"start": 0, "end": 9, "title": "Chapter 2", "kind": "chapter"},
            {"start": 20, "end": 29, "title": "Chapter 2", "kind": "chapter"},
        ]
        issues_dup, _ = check_chapter_numbering("src-3", dup_adopted)
        self.assertEqual(len(issues_dup), 1)
        self.assertTrue(issues_dup[0]["blocking"])
        self.assertEqual(issues_dup[0]["reason"], "疑似重复章节")

        # 无编号章题不阻断但计数 unnumbered
        unnum_adopted = [
            {"start": 0, "end": 2, "title": "楔子", "kind": "chapter"},
            {"start": 10, "end": 19, "title": "Chapter 5", "kind": "chapter"},
            {"start": 30, "end": 39, "title": "Chapter 6", "kind": "chapter"},
        ]
        issues_ok, unnum_cnt = check_chapter_numbering("src-4", unnum_adopted)
        self.assertEqual(issues_ok, [])
        self.assertEqual(unnum_cnt, 1)

    def test_chapter_numbering_evidence_classification_and_cli_input_error(self):
        import contextlib

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = root / ".env"
            cfg.write_text("DEEPSEEK_API_KEY=test\n", encoding="utf-8")

            # 1) 区间内无“第二章”：complete == True，issues == 0，产出 1 条 blocking == False 的 review
            t1 = root / "t1.txt"
            t1.write_text("第一章 甲\n正文。\n第三章 丙\n正文。\n", encoding="utf-8")
            with patch("entitytrace.pipeline.Models", FakeModels):
                _, out1, _ = run_pipeline(
                    t1, root / "db1.sqlite3", root / "out1", cfg, stop_after="chapters", log=lambda _: None
                )
            man1 = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
            rev1 = json.loads((out1 / "review.json").read_text(encoding="utf-8"))
            self.assertTrue(man1["state"]["chapters"]["complete"])
            self.assertEqual(man1["state"]["chapters"]["issues"], 0)
            self.assertEqual(man1["counts"]["chapter_path_counts"]["numbering_issues"], 0)
            ch_revs1 = [r for r in rev1["items"] if r["stage"] == "chapters"]
            self.assertEqual(len(ch_revs1), 1)
            self.assertFalse(ch_revs1[0]["blocking"])
            self.assertEqual(ch_revs1[0]["reason"], "原文章号跳号：区间内未找到缺失章号")
            self.assertEqual(ch_revs1[0]["missing_count"], 1)

            # 2) 区间内含“第二章”：complete == False，issues == 1，blocking == True，CLI 错误类别为 input（retryable == False，退出码 1）
            t2 = root / "t2.txt"
            t2.write_text("第一章 甲\n正文第二章 乙未换行。\n第三章 丙\n正文。\n", encoding="utf-8")
            out_buf, err_buf = io.StringIO(), io.StringIO()
            with patch("entitytrace.pipeline.Models", FakeModels), \
                 contextlib.redirect_stdout(out_buf), \
                 contextlib.redirect_stderr(err_buf):
                rc2 = cli_main([
                    "run", str(t2),
                    "--db", str(root / "db2.sqlite3"),
                    "--output", str(root / "out2"),
                    "--config", str(cfg),
                    "--stop-after", "chapters",
                    "--json",
                ])
            self.assertEqual(rc2, 1)
            receipt2 = json.loads(out_buf.getvalue().strip())
            self.assertFalse(receipt2["stages"]["chapters"]["complete"])
            self.assertEqual(receipt2["error"]["category"], "input")
            self.assertFalse(receipt2["error"]["retryable"])
            self.assertIn("review --stage chapters", receipt2["error"]["message"])
            rev2 = json.loads(Path(receipt2["files"]["review"]).read_text(encoding="utf-8"))
            ch_revs2 = [r for r in rev2["items"] if r["stage"] == "chapters"]
            self.assertEqual(len(ch_revs2), 1)
            self.assertTrue(ch_revs2[0]["blocking"])
            self.assertEqual(ch_revs2[0]["reason"], "章号序列不连续：区间内存在缺失章号的疑似标题")
            self.assertEqual(len(ch_revs2[0]["evidence"]), 3)

            # 3) 含卷标题后章号从 10 回退到 8 且后接 9、章题不同：complete == True，issues == 0，产出 1 条 blocking == False 的 review
            t3 = root / "t3.txt"
            t3.write_text(
                "第八章 晨雾\n正文一。\n第九章 山雨\n正文二。\n第十章 归帆\n正文三。\n"
                "第二卷 远行\n第八章 新途\n正文四。\n第九章 渡口\n正文五。\n",
                encoding="utf-8",
            )
            with patch("entitytrace.pipeline.Models", FakeModels):
                _, out3, _ = run_pipeline(
                    t3, root / "db3.sqlite3", root / "out3", cfg, stop_after="chapters", log=lambda _: None
                )
            man3 = json.loads((out3 / "manifest.json").read_text(encoding="utf-8"))
            rev3 = json.loads((out3 / "review.json").read_text(encoding="utf-8"))
            self.assertTrue(man3["state"]["chapters"]["complete"])
            self.assertEqual(man3["state"]["chapters"]["issues"], 0)
            ch_revs3 = [r for r in rev3["items"] if r["stage"] == "chapters"]
            self.assertEqual(len(ch_revs3), 1)
            self.assertFalse(ch_revs3[0]["blocking"])
            self.assertEqual(ch_revs3[0]["reason"], "原文章号回退后连续：疑似作者重复编号")

            # 4) 重复章题 第二章 乙 ... 第二章 乙：complete == False，issues == 1，blocking == True
            t4 = root / "t4.txt"
            t4.write_text("第一章 甲\n正文一。\n第二章 乙\n正文二。\n第二章 乙\n正文二重复。\n", encoding="utf-8")
            with patch("entitytrace.pipeline.Models", FakeModels):
                _, out4, _ = run_pipeline(
                    t4, root / "db4.sqlite3", root / "out4", cfg, stop_after="chapters", log=lambda _: None
                )
            man4 = json.loads((out4 / "manifest.json").read_text(encoding="utf-8"))
            rev4 = json.loads((out4 / "review.json").read_text(encoding="utf-8"))
            self.assertFalse(man4["state"]["chapters"]["complete"])
            self.assertEqual(man4["state"]["chapters"]["issues"], 1)
            ch_revs4 = [r for r in rev4["items"] if r["stage"] == "chapters"]
            self.assertEqual(len(ch_revs4), 1)
            self.assertTrue(ch_revs4[0]["blocking"])
            self.assertEqual(ch_revs4[0]["reason"], "疑似重复章节")

    def test_chapter_numbering_restarts_from_one_after_volume_without_blocking(self):
        adopted = [
            {"start": 0, "end": 3, "title": "第一卷", "kind": "volume"},
            {"start": 10, "end": 13, "title": "第一章", "kind": "chapter"},
            {"start": 30, "end": 33, "title": "第二章", "kind": "chapter"},
            {"start": 50, "end": 53, "title": "第二卷", "kind": "volume"},
            {"start": 60, "end": 63, "title": "第一章", "kind": "chapter"},
            {"start": 80, "end": 83, "title": "第二章", "kind": "chapter"},
        ]
        issues, unnumbered = check_chapter_numbering("src-vol", adopted)
        self.assertEqual(issues, [])
        self.assertEqual(unnumbered, 0)

    def test_unknown_candidate_id_raises_format_model_error(self):
        cands = collect_candidates("第一章 启程\n天气晴朗。\n")

        class BadIdModels:
            def choices(s, *args, **kwargs):
                raise AssertionError("章节阶段禁止调用 choices")

            def generate(s, system, state, stage, validate, *args, **kwargs):
                bad = {"headings": [{"id": "u999999_999999", "kind": "chapter"}]}
                validate(bad)
                return bad, "req-bad"

        with self.assertRaises(ModelError):
            scan_window(BadIdModels(), cands)

    def test_chapter_stage_never_calls_choices(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "novel.txt"
            text.write_text("第一章 山路\n林舟前行。\n第二章 渡河\n林舟登船。\n", encoding="utf-8")
            cfg = root / ".env"
            cfg.write_text("DEEPSEEK_API_KEY=test\n", encoding="utf-8")

            class StrictNoChoicesModels(FakeModels):
                def choices(self, *args, **kwargs):
                    raise AssertionError("章节阶段一旦调用 choices 即判定失败")

            with patch("entitytrace.pipeline.Models", StrictNoChoicesModels):
                _, output, complete = run_pipeline(
                    text, root / "db.sqlite3", root / "out", cfg,
                    stop_after="chapters", log=lambda _: None,
                )
            self.assertFalse(complete)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["state"]["chapters"]["complete"])
            self.assertEqual(manifest["state"]["chapters"]["issues"], 0)
            self.assertEqual(
                manifest["counts"]["chapter_path_counts"],
                {
                    "candidates": 4,
                    "labeled": 2,
                    "adopted": 2,
                    "dropped_no_body": 0,
                    "unnumbered": 0,
                    "numbering_issues": 0,
                    "label_requests": 1,
                },
            )

    def test_chapter_gap_fill_and_volume_rekind_rules(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def run_custom_chapters(tag: str, text_str: str, label_fn):
                store = Store(root / f"{tag}.sqlite3")
                try:
                    p = root / f"{tag}.txt"
                    p.write_bytes(text_str.encode("utf-8"))
                    src = import_text(store, p)
                    store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (tag, src["id"], "{}", "{}", "now", "now"))

                    class CustomModels:
                        def choices(s, *args, **kwargs):
                            raise AssertionError("章节阶段禁止调用 choices")

                        def generate(s, system, state, stage, validate, *args, **kwargs):
                            val = {"headings": label_fn(state["candidates"])}
                            validate(val)
                            return val, f"req-{tag}"

                    out = detect_chapters(store, CustomModels(), src, tag, lambda _: None)
                    decs = store.records(tag, "decision")
                    revs = store.records(tag, "review")
                    return out, decs, revs
                finally:
                    store.close()

            # 1) 章题标签错落在上一章末句 -> 补齐第二章，末句因无正文丢弃回到上一章正文，unnumbered == 0，complete == True
            text_misplaced = (
                "第一章 晨钟\n"
                "山门开启。\n"
                "众人拾级而上。\n"
                "第二章 暮鼓\n"
                "夜色渐深。\n"
                "第三章 远钟\n"
                "晨光微露。\n"
            )
            out1, decs1, revs1 = run_custom_chapters(
                "misplaced",
                text_misplaced,
                lambda cands: [
                    {"id": c["id"], "kind": "chapter"}
                    for c in cands
                    if c["text"] in {"第一章 晨钟", "众人拾级而上。", "第三章 远钟"}
                ],
            )
            self.assertTrue(out1["complete"])
            self.assertEqual(out1["issues"], 0)
            self.assertEqual(out1["path_counts"]["unnumbered"], 0)
            self.assertEqual(out1["path_counts"]["labeled"], 3)
            self.assertEqual(out1["path_counts"]["adopted"], 3)
            self.assertEqual(out1["path_counts"]["dropped_no_body"], 1)
            self.assertEqual([c["title"] for c in out1["chapters"]], ["第一章 晨钟", "第二章 暮鼓", "第三章 远钟"])
            res_map1 = {d["question_id"]: (d["result"], d["request_id"]) for d in decs1}
            cands1 = {c["title"]: c["id"] for c in collect_candidates(text_misplaced)}
            self.assertEqual(res_map1[cands1["第二章 暮鼓"]], ("gap_filled", None))
            self.assertEqual(res_map1[cands1["众人拾级而上。"]], ("dropped_no_body", "req-misplaced"))
            self.assertEqual(revs1, [])

            # 2) 单纯漏标与连续两章漏标 -> 全部补齐
            text_multi_gap = (
                "第一章 起步\n正文一。\n"
                "第二章 涉水\n正文二。\n"
                "第三章 登山\n正文三。\n"
                "第四章 穿林\n正文四。\n"
                "第五章 抵城\n正文五。\n"
            )
            out2, decs2, revs2 = run_custom_chapters(
                "multigap",
                text_multi_gap,
                lambda cands: [
                    {"id": c["id"], "kind": "chapter"}
                    for c in cands
                    if c["text"] in {"第一章 起步", "第五章 抵城"}
                ],
            )
            self.assertTrue(out2["complete"])
            self.assertEqual(out2["issues"], 0)
            self.assertEqual([c["title"] for c in out2["chapters"]], [
                "第一章 起步", "第二章 涉水", "第三章 登山", "第四章 穿林", "第五章 抵城"
            ])
            gap_filled_decs2 = [d for d in decs2 if d["result"] == "gap_filled"]
            self.assertEqual(len(gap_filled_decs2), 3)
            self.assertTrue(all(d["request_id"] is None for d in gap_filled_decs2))
            self.assertEqual(revs2, [])

            # 3a) 缺失章号对应两行候选 -> 不补且阻断
            text_dup_cand = (
                "第一章 起步\n正文一。\n"
                "第二章 歧路甲\n正文甲。\n"
                "第二章 歧路乙\n正文乙。\n"
                "第三章 终途\n正文三。\n"
            )
            out3a, decs3a, revs3a = run_custom_chapters(
                "dupcand",
                text_dup_cand,
                lambda cands: [
                    {"id": c["id"], "kind": "chapter"}
                    for c in cands
                    if c["text"] in {"第一章 起步", "第三章 终途"}
                ],
            )
            self.assertFalse(out3a["complete"])
            self.assertEqual(out3a["issues"], 1)
            self.assertFalse(any(d["result"] == "gap_filled" for d in decs3a))
            self.assertTrue(revs3a[0]["blocking"])

            # 3b) 缺失章号只在句中出现、不是候选整行 -> 不补且阻断
            text_inline = "第一章 起步\n他在第二章的批注里写下结论。\n第三章 终途\n正文三。\n"
            out3b, decs3b, revs3b = run_custom_chapters(
                "inline",
                text_inline,
                lambda cands: [
                    {"id": c["id"], "kind": "chapter"}
                    for c in cands
                    if c["text"].startswith("第")
                ],
            )
            self.assertFalse(out3b["complete"])
            self.assertEqual(out3b["issues"], 1)
            self.assertFalse(any(d["result"] == "gap_filled" for d in decs3b))
            self.assertTrue(revs3b[0]["blocking"])

            # 3c) 缺失章号行已被模型标注后丢弃 -> 不补且阻断
            text_labeled_dropped = (
                "第一章 起步\n正文一。\n"
                "第二章 空目\n"
                "第三章 终途\n正文三。\n"
            )
            out3c, decs3c, revs3c = run_custom_chapters(
                "labeled_dropped",
                text_labeled_dropped,
                lambda cands: [
                    {"id": c["id"], "kind": "chapter"}
                    for c in cands
                    if c["text"].startswith("第")
                ],
            )
            self.assertFalse(out3c["complete"])
            self.assertEqual(out3c["issues"], 1)
            self.assertFalse(any(d["result"] == "gap_filled" for d in decs3c))
            self.assertTrue(any(d["result"] == "dropped_no_body" for d in decs3c))
            self.assertTrue(revs3c[0]["blocking"])

            # 4a) 作者跳号（原文无缺失章号）-> 仍为非阻断提示
            text_author_skip = "第一章 起步\n正文平稳无缺章。\n第三章 终途\n正文三。\n"
            out4a, decs4a, revs4a = run_custom_chapters(
                "author_skip",
                text_author_skip,
                lambda cands: [
                    {"id": c["id"], "kind": "chapter"}
                    for c in cands
                    if c["text"].startswith("第")
                ],
            )
            self.assertTrue(out4a["complete"])
            self.assertEqual(out4a["issues"], 0)
            self.assertFalse(any(d["result"] == "gap_filled" for d in decs4a))
            self.assertEqual(len(revs4a), 1)
            self.assertFalse(revs4a[0]["blocking"])

            # 4b) 英文 CHAPTER N 风格缺口补齐
            text_en_gap = (
                "CHAPTER 1. Loomings\nCall me Ishmael.\n"
                "CHAPTER 2. The Carpet-Bag\nI stuffed a shirt.\n"
                "CHAPTER 3. The Spouter-Inn\nEntering that gable-ended Spouter-Inn.\n"
            )
            out4b, decs4b, revs4b = run_custom_chapters(
                "en_gap",
                text_en_gap,
                lambda cands: [
                    {"id": c["id"], "kind": "chapter"}
                    for c in cands
                    if c["text"].startswith("CHAPTER 1.") or c["text"].startswith("CHAPTER 3.")
                ],
            )
            self.assertTrue(out4b["complete"])
            self.assertEqual(out4b["issues"], 0)
            self.assertEqual(len(out4b["chapters"]), 3)
            self.assertEqual(sum(1 for d in decs4b if d["result"] == "gap_filled"), 1)
            self.assertEqual(revs4b, [])

            # 5) 无正文的“第二卷”标为 chapter 且后接章 -> 保留为 volume、result=adopted_as_volume；后接卷题或位于文末 -> 照旧丢弃
            text_vol = (
                "第一卷 潜龙\n"
                "第一章 初入\n正文一。\n"
                "第二卷 乘风\n"
                "第二章 远航\n正文二。\n"
                "第三卷 虚卷\n"
                "第四卷 实卷\n"
                "第三章 登顶\n正文三。\n"
                "第五卷 尾空卷\n"
            )
            out5, decs5, revs5 = run_custom_chapters(
                "vol_rekind",
                text_vol,
                lambda cands: [
                    {"id": c["id"], "kind": "volume" if "第四卷" in c["text"] else "chapter"}
                    for c in cands
                    if c["text"].startswith("第")
                ],
            )
            self.assertTrue(out5["complete"])
            self.assertEqual(out5["issues"], 0)
            kept_kinds5 = [(c["title"], c["kind"]) for c in out5["chapters"]]
            self.assertEqual(kept_kinds5, [
                ("第一卷 潜龙", "volume"),
                ("第一章 初入", "chapter"),
                ("第二卷 乘风", "volume"),
                ("第二章 远航", "chapter"),
                ("第四卷 实卷", "volume"),
                ("第三章 登顶", "chapter"),
            ])
            cands5 = {c["title"]: c["id"] for c in collect_candidates(text_vol)}
            dec_by_qid5 = {d["question_id"]: d for d in decs5}
            self.assertEqual(dec_by_qid5[cands5["第一卷 潜龙"]]["kind"], "chapter")
            self.assertEqual(dec_by_qid5[cands5["第一卷 潜龙"]]["result"], "adopted_as_volume")
            self.assertEqual(dec_by_qid5[cands5["第二卷 乘风"]]["kind"], "chapter")
            self.assertEqual(dec_by_qid5[cands5["第二卷 乘风"]]["result"], "adopted_as_volume")
            self.assertEqual(dec_by_qid5[cands5["第三卷 虚卷"]]["result"], "dropped_no_body")
            self.assertEqual(dec_by_qid5[cands5["第五卷 尾空卷"]]["result"], "dropped_no_body")
            self.assertEqual(revs5, [])

    def test_chapter_labeling_concurrency_byte_invariance(self):
        from entitytrace.util import dumps

        # 构造超過 240 条候选（产生 3 个批次），验证并发 1 与并发 8 下章节表版本与决策记录逐字节相同
        lines = []
        for idx in range(1, 130):
            lines.append(f"第{idx}章 风起\n短行正文{idx}。\n")
        text = "".join(lines)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "multi_batch.txt"
            path.write_text(text, encoding="utf-8")

            class DeterministicBatchModels:
                def choices(s, *args, **kwargs):
                    raise AssertionError("章节阶段禁止调用 choices")

                def generate(s, system, state, stage, validate, *args, **kwargs):
                    cands = state["candidates"]
                    req_id = digest(["batch-req", [c["id"] for c in cands]])
                    val = {
                        "headings": [
                            {"id": c["id"], "kind": "chapter"}
                            for c in cands
                            if c["text"].startswith("第")
                        ]
                    }
                    validate(val)
                    return val, req_id

            results = {}
            for conc in (1, 8):
                store = Store(root / f"c_{conc}.sqlite3")
                try:
                    source = import_text(store, path)
                    rid = f"run_c{conc}"
                    store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (rid, source["id"], "{}", "{}", "now", "now"))
                    out = detect_chapters(store, DeterministicBatchModels(), source, rid, lambda _: None, concurrency=conc)
                    decs = store.records(rid, "decision")
                    chs = store.records(rid, "chapter")
                    results[conc] = (out["chapter_set_version"], dumps(decs), dumps(chs), out["path_counts"])
                finally:
                    store.close()

            self.assertGreaterEqual(results[1][3]["label_requests"], 2)
            self.assertEqual(results[1][0], results[8][0])
            self.assertEqual(results[1][1], results[8][1])
            self.assertEqual(results[1][2], results[8][2])
            self.assertEqual(results[1][3], results[8][3])


class CredentialDecouplingAndTemperatureTests(unittest.TestCase):
    def test_generate_without_temperature_preserves_exact_payload_and_cache_key(self):
        from entitytrace.util import dumps

        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "db.sqlite3")
            try:
                sent_payloads: list[dict] = []

                def opener(request, **kwargs):
                    sent_payloads.append(json.loads(request.data.decode("utf-8")))
                    body = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")
                model = Models(store, "r", "i", {"discovery": svc}, opener=opener)
                sys_prompt = "测试提示词"
                user_state = {"units": [{"id": "u0_5", "text": "测试"}]}
                expected_legacy_payload = {
                    "model": "deepseek-flash",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": dumps(user_state)},
                    ],
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                    "max_tokens": 8000,
                }
                expected_legacy_id = digest(["http-v1", svc.public(), "/chat/completions", expected_legacy_payload])

                _, req_id_default = model.generate(sys_prompt, user_state, "discovery", lambda v: None)
                self.assertEqual(req_id_default, expected_legacy_id)
                self.assertNotIn("temperature", sent_payloads[0])
                self.assertEqual(dumps(sent_payloads[0]), dumps(expected_legacy_payload))

                _, req_id_temp0 = model.generate(sys_prompt, user_state, "chapter-label", lambda v: None, temperature=0)
                self.assertNotEqual(req_id_temp0, expected_legacy_id)
                self.assertEqual(sent_payloads[1]["temperature"], 0)
            finally:
                store.close()

    def test_roster_loads_without_typesafe_key_and_groups_v2_errors_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "input.txt"
            text.write_text("第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n", encoding="utf-8")
            deepseek_only_cfg = root / "deepseek.env"
            deepseek_only_cfg.write_text("DEEPSEEK_API_KEY=ds_only_secret\n", encoding="utf-8")
            both_keys_cfg = root / "both.env"
            both_keys_cfg.write_text("DEEPSEEK_API_KEY=ds_only_secret\nTYPESAFE_API_KEY=ts_extra_secret\n", encoding="utf-8")

            clean_env = {k: v for k, v in os.environ.items() if not k.startswith("TYPESAFE_") and not k.startswith("DEEPSEEK_")}
            with patch.dict(os.environ, clean_env, clear=True):
                services = load_services(deepseek_only_cfg)
                self.assertEqual(set(services), {"discovery"})

                with self.assertRaises(ValueError) as ctx:
                    load_services(deepseek_only_cfg, require_decision=True)
                self.assertIn("groups-v2", str(ctx.exception))
                self.assertIn("TypeSafe/Jev", str(ctx.exception))

                FakeModels.fail_once = False
                with patch("entitytrace.pipeline.Models", FakeModels):
                    rid_ds_only, out_ds_only, ok1 = run_pipeline(
                        text, root / "db1.sqlite3", root / "out1", deepseek_only_cfg, log=lambda _: None,
                    )
                    rid_with_ts, _, ok2 = run_pipeline(
                        text, root / "db2.sqlite3", root / "out2", both_keys_cfg, log=lambda _: None,
                    )
                self.assertTrue(ok1)
                self.assertTrue(ok2)
                # 默认 roster 流程彻底忽略本机是否配有 TypeSafe，run_id 恒定不变
                self.assertEqual(rid_ds_only, rid_with_ts)
                manifest = json.loads((out_ds_only / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(set(manifest["config"]["models"]), {"discovery"})

                with patch("entitytrace.pipeline.Models", FakeModels):
                    with self.assertRaises(ValueError) as pipe_ctx:
                        run_pipeline(
                            text, root / "db3.sqlite3", root / "out3", deepseek_only_cfg,
                            identity_algorithm="groups-v2", log=lambda _: None,
                        )
                self.assertIn("groups-v2", str(pipe_ctx.exception))
                self.assertIn("TypeSafe/Jev", str(pipe_ctx.exception))


    def test_failed_validation_request_usage_counted_in_metrics(self):
        from entitytrace.pipeline import metrics

        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "db.sqlite3")
            try:
                calls = []

                def opener_fail_then_succeed(request, **kwargs):
                    calls.append(1)
                    if len(calls) == 1:
                        body = {
                            "choices": [{"finish_reason": "length", "message": {"content": "{\"entities\":"}}],
                            "usage": {"prompt_tokens": 1000, "completion_tokens": 8000},
                        }
                    else:
                        body = {
                            "choices": [{"finish_reason": "stop", "message": {"content": "{\"ok\": true}"}}],
                            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
                        }
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")
                with patch("entitytrace.models.time.sleep", lambda _: None):
                    model = Models(store, "run-fail-ok", "inv-1", {"discovery": svc}, attempts=3, opener=opener_fail_then_succeed)
                    out, _ = model.generate("sys", {"x": 1}, "roster-c1-chapter", lambda v: None)
                self.assertEqual(out, {"ok": True})
                m = metrics(store, "run-fail-ok")
                self.assertEqual(m["input_tokens_reported"], 2000)
                self.assertEqual(m["output_tokens_reported"], 8500)
                self.assertEqual(m["network_attempts"], 2)
                self.assertEqual(m["successful_requests"], 1)
                self.assertEqual(m["failed_attempts"], 1)
                failed_rows = store.rows("SELECT detail FROM calls WHERE run_id=? AND status='failed'", ("run-fail-ok",))
                self.assertEqual(len(failed_rows), 1)
                failed_detail = json.loads(failed_rows[0]["detail"])
                self.assertEqual(failed_detail["usage"], {"prompt_tokens": 1000, "completion_tokens": 8000})
                self.assertIn("validation_error", failed_detail)

                # 0 次失败时与历史结果完全一致
                def opener_always_succeed(request, **kwargs):
                    body = {
                        "choices": [{"finish_reason": "stop", "message": {"content": "{\"ok\": 2}"}}],
                        "usage": {"prompt_tokens": 600, "completion_tokens": 300},
                    }
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                model_ok = Models(store, "run-zero-fail", "inv-2", {"discovery": svc}, attempts=3, opener=opener_always_succeed)
                model_ok.generate("sys", {"x": 2}, "roster-c1-chapter", lambda v: None)
                m_ok = metrics(store, "run-zero-fail")
                self.assertEqual(m_ok["input_tokens_reported"], 600)
                self.assertEqual(m_ok["output_tokens_reported"], 300)
                self.assertEqual(m_ok["network_attempts"], 1)
                self.assertEqual(m_ok["successful_requests"], 1)
                self.assertEqual(m_ok["failed_attempts"], 0)
            finally:
                store.close()

    def test_all_stages_default_temperature_zero_and_fallback_on_output_error_only(self):
        from entitytrace.discovery import discover
        from entitytrace.roster import resolve_roster

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = Store(root / "db.sqlite3")
            try:
                svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")

                # (2) 第一次尝试在 temperature=0 下返回截断或格式错误时，第二次尝试自动移除 temperature，
                #     成功后以首次 request_id 落库并在 calls.detail 记录 temperature_sent
                sent_payloads_out_err: list[dict] = []

                def opener_output_err_then_ok(request, **kwargs):
                    p = json.loads(request.data.decode("utf-8"))
                    sent_payloads_out_err.append(p)
                    if len(sent_payloads_out_err) == 1:
                        body = {
                            "choices": [{"finish_reason": "length", "message": {"content": "{\"ok\":"}}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 8000},
                        }
                    else:
                        body = {
                            "choices": [{"finish_reason": "stop", "message": {"content": "{\"ok\": true}"}}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                        }
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                with patch("entitytrace.models.time.sleep", lambda _: None):
                    m1 = Models(store, "run-t0-fallback", "inv-t1", {"discovery": svc}, attempts=3, opener=opener_output_err_then_ok)
                    val1, req_id1 = m1.generate("sys", {"u": 1}, "roster-c1", lambda v: None, temperature=0)

                self.assertEqual(val1, {"ok": True})
                self.assertEqual(len(sent_payloads_out_err), 2)
                self.assertEqual(sent_payloads_out_err[0].get("temperature"), 0)
                self.assertNotIn("temperature", sent_payloads_out_err[1])
                # 验证缓存键仍为首次含 temperature=0 的 request_id，再次调用命中缓存
                self.assertIsNotNone(store.cache(req_id1))
                val1_cached, req_id1_cached = m1.generate("sys", {"u": 1}, "roster-c1", lambda v: None, temperature=0)
                self.assertEqual(val1_cached, {"ok": True})
                self.assertEqual(req_id1_cached, req_id1)
                self.assertEqual(len(sent_payloads_out_err), 2)

                call_rows_1 = store.rows("SELECT attempt, status, detail FROM calls WHERE run_id=? ORDER BY id", ("run-t0-fallback",))
                self.assertEqual(len(call_rows_1), 3)
                d_att1 = json.loads(call_rows_1[0]["detail"])
                d_att2 = json.loads(call_rows_1[1]["detail"])
                self.assertEqual(call_rows_1[0]["status"], "failed")
                self.assertEqual(d_att1["temperature_sent"], 0)
                self.assertEqual(call_rows_1[1]["status"], "success")
                self.assertIsNone(d_att2["temperature_sent"])

                # (3) 第一次尝试遇 HTTP 503 或网络错误时，第二次尝试仍带 temperature=0
                sent_payloads_http_err: list[dict] = []

                def opener_http_503_then_ok(request, **kwargs):
                    p = json.loads(request.data.decode("utf-8"))
                    sent_payloads_http_err.append(p)
                    if len(sent_payloads_http_err) == 1:
                        raise urllib.error.HTTPError(
                            request.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"busy")
                        )
                    body = {
                        "choices": [{"finish_reason": "stop", "message": {"content": "{\"ok\": 503}"}}],
                        "usage": {"prompt_tokens": 80, "completion_tokens": 20},
                    }
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                with patch("entitytrace.models.time.sleep", lambda _: None):
                    m2 = Models(store, "run-t0-http503", "inv-t2", {"discovery": svc}, attempts=3, opener=opener_http_503_then_ok)
                    val2, _ = m2.generate("sys", {"u": 2}, "discovery", lambda v: None, temperature=0)

                self.assertEqual(val2, {"ok": 503})
                self.assertEqual(len(sent_payloads_http_err), 2)
                self.assertEqual(sent_payloads_http_err[0].get("temperature"), 0)
                self.assertEqual(sent_payloads_http_err[1].get("temperature"), 0)
                call_rows_2 = store.rows("SELECT attempt, status, detail FROM calls WHERE run_id=? ORDER BY id", ("run-t0-http503",))
                self.assertEqual(json.loads(call_rows_2[0]["detail"])["temperature_sent"], 0)
                self.assertEqual(json.loads(call_rows_2[1]["detail"])["temperature_sent"], 0)

                # (1) 验证发现与 roster 五个阶段（roster-c1, roster-c2-review, roster-c2-propose, roster-c2-verify, roster-c3-link）
                #     默认发起请求体全部包含 "temperature": 0
                stage_payloads: dict[str, list[dict]] = {}
                book_path = root / "book_t0.txt"
                book_text = (
                    "第一章\n沈墨在城门遇见沈公子与一位老者。\n"
                    "第二章\n沈墨与沈公子、韩立重逢，韩立提起沈墨。\n"
                )
                book_path.write_bytes(book_text.encode("utf-8"))
                source = import_text(store, book_path)
                i1, i2 = book_text.index("第一章"), book_text.index("第二章")
                _, chapters = build_chapters(
                    source,
                    [
                        {"start": i1, "end": i1 + 3, "title": "第一章", "kind": "chapter"},
                        {"start": i2, "end": i2 + 3, "title": "第二章", "kind": "chapter"},
                    ],
                )
                run_s = "run-all-stages-t0"
                store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_s, source["id"], "{}", "{}", "now", "now"))
                for ch in chapters:
                    store.put(run_s, "chapter", ch)

                class StageInspectingModels(Models):
                    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
                        stage_payloads.setdefault(stage, []).append({"temperature": temperature, "max_tokens": max_tokens, "state": state})
                        return super().generate(system, state, stage, validate, max_tokens=max_tokens, temperature=temperature)

                sent_by_http: list[dict] = []

                def opener_all_stages(request, **kwargs):
                    p = json.loads(request.data.decode("utf-8"))
                    sent_by_http.append(p)
                    user_obj = json.loads(p["messages"][1]["content"])
                    if "section_kind" in user_obj:
                        uid = user_obj["units"][0]["id"]
                        if user_obj["chapter_title"] == "第一章":
                            content = {
                                "mentions": [
                                    {"text": "沈墨", "kind": "person", "refs": [uid]},
                                    {"text": "沈公子", "kind": "person", "refs": [uid]},
                                    {"text": "老者", "kind": "person", "refs": [uid]},
                                ],
                                "links": [],
                            }
                        else:
                            content = {
                                "mentions": [
                                    {"text": "沈墨", "kind": "person", "refs": [uid]},
                                    {"text": "沈公子", "kind": "person", "refs": [uid]},
                                    {"text": "韩立", "kind": "person", "refs": [uid]},
                                ],
                                "links": [],
                            }
                    elif "surfaces" in user_obj:
                        uid = user_obj["units"][0]["id"]
                        if user_obj["chapter_title"] == "第一章":
                            content = {
                                "entities": [
                                    {
                                        "local_id": "e1",
                                        "name": "沈墨",
                                        "kind": "person",
                                        "surfaces": [
                                            {"surface": "沈墨", "surface_type": "proper_name"},
                                            {"surface": "沈公子", "surface_type": "proper_name"},
                                        ],
                                        "evidence_ref": uid,
                                    },
                                    {
                                        "local_id": "e2",
                                        "name": "老者",
                                        "kind": "person",
                                        "surfaces": [{"surface": "老者", "surface_type": "description"}],
                                        "evidence_ref": uid,
                                    },
                                ],
                                "assignments": [],
                                "not_entity": [],
                            }
                        else:
                            content = {
                                "entities": [
                                    {
                                        "local_id": "e1",
                                        "name": "沈公子",
                                        "kind": "person",
                                        "surfaces": [
                                            {"surface": "沈墨", "surface_type": "proper_name"},
                                            {"surface": "沈公子", "surface_type": "proper_name"},
                                        ],
                                        "evidence_ref": uid,
                                    },
                                    {
                                        "local_id": "e2",
                                        "name": "韩立",
                                        "kind": "person",
                                        "surfaces": [{"surface": "韩立", "surface_type": "proper_name"}],
                                        "evidence_ref": uid,
                                    },
                                ],
                                "assignments": [],
                                "not_entity": [],
                            }
                    elif "rows" in user_obj:
                        rids = [r["row_id"] for r in user_obj["rows"]]
                        content = {
                            "groups": [{"group_id": "g1", "name": "沈墨", "kind": "person", "row_ids": rids, "member_ids": [], "reason": "同指"}],
                            "unresolved_row_ids": [],
                            "unresolved_member_ids": [],
                        }
                    elif "entries" in user_obj:
                        eids = [e["entry_id"] for e in user_obj["entries"]]
                        if len(eids) >= 2:
                            content = {"proposals": [{"left": eids[0], "right": eids[1], "reason": "尝试提议"}]}
                        else:
                            content = {"proposals": []}
                    elif "proposal_reason" in user_obj:
                        content = {"verdict": "different", "reason": "不同人"}
                    elif "items" in user_obj:
                        content = {
                            "links": [
                                {"mention_id": it["mention_id"], "selected_entity_id": "none", "reason": "无同指"}
                                for it in user_obj["items"]
                            ]
                        }
                    else:
                        content = {}
                    body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content, ensure_ascii=False)}}]}
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                m_all = StageInspectingModels(store, run_s, "inv-all", {"discovery": svc}, discovery_concurrency=1, opener=opener_all_stages)
                d_out = discover(store, m_all, source, run_s, chapters, 9000, 1, lambda *_: None)
                self.assertTrue(d_out["complete"])
                r_out = resolve_roster(store, m_all, source, run_s, concurrency=1, log=lambda *_: None, task_chars=9000)
                self.assertTrue(r_out["complete"])

                expected_stages = {"discovery", "roster-c1", "roster-c2-review", "roster-c2-propose", "roster-c2-verify", "roster-c3-link"}
                self.assertEqual(set(stage_payloads.keys()), expected_stages)
                for st_name, recs in stage_payloads.items():
                    self.assertTrue(len(recs) >= 1, st_name)
                    for r in recs:
                        self.assertEqual(r["temperature"], 0, f"{st_name} temperature != 0")
                for hp in sent_by_http:
                    self.assertEqual(hp.get("temperature"), 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
