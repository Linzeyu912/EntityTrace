from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.config import Service
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
                unit = next(u for u in state["units"] if u["id"] == key)
                choice = "chapter" if unit["text"].startswith("第") else "body"
            answers[key] = {"type": "choice", "choice": choice, "confidence": 1,
                            "probabilities": {k: int(k == choice) for k in question["criteria"]}}
        return answers, digest([stage, state, questions])

    def generate(self, system, state, stage, validate, *args, **kwargs):
        type(self).generated += 1
        if type(self).fail_once:
            type(self).fail_once = False
            raise ModelError("模拟单任务失败")
        if stage == "roster-c1":
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
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = root / "input.txt"
            text.write_text("第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n", encoding="utf-8")
            config = root / ".env"
            config.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            db_path = root / "db.sqlite3"
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


if __name__ == "__main__":
    unittest.main()
