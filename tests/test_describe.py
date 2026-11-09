from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.cli import main as cli_main
from entitytrace.discovery import plan_tasks
from entitytrace.describe import (
    D2_MAX_REQUESTS_PER_ENTITY,
    FACETS,
    MAX_QUOTE_CHARS,
    build_d1_tasks,
    build_describe_id,
    build_d2_tasks,
    build_windows,
    check_output,
    ground_output,
    locate_quote,
    pack_windows,
    run_describe,
)
from entitytrace.models import ModelError
from entitytrace.roster import VERSION as ROSTER_VERSION
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import dumps

TEXT = (
    "第一章 山门\n"
    "沈墨身长八尺，面如冠玉，穿着一领青衫。\n"
    "沈墨手中铁剑横在膝上，山门隐在云雾里。\n"
    "第二章 夜行\n"
    "沈墨换了白衣，铁剑仍在。\n"
    "山门大开，沈墨运起御剑术。\n"
)

SURFACES = {
    "e_shen": ("沈墨", "person"),
    "e_jian": ("铁剑", "object"),
    "e_men": ("山门", "place"),
    "e_skill": ("御剑术", "ability"),
}


class _Discovery:
    provider = "example"
    model = "example/model"
    extra_body = {}


SERVICES = {"discovery": _Discovery()}


class FakeModels:
    """记录调用并按输入片段返回预置描述事实，零网络请求。"""

    def __init__(self, services, handler):
        self.services = services
        self.handler = handler
        self.calls: list[dict] = []

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        self.calls.append({"stage": stage, "state": state, "temperature": temperature})
        value = self.handler(stage, state, len(self.calls))
        validate(value)
        return value, f"req_{len(self.calls)}"


def _span(text: str, title: str, kind: str) -> dict:
    start = text.index(title)
    return {"start": start, "end": start + len(title), "title": title, "kind": kind}


def _find_all(text: str, surface: str) -> list[int]:
    found, cursor = [], 0
    while True:
        at = text.find(surface, cursor)
        if at < 0:
            return found
        found.append(at)
        cursor = at + 1


def _make_task(
    text: str,
    start: int,
    end: int,
    entity_kinds: dict[str, str],
    mentions_by_entity: dict[str, list[dict]] | None = None,
    default_entity_id: str | None = None,
) -> dict:
    return {
        "id": "task-1",
        "stage": "describe-d1" if default_entity_id is None else "describe-d2",
        "entity_id": default_entity_id,
        "start": start,
        "end": end,
        "segments": [{"start": start, "end": end}],
        "entity_kinds": entity_kinds,
        "mentions_by_entity": mentions_by_entity or {},
        "default_entity_id": default_entity_id,
        "text": text,
    }


def _chapters_for(text: str) -> list[dict]:
    return [{"id": "ch1", "start": 0, "end": len(text), "body_start": 0}]


class QuoteVerificationTests(unittest.TestCase):
    def test_unique_occurrence_and_address(self):
        text = "沈墨身长八尺，面如冠玉。"
        task = _make_task(text, 0, len(text), {"e1": "person"},
                          {"e1": [{"start": 0, "end": 2, "text": "沈墨"}]})
        result = ground_output("did", task, {"facts": [{
            "entity_id": "e1", "facet": "build", "value": "高大", "quote": "身长八尺",
            "source": "narration",
        }], "events": []}, _chapters_for(text), {"e1"})
        self.assertEqual(len(result["facts"]), 1)
        fact = result["facts"][0]
        self.assertEqual((fact["start"], fact["end"]), (2, 6))
        self.assertEqual(text[fact["start"]:fact["end"]], "身长八尺")
        self.assertEqual(fact["chapter_id"], "ch1")

    def test_multiple_occurrences_pick_nearest_mention(self):
        text = "甲乙丙，丁戊己，甲乙丙，庚辛壬。"
        segments = [{"start": 0, "end": len(text)}]
        first = locate_quote(text, segments, "甲乙丙", [{"start": 0, "end": 3}])
        second = locate_quote(text, segments, "甲乙丙", [{"start": 9, "end": 12}])
        self.assertEqual(first, (0, 3))
        self.assertEqual(second, (8, 11))
        # 无提及坐标时退化为最靠前的一处，结果仍确定。
        self.assertEqual(locate_quote(text, segments, "甲乙丙", []), (0, 3))

    def test_not_found_becomes_review(self):
        text = "沈墨身长八尺。"
        task = _make_task(text, 0, len(text), {"e1": "person"}, {"e1": []})
        result = ground_output("did", task, {"facts": [{
            "entity_id": "e1", "facet": "build", "value": "高大", "quote": "并不存在",
            "source": "narration",
        }], "events": []}, _chapters_for(text), {"e1"})
        self.assertEqual(result["facts"], [])
        self.assertEqual([r["reason"] for r in result["reviews"]], ["quote-not-found"])

    def test_quote_too_long_becomes_review(self):
        quote = "长" * (MAX_QUOTE_CHARS + 1)
        text = "沈墨" + quote + "。"
        task = _make_task(text, 0, len(text), {"e1": "person"}, {"e1": []})
        result = ground_output("did", task, {"facts": [{
            "entity_id": "e1", "facet": "build", "value": "很长", "quote": quote,
            "source": "narration",
        }], "events": []}, _chapters_for(text), {"e1"})
        self.assertEqual(result["facts"], [])
        self.assertEqual([r["reason"] for r in result["reviews"]], ["quote-too-long"])

    def test_quote_must_stay_inside_request_segments(self):
        text = "沈墨身长八尺。远处另有一段。"
        task = _make_task(text, 0, 7, {"e1": "person"}, {"e1": []})
        result = ground_output("did", task, {"facts": [{
            "entity_id": "e1", "facet": "build", "value": "错误", "quote": "远处",
            "source": "narration",
        }], "events": []}, _chapters_for(text), {"e1"})
        self.assertEqual([r["reason"] for r in result["reviews"]], ["quote-not-found"])


class FieldReviewTests(unittest.TestCase):
    def setUp(self):
        self.text = "沈墨身长八尺，面如冠玉。"
        self.task = _make_task(self.text, 0, len(self.text), {"e1": "person"},
                               {"e1": [{"start": 0, "end": 2, "text": "沈墨"}]})
        self.chapters = _chapters_for(self.text)

    def test_unknown_entity_and_bad_facet(self):
        value = {"facts": [
            {"entity_id": "e_missing", "facet": "age", "value": "二十", "quote": "沈墨",
             "source": "narration"},
            {"entity_id": "e1", "facet": "scale", "value": "宏大", "quote": "沈墨",
             "source": "narration"},
        ], "events": []}
        result = ground_output("did", self.task, value, self.chapters, {"e1"})
        self.assertEqual([r["reason"] for r in result["reviews"]], ["unknown-entity", "bad-facet"])

    def test_bad_enum_and_quote_reviews(self):
        value = {"facts": [
            {"entity_id": "e1", "facet": "build", "value": "高大", "quote": "身长八尺",
             "source": "narration2"},
            {"entity_id": "e1", "facet": "build", "value": "", "quote": "身长八尺",
             "source": "narration"},
            {"entity_id": "e1", "facet": "build", "value": "高大", "source": "narration"},
        ], "events": [
            {"entity_id": "e1", "event_type": "fly", "splits_entity": False, "value": "飞",
             "quote": "身长八尺"},
            {"entity_id": "e1", "event_type": "injury", "splits_entity": "false", "value": "受伤",
             "quote": "身长八尺"},
        ]}
        result = ground_output("did", self.task, value, self.chapters, {"e1"})
        self.assertEqual(
            [r["reason"] for r in result["reviews"]],
            ["bad-enum", "bad-enum", "quote-not-found", "bad-enum", "bad-enum"],
        )

    def test_event_and_holder_persist(self):
        value = {"facts": [{
            "entity_id": "e1", "facet": "costume", "value": "青衫", "quote": "身长八尺",
            "source": "poem", "holder_entity_id": "e_unknown",
        }], "events": [{
            "entity_id": "e1", "event_type": "costume_change", "splits_entity": True,
            "value": "换衣", "quote": "面如冠玉",
        }]}
        result = ground_output("did", self.task, value, self.chapters, {"e1"})
        self.assertEqual(len(result["facts"]), 1)
        self.assertIsNone(result["facts"][0]["holder_entity_id"])
        self.assertEqual(len(result["events"]), 1)
        self.assertTrue(result["events"][0]["splits_entity"])

    def test_d2_defaults_missing_entity_id_to_current_entity(self):
        task = _make_task(self.text, 0, len(self.text), {"e1": "person"},
                          {"e1": [{"start": 0, "end": 2, "text": "沈墨"}]}, default_entity_id="e1")
        value = {"facts": [{
            "facet": "hair", "value": "黑发", "quote": "身长八尺", "source": "narration",
        }], "events": []}
        result = ground_output("did", task, value, self.chapters, {"e1"})
        self.assertEqual(result["facts"][0]["entity_id"], "e1")

    def test_structural_invalid_output(self):
        with self.assertRaises(ModelError):
            check_output({"facts": []})
        with self.assertRaises(ModelError):
            check_output({"facts": ["bad"], "events": []})


class D1TaskTests(unittest.TestCase):
    def test_skip_chunks_without_top_entities(self):
        text = "第一章\n" + ("无人。\n" * 80) + ("沈墨现身。\n" * 6)
        source = {"id": "s1", "text": text}
        chapters = [{"id": "ch1", "chapter_set_version": "v1", "title": "第一章",
                     "kind": "chapter", "start": 0, "body_start": 4, "end": len(text)}]
        shen_start = text.index("沈墨现身")
        top = [{"entity_id": "e1", "kind": "person", "kind_rank": 1, "name": "沈墨"}]
        mentions = {"e1": [{"start": shen_start, "end": shen_start + 2, "text": "沈墨",
                            "chapter_id": "ch1"}]}
        tasks = build_d1_tasks(source, chapters, top, mentions, "did", task_chars=50)
        self.assertGreaterEqual(len(tasks), 1)
        self.assertLess(len(tasks), len(plan_tasks(source, chapters, 50)))
        self.assertTrue(all(task["state"]["entities"] for task in tasks))
        target = [task for task in tasks if task["start"] <= shen_start < task["end"]]
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0]["state"]["entities"], [{
            "entity_id": "e1", "kind": "person", "surfaces_in_chunk": ["沈墨"],
        }])
        self.assertEqual(target[0]["entity_kinds"], {"e1": "person"})


class D2WindowTests(unittest.TestCase):
    def test_window_merge_and_boundaries(self):
        text = "甲" * 2000
        chapters = [
            {"id": "c1", "start": 0, "body_start": 10, "end": 1000},
            {"id": "c2", "start": 1000, "body_start": 1010, "end": 2000},
        ]
        mentions = [
            {"start": 100, "end": 102, "text": "甲甲", "chapter_id": "c1"},
            {"start": 180, "end": 182, "text": "甲甲", "chapter_id": "c1"},
            {"start": 1500, "end": 1502, "text": "甲甲", "chapter_id": "c2"},
        ]
        windows = build_windows(text, chapters, mentions, window=200)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["start"], 10)
        self.assertEqual(windows[0]["end"], 382)
        self.assertEqual(windows[1]["start"], 1300)
        self.assertEqual(windows[1]["end"], 1702)

    def test_pack_respects_character_budget(self):
        text = "乙" * 5000
        windows = [
            {"start": 0, "end": 300, "chapter_id": "c1"},
            {"start": 300, "end": 600, "chapter_id": "c1"},
            {"start": 600, "end": 900, "chapter_id": "c1"},
        ]
        packages = pack_windows(text, windows, pack_chars=700)
        self.assertEqual(len(packages), 2)
        for package in packages:
            size = sum(len(f"--- 片段 start={w['start']} end={w['end']} ---\n")
                       + (w["end"] - w["start"]) for w in package)
            self.assertLessEqual(size, 700)

    def test_truncation_is_deterministic_and_layered(self):
        text = "丙" * 20000
        chapters = [
            {"id": f"c{i}", "start": i * 2000, "body_start": i * 2000, "end": (i + 1) * 2000}
            for i in range(10)
        ]
        mentions = []
        for i in range(10):
            for j in range(4):
                start = i * 2000 + 100 + j * 500
                mentions.append({"start": start, "end": start + 2, "text": "丙丙",
                                 "chapter_id": f"c{i}"})
        top = [{"entity_id": "e1", "kind": "person", "kind_rank": 1, "name": "丙"}]
        tasks_a, truncated_a = build_d2_tasks(
            {"id": "s1", "text": text}, chapters, top, {"e1": mentions}, "did", pack_chars=50,
        )
        tasks_b, truncated_b = build_d2_tasks(
            {"id": "s1", "text": text}, chapters, top, {"e1": mentions}, "did", pack_chars=50,
        )
        self.assertEqual(truncated_a, truncated_b)
        self.assertEqual(len(truncated_a), 1)
        item = truncated_a[0]
        self.assertEqual(item["entity_id"], "e1")
        self.assertEqual(item["windows_total"], 40)
        self.assertEqual(item["windows_used"], D2_MAX_REQUESTS_PER_ENTITY)
        self.assertEqual(len(tasks_a), D2_MAX_REQUESTS_PER_ENTITY)
        self.assertEqual(
            [[(s["start"], s["end"]) for s in task["segments"]] for task in tasks_a],
            [[(s["start"], s["end"]) for s in task["segments"]] for task in tasks_b],
        )
        self.assertTrue(all(len(task["segments"]) == 1 for task in tasks_a))

    def test_pack_boundaries_merge_adjacent_windows(self):
        text = "丁" * 3000
        chapters = [{"id": "c1", "start": 0, "body_start": 0, "end": 3000}]
        mentions = [
            {"start": 100, "end": 102, "text": "丁丁", "chapter_id": "c1"},
            {"start": 302, "end": 304, "text": "丁丁", "chapter_id": "c1"},
        ]
        # 两个窗口 [0,302) 与 [102,504) 重叠；相邻边界同样合并。
        windows = build_windows(text, chapters, mentions, window=200)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], {"start": 0, "end": 504, "chapter_id": "c1"})


class FixtureCase(unittest.TestCase):
    TEXT = TEXT

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "describe.sqlite3"
        self.out_path = self.root / "describe.json"
        self.run_id = "run_describe_fixture_0001"
        self._build_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def _build_fixture(self):
        book = self.root / "book.txt"
        book.write_bytes(self.TEXT.encode("utf-8"))
        store = Store(self.db_path)
        try:
            source = import_text(store, book)
            _, chapters = build_chapters(source, [
                _span(self.TEXT, "第一章 山门", "chapter"),
                _span(self.TEXT, "第二章 夜行", "chapter"),
            ])
            config = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {"complete": True, "chapter_set_version": chapters[0]["chapter_set_version"]},
                "discovery": {"complete": True},
            }
            store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",
                          (self.run_id, source["id"], dumps(config), dumps(state), "now", "now"))
            for chapter in chapters:
                store.put(self.run_id, "chapter", chapter)

            entities, rows = [], []
            for entity_id, (surface, kind) in SURFACES.items():
                positions = _find_all(self.TEXT, surface)
                if surface == "山门":
                    positions = [p for p in positions if p >= chapters[0]["body_start"]]
                mentions = []
                chapter_stats: dict[str, dict] = {}
                surface_counts: dict[str, int] = {}
                for pos in positions:
                    chapter = next(ch for ch in chapters if ch["start"] <= pos < ch["end"])
                    mid = f"m_{entity_id}_{pos}"
                    mentions.append({
                        "id": mid, "chapter_id": chapter["id"], "start": pos, "end": pos + len(surface),
                        "text": surface, "link_method": "chapter_local",
                    })
                    stat = chapter_stats.setdefault(chapter["id"], {"count": 0, "min_start": pos})
                    stat["count"] += 1
                    stat["min_start"] = min(stat["min_start"], pos)
                    surface_counts[surface] = surface_counts.get(surface, 0) + 1
                entities.append({
                    "id": entity_id, "name": surface, "kind": kind,
                    "mention_count": len(mentions),
                    "first_chapter_id": mentions[0]["chapter_id"] if mentions else None,
                    "identity_status": "linked_by_roster",
                    "surface_counts": surface_counts,
                    "chapter_stats": chapter_stats,
                })
                header = {
                    "id": entity_id, "name": surface, "names": [], "aliases": [],
                    "kind": kind, "identity_status": "linked_by_roster",
                    "mention_count": len(mentions),
                    "first_chapter_id": mentions[0]["chapter_id"] if mentions else None,
                    "link_methods": {"chapter_local": len(mentions)},
                }
                rows.append((entity_id, json.dumps(header, ensure_ascii=False),
                             json.dumps(mentions, ensure_ascii=False)))
            meta = {
                "result_revision": "rev-1", "status": "completed", "source_id": source["id"],
                "chapters": chapters, "counts": {}, "review_counts": {},
                "snapshot_version": 2,
            }
            store.save_query_snapshot(
                self.run_id, store.query_fingerprint(self.run_id), meta, entities, [], rows,
            )
        finally:
            store.close()

    def _handler(self, stage, state, call_index):
        text = state.get("text") or "".join(seg["text"] for seg in state.get("segments", []))
        facts = []
        if "身长八尺" in text:
            facts.append({
                "entity_id": "e_shen", "facet": "build", "value": "身长八尺",
                "quote": "身长八尺", "source": "narration",
            })
        if "面如冠玉" in text:
            facts.append({
                "entity_id": "e_shen", "facet": "face", "value": "面如冠玉",
                "quote": "面如冠玉", "source": "narration",
            })
        return {"facts": facts, "events": []}


class RunDescribeTests(FixtureCase):
    def test_stable_id_and_resume(self):
        models = FakeModels(SERVICES, self._handler)
        store = Store(self.db_path)
        try:
            first = run_describe(store, self.run_id, "d1", services=SERVICES, models=models,
                                 top_k=10, kinds="person", concurrency=2, out=self.out_path)
            calls_first = len(models.calls)
            self.assertGreater(calls_first, 0)
            self.assertEqual(first["counts"]["facts"], 2)

            second = run_describe(store, self.run_id, "d1", services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=2, out=self.out_path)
            self.assertEqual(len(models.calls), calls_first)
            self.assertEqual(second["describe_id"], first["describe_id"])
            self.assertEqual(second["counts"], first["counts"])
            self.assertEqual(second["new_tasks"], 0)

            d2 = run_describe(store, self.run_id, "d2", services=SERVICES, models=models,
                              top_k=10, kinds="person", concurrency=2)
            self.assertNotEqual(d2["describe_id"], first["describe_id"])
            other_k = run_describe(store, self.run_id, "d1", services=SERVICES, models=models,
                                   top_k=1, kinds="person", concurrency=2)
            self.assertNotEqual(other_k["describe_id"], first["describe_id"])
        finally:
            store.close()

    def test_describe_id_sensitivity(self):
        base = build_describe_id("d1", 20, ["person"], ["e1"], "run", "example", "example/model")
        self.assertEqual(base, build_describe_id("d1", 20, ["person"], ["e1"], "run",
                                                 "example", "example/model"))
        self.assertNotEqual(base, build_describe_id("d2", 20, ["person"], ["e1"], "run",
                                                    "example", "example/model"))
        self.assertNotEqual(base, build_describe_id("d1", 10, ["person"], ["e1"], "run",
                                                    "example", "example/model"))
        self.assertNotEqual(base, build_describe_id("d1", 20, ["person"], ["e2"], "run",
                                                    "example", "example/model"))

    def test_gaps_are_deterministic_zero_request(self):
        models = FakeModels(SERVICES, self._handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, "d1", services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=1, out=self.out_path)
            entity = result["document"]["entities"][0]
            covered = {fact["facet"] for fact in entity["facts"]}
            self.assertEqual(covered, {"build", "face"})
            self.assertEqual(entity["gaps"],
                             [f for f in FACETS["person"] if f not in covered])
            gaps_before = list(entity["gaps"])
            calls_before = len(models.calls)
            again = run_describe(store, self.run_id, "d1", services=SERVICES, models=models,
                                 top_k=10, kinds="person", concurrency=1, out=self.out_path)
            self.assertEqual(len(models.calls), calls_before)
            self.assertEqual(again["document"]["entities"][0]["gaps"], gaps_before)
        finally:
            store.close()

    def test_export_document_shape_and_stats(self):
        models = FakeModels(SERVICES, self._handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, "d2", services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=2, out=self.out_path)
            self.assertTrue(self.out_path.is_file())
            doc = json.loads(self.out_path.read_text(encoding="utf-8"))
            self.assertEqual(set(doc), {"describe_id", "run_id", "method", "top_k", "kinds",
                                        "entities", "reviews", "stats"})
            self.assertEqual(doc["method"], "d2")
            self.assertEqual(doc["top_k"], 10)
            self.assertEqual(set(doc["stats"]), {"requests", "prompt_tokens", "completion_tokens",
                                                 "wall_seconds", "truncated_entities"})
            entity = doc["entities"][0]
            self.assertEqual(set(entity), {"entity_id", "kind", "kind_rank", "name", "facts",
                                           "state_events", "gaps"})
            self.assertGreater(result["counts"]["tasks"], 0)
        finally:
            store.close()

    def test_invalid_output_turns_into_review_not_failure(self):
        def handler(stage, state, call_index):
            raise ModelError("生成返回不是有效 JSON")

        models = FakeModels(SERVICES, handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, "d1", services=SERVICES, models=models,
                                  top_k=10, kinds="person", concurrency=1)
            self.assertGreaterEqual(result["counts"]["reviews"], 1)
            self.assertEqual(result["counts"]["facts"], 0)
            store_rows = store.rows(
                "SELECT reason, status FROM describe_reviews r JOIN describe_tasks t "
                "ON r.describe_id=t.describe_id AND r.task_id=t.id WHERE r.describe_id=?",
                (result["describe_id"],),
            )
            self.assertTrue(store_rows)
            self.assertTrue(all(row["reason"] == "invalid-output" and row["status"] == "completed"
                                for row in store_rows))
        finally:
            store.close()

    def test_topk_file_filters_and_validates(self):
        topk_path = self.root / "topk.json"
        topk_path.write_text(json.dumps({
            "entities": [
                {"entity_id": "e_shen", "kind": "person", "kind_rank": 1},
                {"entity_id": "e_jian", "kind": "object", "kind_rank": 1},
                {"entity_id": "e_men", "kind": "place", "kind_rank": 2},
            ]
        }, ensure_ascii=False), encoding="utf-8")
        models = FakeModels(SERVICES, self._handler)
        store = Store(self.db_path)
        try:
            result = run_describe(store, self.run_id, "d2", services=SERVICES, models=models,
                                  top_k=1, kinds="person,object", topk_file=topk_path,
                                  concurrency=1)
            ids = [entity["entity_id"] for entity in result["document"]["entities"]]
            self.assertEqual(ids, ["e_shen", "e_jian"])
            bad = self.root / "bad_topk.json"
            bad.write_text(json.dumps({"entities": [
                {"entity_id": "e_not_exist", "kind": "person", "kind_rank": 1},
            ]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_describe(store, self.run_id, "d2", services=SERVICES, models=models,
                             top_k=1, kinds="person", topk_file=bad, concurrency=1)
        finally:
            store.close()

    def test_legacy_tables_are_created_without_touching_existing_rows(self):
        store = Store(self.db_path)
        try:
            for table in ("describe_reviews", "state_events", "facts", "describe_tasks", "describe_runs"):
                store.execute(f"DROP TABLE {table}")
            store.execute("INSERT INTO sources VALUES(?,?,?,?)",
                          ("legacy_source", dumps({"id": "legacy_source"}), "旧原文", b"old"))
            store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",
                          ("legacy_run", "legacy_source", "{}", "{}", "t0", "t0"))
        finally:
            store.close()
        reopened = Store(self.db_path)
        try:
            tables = {row["name"] for row in reopened.rows(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"describe_runs", "describe_tasks", "facts", "state_events",
                             "describe_reviews"} <= tables)
            self.assertEqual(reopened.rows("SELECT id FROM runs WHERE id=?", ("legacy_run",))[0]["id"],
                             "legacy_run")
            self.assertEqual(reopened.rows("SELECT text FROM sources WHERE id=?",
                                           ("legacy_source",))[0]["text"], "旧原文")
        finally:
            reopened.close()


class CliContractTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.config_path = self.root / "config.env"
        self.config_path.write_text("DEEPSEEK_API_KEY=test_key\n", encoding="utf-8")

    def test_describe_cli_single_line_json(self):
        class CliFakeModels:
            calls: list[dict] = []

            def __init__(self, store, run_id, invocation, services, *args, **kwargs):
                self.services = services

            def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
                type(self).calls.append({"stage": stage})
                value = {"facts": [], "events": []}
                if "身长八尺" in state.get("text", ""):
                    value["facts"].append({
                        "entity_id": "e_shen", "facet": "build", "value": "高大",
                        "quote": "身长八尺", "source": "narration",
                    })
                validate(value)
                return value, "req_cli"

        out_buf, err_buf = io.StringIO(), io.StringIO()
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith("DEEPSEEK_") and not k.startswith("TYPESAFE_")
                     and not k.startswith("ENTITYTRACE_") and k != "GENERATOR_PROVIDER"}
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("entitytrace.describe.Models", CliFakeModels), \
             contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = cli_main(["describe", self.run_id, "--db", str(self.db_path), "--method", "d1",
                           "--top-k", "10", "--kinds", "person", "--concurrency", "1",
                           "--config", str(self.config_path), "--out", str(self.out_path)])
        self.assertEqual(rc, 0, err_buf.getvalue())
        lines = out_buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1, out_buf.getvalue())
        receipt = json.loads(lines[0])
        self.assertEqual(set(receipt), {"ok", "describe_id", "run_id", "method", "counts", "out"})
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["method"], "d1")
        self.assertEqual(receipt["run_id"], self.run_id)
        self.assertEqual(set(receipt["counts"]), {"tasks", "facts", "events", "reviews"})
        self.assertEqual(receipt["counts"]["facts"], 1)
        self.assertTrue(Path(receipt["out"]).is_file())
        self.assertEqual(len(CliFakeModels.calls), receipt["counts"]["tasks"])


if __name__ == "__main__":
    unittest.main()
