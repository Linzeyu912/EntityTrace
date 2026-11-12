from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.pipeline import build_export_bundle
from entitytrace.propagation import propagate_same_task_proper
from entitytrace.roster import (
    VERSION as ROSTER_VERSION,
    _sanitize_c1_result_dirty_surfaces,
    apply_c1_task_result,
    resolve_surface_direct,
)
from entitytrace.storage import Store
from entitytrace.text import (
    detect_noise_spans,
    dirty_surface_reasons,
    is_dirty_surface,
    line_segments,
    units,
)
from entitytrace.util import digest, dumps


def legacy_units(text: str, start: int = 0, end: int | None = None, limit: int = 500) -> list[dict]:
    """改造前逐位硬切实现的本地副本，仅用于对照旧行为。"""
    end = len(text) if end is None else end
    output = []
    cursor = start
    for line in text[start:end].splitlines(keepends=True):
        for offset in range(0, len(line), limit):
            a, b = cursor + offset, min(cursor + offset + limit, cursor + len(line))
            if text[a:b].strip():
                output.append({"id": f"u{a}_{b}", "start": a, "end": b, "text": text[a:b],
                               "complete_line": offset == 0 and b == cursor + len(line)})
        cursor += len(line)
    return output


def coverage_ok(line: str, segments: list[tuple[int, int]]) -> bool:
    if not segments:
        return False
    if segments[0][0] != 0 or segments[-1][1] != len(line):
        return False
    if any(segments[i][1] != segments[i + 1][0] for i in range(len(segments) - 1)):
        return False
    return "".join(line[a:b] for a, b in segments) == line


def make_chapters(text: str, headings: list[str]) -> list[dict]:
    chapters = []
    cursor = 0
    for index, heading in enumerate(headings):
        start = text.index(heading, cursor)
        end = text.index(headings[index + 1], start) if index + 1 < len(headings) else len(text)
        chapters.append({"id": f"ch{index + 1}", "start": start, "end": end,
                         "body_start": start + len(heading), "title": heading, "kind": "chapter",
                         "chapter_set_version": "csv-test"})
        cursor = start + len(heading)
    return chapters


class LineSegmentationTests(unittest.TestCase):
    def test_short_line_matches_legacy_units_exactly(self):
        text = "第一行正文。\n第二行略长一些的正文！\n"
        self.assertEqual(units(text, limit=500), legacy_units(text, limit=500))
        for line in text.splitlines(keepends=True):
            self.assertEqual(line_segments(line, 500), [(0, len(line))])

    def test_empty_line_and_invalid_limit(self):
        self.assertEqual(line_segments("", 500), [(0, 0)])
        with self.assertRaises(ValueError):
            line_segments("abc", 0)
        with self.assertRaises(ValueError):
            line_segments("abc", -5)

    def test_chinese_long_line_prefers_sentence_end(self):
        line = "他走进屋里。" * 200
        segments = line_segments(line, 100)
        self.assertTrue(coverage_ok(line, segments))
        self.assertTrue(all(len(line[a:b]) <= 100 for a, b in segments))
        for a, b in segments[:-1]:
            self.assertEqual(line[b - 1], "。")

    def test_closing_quote_is_included_in_cut(self):
        line = "甲乙，" * 30 + "结束。”" + "后续正文，" * 100
        segments = line_segments(line, 120)
        self.assertTrue(coverage_ok(line, segments))
        self.assertTrue(any(line[b - 1] == "”" for _a, b in segments[:-1]))

    def test_chinese_secondary_comma_when_no_sentence_end(self):
        line = "甲乙丙丁，" * 120
        segments = line_segments(line, 100)
        self.assertTrue(coverage_ok(line, segments))
        for a, b in segments[:-1]:
            self.assertIn(line[b - 1], "，、")

    def test_english_abbreviation_dot_is_not_a_boundary(self):
        line = ("Mr. Bingley and Mrs. Bennet walked into the room. " * 6) + "St. James Park was quiet."
        segments = line_segments(line, 80)
        self.assertTrue(coverage_ok(line, segments))
        for a, b in segments:
            chunk = line[a:b].rstrip()
            self.assertFalse(chunk.endswith("Mr."))
            self.assertFalse(chunk.endswith("Mrs."))
            self.assertFalse(chunk.endswith("St."))

    def test_english_proper_noun_space_is_not_split(self):
        line = "Elizabeth Bennet said nothing at all. " * 40
        segments = line_segments(line, 100)
        self.assertTrue(coverage_ok(line, segments))
        for a, b in segments[:-1]:
            self.assertFalse(line[a:b].rstrip().endswith("Elizabeth"))

    def test_extreme_line_without_punctuation_falls_back_to_limit(self):
        line = "无" * 1300
        segments = line_segments(line, 500)
        self.assertTrue(coverage_ok(line, segments))
        self.assertEqual([b - a for a, b in segments], [500, 500, 300])

    def test_units_offsets_are_code_point_exact(self):
        text = ("长句一。" * 90) + "\n" + ("Long sentence here. " * 40) + "\n"
        for unit in units(text, limit=120):
            self.assertEqual(unit["id"], f"u{unit['start']}_{unit['end']}")
            self.assertEqual(text[unit["start"]:unit["end"]], unit["text"])


class NoiseGuardTests(unittest.TestCase):
    def test_url_and_bare_domain_lines_are_masked(self):
        text = "正文一行。\nhttps://www.example-novel.com/book/1.html\nwww.pirate-novel.cn 手打更新\n"
        spans = detect_noise_spans(text)
        masked = [text[a:b] for a, b in spans]
        self.assertEqual(len(masked), 2)
        self.assertTrue(all("正文一行" not in item for item in masked))

    def test_promo_lines_are_masked(self):
        text = "求月票！求推荐票！感谢各位书友！\n感谢盟主“青山不改”的十万打赏，加更一章！\n书友群：123456789，公众号：某某读书\n"
        spans = detect_noise_spans(text)
        self.assertEqual(len(spans), 3)

    def test_template_and_navigation_lines_are_masked(self):
        text = "本书由某某小说网整理，转载请注明出处。\n更多精彩小说请访问某某读书网\n上一章\n12/36\n目录\n"
        spans = detect_noise_spans(text)
        self.assertEqual(len(spans), 5)

    def test_author_note_block_and_unclosed_note(self):
        text = (
            "正文段落一。\n"
            "作者的话：今天家里有事，请假一天。\n"
            "今天先更到这里，明天补上。\n"
            "以上。\n"
            "正文段落二。\n"
            "PS：求个月票，谢谢大家。\n"
            "正文段落三。\n"
        )
        spans = detect_noise_spans(text)
        masked = [text[a:b] for a, b in spans]
        self.assertIn("作者的话：今天家里有事，请假一天。\n", masked)
        self.assertIn("以上。\n", masked)
        self.assertEqual([item for item in masked if item.startswith("PS：")], ["PS：求个月票，谢谢大家。\n"])
        self.assertEqual(len(masked), 4)
        self.assertNotIn("正文段落二。\n", masked)

    def test_repeated_chapter_tail_requires_four_chapters(self):
        short = "".join(f"第{i}章\n正文第{i}段，山色渐深。\n本章完，感谢观看支持\n" for i in range(1, 4))
        short_chapters = make_chapters(short, [f"第{i}章" for i in range(1, 4)])
        self.assertEqual(detect_noise_spans(short, short_chapters), [])

        long_text = "".join(f"第{i}章\n正文第{i}段，山色渐深。\n本章完，感谢观看支持\n" for i in range(1, 7))
        long_chapters = make_chapters(long_text, [f"第{i}章" for i in range(1, 7)])
        spans = detect_noise_spans(long_text, long_chapters)
        masked = [long_text[a:b] for a, b in spans]
        self.assertEqual(len(masked), 6)
        self.assertTrue(all(item.startswith("本章完") for item in masked))

    def test_negative_body_words_are_not_masked(self):
        text = (
            "他深入群山之后，才发现这片山脉远比传闻中辽阔。\n"
            "他关注着远处那道若有若无的剑光，久久没有移开视线。\n"
            "韩立从怀中取出一枚青色玉简，收藏在袖中。\n"
            "他订阅邸报已有三年。\n"
        )
        self.assertEqual(detect_noise_spans(text), [])

    def test_chapter_heading_lines_are_protected(self):
        text = "第一章 求月票\n正文一行。\n目录\n正文二行。\n第三章节\n正文三行。\n第四章\n正文四行。\n"
        chapters = make_chapters(text, ["第一章 求月票", "目录", "第三章节", "第四章"])
        spans = detect_noise_spans(text, chapters)
        masked = [text[a:b] for a, b in spans]
        self.assertEqual(masked, [])
        for chapter in chapters:
            self.assertFalse(any(a < chapter["start"] < b for a, b in spans))

    def test_units_skip_spans_preserves_offsets_and_coverage(self):
        text = "正文甲。\n求月票！\n正文乙。\n"
        spans = detect_noise_spans(text)
        self.assertEqual(len(spans), 1)
        kept = units(text, limit=500, skip_spans=spans)
        self.assertEqual([u["text"] for u in kept], ["正文甲。\n", "正文乙。\n"])
        for unit in kept:
            self.assertEqual(text[unit["start"]:unit["end"]], unit["text"])
        self.assertEqual(units(text, limit=500, skip_spans=None), legacy_units(text, limit=500))

    def test_plan_tasks_attaches_masked_spans_only_when_needed(self):
        text = "第一章\n韩立走进药园。\n求月票！\n第二章\n南宫婉抬头看天。\n"
        chapters = make_chapters(text, ["第一章", "第二章"])
        source = {"id": digest(["source", text]), "text": text}
        tasks = plan_tasks(source, chapters)
        noisy = [t for t in tasks if t["chapter_title"] == "第一章"]
        clean = [t for t in tasks if t["chapter_title"] == "第二章"]
        self.assertEqual(len(noisy), 1)
        self.assertEqual(len(clean), 1)
        self.assertIn("masked_spans", noisy[0])
        self.assertFalse(any("求月票" in unit["text"] for unit in noisy[0]["units"]))
        self.assertNotIn("masked_spans", clean[0])
        self.assertEqual(set(clean[0]), {"id", "stage", "start", "end", "chapter_id", "chapter_title",
                                         "section_kind", "units"})


class DirtySurfaceTests(unittest.TestCase):
    def test_sentence_punctuation_is_dirty(self):
        self.assertIn("sentence_punct", dirty_surface_reasons("身材魁梧，头戴绿色斗篷之人"))
        self.assertTrue(is_dirty_surface("身材魁梧，头戴绿色斗篷之人"))

    def test_action_suffix_is_dirty_with_length_boundary(self):
        self.assertIn("action_suffix", dirty_surface_reasons("韩立说道"))
        self.assertIn("action_suffix", dirty_surface_reasons("老者看着"))
        self.assertEqual(dirty_surface_reasons("王道"), ())
        self.assertEqual(dirty_surface_reasons("说道"), ())

    def test_long_fragment_is_dirty(self):
        self.assertIn("long_fragment", dirty_surface_reasons("蓝色的是临时待在本城的修士所戴的"))
        self.assertFalse(is_dirty_surface("韩立"))

    def test_book_title_and_interpunct_are_exempt(self):
        self.assertEqual(dirty_surface_reasons("《太上无极混元教主元始天尊》"), ())
        self.assertEqual(dirty_surface_reasons("亚历山大·大仲马与他的朋友们"), ())

    def test_deity_rank_suffix_is_exempt(self):
        self.assertEqual(dirty_surface_reasons("雷部二十四员催云助雨护法天君"), ())
        self.assertEqual(dirty_surface_reasons("火部五位正神外设四员副将真君"), ())
        self.assertEqual(dirty_surface_reasons("五岳四渎诸路护国佑民无上天尊"), ())
        self.assertEqual(dirty_surface_reasons("金龙如意正一龙虎玄坛真君之神"), ())
        self.assertEqual(dirty_surface_reasons("主痘碧霞元君"), ())

    def test_twelve_character_name_is_kept(self):
        self.assertEqual(dirty_surface_reasons("一二三四五六七八九十十一"), ())


class GroundingGuardTests(unittest.TestCase):
    def _source(self, text: str) -> dict:
        return {"id": digest(["source", text]), "text": text}

    def test_ground_result_filters_dirty_mentions_and_links(self):
        text = "韩立说道走进屋中。韩立点头。\n"
        source = self._source(text)
        task = {"id": "t1", "chapter_id": "ch1", "start": 0, "end": len(text),
                "units": [{"id": "u0_9", "start": 0, "end": 9, "text": text[0:9]},
                          {"id": "u9_15", "start": 9, "end": 15, "text": text[9:15]}]}
        value = {
            "mentions": [
                {"text": "韩立说道", "kind": "person", "refs": ["u0_9"]},
                {"text": "韩立", "kind": "person", "refs": ["u9_15"]},
            ],
            "links": [
                {"left": "韩立说道", "right": "韩立", "ref": "u0_9"},
                {"left": "韩立", "right": "韩立", "ref": "u9_15"},
            ],
        }
        result = ground_result(source, task, value, "req1")
        self.assertEqual([o["status"] for o in result["observations"]], ["filtered_dirty_surface", "grounded"])
        self.assertEqual(result["observations"][0]["mention_ids"], [])
        self.assertEqual([m["text"] for m in result["mentions"]], ["韩立"])
        self.assertEqual(len(result["links"]), 1)
        self.assertFalse(any(r for r in result["reviews"] if r.get("status") == "pending"
                             and r.get("reason") in {"原文引用内找不到完整称呼", "模型没有提供原文引用"}))

    def test_ground_result_recovers_mention_across_legacy_hard_cut(self):
        prefix = "天地玄黄。" * 99 + "宇宙洪荒"
        text = prefix + "韩立" + "继续赶路。" * 120 + "\n"
        source = self._source(text)
        self.assertEqual(len(prefix), 499)
        task = {"id": "t2", "chapter_id": "ch1", "start": 0, "end": len(text),
                "units": units(text, 0, len(text), limit=500)}
        target = next(u for u in task["units"] if u["start"] <= 499 and 501 <= u["end"])
        value = {"mentions": [{"text": "韩立", "kind": "person", "refs": [target["id"]]}], "links": []}
        result = ground_result(source, task, value, "req2")
        self.assertEqual([m["start"] for m in result["mentions"]], [499])
        self.assertEqual(result["observations"][0]["status"], "grounded")
        self.assertFalse(any(r["reason"] == "原文引用内找不到完整称呼" for r in result["reviews"]))


class RosterDirtySurfaceTests(unittest.TestCase):
    def _task(self, text: str) -> dict:
        return {"id": "c1task", "chapter_id": "ch1", "start": 0, "end": len(text),
                "units": [{"id": "u0_8", "start": 0, "end": 8, "text": text[0:8]},
                          {"id": "u8_13", "start": 8, "end": 13, "text": text[8:13]}]}

    def _mention(self, text: str, start: int, end: int) -> dict:
        return {"id": digest(["mention", start, end, text]), "text": text, "start": start, "end": end,
                "address": {"text_version_id": "src", "offset_unit": "unicode_code_point",
                            "start": start, "end": end},
                "chapter_id": "ch1", "provisional_kinds": ["person"], "observation_ids": [], "evidence": []}

    def test_apply_c1_result_marks_dirty_surface_not_entity(self):
        text = "韩立说道走来。\n韩立点头。\n"
        task = self._task(text)
        dirty = self._mention("韩立说道", 0, 4)
        clean = self._mention("韩立", 8, 10)
        raw = {
            "entities": [
                {"local_id": "e1", "name": "韩立说道", "kind": "person",
                 "surfaces": [{"surface": "韩立说道", "surface_type": "proper_name"},
                              {"surface": "韩立", "surface_type": "proper_name"}],
                 "evidence_ref": "u0_8"},
                {"local_id": "e2", "name": "韩立说道", "kind": "person",
                 "surfaces": [{"surface": "韩立说道", "surface_type": "proper_name"}],
                 "evidence_ref": "u0_8"},
            ],
            "assignments": [],
            "not_entity": [],
        }
        result = apply_c1_task_result({"id": "src", "text": text}, task, [dirty, clean], raw, "req")
        outcomes = {out["mention_id"]: out for out in result["mention_outcomes"]}
        self.assertEqual(outcomes[dirty["id"]]["status"], "not_entity")
        self.assertIsNone(outcomes[dirty["id"]]["local_id"])
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["name"], "韩立")
        self.assertNotIn("韩立说道", result["entities"][0]["surfaces"])
        self.assertEqual(result["reviews"], [])
        self.assertNotIn("韩立说道", result["repaired_surfaces"])

    def test_sanitize_cache_fast_path_returns_same_object(self):
        mentions = [self._mention("韩立", 8, 10)]
        cached = {"task_id": "c1task", "request_id": "req", "entities": [], "mention_outcomes": [],
                  "weak_norms": [], "declared_proper_norms": ["韩立"], "repaired_surfaces": [], "reviews": []}
        self.assertIs(_sanitize_c1_result_dirty_surfaces(cached, mentions), cached)

    def test_sanitize_cache_cleans_dirty_entity_and_outcome(self):
        text = "韩立说道走来。\n韩立点头。\n"
        dirty = self._mention("韩立说道", 0, 4)
        clean = self._mention("韩立", 8, 10)
        cached = {
            "task_id": "c1task",
            "request_id": "req",
            "entities": [
                {"local_id": "e1", "name": "韩立说道", "kind": "person",
                 "surfaces": ["韩立说道", "韩立"], "surface_types": {"韩立说道": "proper_name", "韩立": "proper_name"},
                 "proper_surfaces": ["韩立说道", "韩立"], "mention_ids": [dirty["id"], clean["id"]]},
                {"local_id": "e2", "name": "韩立说道", "kind": "person", "surfaces": ["韩立说道"],
                 "surface_types": {"韩立说道": "proper_name"}, "proper_surfaces": ["韩立说道"],
                 "mention_ids": [dirty["id"]]},
            ],
            "mention_outcomes": [
                {"mention_id": dirty["id"], "task_id": "c1task", "status": "assigned", "local_id": "e1", "request_id": "req"},
                {"mention_id": clean["id"], "task_id": "c1task", "status": "assigned", "local_id": "e1", "request_id": "req"},
            ],
            "weak_norms": ["韩立说道"],
            "declared_proper_norms": ["韩立说道", "韩立"],
            "repaired_surfaces": ["韩立说道"],
            "reviews": [
                {"id": "r1", "stage": "roster-c1-repair", "surface": "韩立说道", "status": "pending"},
                {"id": "r2", "stage": "roster-c3-link", "surface": "韩立", "status": "pending"},
            ],
        }
        cleaned = _sanitize_c1_result_dirty_surfaces(cached, [dirty, clean])
        self.assertEqual(len(cleaned["entities"]), 1)
        self.assertEqual(cleaned["entities"][0]["name"], "韩立")
        self.assertEqual(cleaned["entities"][0]["mention_ids"], [clean["id"]])
        self.assertEqual(cleaned["declared_proper_norms"], ["韩立"])
        self.assertEqual(cleaned["weak_norms"], [])
        self.assertEqual([rv["id"] for rv in cleaned["reviews"]], ["r2"])
        outcomes = {out["mention_id"]: out for out in cleaned["mention_outcomes"]}
        self.assertEqual(outcomes[dirty["id"]]["status"], "not_entity")
        self.assertIsNone(outcomes[dirty["id"]]["local_id"])
        self.assertEqual(outcomes[clean["id"]]["status"], "assigned")
        # 入参不被修改
        self.assertEqual(len(cached["entities"]), 2)
        self.assertEqual(cached["entities"][0]["mention_ids"], [dirty["id"], clean["id"]])

    def test_resolve_surface_direct_exports_dirty_placeholder(self):
        text = "韩立说道走来。韩立点头。韩立离开。\n"
        dirty = self._mention("韩立说道", 0, 4)
        clean_a = self._mention("韩立", 8, 10)
        clean_b = self._mention("韩立", 13, 15)
        mentions = [dirty, clean_a, clean_b]
        links = [{"id": "link1", "left": "韩立说道", "right": "韩立", "ref": "u0_8"}]
        result = resolve_surface_direct({"id": "src", "text": text}, [{"id": "ch1", "start": 0, "end": len(text)}],
                                       mentions, links, [])
        self.assertEqual(len(result["entities"]), 2)
        placeholder = next(e for e in result["entities"] if e["identity_status"] == "not_entity_suggested")
        self.assertEqual(placeholder["name"], "韩立说道")
        self.assertEqual(placeholder["kind"], "unknown")
        self.assertEqual(placeholder["aliases"], [])
        self.assertEqual(placeholder["mention_ids"], [dirty["id"]])
        by_id = {m["id"]: m for m in result["mentions"]}
        self.assertEqual(by_id[dirty["id"]]["link_method"], "not_entity_suggested")
        self.assertEqual(by_id[dirty["id"]]["link_refs"], [])
        self.assertEqual(len({by_id[clean_a["id"]]["entity_id"], by_id[clean_b["id"]]["entity_id"]}), 1)
        self.assertEqual(by_id[clean_a["id"]]["link_method"], "surface_direct")


class PropagationMaskTests(unittest.TestCase):
    def _fixture(self, masked: bool):
        body = "韩立看见药园。\n"
        noise = "感谢韩立道友的盟主打赏，加更一章！\n"
        text = body + noise
        source = {"id": "src", "text": text}
        chapter = {"id": "ch1", "start": 0, "end": len(text)}
        task = {"id": "t1", "chapter_id": "ch1", "start": 0, "end": len(text),
                "units": units(text, 0, len(text), limit=500)}
        if masked:
            task["masked_spans"] = [(len(body), len(text))]
        first = {"id": "m1", "text": "韩立", "start": 0, "end": 2,
                 "address": {"text_version_id": "src", "offset_unit": "unicode_code_point", "start": 0, "end": 2},
                 "chapter_id": "ch1", "task_id": "t1", "provisional_kinds": ["person"],
                 "observation_ids": [], "evidence": [], "entity_id": "e1",
                 "link_method": "chapter_local", "link_refs": []}
        entities = [{"id": "e1", "name": "韩立", "names": ["韩立"], "aliases": [],
                     "mention_ids": ["m1"], "mention_count": 1, "start": 0, "first_chapter_id": "ch1",
                     "kind": "person", "identity_status": "linked_by_roster", "link_methods": {"chapter_local": 1}}]
        c1 = {"t1": {"declared_proper_norms": ["韩立"], "weak_norms": [], "request_id": "req"}}
        summary = {"algorithm": ROSTER_VERSION, "entity_count": 1, "active_entity_count": 1,
                   "singleton_count": 1, "link_methods": {"chapter_local": 1}}
        return source, [chapter], [task], c1, entities, [first], summary

    def test_propagation_skips_masked_noise_line(self):
        source, chapters, tasks, c1, entities, mentions, summary = self._fixture(masked=True)
        result = propagate_same_task_proper(source, chapters, tasks, c1, entities, mentions, [], summary)
        self.assertEqual(result[4], [])
        self.assertEqual(result[1], mentions)

    def test_propagation_without_mask_would_complete(self):
        source, chapters, tasks, c1, entities, mentions, summary = self._fixture(masked=False)
        result = propagate_same_task_proper(source, chapters, tasks, c1, entities, mentions, [], summary)
        self.assertEqual(len(result[4]), 1)
        self.assertEqual(result[4][0]["start"], source["text"].index("韩立", 1))


class ExportCoverageTests(unittest.TestCase):
    def test_build_export_bundle_covers_dirty_mentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input.txt"
            text = "第一章\n韩立说道走来。韩立点头。\n"
            path.write_text(text, encoding="utf-8")
            store = Store(root / "run.sqlite3")
            try:
                from entitytrace.text import import_text

                source = import_text(store, path)
                body = source["text"]
                state = {"chapters": {"complete": True}, "discovery": {"complete": True},
                         "identity": {"complete": True}}
                store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",
                              ("run1", source["id"], dumps({"identity_algorithm": ROSTER_VERSION}),
                               dumps(state), "now", "now"))
                chapter = {"id": "ch1", "start": 0, "end": len(body), "kind": "chapter"}
                store.put("run1", "chapter", chapter)
                store.put("run1", "observation", {"id": "obs1", "task_id": "t1", "status": "grounded"})
                dirty_text = "韩立说道"
                clean_text = "韩立"
                dirty_start = body.index(dirty_text)
                clean_start = body.index(clean_text, dirty_start + len(dirty_text))
                dirty = {"id": digest(["mention", dirty_start, dirty_start + 4, dirty_text]), "text": dirty_text,
                         "start": dirty_start, "end": dirty_start + 4, "chapter_id": "ch1", "task_id": "t1",
                         "address": {"text_version_id": source["id"], "offset_unit": "unicode_code_point",
                                     "start": dirty_start, "end": dirty_start + 4},
                         "evidence": [], "provisional_kinds": ["person"], "observation_ids": ["obs1"]}
                clean = {"id": digest(["mention", clean_start, clean_start + 2, clean_text]), "text": clean_text,
                         "start": clean_start, "end": clean_start + 2, "chapter_id": "ch1", "task_id": "t1",
                         "address": {"text_version_id": source["id"], "offset_unit": "unicode_code_point",
                                     "start": clean_start, "end": clean_start + 2},
                         "evidence": [], "provisional_kinds": ["person"], "observation_ids": ["obs1"]}
                store.put("run1", "mention", dirty)
                store.put("run1", "mention", clean)
                placeholder_id = digest(["roster-not-entity", source["id"], [dirty_start, dirty_start + 4], [dirty_text]])
                clean_entity_id = digest(["roster-entity", source["id"], [clean_start, clean_start + 2], [clean_text]])
                roster_out = {
                    "id": "roster_output",
                    "start": 0,
                    "entities": [
                        {"id": clean_entity_id, "name": clean_text, "names": [clean_text], "aliases": [clean_text],
                         "mention_ids": [clean["id"]], "mention_count": 1, "start": clean_start,
                         "first_chapter_id": "ch1", "kind": "person", "identity_status": "unresolved_singleton",
                         "link_methods": {"singleton": 1}},
                        {"id": placeholder_id, "name": dirty_text, "names": [dirty_text], "aliases": [],
                         "mention_ids": [dirty["id"]], "mention_count": 1, "start": dirty_start,
                         "first_chapter_id": "ch1", "kind": "unknown", "identity_status": "not_entity_suggested",
                         "link_methods": {"not_entity_suggested": 1}},
                    ],
                    "mentions": [
                        {**dirty, "entity_id": placeholder_id, "link_method": "not_entity_suggested",
                         "link_refs": [], "identity_status": "not_entity_suggested"},
                        {**clean, "entity_id": clean_entity_id, "link_method": "singleton", "link_refs": []},
                    ],
                    "reviews": [],
                    "ambiguous_surfaces": [],
                    "roster_summary": {"algorithm": ROSTER_VERSION, "entity_count": 2, "active_entity_count": 1,
                                       "singleton_count": 2, "link_methods": {"not_entity_suggested": 1, "singleton": 1},
                                       "c3_rejected_name_mismatch": 0, "identity_review_count": 0,
                                       "ambiguous_surface_count": 0},
                }
                store.put("run1", "roster_output", roster_out)
                bundle = build_export_bundle(store, "run1")
                result = bundle["result"]
                mention_ids = {m["id"] for m in result["mentions"]}
                self.assertEqual(mention_ids, {dirty["id"], clean["id"]})
                self.assertEqual(sorted(mid for e in result["entities"] for mid in e["mention_ids"]),
                                 sorted(mention_ids))
                self.assertEqual(result["status"], "completed")
                by_id = {m["id"]: m for m in result["mentions"]}
                self.assertEqual(by_id[dirty["id"]]["identity_status"], "not_entity_suggested")
                self.assertEqual(by_id[dirty["id"]]["link_method"], "not_entity_suggested")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
