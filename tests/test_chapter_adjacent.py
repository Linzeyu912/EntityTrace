from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import (
    collect_candidates,
    detect_chapters,
    filter_headings_with_body,
)
from entitytrace.storage import Store
from entitytrace.text import import_text


class ChapterAdjacentConflictTests(unittest.TestCase):
    def test_numbered_former_preferred_over_unnumbered_latter_and_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = Store(root / "db.sqlite3")
            try:
                path = root / "case1.txt"
                path.write_text(
                    "第一章 渡口\n"
                    "北风吹过栈桥。\n"
                    "第二章 山馆\n"
                    "话说当日旅人投宿山馆。\n"
                    "炉火正旺，众人围坐夜谈。\n"
                    "第三章 破晓\n"
                    "晨钟响起。\n",
                    encoding="utf-8",
                )
                source = import_text(store, path)
                store.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                    ("r-adj", source["id"], "{}", "{}", "now", "now"),
                )

                class AdjacentConflictModels:
                    def choices(s, *args, **kwargs):
                        raise AssertionError("章节阶段禁止调用 choices")

                    def generate(s, system, state, stage, validate, *args, **kwargs):
                        cands = state["candidates"]
                        by_text = {c["text"]: c["id"] for c in cands}
                        val = {
                            "headings": [
                                {"id": by_text["第一章 渡口"], "kind": "chapter"},
                                {"id": by_text["第二章 山馆"], "kind": "chapter"},
                                {"id": by_text["话说当日旅人投宿山馆。"], "kind": "chapter"},
                                {"id": by_text["第三章 破晓"], "kind": "chapter"},
                            ]
                        }
                        validate(val)
                        return val, "req-adj-1"

                outcome = detect_chapters(store, AdjacentConflictModels(), source, "r-adj", lambda _: None)
                self.assertTrue(outcome["complete"])
                self.assertEqual(outcome["issues"], 0)
                self.assertEqual(
                    outcome["path_counts"],
                    {
                        "candidates": 7,
                        "labeled": 4,
                        "adopted": 3,
                        "dropped_no_body": 1,
                        "unnumbered": 0,
                        "numbering_issues": 0,
                        "label_requests": 1,
                    },
                )
                self.assertEqual(
                    [c["title"] for c in outcome["chapters"]],
                    ["第一章 渡口", "第二章 山馆", "第三章 破晓"],
                )
                ch2 = outcome["chapters"][1]
                self.assertIn("话说当日旅人投宿山馆。", source["text"][ch2["body_start"]:ch2["end"]])
                decisions = store.records("r-adj", "decision")
                cands_by_id = {c["id"]: c["title"] for c in collect_candidates(source["text"])}
                res_by_title = {cands_by_id[rec["question_id"]]: rec["result"] for rec in decisions}
                self.assertEqual(
                    res_by_title,
                    {
                        "第一章 渡口": "adopted",
                        "第二章 山馆": "adopted",
                        "话说当日旅人投宿山馆。": "dropped_unnumbered_adjacent",
                        "第三章 破晓": "adopted",
                    },
                )
            finally:
                store.close()

    def test_unnumbered_former_and_numbered_latter_keeps_latter(self):
        text = (
            "Chapter 1\n"
            "The ship sailed out.\n"
            "And so the watch ended.\n"
            "Chapter 2\n"
            "Morning came upon the sea.\n"
        )
        cands = collect_candidates(text)
        labeled = [
            {**c, "kind": "chapter", "request_id": "req-2"}
            for c in cands
            if c["title"] in {"Chapter 1", "And so the watch ended.", "Chapter 2"}
        ]
        kept, dropped = filter_headings_with_body(text, labeled)
        self.assertEqual([h["title"] for h in kept], ["Chapter 1", "Chapter 2"])
        self.assertEqual([h["title"] for h in dropped], ["And so the watch ended."])
        self.assertEqual(dropped[0]["drop_reason"], "dropped_no_body")

    def test_both_numbered_or_both_unnumbered_preserves_existing_behavior(self):
        text_both_num = "第一章 甲\n第二章 乙\n正文内容在此。\n"
        cands_bn = collect_candidates(text_both_num)
        labeled_bn = [
            {**c, "kind": "chapter", "request_id": "req-3a"}
            for c in cands_bn
            if c["title"] in {"第一章 甲", "第二章 乙"}
        ]
        kept_bn, dropped_bn = filter_headings_with_body(text_both_num, labeled_bn)
        self.assertEqual([h["title"] for h in kept_bn], ["第二章 乙"])
        self.assertEqual([h["title"] for h in dropped_bn], ["第一章 甲"])
        self.assertEqual(dropped_bn[0]["drop_reason"], "dropped_no_body")

        text_both_unnum = "楔子甲\n楔子乙\n正文内容在此。\n"
        cands_bu = collect_candidates(text_both_unnum)
        labeled_bu = [
            {**c, "kind": "chapter", "request_id": "req-3b"}
            for c in cands_bu
            if c["title"] in {"楔子甲", "楔子乙"}
        ]
        kept_bu, dropped_bu = filter_headings_with_body(text_both_unnum, labeled_bu)
        self.assertEqual([h["title"] for h in kept_bu], ["楔子乙"])
        self.assertEqual([h["title"] for h in dropped_bu], ["楔子甲"])
        self.assertEqual(dropped_bu[0]["drop_reason"], "dropped_no_body")

    def test_numbered_former_without_body_after_demotion_preserves_existing_behavior(self):
        # 情况 A：前有章号、后无章号，且两者之后直到文末也没有任何正文 -> 两者均按 dropped_no_body 丢弃
        text_end = "第一章 空题\n无号副题\n\n"
        cands_end = collect_candidates(text_end)
        labeled_end = [
            {**c, "kind": "chapter", "request_id": "req-4a"}
            for c in cands_end
            if c["title"] in {"第一章 空题", "无号副题"}
        ]
        kept_end, dropped_end = filter_headings_with_body(text_end, labeled_end)
        self.assertEqual(kept_end, [])
        self.assertEqual([h["title"] for h in dropped_end], ["第一章 空题", "无号副题"])
        self.assertTrue(all(h["drop_reason"] == "dropped_no_body" for h in dropped_end))

        # 情况 B：同一位置重叠的带号章题与无号章题（两者自身范围后均紧接空白且后方被已丢弃标题覆盖）：
        # 先构造 next_kept 为无号章，且当改判时 h.end 到 after_next 无非标题正文，验证守卫分支维持现行
        text_guard = "第一章 空章\n\n"
        labeled_guard = [
            {
                "id": "g1",
                "start": 0,
                "end": len("第一章 空章"),
                "unit_end": len("第一章 空章\n"),
                "title": "第一章 空章",
                "kind": "chapter",
                "request_id": "req-4b",
            },
            {
                "id": "g2",
                "start": 0,
                "end": 0,
                "unit_end": 0,
                "title": "无号重影",
                "kind": "chapter",
                "request_id": "req-4b",
            },
        ]
        # 按 start 排序后 reversed 顺序为 g2 -> g1：g2 因 [0, len) 有 "第一章 空章" 被 keep；
        # 随后处理 g1 时，g1.end 到 g2.start(0) 无正文，且改判检查 [g1.end, len) 全为空白，故不触发改判，g1 维持 dropped_no_body
        kept_g, dropped_g = filter_headings_with_body(text_guard, labeled_guard)
        self.assertEqual([h["id"] for h in kept_g], ["g2"])
        self.assertEqual([h["id"] for h in dropped_g], ["g1"])
        self.assertEqual(dropped_g[0]["drop_reason"], "dropped_no_body")


if __name__ == "__main__":
    unittest.main()
