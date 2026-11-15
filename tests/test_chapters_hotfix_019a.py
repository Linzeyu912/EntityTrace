from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import (
    BATCH_MAX_CANDIDATES,
    BATCH_MAX_NUMBERED_HEADINGS,
    _body_backed_numbered_ids,
    _check_labels,
    collect_candidates,
    detect_chapters,
    scan_window,
    split_candidate_batches,
)
from entitytrace.models import ModelError
from entitytrace.storage import Store
from entitytrace.text import import_text


def _candidates(count: int) -> list[dict]:
    """构造只含标题与占位位置的候选序列，用于分批纯函数测试。"""
    items: list[dict] = []
    for idx in range(count):
        title = f"路人闲话{idx}。"
        start = idx * 40
        items.append({
            "id": f"u{idx:06d}",
            "start": start,
            "end": start + len(title),
            "unit_end": start + len(title) + 1,
            "title": title,
            "prev": "",
            "next": "",
        })
    return items


def _numbered_candidates(count: int, with_body: bool = True) -> list[dict]:
    items: list[dict] = []
    for idx in range(count):
        title = f"第{idx + 1}章 标题{idx}"
        start = idx * 40
        items.append({
            "id": f"u{idx:06d}",
            "start": start,
            "end": start + len(title),
            "unit_end": start + len(title) + 1,
            "title": title,
            "prev": "",
            "next": "正文随即展开。" if with_body else "",
        })
    return items


def _toc_text(count: int = 30) -> str:
    return "\n".join(f"第{i}章 目次{i}" for i in range(1, count + 1)) + "\n"


def _toc_and_body_text() -> str:
    toc = "\n".join(f"第{i}章 目次{i}" for i in range(1, 31))
    body = "".join(
        f"第{i}章 正文标题{i}\n这是第{i}章的正文内容，人物在此登场。\n"
        for i in range(1, 5)
    )
    return toc + "\n\n" + body


MISLABELED_TEXT = (
    "第一卷 初入\n"
    "第一章 启程\n"
    "少年走出山村。\n"
    "第二章 入城\n"
    "城门下人来人往。\n"
    "第三章 夜宿\n"
    "客栈灯火通明。\n"
    "第二卷 风起\n"
    "第四章 孤岛\n"
    "四面皆是海水。\n"
    "第五章 归途\n"
    "船队缓缓靠岸。\n"
    "第六章 终局\n"
    "故事就此落幕。\n"
)


def _mislabeled_kinds(cands: list[dict]) -> list[dict]:
    mapping = {
        "第一卷 初入": "volume",
        "第一章 启程": "chapter",
        "第二卷 风起": "chapter",
        "第四章 孤岛": "volume",
        "第六章 终局": "chapter",
    }
    return [
        {"id": c["id"], "kind": mapping[c["text"]]}
        for c in cands if c["text"] in mapping
    ]


def _mislabeled_extra_kinds(cands: list[dict]) -> list[dict]:
    mapping = {
        "第一卷 初入": "volume",
        "第一章 启程": "chapter",
        "第二卷 风起": "chapter",
        "第四章 孤岛": "extra",
        "第六章 终局": "chapter",
    }
    return [
        {"id": c["id"], "kind": mapping[c["text"]]}
        for c in cands if c["text"] in mapping
    ]


def _multi_volume_text() -> str:
    parts: list[str] = []
    for volume in range(1, 4):
        parts.append(f"第{volume}卷 卷名{volume}\n")
        for number in range(1, 9):
            parts.append(f"第{number}章 卷{volume}章{number}\n")
            parts.append(f"此处正文承接上文，人物依次登场（卷{volume}章{number}）。\n")
    return "".join(parts)


def _multi_volume_kinds(cands: list[dict]) -> list[dict]:
    result: list[dict] = []
    for c in cands:
        text = c["text"]
        if text.startswith("第") and "卷 " in text and "章 " not in text:
            result.append({"id": c["id"], "kind": "volume"})
            continue
        if text == "第五章 卷1章5" or text == "第二章 卷3章2":
            continue
        if text == "第三章 卷2章3":
            result.append({"id": c["id"], "kind": "volume"})
            continue
        result.append({"id": c["id"], "kind": "chapter"})
    return result


class _ScriptedModels:
    def __init__(self, choose):
        self._choose = choose
        self.calls = 0

    def choices(self, *args, **kwargs):
        raise AssertionError("章节阶段不调用 choices")

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        self.calls += 1
        cands = state["candidates"]
        value = {"headings": self._choose(cands)}
        validate(value)
        return value, f"req-{self.calls}"


def _detect(root: Path, name: str, text: str, choose, concurrency: int = 1):
    store = Store(root / f"{name}.sqlite3")
    try:
        path = root / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        source = import_text(store, path)
        store.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?)",
            (name, source["id"], "{}", "{}", "now", "now"),
        )
        models = _ScriptedModels(choose)
        outcome = detect_chapters(store, models, source, name, lambda _: None, concurrency=concurrency)
        return outcome, store.records(name, "decision"), store.records(name, "review")
    finally:
        store.close()


class SplitCandidateBatchesTests(unittest.TestCase):
    def test_low_density_matches_fixed_slice(self):
        self.assertEqual(BATCH_MAX_CANDIDATES, 400)
        self.assertEqual(BATCH_MAX_NUMBERED_HEADINGS, 120)
        cands = []
        for idx in range(1200):
            numbered = idx % 400 < 60
            title = f"第{idx + 1}章 标题{idx}" if numbered else f"路人闲话{idx}。"
            start = idx * 40
            cands.append({
                "id": f"u{idx:06d}",
                "start": start,
                "end": start + len(title),
                "unit_end": start + len(title) + 1,
                "title": title,
                "prev": "",
                "next": "",
            })
        expected = [cands[i:i + BATCH_MAX_CANDIDATES] for i in range(0, len(cands), BATCH_MAX_CANDIDATES)]
        batches = split_candidate_batches(cands)
        self.assertEqual([[c["id"] for c in b] for b in batches],
                         [[c["id"] for c in b] for b in expected])
        self.assertEqual([len(b) for b in batches], [400, 400, 400])

    def test_high_density_respects_numbered_cap_and_preserves_order(self):
        chapters = [f"第{i}章 标题{i}" for i in range(1, 601)]
        fillers = [f"路旁行人第{i}次经过。" for i in range(1, 601)]
        titles: list[str] = []
        for idx in range(600):
            titles.append(chapters[idx])
            titles.append(fillers[idx])
        cands = []
        for idx, title in enumerate(titles):
            start = idx * 40
            cands.append({
                "id": f"u{idx:06d}",
                "start": start,
                "end": start + len(title),
                "unit_end": start + len(title) + 1,
                "title": title,
                "prev": "",
                "next": "",
            })
        batches = split_candidate_batches(cands)
        self.assertGreaterEqual(len(batches), 5)
        for batch in batches:
            numbered = sum(1 for c in batch if c["title"].startswith("第") and "章" in c["title"])
            self.assertLessEqual(numbered, BATCH_MAX_NUMBERED_HEADINGS)
        flattened = [c["id"] for batch in batches for c in batch]
        self.assertEqual(flattened, [c["id"] for c in cands])

    def test_unnumbered_candidates_are_capped_by_total_size(self):
        cands = _candidates(1000)
        batches = split_candidate_batches(cands)
        self.assertEqual([len(b) for b in batches], [400, 400, 200])
        self.assertEqual([c["id"] for b in batches for c in b], [c["id"] for c in cands])


class BodyBackedNumberedTests(unittest.TestCase):
    def test_toc_rows_excluded_and_body_chapters_included(self):
        text = _toc_and_body_text()
        cands = collect_candidates(text)
        toc_end = text.index("\n\n")
        toc_ids = {c["id"] for c in cands if c["start"] < toc_end}
        body_ids = {
            c["id"] for c in cands
            if c["start"] >= toc_end and c["title"].startswith("第") and "正文标题" in c["title"]
        }
        backed = _body_backed_numbered_ids(text, cands)
        self.assertEqual(len(toc_ids), 30)
        self.assertEqual(backed & toc_ids, set())
        self.assertEqual(backed, body_ids)

    def test_heading_without_following_body_is_not_counted(self):
        text = "第一章 空目\n第二章 空目\n"
        cands = collect_candidates(text)
        self.assertEqual(_body_backed_numbered_ids(text, cands), set())


class CheckLabelsTests(unittest.TestCase):
    def test_two_argument_call_still_works(self):
        _check_labels({"headings": []}, {"a"})
        _check_labels({"headings": [{"id": "a", "kind": "chapter"}]}, {"a", "b"})

    def test_expected_coverage_is_accepted(self):
        allowed = {f"u{i:03d}" for i in range(20)}
        expected = set(sorted(allowed)[:20])
        headings = [{"id": uid, "kind": "chapter"} for uid in sorted(expected)[:19]]
        _check_labels({"headings": headings}, allowed, expected)

    def test_toc_under_reported_is_not_treated_as_severe(self):
        text = _toc_text()
        cands = collect_candidates(text)
        expected = _body_backed_numbered_ids(text, cands)
        self.assertEqual(expected, set())
        _check_labels({"headings": []}, {c["id"] for c in cands}, expected)

    def test_severe_under_labeling_raises(self):
        cands = _numbered_candidates(20)
        expected = {c["id"] for c in cands}
        self.assertEqual(len(expected), 20)
        headings = [{"id": c["id"], "kind": "chapter"} for c in cands[:2]]
        with self.assertRaises(ModelError) as ctx:
            _check_labels({"headings": headings}, {c["id"] for c in cands}, expected)
        self.assertIn("严重漏标", str(ctx.exception))


class _CollapsingModels:
    def __init__(self):
        self.sizes: list[int] = []

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        cands = state["candidates"]
        self.sizes.append(len(cands))
        if len(cands) > 20:
            raise ModelError("章节候选批响应严重漏标")
        value = {"headings": [{"id": c["id"], "kind": "chapter"} for c in cands]}
        validate(value)
        return value, f"req-{cands[0]['id']}"


class _WrappedCollapsingModels:
    def __init__(self):
        self.sizes: list[int] = []

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        cands = state["candidates"]
        self.sizes.append(len(cands))
        if len(cands) > 20:
            value = {"headings": [{"id": c["id"], "kind": "chapter"} for c in cands[:2]]}
            try:
                validate(value)
            except ModelError as exc:
                raise ModelError("discovery 返回格式不符合约定", None, None) from exc
            raise AssertionError("严重漏标必须被校验拦截")
        value = {"headings": [{"id": c["id"], "kind": "chapter"} for c in cands]}
        validate(value)
        return value, f"req-{cands[0]['id']}"


class _AlwaysCollapsingModels:
    def __init__(self):
        self.calls = 0

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        self.calls += 1
        raise ModelError("章节候选批响应严重漏标")


class _HttpFailingModels:
    def __init__(self, status, code):
        self.status = status
        self.code = code
        self.calls = 0

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        self.calls += 1
        raise ModelError("模型服务请求失败", self.status, self.code)


class ScanWindowSplitTests(unittest.TestCase):
    def test_direct_severe_under_labeling_splits_and_keeps_request_ids(self):
        text = ""
        for i in range(1, 21):
            text += f"第{i}章 标题{i}\n正文第{i}段。\n"
        cands = collect_candidates(text)
        self.assertEqual(len(cands), 40)
        models = _CollapsingModels()
        labeled, request_id = scan_window(models, cands, text=text)
        self.assertEqual(models.sizes, [40, 20, 20])
        self.assertEqual(len(labeled), 40)
        self.assertEqual(len({item["request_id"] for item in labeled}), 2)
        self.assertIn(request_id, {item["request_id"] for item in labeled})

    def test_wrapped_format_error_also_triggers_split(self):
        text = ""
        for i in range(1, 21):
            text += f"第{i}章 标题{i}\n正文第{i}段。\n"
        cands = collect_candidates(text)
        models = _WrappedCollapsingModels()
        labeled, _ = scan_window(models, cands, text=text)
        self.assertEqual(models.sizes, [40, 20, 20])
        self.assertEqual(len(labeled), 40)

    def test_small_batch_rethrows_without_recursion(self):
        cands = _numbered_candidates(20)
        models = _AlwaysCollapsingModels()
        with self.assertRaises(ModelError) as ctx:
            scan_window(models, cands, text="")
        self.assertIn("严重漏标", str(ctx.exception))
        self.assertEqual(models.calls, 1)

    def test_provider_errors_are_never_treated_as_under_labeling(self):
        cands = _numbered_candidates(40)
        for status, code in ((401, "auth"), (429, "rate_limit")):
            models = _HttpFailingModels(status, code)
            with self.assertRaises(ModelError) as ctx:
                scan_window(models, cands, text="")
            self.assertEqual(ctx.exception.status, status)
            self.assertEqual(ctx.exception.code, code)
            self.assertEqual(models.calls, 1)


class DetectChaptersHotfixTests(unittest.TestCase):
    def test_mislabeled_volume_chapter_is_rescued_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outcome, decisions, _ = _detect(root, "rescue_volume", MISLABELED_TEXT, _mislabeled_kinds)
            self.assertTrue(outcome["complete"])
            self.assertEqual(outcome["issues"], 0)
            self.assertEqual(
                [(c["title"], c["kind"]) for c in outcome["chapters"]],
                [
                    ("第一卷 初入", "volume"),
                    ("第一章 启程", "chapter"),
                    ("第二章 入城", "chapter"),
                    ("第三章 夜宿", "chapter"),
                    ("第二卷 风起", "volume"),
                    ("第四章 孤岛", "chapter"),
                    ("第五章 归途", "chapter"),
                    ("第六章 终局", "chapter"),
                ],
            )
            ids = [d["question_id"] for d in decisions]
            self.assertEqual(len(ids), len(set(ids)))
            rescued = [
                d for d in decisions
                if d["kind"] == "chapter" and d["result"] == "gap_filled"
            ]
            self.assertEqual(len(rescued), 4)
            upgraded = [d for d in rescued if d["request_id"] is not None]
            self.assertEqual(len(upgraded), 1)
            self.assertEqual(upgraded[0]["request_id"], "req-1")
            appended = [d for d in rescued if d["request_id"] is None]
            self.assertEqual(len(appended), 3)
            demoted = [d for d in decisions if d["result"] == "adopted_as_volume"]
            self.assertEqual(len(demoted), 1)
            self.assertEqual(demoted[0]["kind"], "chapter")

    def test_mislabeled_extra_chapter_is_rescued_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outcome, decisions, _ = _detect(root, "rescue_extra", MISLABELED_TEXT, _mislabeled_extra_kinds)
            self.assertTrue(outcome["complete"])
            self.assertEqual(
                [(c["title"], c["kind"]) for c in outcome["chapters"]],
                [
                    ("第一卷 初入", "volume"),
                    ("第一章 启程", "chapter"),
                    ("第二章 入城", "chapter"),
                    ("第三章 夜宿", "chapter"),
                    ("第二卷 风起", "volume"),
                    ("第四章 孤岛", "chapter"),
                    ("第五章 归途", "chapter"),
                    ("第六章 终局", "chapter"),
                ],
            )
            ids = [d["question_id"] for d in decisions]
            self.assertEqual(len(ids), len(set(ids)))

    def test_rescued_position_has_exactly_one_gap_filled_decision(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outcome, decisions, _ = _detect(root, "single_decision", MISLABELED_TEXT, _mislabeled_kinds)
            by_id = {}
            for item in decisions:
                by_id.setdefault(item["question_id"], []).append(item)
            self.assertTrue(all(len(rows) == 1 for rows in by_id.values()))
            fills = [rows[0] for rows in by_id.values() if rows[0]["result"] == "gap_filled"]
            self.assertEqual(len(fills), 4)
            self.assertEqual(len({row["start"] for row in fills}), 4)
            counts = outcome["path_counts"]
            self.assertEqual(
                sorted(counts),
                sorted([
                    "candidates", "labeled", "adopted", "dropped_no_body",
                    "unnumbered", "numbering_issues", "label_requests",
                ]),
            )
            self.assertTrue(all(isinstance(value, int) for value in counts.values()))

    def test_labeled_chapter_without_body_is_still_dropped_and_blocks(self):
        text = (
            "第一章 起步\n正文一。\n"
            "第二章 空目\n"
            "第三章 终途\n正文三。\n"
        )

        def choose(cands):
            return [
                {"id": c["id"], "kind": "chapter"}
                for c in cands if c["text"].startswith("第")
            ]

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outcome, decisions, reviews = _detect(root, "empty_chapter", text, choose)
            self.assertFalse(outcome["complete"])
            self.assertEqual(outcome["issues"], 1)
            self.assertFalse(any(d["result"] == "gap_filled" for d in decisions))
            self.assertTrue(any(d["result"] == "dropped_no_body" for d in decisions))
            self.assertTrue(reviews[0]["blocking"])

    def test_multi_volume_structure_is_complete_after_gap_fill(self):
        text = _multi_volume_text()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outcome, decisions, reviews = _detect(root, "multi_volume", text, _multi_volume_kinds)
            self.assertTrue(outcome["complete"])
            self.assertEqual(outcome["issues"], 0)
            self.assertEqual(outcome["path_counts"]["unnumbered"], 0)
            self.assertEqual(outcome["path_counts"]["numbering_issues"], 0)
            self.assertEqual(reviews, [])
            chapters = outcome["chapters"]
            self.assertEqual(sum(1 for c in chapters if c["kind"] == "volume"), 3)
            self.assertEqual(sum(1 for c in chapters if c["kind"] == "chapter"), 24)
            for chapter in chapters:
                if chapter["kind"] == "chapter":
                    self.assertIsNotNone(chapter["parent_volume_id"])
            ids = [d["question_id"] for d in decisions]
            self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
