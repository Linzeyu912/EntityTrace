from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.check_type_gold import check_type_gold

ROOT = Path(__file__).resolve().parents[1]
REAL_GOLD_DIR = ROOT / "test/entity-gold/fengshen/reference"
REAL_TEXT_PATH = ROOT / "test/text-library/texts/07_中文长篇_封神演义_100回.txt"


def _make_mini_fixture(tmp_dir: Path) -> tuple[Path, Path, dict, dict, dict]:
    text_content = (
        "第一回 纣王女娲宫进香\r\n"
        "话说纣王驾临朝歌女娲宫降香，赐龙凤剑一口，又有宝剑一把。\r\n"
        "第二回 冀州侯苏护反商\r\n"
        "且说苏护在朝歌午门外饮恨，率成汤兵马奔回冀州，路见白鱼跃水。\r\n"
    )
    raw_bytes = text_content.encode("utf-8")
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    text_path = tmp_dir / "mini.txt"
    text_path.write_bytes(raw_bytes)

    ch1_body_start = len("第一回 纣王女娲宫进香")
    ch2_start = text_content.index("第二回")
    ch2_body_start = ch2_start + len("第二回 冀州侯苏护反商")
    ch2_end = len(text_content)

    s_zhaoge1 = text_content.index("朝歌", ch1_body_start, ch2_start)
    s_lfj = text_content.index("龙凤剑", ch1_body_start, ch2_start)
    s_bj = text_content.index("宝剑", ch1_body_start, ch2_start)
    s_suhu = text_content.index("苏护", ch2_body_start, ch2_end)
    s_ct = text_content.index("成汤", ch2_body_start, ch2_end)
    s_by = text_content.index("白鱼", ch2_body_start, ch2_end)

    meta = {
        "book": "mini_book",
        "source_path": "mini.txt",
        "source_sha256": sha256,
        "guideline": "0.14a-T1-T5",
        "created_at": "2026-11-03T12:00:00+08:00",
        "labeler": "agent:claude-opus-4-6",
        "human_review": "pending",
    }
    prov = {
        "note": "手工核对上下文",
        "guideline": "0.14a-T1-T5",
        "labeler": "agent:claude-opus-4-6",
        "human_review": "pending",
    }

    roster = {
        **meta,
        "entities": [
            {
                "gold_id": "G1",
                "canonical_name": "朝歌",
                "entity_type": "place",
                "sub_type": "city",
                "aliases": ["朝歌"],
                "representative_mentions": [
                    {"chapter_ordinal": 1, "start": s_zhaoge1, "end": s_zhaoge1 + 2, "text": "朝歌"}
                ],
                **prov,
            },
            {
                "gold_id": "G2",
                "canonical_name": "龙凤剑",
                "entity_type": "item",
                "sub_type": "weapon",
                "aliases": ["龙凤剑"],
                "representative_mentions": [
                    {"chapter_ordinal": 1, "start": s_lfj, "end": s_lfj + 3, "text": "龙凤剑"}
                ],
                **prov,
            },
            {
                "gold_id": "G3",
                "canonical_name": "苏护",
                "entity_type": "person",
                "sub_type": "official_general",
                "aliases": ["苏护"],
                "representative_mentions": [
                    {"chapter_ordinal": 2, "start": s_suhu, "end": s_suhu + 2, "text": "苏护"}
                ],
                **prov,
            },
            {
                "gold_id": "G4",
                "canonical_name": "成汤",
                "entity_type": "org",
                "sub_type": "dynasty",
                "aliases": ["成汤"],
                "representative_mentions": [
                    {"chapter_ordinal": 2, "start": s_ct, "end": s_ct + 2, "text": "成汤"}
                ],
                **prov,
            },
        ],
    }

    mentions = {
        **meta,
        "sampled_chapters": [
            {"ordinal": 1, "body_start": ch1_body_start, "end": ch2_start},
            {"ordinal": 2, "body_start": ch2_body_start, "end": ch2_end},
        ],
        "mentions": [
            {
                "mention_id": "M1",
                "chapter_ordinal": 1,
                "start": s_zhaoge1,
                "end": s_zhaoge1 + 2,
                "text": "朝歌",
                "entity_type": "place",
                "gold_id": "G1",
                "refers_to": None,
                **prov,
            },
            {
                "mention_id": "M2",
                "chapter_ordinal": 1,
                "start": s_lfj,
                "end": s_lfj + 3,
                "text": "龙凤剑",
                "entity_type": "item",
                "gold_id": "G2",
                "refers_to": None,
                **prov,
            },
            {
                "mention_id": "M3",
                "chapter_ordinal": 1,
                "start": s_bj,
                "end": s_bj + 2,
                "text": "宝剑",
                "entity_type": "generic",
                "gold_id": None,
                "refers_to": "G2",
                **prov,
            },
            {
                "mention_id": "M4",
                "chapter_ordinal": 2,
                "start": s_by,
                "end": s_by + 2,
                "text": "白鱼",
                "entity_type": "not_entity",
                "gold_id": None,
                "refers_to": None,
                **prov,
            },
        ],
        "skips": [],
    }

    hard_cases = {
        **meta,
        "cases": [
            {
                "case_id": "HC1",
                "category": "generic_and_not_entity_trap",
                "title": "龙凤剑与宝剑",
                "expected_judgment": "proper_vs_generic",
                "target_gold_ids": ["G2"],
                "mentions": [
                    {
                        "mention_id": "HM1",
                        "chapter_ordinal": 1,
                        "start": s_lfj,
                        "end": s_lfj + 3,
                        "text": "龙凤剑",
                        "entity_type": "item",
                        "gold_id": "G2",
                        "refers_to": None,
                        **prov,
                    },
                    {
                        "mention_id": "HM2",
                        "chapter_ordinal": 1,
                        "start": s_bj,
                        "end": s_bj + 2,
                        "text": "宝剑",
                        "entity_type": "generic",
                        "gold_id": None,
                        "refers_to": "G2",
                        **prov,
                    },
                ],
                **prov,
            }
        ],
    }

    gold_dir = tmp_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    return gold_dir, text_path, roster, mentions, hard_cases


def _write_gold(gold_dir: Path, roster: dict, mentions: dict, hard_cases: dict) -> None:
    (gold_dir / "roster.json").write_text(json.dumps(roster, ensure_ascii=False), encoding="utf-8")
    (gold_dir / "mentions.json").write_text(json.dumps(mentions, ensure_ascii=False), encoding="utf-8")
    (gold_dir / "hard_cases.json").write_text(json.dumps(hard_cases, ensure_ascii=False), encoding="utf-8")


class TestTypeGoldCheck(unittest.TestCase):
    def test_real_fengshen_gold_passes(self) -> None:
        res = check_type_gold(REAL_GOLD_DIR, REAL_TEXT_PATH)
        self.assertTrue(res["ok"], msg=f"Errors: {res['errors']}")
        self.assertEqual(res["error_count"], 0)
        self.assertGreaterEqual(res["summary"]["roster_total"], 140)
        self.assertLessEqual(res["summary"]["roster_total"], 220)
        self.assertEqual(res["summary"]["sampled_chapters"], [8, 24, 40, 56, 72, 88])
        self.assertLessEqual(res["summary"]["mentions_total"], 1200)
        self.assertGreaterEqual(res["summary"]["hard_cases_total"], 40)

    def test_mini_fixture_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gold_dir, text_path, roster, mentions, hard_cases = _make_mini_fixture(Path(td))
            _write_gold(gold_dir, roster, mentions, hard_cases)
            res = check_type_gold(gold_dir, text_path)
            self.assertTrue(res["ok"], msg=f"Errors: {res['errors']}")

    def test_detects_slice_mismatch_and_out_of_body(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gold_dir, text_path, roster, mentions, hard_cases = _make_mini_fixture(Path(td))
            bad_mentions = copy.deepcopy(mentions)
            # 1) 切片文字与原文不符
            bad_mentions["mentions"][0]["text"] = "西岐"
            # 2) 坐标落在章题上（小于 body_start）
            bad_mentions["mentions"][1]["start"] = 0
            bad_mentions["mentions"][1]["end"] = 3
            bad_mentions["mentions"][1]["text"] = "第一回"
            _write_gold(gold_dir, roster, bad_mentions, hard_cases)
            res = check_type_gold(gold_dir, text_path)
            self.assertFalse(res["ok"])
            joined = "\n".join(res["errors"])
            self.assertIn("原文切片不一致", joined)
            self.assertIn("未落在第 1 回正文区间", joined)

    def test_detects_missing_gold_id_and_invalid_generic_gold_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gold_dir, text_path, roster, mentions, hard_cases = _make_mini_fixture(Path(td))
            bad_mentions = copy.deepcopy(mentions)
            # 1) gold_id 不在 roster 中
            bad_mentions["mentions"][0]["gold_id"] = "G_NON_EXISTENT"
            # 2) generic 的 gold_id 非 null
            bad_mentions["mentions"][2]["gold_id"] = "G2"
            # 3) refers_to 不在 roster 中
            bad_mentions["mentions"][2]["refers_to"] = "G_BAD_REF"
            _write_gold(gold_dir, roster, bad_mentions, hard_cases)
            res = check_type_gold(gold_dir, text_path)
            self.assertFalse(res["ok"])
            joined = "\n".join(res["errors"])
            self.assertIn("G_NON_EXISTENT", joined)
            self.assertIn("gold_id 必须为 null", joined)
            self.assertIn("G_BAD_REF", joined)

    def test_detects_duplicate_and_overlapping_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gold_dir, text_path, roster, mentions, hard_cases = _make_mini_fixture(Path(td))
            bad_mentions = copy.deepcopy(mentions)
            dup = copy.deepcopy(bad_mentions["mentions"][0])
            dup["mention_id"] = "M_DUP"
            bad_mentions["mentions"].append(dup)
            _write_gold(gold_dir, roster, bad_mentions, hard_cases)
            res = check_type_gold(gold_dir, text_path)
            self.assertFalse(res["ok"])
            self.assertIn("重复区间", "\n".join(res["errors"]))


if __name__ == "__main__":
    unittest.main()
