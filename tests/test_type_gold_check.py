from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.check_type_gold import REQUIRED_FENGSHEN_HC_TAGS, check_type_gold

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
    s_zhaoge2 = text_content.index("朝歌", ch2_body_start, ch2_end)
    s_ct = text_content.index("成汤", ch2_body_start, ch2_end)
    s_by = text_content.index("白鱼", ch2_body_start, ch2_end)

    meta = {
        "book": "mini_book",
        "source_path": "mini.txt",
        "source_sha256": sha256,
        "guideline": "0.14a-T1-T5",
        "created_at": "2026-11-03T12:00:00+08:00",
        "labeler": "agent:subagent-test",
        "human_review": "pending",
    }
    prov = {
        "note": "手工核对上下文",
        "guideline": "0.14a-T1-T5",
        "labeler": "agent:subagent-test",
        "human_review": "pending",
    }

    roster = {
        **meta,
        "entities": [
            {
                "gold_id": "G1",
                "canonical_name": "朝歌",
                "entity_type": "place",
                "kind": "place",
                "alt_kinds": [],
                "policy": "T1",
                "sub_type": "city",
                "aliases": ["朝歌"],
                "representative_mentions": [
                    {
                        "chapter_ordinal": 1,
                        "start": s_zhaoge1,
                        "end": s_zhaoge1 + 2,
                        "text": "朝歌",
                        "clue": "驾临朝歌女娲宫",
                    }
                ],
                **prov,
            },
            {
                "gold_id": "G2",
                "canonical_name": "龙凤剑",
                "entity_type": "item",
                "kind": "object",
                "alt_kinds": ["ability"],
                "policy": "T1",
                "sub_type": "weapon",
                "aliases": ["龙凤剑"],
                "representative_mentions": [
                    {
                        "chapter_ordinal": 1,
                        "start": s_lfj,
                        "end": s_lfj + 3,
                        "text": "龙凤剑",
                        "clue": "赐龙凤剑一口",
                    }
                ],
                **prov,
            },
            {
                "gold_id": "G3",
                "canonical_name": "苏护",
                "entity_type": "person",
                "kind": "person",
                "alt_kinds": [],
                "policy": "T1",
                "sub_type": "official_general",
                "aliases": ["苏护"],
                "representative_mentions": [
                    {
                        "chapter_ordinal": 2,
                        "start": s_suhu,
                        "end": s_suhu + 2,
                        "text": "苏护",
                        "clue": "且说苏护在朝歌",
                    }
                ],
                **prov,
            },
            {
                "gold_id": "G4",
                "canonical_name": "成汤",
                "entity_type": "org",
                "kind": "organization",
                "alt_kinds": ["place"],
                "policy": "T1",
                "sub_type": "dynasty",
                "aliases": ["成汤"],
                "representative_mentions": [
                    {
                        "chapter_ordinal": 2,
                        "start": s_ct,
                        "end": s_ct + 2,
                        "text": "成汤",
                        "clue": "率成汤兵马",
                    }
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
                "clue": "驾临朝歌女娲宫",
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
                "clue": "赐龙凤剑一口",
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
                "clue": "又有宝剑一把",
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
                "clue": "路见白鱼跃水",
                **prov,
            },
            {
                "mention_id": "M5",
                "chapter_ordinal": 2,
                "start": s_zhaoge2,
                "end": s_zhaoge2 + 2,
                "text": "朝歌",
                "entity_type": "place",
                "gold_id": "G1",
                "refers_to": None,
                "clue": "苏护在朝歌午门",
                **prov,
            },
            {
                "mention_id": "M6",
                "chapter_ordinal": 2,
                "start": s_ct,
                "end": s_ct + 2,
                "text": "成汤",
                "entity_type": "org",
                "gold_id": "G4",
                "refers_to": None,
                "clue": "率成汤兵马",
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
                "type": "same",
                "tags": ["custom_test_tag"],
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
                        "clue": "赐龙凤剑一口",
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
                        "clue": "又有宝剑一把",
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

    def test_new_checks_1_to_8_and_surface_full_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gold_dir, text_path, roster, mentions, hard_cases = _make_mini_fixture(Path(td))

            # Check 1: alias 0 occurrences in text + Check 7: invalid kind/alt_kinds/policy + Check 8: empty labeler
            bad_roster = copy.deepcopy(roster)
            bad_roster["entities"][0]["aliases"].append("不存在别名")
            bad_roster["entities"][1]["kind"] = "invalid_kind"
            bad_roster["entities"][2]["alt_kinds"] = ["person"]  # duplicate of main kind
            bad_roster["entities"][3]["policy"] = "T3"
            bad_roster["entities"][3]["labeler"] = "agent:"

            # Check 2: cross-layer mismatch + Check 3: strict substring of longer registered name + Check 4: invalid clue + Surface full coverage
            bad_roster["entities"].append({
                "gold_id": "G5",
                "canonical_name": "朝歌女娲宫",
                "entity_type": "place",
                "kind": "place",
                "alt_kinds": [],
                "policy": "T1",
                "aliases": ["朝歌女娲宫"],
                "representative_mentions": [
                    bad_roster["entities"][0]["representative_mentions"][0]
                ],
                "note": "长名称",
                "guideline": "0.14a-T1-T5",
                "labeler": "agent:subagent-test",
                "human_review": "pending",
            })
            bad_mentions = copy.deepcopy(mentions)
            # Remove M5 ("朝歌" in Ch 2) to trigger surface full-coverage error
            bad_mentions["mentions"] = [m for m in bad_mentions["mentions"] if m["mention_id"] != "M5"]
            # Make M3 ("宝剑") have clue identical to text (0 chars outside interval -> fails Check 4)
            bad_mentions["mentions"][2]["clue"] = "宝剑"

            # Check 5 & 6: hard_case type mismatch & missing Fengshen required tags
            bad_hc = copy.deepcopy(hard_cases)
            # Cross-layer mismatch with M3 (M3 has refers_to="G2", HM2 has refers_to=None)
            bad_hc["cases"][0]["mentions"][1]["refers_to"] = None
            # Now HC1 has type="same" but HM2 has refers_to=None -> fails Check 5

            _write_gold(gold_dir, bad_roster, bad_mentions, bad_hc)
            res = check_type_gold(gold_dir, text_path, require_fengshen_checklist=True)
            self.assertFalse(res["ok"])
            joined = "\n".join(res["errors"])
            self.assertIn("在原文中未逐字出现 (0 次)", joined)
            self.assertIn("非法或缺失 7 类 kind", joined)
            self.assertIn("alt_kinds 存在重复或包含主类 kind", joined)
            self.assertIn("policy 必须为 'T1' 或 'T2'", joined)
            self.assertIn("labeler 必须为非空 'agent:<model>' 格式", joined)
            self.assertIn("严格落在更长已登记名称 '朝歌女娲宫'", joined)
            self.assertIn("称呼全覆盖未通过", joined)
            self.assertIn("clue 长度必须为 4—20 码点", joined)
            self.assertIn("跨层区间不一致", joined)
            self.assertIn("type='same' 要求所有提及指向恰好 1 个有效所指", joined)
            self.assertIn("缺少 0.14b 规格一第 4 项要求的难例标签", joined)
            self.assertTrue(len(REQUIRED_FENGSHEN_HC_TAGS) >= 18)


if __name__ == "__main__":
    unittest.main()
