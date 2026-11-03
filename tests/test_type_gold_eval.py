from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_type_gold import ROOT, evaluate_type_gold


def _build_synthetic_gold_and_runs(tmp_dir: Path) -> tuple[Path, Path, Path]:
    gold_dir = tmp_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    roster = {
        "book": "synth_book",
        "source_path": "synth.txt",
        "source_sha256": "deadbeef",
        "guideline": "0.14a-T1-T5",
        "created_at": "2026-11-03T12:00:00+08:00",
        "labeler": "agent:claude-opus-4-6",
        "human_review": "pending",
        "entities": [
            {
                "gold_id": "G_PER_1",
                "canonical_name": "姜子牙",
                "entity_type": "person",
                "sub_type": "deity_immortal",
                "aliases": ["姜子牙", "子牙"],
                "representative_mentions": [
                    {"chapter_ordinal": 8, "start": 10, "end": 13, "text": "姜子牙"}
                ],
                "note": "rep",
                "guideline": "0.14a-T1-T5",
                "labeler": "agent:claude-opus-4-6",
                "human_review": "pending",
            },
            {
                "gold_id": "G_PER_2",
                "canonical_name": "商容",
                "entity_type": "person",
                "sub_type": "official_general",
                "aliases": ["商容"],
                "representative_mentions": [
                    {"chapter_ordinal": 8, "start": 20, "end": 22, "text": "商容"}
                ],
                "note": "rep",
                "guideline": "0.14a-T1-T5",
                "labeler": "agent:claude-opus-4-6",
                "human_review": "pending",
            },
            {
                "gold_id": "G_PLC_1",
                "canonical_name": "朝歌",
                "entity_type": "place",
                "sub_type": "city",
                "aliases": ["朝歌"],
                "representative_mentions": [
                    {"chapter_ordinal": 8, "start": 30, "end": 32, "text": "朝歌"}
                ],
                "note": "rep",
                "guideline": "0.14a-T1-T5",
                "labeler": "agent:claude-opus-4-6",
                "human_review": "pending",
            },
            {
                "gold_id": "G_ORG_1",
                "canonical_name": "成汤",
                "entity_type": "org",
                "sub_type": "dynasty",
                "aliases": ["成汤"],
                "representative_mentions": [
                    {"chapter_ordinal": 8, "start": 40, "end": 42, "text": "成汤"}
                ],
                "note": "rep",
                "guideline": "0.14a-T1-T5",
                "labeler": "agent:claude-opus-4-6",
                "human_review": "pending",
            },
            {
                "gold_id": "G_ITM_1",
                "canonical_name": "打神鞭",
                "entity_type": "item",
                "sub_type": "talisman",
                "aliases": ["打神鞭"],
                "representative_mentions": [
                    {"chapter_ordinal": 8, "start": 50, "end": 53, "text": "打神鞭"}
                ],
                "note": "rep",
                "guideline": "0.14a-T1-T5",
                "labeler": "agent:claude-opus-4-6",
                "human_review": "pending",
            },
        ],
    }

    mentions = {
        "book": "synth_book",
        "source_path": "synth.txt",
        "source_sha256": "deadbeef",
        "guideline": "0.14a-T1-T5",
        "created_at": "2026-11-03T12:00:00+08:00",
        "labeler": "agent:claude-opus-4-6",
        "human_review": "pending",
        "sampled_chapters": [
            {"ordinal": 8, "body_start": 0, "end": 200},
            {"ordinal": 24, "body_start": 200, "end": 400},
        ],
        "mentions": [
            {
                "mention_id": "M1",
                "chapter_ordinal": 8,
                "start": 30,
                "end": 32,
                "text": "朝歌",
                "entity_type": "place",
                "gold_id": "G_PLC_1",
                "refers_to": None,
                "note": "place",
            },
            {
                "mention_id": "M2",
                "chapter_ordinal": 8,
                "start": 40,
                "end": 42,
                "text": "成汤",
                "entity_type": "org",
                "gold_id": "G_ORG_1",
                "refers_to": None,
                "note": "org",
            },
            {
                "mention_id": "M3",
                "chapter_ordinal": 8,
                "start": 50,
                "end": 54,
                "text": "此打神鞭",
                "entity_type": "item",
                "gold_id": "G_ITM_1",
                "refers_to": None,
                "note": "item (pred in run2 has 51..54 overlap 3/4 >= 50%)",
            },
            {
                "mention_id": "M4",
                "chapter_ordinal": 8,
                "start": 60,
                "end": 62,
                "text": "宝剑",
                "entity_type": "generic",
                "gold_id": None,
                "refers_to": None,
                "note": "generic sword ch8",
            },
            {
                "mention_id": "M5",
                "chapter_ordinal": 24,
                "start": 260,
                "end": 262,
                "text": "宝剑",
                "entity_type": "generic",
                "gold_id": None,
                "refers_to": None,
                "note": "generic sword ch24",
            },
            {
                "mention_id": "M6",
                "chapter_ordinal": 8,
                "start": 80,
                "end": 82,
                "text": "白鱼",
                "entity_type": "not_entity",
                "gold_id": None,
                "refers_to": None,
                "note": "not_entity fish",
            },
        ],
        "skips": [],
    }

    hard_cases = {
        "book": "synth_book",
        "source_path": "synth.txt",
        "source_sha256": "deadbeef",
        "guideline": "0.14a-T1-T5",
        "created_at": "2026-11-03T12:00:00+08:00",
        "labeler": "agent:claude-opus-4-6",
        "human_review": "pending",
        "cases": [
            {
                "case_id": "HC_SPLIT",
                "category": "reincarnation_daoist_secular",
                "type": "same",
                "title": "姜子牙与子牙被过分拆",
                "expected_judgment": "same_entity",
                "target_gold_ids": ["G_PER_1"],
                "mentions": [
                    {
                        "mention_id": "HM1",
                        "chapter_ordinal": 8,
                        "start": 10,
                        "end": 13,
                        "text": "姜子牙",
                        "entity_type": "person",
                        "gold_id": "G_PER_1",
                        "refers_to": None,
                        "note": "m1",
                    },
                    {
                        "mention_id": "HM2",
                        "chapter_ordinal": 8,
                        "start": 100,
                        "end": 102,
                        "text": "子牙",
                        "entity_type": "person",
                        "gold_id": "G_PER_1",
                        "refers_to": None,
                        "note": "m2",
                    },
                ],
                "note": "test split",
            },
            {
                "case_id": "HC_PASS",
                "category": "alias_fullname_abbrev",
                "type": "same",
                "title": "朝歌两处提及正确归并",
                "expected_judgment": "same_entity",
                "target_gold_ids": ["G_PLC_1"],
                "mentions": [
                    {
                        "mention_id": "HM3",
                        "chapter_ordinal": 8,
                        "start": 30,
                        "end": 32,
                        "text": "朝歌",
                        "entity_type": "place",
                        "gold_id": "G_PLC_1",
                        "refers_to": None,
                        "note": "m3",
                    },
                    {
                        "mention_id": "HM4",
                        "chapter_ordinal": 8,
                        "start": 110,
                        "end": 112,
                        "text": "朝歌",
                        "entity_type": "place",
                        "gold_id": "G_PLC_1",
                        "refers_to": None,
                        "note": "m4",
                    },
                ],
                "note": "test pass",
            },
            {
                "case_id": "HC_DIFF",
                "category": "shared_title_multi_referent",
                "type": "different",
                "title": "姜子牙与商容不得合并",
                "expected_judgment": "different",
                "target_gold_ids": ["G_PER_1", "G_PER_2"],
                "mentions": [
                    {
                        "mention_id": "HM5",
                        "chapter_ordinal": 8,
                        "start": 10,
                        "end": 13,
                        "text": "姜子牙",
                        "entity_type": "person",
                        "gold_id": "G_PER_1",
                        "refers_to": None,
                        "note": "m5",
                    },
                    {
                        "mention_id": "HM6",
                        "chapter_ordinal": 8,
                        "start": 20,
                        "end": 22,
                        "text": "商容",
                        "entity_type": "person",
                        "gold_id": "G_PER_2",
                        "refers_to": None,
                        "note": "m6",
                    },
                ],
                "note": "test different",
            },
            {
                "case_id": "HC_GEN",
                "category": "generic_and_not_entity_trap",
                "type": "generic_no_merge",
                "title": "跨章泛称宝剑不得合并",
                "expected_judgment": "generic_no_merge",
                "target_gold_ids": [],
                "mentions": [
                    {
                        "mention_id": "HM7",
                        "chapter_ordinal": 8,
                        "start": 60,
                        "end": 62,
                        "text": "宝剑",
                        "entity_type": "generic",
                        "gold_id": None,
                        "refers_to": None,
                        "note": "gen ch8",
                    },
                    {
                        "mention_id": "HM8",
                        "chapter_ordinal": 24,
                        "start": 260,
                        "end": 262,
                        "text": "宝剑",
                        "entity_type": "generic",
                        "gold_id": None,
                        "refers_to": None,
                        "note": "gen ch24",
                    },
                ],
                "note": "test generic_no_merge",
            },
            {
                "case_id": "HC021",
                "category": "reincarnation_daoist_secular",
                "type": "policy_pending",
                "title": "老子一气化三清待定",
                "expected_judgment": "policy_pending",
                "target_gold_ids": ["G_PER_1"],
                "mentions": [
                    {
                        "mention_id": "HM9",
                        "chapter_ordinal": 8,
                        "start": 10,
                        "end": 13,
                        "text": "姜子牙",
                        "entity_type": "person",
                        "gold_id": "G_PER_1",
                        "refers_to": None,
                        "note": "p1",
                    },
                    {
                        "mention_id": "HM10",
                        "chapter_ordinal": 8,
                        "start": 100,
                        "end": 102,
                        "text": "子牙",
                        "entity_type": "person",
                        "gold_id": "G_PER_1",
                        "refers_to": None,
                        "note": "p2",
                    },
                ],
                "note": "policy_pending listed only",
            },
        ],
    }

    (gold_dir / "roster.json").write_text(json.dumps(roster, ensure_ascii=False), encoding="utf-8")
    (gold_dir / "mentions.json").write_text(json.dumps(mentions, ensure_ascii=False), encoding="utf-8")
    (gold_dir / "hard_cases.json").write_text(json.dumps(hard_cases, ensure_ascii=False), encoding="utf-8")

    # Run 1:
    # - 姜子牙 (10..13) in E_PER_A, 子牙 (100..102) in E_PER_B -> over-split on G_PER_1!
    # - 商容 (20..22) also merged into E_PER_A -> over-merge on E_PER_A (G_PER_1 + G_PER_2)!
    # - 朝歌 (30..32) and (110..112) in E_PLC_1 (kind="place") -> HC_PASS passes!
    # - 成汤 (40..42) in E_ORG_AS_PLC (kind="place" instead of "org", alt_kinds=["place"] for dynasty)
    # - 宝剑 (60..62 ch8) and (260..262 ch24) and 白鱼 (80..82 not_entity) all merged into E_POLLUTE (kind="object") -> cross-chapter generic overmerge + not_entity linked!
    run1_doc = {
        "run_id": "run1_id",
        "status": "completed_with_review",
        "chapters": [
            {"id": "ch8", "ordinal": 8, "start": 0, "end": 200},
            {"id": "ch24", "ordinal": 24, "start": 200, "end": 400},
        ],
        "entities": [
            {
                "id": "E_PER_A",
                "name": "姜子牙",
                "names": ["姜子牙", "商容"],
                "aliases": [],
                "kind": "person",
                "mention_ids": ["pm1", "pm2"],
                "mention_count": 2,
                "identity_status": "confirmed",
            },
            {
                "id": "E_PER_B",
                "name": "子牙",
                "names": ["子牙"],
                "aliases": [],
                "kind": "person",
                "mention_ids": ["pm3"],
                "mention_count": 1,
                "identity_status": "unresolved_singleton",
            },
            {
                "id": "E_PLC_1",
                "name": "朝歌",
                "names": ["朝歌"],
                "aliases": [],
                "kind": "place",
                "mention_ids": ["pm4", "pm5"],
                "mention_count": 2,
                "identity_status": "confirmed",
            },
            {
                "id": "E_ORG_AS_PLC",
                "name": "成汤",
                "names": ["成汤"],
                "aliases": [],
                "kind": "place",
                "mention_ids": ["pm6"],
                "mention_count": 1,
                "identity_status": "confirmed",
            },
            {
                "id": "E_POLLUTE",
                "name": "宝剑",
                "names": ["宝剑", "白鱼"],
                "aliases": [],
                "kind": "object",
                "mention_ids": ["pm7", "pm8", "pm9"],
                "mention_count": 3,
                "identity_status": "confirmed",
            },
        ],
        "mentions": [
            {"id": "pm1", "start": 10, "end": 13, "text": "姜子牙", "entity_id": "E_PER_A", "chapter_id": "ch8"},
            {"id": "pm2", "start": 20, "end": 22, "text": "商容", "entity_id": "E_PER_A", "chapter_id": "ch8"},
            {"id": "pm3", "start": 100, "end": 102, "text": "子牙", "entity_id": "E_PER_B", "chapter_id": "ch8"},
            {"id": "pm4", "start": 30, "end": 32, "text": "朝歌", "entity_id": "E_PLC_1", "chapter_id": "ch8"},
            {"id": "pm5", "start": 110, "end": 112, "text": "朝歌", "entity_id": "E_PLC_1", "chapter_id": "ch8"},
            {"id": "pm6", "start": 40, "end": 42, "text": "成汤", "entity_id": "E_ORG_AS_PLC", "chapter_id": "ch8"},
            {"id": "pm7", "start": 60, "end": 62, "text": "宝剑", "entity_id": "E_POLLUTE", "chapter_id": "ch8"},
            {"id": "pm8", "start": 260, "end": 262, "text": "宝剑", "entity_id": "E_POLLUTE", "chapter_id": "ch24"},
            {"id": "pm9", "start": 80, "end": 82, "text": "白鱼", "entity_id": "E_POLLUTE", "chapter_id": "ch8"},
        ],
    }

    # Run 2: merges 姜子牙 and 子牙 properly, separates 商容, does not pollute 宝剑/白鱼, and has overlap match for 打神鞭 (51..54 vs 50..54)
    run2_doc = {
        "run_id": "run2_id",
        "status": "completed",
        "chapters": [
            {"id": "ch8", "ordinal": 8, "start": 0, "end": 200},
            {"id": "ch24", "ordinal": 24, "start": 200, "end": 400},
        ],
        "entities": [
            {
                "id": "E_PER_1",
                "name": "姜子牙",
                "names": ["姜子牙", "子牙"],
                "aliases": [],
                "kind": "person",
                "mention_ids": ["pm1", "pm3"],
                "mention_count": 2,
                "identity_status": "confirmed",
            },
            {
                "id": "E_PER_2",
                "name": "商容",
                "names": ["商容"],
                "aliases": [],
                "kind": "person",
                "mention_ids": ["pm2"],
                "mention_count": 1,
                "identity_status": "confirmed",
            },
            {
                "id": "E_PLC_1",
                "name": "朝歌",
                "names": ["朝歌"],
                "aliases": [],
                "kind": "place",
                "mention_ids": ["pm4", "pm5"],
                "mention_count": 2,
                "identity_status": "confirmed",
            },
            {
                "id": "E_ITM_1",
                "name": "打神鞭",
                "names": ["打神鞭"],
                "aliases": [],
                "kind": "object",
                "mention_ids": ["pm10"],
                "mention_count": 1,
                "identity_status": "confirmed",
            },
        ],
        "mentions": [
            {"id": "pm1", "start": 10, "end": 13, "text": "姜子牙", "entity_id": "E_PER_1", "chapter_id": "ch8"},
            {"id": "pm2", "start": 20, "end": 22, "text": "商容", "entity_id": "E_PER_2", "chapter_id": "ch8"},
            {"id": "pm3", "start": 100, "end": 102, "text": "子牙", "entity_id": "E_PER_1", "chapter_id": "ch8"},
            {"id": "pm4", "start": 30, "end": 32, "text": "朝歌", "entity_id": "E_PLC_1", "chapter_id": "ch8"},
            {"id": "pm5", "start": 110, "end": 112, "text": "朝歌", "entity_id": "E_PLC_1", "chapter_id": "ch8"},
            {"id": "pm10", "start": 51, "end": 54, "text": "打神鞭", "entity_id": "E_ITM_1", "chapter_id": "ch8"},
        ],
    }

    r1_path = tmp_dir / "run1" / "result.json"
    r2_path = tmp_dir / "run2" / "result.json"
    r1_path.parent.mkdir(parents=True, exist_ok=True)
    r2_path.parent.mkdir(parents=True, exist_ok=True)
    r1_path.write_text(json.dumps(run1_doc, ensure_ascii=False), encoding="utf-8")
    r2_path.write_text(json.dumps(run2_doc, ensure_ascii=False), encoding="utf-8")
    return gold_dir, r1_path, r2_path


class TestTypeGoldEval(unittest.TestCase):
    def test_confusion_matrix_pollution_b3_spec2_and_noise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gold_dir, r1_path, r2_path = _build_synthetic_gold_and_runs(Path(td))
            report = evaluate_type_gold([r1_path, r2_path], gold_dir)

            self.assertEqual(len(report["runs"]), 2)
            r1_strict = report["runs"][0]["strict"]
            cm_ex = r1_strict["confusion_matrix"]["exhaustive_layer"]
            self.assertEqual(cm_ex["place"]["place"], 1)
            self.assertEqual(cm_ex["org"]["place"], 1)
            self.assertEqual(cm_ex["item"]["unmatched"], 1)
            self.assertEqual(cm_ex["generic"]["item"], 2)
            self.assertEqual(cm_ex["not_entity"]["item"], 1)

            # Strict vs Overlap (>= 50%) on Run 2 for M3 (50..54 vs 51..54)
            r2_strict_rec = report["runs"][1]["strict"]["spec2_metrics"]["item1_discovery_recall"]["exhaustive_layer"]
            r2_ov_rec = report["runs"][1]["overlap"]["spec2_metrics"]["item1_discovery_recall"]["exhaustive_layer"]
            self.assertEqual(r2_strict_rec["by_4class"]["item"]["numerator"], 0)
            self.assertEqual(r2_ov_rec["by_4class"]["item"]["numerator"], 1)

            # Spec 2 Item 2 (link precision): Run 1 has E_POLLUTE with 2 cross-chapter generics + 1 not_entity
            s1_lp = r1_strict["spec2_metrics"]["item2_link_precision"]["exhaustive_layer"]["total"]
            self.assertIn("M6", s1_lp["wrong_not_entity_ids"])
            self.assertIn("M4", s1_lp["wrong_cross_chapter_generic_ids"])
            self.assertIn("M5", s1_lp["wrong_cross_chapter_generic_ids"])

            # Spec 2 Item 4 (generic cross-chapter overmerge): 2/2 in Run 1, 0/0 in Run 2
            s1_gen = r1_strict["spec2_metrics"]["item4_generic_cross_chapter_overmerge"]["exhaustive_layer"]
            self.assertEqual(s1_gen["overmerge_rate_among_matched"]["numerator"], 2)
            self.assertEqual(s1_gen["overmerge_rate_among_matched"]["denominator"], 2)
            self.assertAlmostEqual(s1_gen["overmerge_rate_among_matched"]["value"], 1.0)

            # Spec 2 Item 6 (hard cases by type: same, different, generic_no_merge, policy_pending listed only)
            s1_hc = r1_strict["spec2_metrics"]["item6_hard_cases_by_type"]
            self.assertEqual(s1_hc["same"]["numerator"], 1)
            self.assertEqual(s1_hc["same"]["denominator"], 2)
            self.assertEqual(s1_hc["different"]["numerator"], 0)
            self.assertEqual(s1_hc["different"]["denominator"], 1)
            self.assertEqual(s1_hc["generic_no_merge"]["numerator"], 0)
            self.assertEqual(s1_hc["generic_no_merge"]["denominator"], 1)
            self.assertEqual(s1_hc["policy_pending"]["count"], 1)
            # Scored total excludes policy_pending (2 + 1 + 1 = 4)
            self.assertEqual(s1_hc["scored_total"]["denominator"], 4)

            # Spec 2 Item 9 (noise and co-matched partition B3)
            noise = report["noise"]
            self.assertIsNotNone(noise)
            co_b3 = noise["exhaustive_co_matched_partition_b3"]["strict_all_mentions"]["total"]
            self.assertEqual(co_b3["co_matched_count"], 1)
            self.assertAlmostEqual(co_b3["b3_f1_vs_run1"], 1.0)

    def test_not_imported_by_production_code(self) -> None:
        pkg_dir = ROOT / "entitytrace"
        forbidden = ("evaluate_type_gold", "check_type_gold", "entity-gold/fengshen", "entity-gold\\fengshen")
        for py_file in pkg_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, content, f"Production file {py_file} must not reference {token}")


if __name__ == "__main__":
    unittest.main()
