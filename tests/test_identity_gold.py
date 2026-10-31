from __future__ import annotations

import ast
import unittest
from pathlib import Path

from tools.evaluate_identity_gold import evaluate_result, match_mentions

ROOT = Path(__file__).resolve().parents[1]


class IdentityGoldEvaluationTests(unittest.TestCase):
    def test_span_matching_exact_and_half_overlap(self):
        gold = [
            {"sample_id": "s1", "start": 10, "end": 20, "surface": "Elizabeth", "gold_id": "g1"},
            {"sample_id": "s2", "start": 30, "end": 40, "surface": "Mr. Darcy", "gold_id": "g2"},
            {"sample_id": "s3", "start": 50, "end": 60, "surface": "Bingley", "gold_id": "g3"},
            {"sample_id": "s4", "start": 70, "end": 80, "surface": "not_ent", "gold_id": "not_entity"},
            {"sample_id": "s5", "start": 90, "end": 100, "surface": "unc", "gold_id": "uncertain"},
        ]
        pred = [
            {"id": "p1", "start": 10, "end": 20, "text": "Elizabeth"},
            {"id": "p2_short", "start": 30, "end": 35, "text": "Mr."},
            {"id": "p2_better", "start": 32, "end": 40, "text": "Darcy"},
            {"id": "p3_too_short", "start": 50, "end": 54, "text": "Bing"},
            {"id": "p4_not_ent", "start": 70, "end": 80, "text": "not_ent"},
        ]
        matched = match_mentions(gold, pred, strict_only=False)
        self.assertEqual(matched["s1"]["id"], "p1")
        # s2 [30,40) 与 p2_short 重叠 5，与 p2_better 重叠 8，应选重叠更大的 p2_better
        self.assertEqual(matched["s2"]["id"], "p2_better")
        # s3 [50,60) 与 p3_too_short 重叠 4 < 50% (5)，不应匹配
        self.assertNotIn("s3", matched)
        # s4 not_entity 也参与位置匹配以便检查是否误入多成员实体；s5 uncertain 不参与
        self.assertEqual(matched["s4"]["id"], "p4_not_ent")
        self.assertNotIn("s5", matched)

        strict = match_mentions(gold, pred, strict_only=True)
        self.assertEqual(set(strict), {"s1", "s4"})

    def test_b3_metrics_and_overmerges(self):
        roster = {
            "book": "synthetic.txt",
            "book_id": "synthetic",
            "source_sha256": "abc",
            "entities": [
                {"gold_id": "g1", "kind": "person", "canonical": "Alice", "aliases": ["Alice"]},
                {"gold_id": "g2", "kind": "person", "canonical": "Bob", "aliases": ["Bob"]},
            ],
        }
        # 构造 g1 10 处提及（拆为 e1: 8 处，e2: 2 处），g2 10 处提及（2 处误入 e1，8 处归 e3）
        gold_mentions = []
        pred_mentions = []
        e1_mids, e2_mids, e3_mids = [], [], []
        for i in range(10):
            s = i * 10
            sid = f"s_g1_{i}"
            mid = f"m_g1_{i}"
            gold_mentions.append({"sample_id": sid, "start": s, "end": s + 5, "surface": "Alice", "gold_id": "g1", "chapter_index": 1})
            pred_mentions.append({"id": mid, "start": s, "end": s + 5, "text": "Alice"})
            if i < 8:
                e1_mids.append(mid)
            else:
                e2_mids.append(mid)
        for i in range(10):
            s = 200 + i * 10
            sid = f"s_g2_{i}"
            mid = f"m_g2_{i}"
            gold_mentions.append({"sample_id": sid, "start": s, "end": s + 3, "surface": "Bob", "gold_id": "g2", "chapter_index": 1})
            pred_mentions.append({"id": mid, "start": s, "end": s + 3, "text": "Bob"})
            if i < 2:
                e1_mids.append(mid)
            else:
                e3_mids.append(mid)

        result_doc = {
            "run_id": "r1",
            "schema_version": 2,
            "source": {"text_sha256": "abc"},
            "mentions": pred_mentions,
            "entities": [
                {"id": "e1", "mention_ids": e1_mids},
                {"id": "e2", "mention_ids": e2_mids},
                {"id": "e3", "mention_ids": e3_mids},
            ],
        }
        mentions_doc = {"mentions": gold_mentions, "labeler": "test", "human_review": "pending"}
        report = evaluate_result(result_doc, roster, mentions_doc)
        pm = report["primary_metrics"]
        self.assertEqual(pm["discovery_recall_sample"]["value"], 1.0)
        self.assertAlmostEqual(pm["b3_precision"], 0.84, places=6)
        self.assertAlmostEqual(pm["b3_recall"], 0.68, places=6)
        self.assertAlmostEqual(pm["major_entity_coverage"]["value"], 0.8, places=6)
        self.assertEqual(len(pm["overmerges"]), 1)
        self.assertEqual(len(pm["major_entity_overmerges"]), 1)
        # G5a：e1 中含 g1(8处>=2) 与 g2(2处>=2)，触发实体级误合并 major_entity_pair_overmerges == 1
        self.assertEqual(len(pm["major_entity_pair_overmerges"]), 1)
        # G5b：共 20 处主要实体抽样提及，其中 g2 的 2 处（s_g2_0, s_g2_1）落入以 g1 为多数的 e1 -> 2/20 = 0.1
        self.assertEqual(pm["major_mention_misassign_rate"]["numerator"], 2)
        self.assertEqual(pm["major_mention_misassign_rate"]["denominator"], 20)
        self.assertAlmostEqual(pm["major_mention_misassign_rate"]["value"], 0.1, places=6)
        self.assertEqual(pm["major_mention_misassign_rate"]["sample_ids"], ["s_g2_0", "s_g2_1"])
        self.assertFalse(pm["major_mention_misassign_rate"]["passed"])

    def test_g5a_and_g5b_single_mention_noise_vs_entity_pair_overmerge(self):
        roster = {
            "book": "synthetic.txt",
            "book_id": "synthetic",
            "source_sha256": "abc",
            "entities": [
                {"gold_id": "g1", "kind": "person", "canonical": "Alice", "aliases": ["Alice"]},
                {"gold_id": "g2", "kind": "person", "canonical": "Bob", "aliases": ["Bob"]},
            ],
        }
        # g1 20 处全部在 e1；g2 20 处中仅 1 处（s_g2_0）误入 e1，其余 19 处在 e2
        gold_mentions = []
        pred_mentions = []
        e1_mids, e2_mids = [], []
        for i in range(20):
            s = i * 10
            sid = f"s_g1_{i}"
            mid = f"m_g1_{i}"
            gold_mentions.append({"sample_id": sid, "start": s, "end": s + 5, "surface": "Alice", "gold_id": "g1", "chapter_index": 1})
            pred_mentions.append({"id": mid, "start": s, "end": s + 5, "text": "Alice"})
            e1_mids.append(mid)
        for i in range(20):
            s = 300 + i * 10
            sid = f"s_g2_{i}"
            mid = f"m_g2_{i}"
            gold_mentions.append({"sample_id": sid, "start": s, "end": s + 3, "surface": "Bob", "gold_id": "g2", "chapter_index": 1})
            pred_mentions.append({"id": mid, "start": s, "end": s + 3, "text": "Bob"})
            if i == 0:
                e1_mids.append(mid)
            else:
                e2_mids.append(mid)

        result_doc = {
            "run_id": "r_g5",
            "schema_version": 3,
            "source": {"text_sha256": "abc"},
            "mentions": pred_mentions,
            "entities": [
                {"id": "e1", "name": "Alice", "mention_ids": e1_mids},
                {"id": "e2", "name": "Bob", "mention_ids": e2_mids},
            ],
        }
        mentions_doc = {"mentions": gold_mentions, "major_entity_ids": ["g1", "g2"], "labeler": "test", "human_review": "pending"}
        report = evaluate_result(result_doc, roster, mentions_doc)
        pm = report["primary_metrics"]
        # 旧字段仍记录 1 处，但 G5a 实体级误合并（各 >=2 处）为 0，G5b 提及级误归率为 1/40 = 0.025 <= 3% 通过
        self.assertEqual(len(pm["major_entity_overmerges"]), 1)
        self.assertEqual(len(pm["major_entity_pair_overmerges"]), 0)
        self.assertEqual(pm["major_mention_misassign_rate"]["numerator"], 1)
        self.assertEqual(pm["major_mention_misassign_rate"]["denominator"], 40)
        self.assertAlmostEqual(pm["major_mention_misassign_rate"]["value"], 0.025, places=6)
        self.assertEqual(pm["major_mention_misassign_rate"]["max_allowed_count"], 1)
        self.assertTrue(pm["major_mention_misassign_rate"]["passed"])
        self.assertEqual(pm["major_mention_misassign_rate"]["sample_ids"], ["s_g2_0"])

    def test_not_entity_linked_and_hard_stratum_and_out_of_roster(self):
        roster = {
            "book": "synthetic.txt",
            "book_id": "synthetic",
            "source_sha256": "abc",
            "entities": [
                {"gold_id": "g1", "kind": "person", "canonical": "Alice", "aliases": ["Alice"]},
            ],
        }
        gold_mentions = [
            {"sample_id": "s1", "start": 0, "end": 5, "surface": "Alice", "stratum": "entity", "gold_id": "g1", "chapter_index": 1},
            {"sample_id": "s2", "start": 10, "end": 18, "surface": "Miss A", "stratum": "hard", "gold_id": "g1", "chapter_index": 1},
            {"sample_id": "s3", "start": 20, "end": 26, "surface": "friend", "stratum": "hard", "gold_id": "x001", "chapter_index": 1},
            {"sample_id": "s4", "start": 30, "end": 34, "surface": "book", "stratum": "hard", "gold_id": "not_entity", "chapter_index": 1},
        ]
        pred_mentions = [
            {"id": "m1", "start": 0, "end": 5, "text": "Alice"},
            {"id": "m2", "start": 10, "end": 18, "text": "Miss A"},
            {"id": "m3", "start": 20, "end": 26, "text": "friend"},
            {"id": "m4", "start": 30, "end": 34, "text": "book"},
        ]
        # m1, m2, m3, m4 全部被放入多成员组 e1
        result_doc = {
            "run_id": "r1",
            "schema_version": 3,
            "source": {"text_sha256": "abc"},
            "mentions": pred_mentions,
            "entities": [{"id": "e1", "name": "Alice", "mention_ids": ["m1", "m2", "m3", "m4"]}],
        }
        mentions_doc = {
            "mentions": gold_mentions,
            "major_entity_ids": ["g1"],
            "out_of_roster_entities": [{"gold_id": "x001", "kind": "person", "canonical": "Other Friend"}],
            "labeler": "test",
            "human_review": "pending",
        }
        report = evaluate_result(result_doc, roster, mentions_doc)
        pm = report["primary_metrics"]
        self.assertEqual(pm["not_entity_linked_count"], 1)
        self.assertEqual(pm["out_of_roster_in_major_entity_count"], 1)
        # 难例层共 3 个样本被并入多成员组 e1（s2=g1 正确，s3=x001 错误，s4=not_entity 错误）-> 1/3
        self.assertEqual(pm["hard_stratum_precision"]["numerator"], 1)
        self.assertEqual(pm["hard_stratum_precision"]["denominator"], 3)
        self.assertAlmostEqual(pm["hard_stratum_precision"]["value"], round(1 / 3, 6), places=6)
        # 由于 g1 仅匹配 2 处（不足最小阈值 4），major_entity_coverage 报告 not_applicable
        self.assertEqual(pm["major_entity_coverage"]["status"], "not_applicable")
        self.assertIsNone(pm["major_entity_coverage"]["value"])

    def test_production_code_does_not_import_gold_or_evaluator(self):
        forbidden = {"evaluate_identity_gold", "build_entity_gold_sample", "entity-gold"}
        for path in (ROOT / "entitytrace").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"生产代码 {path.name} 不得引用金标准或评分工具 {token}")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names]
                    if isinstance(node, ast.ImportFrom) and node.module:
                        names.append(node.module)
                    for name in names:
                        self.assertFalse(name.startswith("tools"), f"生产代码 {path.name} 不得导入 tools 模块")


if __name__ == "__main__":
    unittest.main()
