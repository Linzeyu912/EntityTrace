"""跨时期人生阶段确定性切分与更正后描述零请求重聚专项回归测试。"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.cli import main as cli_main
from entitytrace.corrections import apply_correction
from entitytrace.describe import (
    D1_PROMPT,
    EVENT_VERIFY_PROMPT,
    FACETS,
    PERSISTENCE_PROMPT,
    VERIFY_PROMPT,
    build_describe_id,
    build_entity_stages,
    export_describe_run,
    run_describe,
)
from entitytrace.discovery import plan_tasks
from entitytrace.life_stages import (
    LIFE_STAGE_BUCKET_LABELS,
    LIFE_STAGE_BUCKETS,
    TRIGGER_SOURCE_LIFE_STAGE,
    AgeSignal,
    age_to_bucket,
    infer_life_stage_cuts,
    parse_age_signal,
)
from entitytrace.pipeline import build_export_bundle
from entitytrace.roster import VERSION as ROSTER_VERSION, resolve_roster
from entitytrace.storage import Store
from entitytrace.text import address, import_text
from entitytrace.util import dumps
from tests.test_describe import FakeModels, SERVICES
from tests.test_describe_v6 import _assert_stage_invariants


class _RosterModels:
    def __init__(self, handlers):
        self.handlers = handlers

    def choices(self, *args, **kwargs):
        raise AssertionError("离线测试不应调用 choices")

    def generate(self, system, state, stage, validate, *args, **kwargs):
        val, req_id = self.handlers[stage](state)
        validate(val)
        return val, req_id


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AgeBucketAndParsingTests(unittest.TestCase):
    """凡人五级人生阶段桶映射、数值与定性解析及守卫 1/2 测试。"""

    def test_01_age_to_bucket_boundaries(self):
        self.assertEqual(LIFE_STAGE_BUCKETS, ("child", "teen", "young_adult", "middle", "senior"))
        self.assertEqual(age_to_bucket(0.0), 0)
        self.assertEqual(age_to_bucket(12.0), 0)
        self.assertEqual(age_to_bucket(12.1), 1)
        self.assertEqual(age_to_bucket(17.9), 1)
        self.assertEqual(age_to_bucket(18.0), 2)
        self.assertEqual(age_to_bucket(29.9), 2)
        self.assertEqual(age_to_bucket(30.0), 3)
        self.assertEqual(age_to_bucket(54.9), 3)
        self.assertEqual(age_to_bucket(55.0), 4)
        self.assertEqual(age_to_bucket(120.0), 4)

    def test_02_arabic_numbers_ranges_and_modifiers(self):
        s1 = parse_age_signal("19岁", "他今年19岁")
        self.assertEqual((s1.low, s1.high, s1.bucket_low, s1.bucket_high), (19.0, 19.0, 2, 2))
        s2 = parse_age_signal("25—26岁", "约25—26岁")
        self.assertEqual((s2.low, s2.high, s2.bucket_low, s2.bucket_high), (25.0, 26.0, 2, 2))
        s3 = parse_age_signal("40多岁", "一个40多岁的男子")
        self.assertEqual((s3.bucket_low, s3.bucket_high), (3, 3))
        s4 = parse_age_signal("50开外", "那人50开外")
        self.assertEqual((s4.bucket_low, s4.bucket_high), (3, 4))
        s5 = parse_age_signal("不到22岁", "不到22岁")
        self.assertEqual((s5.bucket_low, s5.bucket_high), (2, 2))
        s6 = parse_age_signal("将近23岁", "将近23岁")
        self.assertEqual((s6.bucket_low, s6.bucket_high), (2, 2))
        s7 = parse_age_signal("刚满16岁", "刚满16岁")
        self.assertEqual((s7.low, s7.high, s7.bucket_low), (16.0, 16.0, 1))
        s8 = parse_age_signal("年近80", "年近80")
        self.assertEqual((s8.bucket_low, s8.bucket_high), (4, 4))
        s9 = parse_age_signal("不到18岁", "不到18岁")
        self.assertEqual((s9.bucket_low, s9.bucket_high), (1, 1))

    def test_03_chinese_consec_approx_xun_you_and_idioms(self):
        self.assertEqual(parse_age_signal("五六岁", "五六岁").low, 5.0)
        self.assertEqual(parse_age_signal("七八岁", "七八岁").high, 8.0)
        self.assertEqual(parse_age_signal("一两岁", "一两岁").high, 2.0)

        sl = parse_age_signal("十来岁", "十来岁的小孩")
        self.assertEqual((sl.low, sl.high, sl.bucket_low, sl.bucket_high), (10.0, 12.0, 0, 0))
        sj = parse_age_signal("十几岁", "才十几岁的娃娃")
        self.assertEqual((sj.low, sj.high, sj.bucket_low, sj.bucket_high), (13.0, 17.0, 1, 1))

        s14 = parse_age_signal("十四五岁", "十四五岁少年")
        self.assertEqual((s14.low, s14.high, s14.bucket_low, s14.bucket_high), (14.0, 15.0, 1, 1))
        s17 = parse_age_signal("十七八岁", "看年纪只有十七八岁的青年")
        self.assertEqual((s17.low, s17.high, s17.bucket_low, s17.bucket_high), (17.0, 18.0, 1, 2))
        s24 = parse_age_signal("二十四五岁", "年纪二十四五")
        self.assertEqual((s24.low, s24.high, s24.bucket_low, s24.bucket_high), (24.0, 25.0, 2, 2))
        s34 = parse_age_signal("三四十岁", "三四十岁年纪")
        self.assertEqual((s34.low, s34.high, s34.bucket_low, s34.bucket_high), (30.0, 40.0, 3, 3))
        s25 = parse_age_signal("二十有五", "小人今年二十有五")
        self.assertEqual((s25.low, s25.high, s25.bucket_low), (25.0, 25.0, 2))

        sx3 = parse_age_signal("年约三旬", "年及三旬")
        sx3p = parse_age_signal("三旬以上", "目今三旬之上")
        self.assertEqual((sx3.bucket_low, sx3p.bucket_low), (3, 3))
        self.assertEqual(parse_age_signal("六旬", "年已六旬").bucket_low, 4)
        self.assertEqual(parse_age_signal("八旬", "年届八旬").bucket_low, 4)

        idiom_expected = [
            ("垂髫", 0), ("总角", 0), ("豆蔻", 1), ("及笄", 1),
            ("弱冠", 2), ("而立", 3), ("不惑", 3), ("知命", 3),
            ("半百", 3), ("花甲", 4), ("古稀", 4), ("耄耋", 4),
        ]
        for word, expected_bucket in idiom_expected:
            sig = parse_age_signal(word, f"年方{word}")
            self.assertIsNotNone(sig, word)
            self.assertEqual(sig.bucket_low, expected_bucket, word)

    def test_04_english_age_expressions(self):
        s1 = parse_age_signal("", "She is almost three-and-twenty!")
        self.assertEqual((s1.bucket_low, s1.bucket_high), (2, 2))
        s2 = parse_age_signal("twenty-seven", "at the age of twenty-seven")
        self.assertEqual((s2.low, s2.high, s2.bucket_low), (27.0, 27.0, 2))
        s3 = parse_age_signal("", "about fifteen or sixteen")
        self.assertEqual((s3.low, s3.high, s3.bucket_low, s3.bucket_high), (15.0, 16.0, 1, 1))
        s4 = parse_age_signal("in his thirties", "a man in his thirties")
        self.assertEqual((s4.low, s4.high, s4.bucket_low, s4.bucket_high), (30.0, 39.0, 3, 3))
        s5 = parse_age_signal("in her forties", "a lady in her forties")
        self.assertEqual((s5.bucket_low, s5.bucket_high), (3, 3))
        s6 = parse_age_signal("in his fifties", "in his fifties")
        self.assertEqual((s6.bucket_low, s6.bucket_high), (3, 4))
        s7 = parse_age_signal("", "though little more than sixteen")
        self.assertEqual((s7.bucket_low, s7.bucket_high), (1, 1))
        s8 = parse_age_signal("", "You cannot be more than twenty, I am sure")
        self.assertEqual((s8.bucket_low, s8.bucket_high), (2, 2))

    def test_05_supernatural_and_huge_ages_rejected(self):
        for val, quo in [
            ("六百余岁", "已活六百余岁"),
            ("四五百年", "但四五百年有吧"),
            ("300岁", "修炼300岁"),
            ("151岁", "高龄151岁"),
            ("六百余岁老翁", "一位六百余岁老翁"),
            ("两千岁", "存活两千岁"),
        ]:
            self.assertIsNone(parse_age_signal(val, quo), (val, quo))

    def test_06_value_priority_over_dual_age_quote(self):
        sig = parse_age_signal("七十二岁", "弟子三十二岁上山，如今虚度七十二岁了。")
        self.assertIsNotNone(sig)
        self.assertEqual((sig.low, sig.high, sig.bucket_low), (72.0, 72.0, 4))

    def test_07_quote_fallback_and_non_age_number_rejection(self):
        # value 无数值时回退 quote 数值
        sig = parse_age_signal("中年男子", "眼前站着一名四十来岁的中年男子")
        self.assertIsNotNone(sig)
        self.assertEqual((sig.low, sig.high, sig.bucket_low), (40.0, 45.0, 3))
        # quote 里的非年龄数字（如“第八层”“three months”）不误判为年龄
        s_vague = parse_age_signal("年纪轻", "小小年纪就已练到了第八层了")
        self.assertIsNotNone(s_vague)
        self.assertTrue(s_vague.is_vague)
        self.assertIsNone(parse_age_signal("外貌不明", "He stayed three months in London."))

    def test_08_qualitative_specific_priority_over_vague_youth(self):
        specific_cases = [
            ("幼童", 0), ("少年", 1), ("青年男子", 2), ("年轻女子", 2),
            ("中年", 3), ("年约中旬", 3), ("年老", 4), ("年纪高迈", 4),
            ("elderly", 4), ("middle-aged", 3),
        ]
        for text_val, bucket in specific_cases:
            sig = parse_age_signal(text_val, text_val)
            self.assertIsNotNone(sig, text_val)
            self.assertFalse(sig.is_vague, text_val)
            self.assertEqual(sig.bucket_low, bucket, text_val)

        for vague in ("年轻", "年纪轻", "年纪轻轻", "很年轻", "年纪不大", "年少", "very young", "youthful"):
            sig = parse_age_signal(vague, vague)
            self.assertIsNotNone(sig, vague)
            self.assertTrue(sig.is_vague, vague)
            self.assertEqual((sig.low, sig.high, sig.bucket_low, sig.bucket_high), (13.0, 29.0, 1, 2))

    def test_09_guard1_relative_and_duration_blocked(self):
        blocked = [
            ("比达西小十余岁", "My sister, who is more than ten years my junior"),
            ("相差五岁", "两人相差五岁"),
            ("最年幼", "though you _are_ the youngest"),
            ("姐妹中最年长者", "My eldest sister has been in town"),
            ("排行老四", "他在家里排行老四"),
            ("约六十余岁", "it had abided for now some sixty years or more"),
            ("年长", "she was older than her brother"),
        ]
        for val, quo in blocked:
            self.assertIsNone(parse_age_signal(val, quo), (val, quo))

    def test_10_guard1_flashback_blocked_unless_present_anchor(self):
        for val, quo in [
            ("十五岁时", "When she was only fifteen there was a gentleman"),
            ("十五岁", "She was then but fifteen"),
            ("八十岁", "八十岁方被文王聘请归国"),
            ("四十岁", "not till he was full forty years old did he change"),
            ("幼时", "幼时曾居山村"),
            ("未修大法时已活六百余岁", "我未修炼玄魂炼妖大法时，便已活了六百余岁"),
            ("转玄魂后又过四五百年", "但四五百年有吧"),
            ("二十岁那年", "二十岁那年他远走他乡"),
        ]:
            self.assertIsNone(parse_age_signal(val, quo), (val, quo))

        # 含明确现在时锚词“如今”不误杀
        sig_now = parse_age_signal("七十二岁", "弟子三十二岁上山，如今虚度七十二岁了。")
        self.assertIsNotNone(sig_now)
        self.assertEqual(sig_now.bucket_low, 4)

    def test_11_guard2_disguise_blocked_and_normal_appearance_preserved(self):
        # 显式伪装词拦截（含同自然段 ±80 码点）
        self.assertIsNone(parse_age_signal("五十余岁", "易容成五十余岁老者"))
        self.assertIsNone(parse_age_signal(
            "五十余岁", "一副五十余岁老者的模样",
            context_after="，这当然不是韩立的真正容貌，只不过为了掩饰自己的容颜永驻。",
        ))
        self.assertIsNone(parse_age_signal("四十岁", "He was forty in disguise."))
        # 跨自然段（含换行）不串扰
        sig_cross = parse_age_signal(
            "四十岁", "他此时已四十岁",
            context_after="\n次日别人易容前来。",
        )
        self.assertIsNotNone(sig_cross)
        self.assertEqual(sig_cross.bucket_low, 3)
        # 严禁泛杀“看起来”“模样”“样子”
        for val, quo in [
            ("看起来三十余岁", "一位看起来三十余岁的普通青年"),
            ("二十许岁青年模样", "还是一副二十许岁的青年模样"),
            ("十七八岁", "十七八岁的样子。"),
        ]:
            self.assertIsNotNone(parse_age_signal(val, quo), (val, quo))

    def test_12_month_and_non_age_units_skipped(self):
        sig = parse_age_signal("去年六月刚满十六岁", "And she was only sixteen last June.")
        self.assertIsNotNone(sig)
        self.assertEqual((sig.low, sig.high, sig.bucket_low), (16.0, 16.0, 1))


class LifeStageInferenceTests(unittest.TestCase):
    """infer_life_stage_cuts 四道守卫与 build_entity_stages 集成不变量测试。"""

    def _chapters(self, n: int = 30) -> list[dict]:
        return [
            {"id": f"ch{i}", "kind": "chapter", "start": (i - 1) * 100, "end": i * 100}
            for i in range(1, n + 1)
        ]

    def _fact(self, fid: str, facet: str, value: str, ch: int, offset_in_ch: int = 10,
              quote: str | None = None, entity_id: str = "e1") -> dict:
        start = (ch - 1) * 100 + offset_in_ch
        q = quote if quote is not None else value
        return {
            "id": fid,
            "entity_id": entity_id,
            "facet": facet,
            "value": value,
            "quote": q,
            "start": start,
            "end": start + len(q),
            "chapter_id": f"ch{ch}",
            "source": "narration",
            "holder_entity_id": None,
            "support": 3,
        }

    def test_13_monotonic_three_stage_growth_restores_two_cuts(self):
        chs = self._chapters(20)
        facts = [
            self._fact("f1", "age", "十来岁", 2, quote="十来岁的小孩"),
            self._fact("f2", "costume", "粗布衣", 3),
            self._fact("f3", "age", "十四岁", 8, quote="已经十四岁了"),
            self._fact("f4", "age", "十六七岁", 12, quote="才十六七岁的年纪"),
            self._fact("f5", "age", "十八岁", 15, quote="已经十八岁了"),
            self._fact("f6", "costume", "青衫", 16),
        ]
        cuts = infer_life_stage_cuts(facts, [], chs)
        self.assertEqual(len(cuts), 2)
        self.assertEqual(cuts[0]["trigger_source"], TRIGGER_SOURCE_LIFE_STAGE)
        self.assertEqual((cuts[0]["from_stage_bucket"], cuts[0]["to_stage_bucket"]), ("child", "teen"))
        self.assertEqual(cuts[0]["source_fact_id"], "f3")
        self.assertEqual(cuts[0]["quote"], "已经十四岁了")
        self.assertEqual((cuts[0]["start"], cuts[0]["end"]), (facts[2]["start"], facts[2]["end"]))
        self.assertEqual((cuts[1]["from_stage_bucket"], cuts[1]["to_stage_bucket"]), ("teen", "young_adult"))
        self.assertEqual(cuts[1]["source_fact_id"], "f5")
        self.assertIn(LIFE_STAGE_BUCKET_LABELS["teen"], cuts[1]["value"])

    def test_14_anti_single_year_birthday_delta_guard(self):
        chs = self._chapters(15)
        # 17 -> 18 仅 +1 岁，拦截
        f_teen_ya = [
            self._fact("f1", "age", "十七岁", 1),
            self._fact("f2", "age", "十八岁", 5),
        ]
        self.assertEqual(infer_life_stage_cuts(f_teen_ya, [], chs), [])
        # 29 -> 31 仅 +2 岁 (< 5.0)，拦截
        f_ya_mid = [
            self._fact("f1", "age", "二十九岁", 1),
            self._fact("f2", "age", "三十一岁", 5),
        ]
        self.assertEqual(infer_life_stage_cuts(f_ya_mid, [], chs), [])
        # 24 -> 35 (+11 >= 5.0)，放行
        f_ya_mid_ok = [
            self._fact("f1", "age", "二十四岁", 1),
            self._fact("f2", "age", "三十五岁", 5),
        ]
        cuts = infer_life_stage_cuts(f_ya_mid_ok, [], chs)
        self.assertEqual(len(cuts), 1)
        self.assertEqual((cuts[0]["from_stage_bucket"], cuts[0]["to_stage_bucket"]), ("young_adult", "middle"))

    def test_15_same_chapter_not_split(self):
        chs = self._chapters(5)
        facts = [
            self._fact("f1", "age", "十岁", 1, offset_in_ch=10),
            self._fact("f2", "age", "二十岁", 1, offset_in_ch=50),
        ]
        self.assertEqual(infer_life_stage_cuts(facts, [], chs), [])

    def test_16_vague_youth_never_triggers_and_narrows_cleanly(self):
        chs = self._chapters(20)
        # 15 -> 很年轻：绝不假切
        self.assertEqual(infer_life_stage_cuts([
            self._fact("f1", "age", "十五岁", 1),
            self._fact("f2", "age", "很年轻", 5),
        ], [], chs), [])
        # 年纪轻轻 -> 十五六岁：无损收窄不切分；随后 -> 二十二岁 可切 teen->young_adult
        cuts_narrow = infer_life_stage_cuts([
            self._fact("f1", "age", "年纪轻轻", 1),
            self._fact("f2", "age", "十五六岁", 2),
            self._fact("f3", "age", "二十二岁", 10),
        ], [], chs)
        self.assertEqual(len(cuts_narrow), 1)
        self.assertEqual((cuts_narrow[0]["from_stage_bucket"], cuts_narrow[0]["to_stage_bucket"]), ("teen", "young_adult"))
        # 年纪轻轻 -> 四十岁：跨入中年期
        cuts_mid = infer_life_stage_cuts([
            self._fact("f1", "age", "年纪轻轻", 1),
            self._fact("f2", "age", "四十岁", 10),
        ], [], chs)
        self.assertEqual(len(cuts_mid), 1)
        self.assertEqual((cuts_mid[0]["from_stage_bucket"], cuts_mid[0]["to_stage_bucket"]), ("young_adult", "middle"))

    def test_17_global_suffix_monotonicity_vetoes_rebound_and_oscillation(self):
        chs = self._chapters(20)
        # 青年 -> 伪装三十余岁 -> 伪装五十余岁 -> 伪装中旬 -> 回落年纪轻/二十多岁
        facts_rebound = [
            self._fact("f1", "age", "二十五岁", 1),
            self._fact("f2", "age", "看起来三十余岁", 5),
            self._fact("f3", "age", "外表五十余岁", 6),
            self._fact("f4", "age", "年约中旬", 9),
            self._fact("f5", "age", "年纪轻", 12),
            self._fact("f6", "age", "二十多岁", 15),
        ]
        self.assertEqual(infer_life_stage_cuts(facts_rebound, [], chs), [])

        # 墨大夫式震荡：60+ -> 37 -> 70+ -> 30+
        facts_osc = [
            self._fact("f1", "age", "六十余岁", 1),
            self._fact("f2", "age", "三十七岁", 4),
            self._fact("f3", "age", "七十许岁老翁", 8),
            self._fact("f4", "age", "三十来岁盛年", 12),
        ]
        self.assertEqual(infer_life_stage_cuts(facts_osc, [], chs), [])

    def test_18_fuzzy_17_18_boundary_does_not_veto_18(self):
        chs = self._chapters(15)
        facts = [
            self._fact("f1", "age", "十岁", 1),
            self._fact("f2", "age", "十四岁", 4),
            self._fact("f3", "age", "十八岁", 8),
            self._fact("f4", "age", "十七八岁", 10),
            self._fact("f5", "age", "二十五岁", 12),
        ]
        cuts = infer_life_stage_cuts(facts, [], chs)
        self.assertEqual([(c["from_stage_bucket"], c["to_stage_bucket"]) for c in cuts], [
            ("child", "teen"), ("teen", "young_adult"),
        ])

    def test_19_existing_split_event_dedup_and_state_advancement(self):
        chs = self._chapters(20)
        facts = [
            self._fact("f1", "age", "十岁", 1),
            self._fact("f2", "age", "十四岁", 5),
            self._fact("f3", "age", "十八岁", 10),
            self._fact("f4", "age", "四十岁", 18),
        ]
        existing = [
            {"id": "ev1", "event_type": "age_jump", "stage_split": True, "start": facts[1]["start"], "end": facts[1]["end"], "chapter_id": "ch5", "value": "十四岁", "quote": "十四岁"},
            {"id": "ev2", "event_type": "age_jump", "stage_split": True, "start": facts[2]["start"], "end": facts[2]["end"], "chapter_id": "ch10", "value": "十八岁", "quote": "十八岁"},
        ]
        cuts = infer_life_stage_cuts(facts, existing, chs)
        self.assertEqual(len(cuts), 1)
        self.assertEqual((cuts[0]["from_stage_bucket"], cuts[0]["to_stage_bucket"]), ("young_adult", "middle"))
        self.assertEqual(cuts[0]["source_fact_id"], "f4")

    def test_20_existing_split_same_chapter_no_intervening_fact_dedup(self):
        chs = self._chapters(10)
        facts = [
            self._fact("f1", "age", "十岁", 1, offset_in_ch=10),
            self._fact("f2", "age", "十五岁", 5, offset_in_ch=10),
            self._fact("f3", "costume", "青衫", 6, offset_in_ch=10),
        ]
        # 既有事件在同章稍后 20 码点处且两者间无其他事实
        existing = [{
            "id": "ev_same_ch", "event_type": "age_jump", "stage_split": True,
            "start": facts[1]["start"] + 20, "end": facts[1]["start"] + 25,
            "chapter_id": "ch5", "value": "长成少年", "quote": "长成少年",
        }]
        self.assertEqual(infer_life_stage_cuts(facts, existing, chs), [])

    def test_21_dominant_long_gap_alignment_moves_cut_start_keeps_verbatim_quote(self):
        chs = self._chapters(30)
        f_old = self._fact("f1", "age", "十九岁", 2, offset_in_ch=10, quote="年方十九岁")
        f_reappear = self._fact("f2", "costume", "黑礼服", 20, offset_in_ch=10, quote="一袭黑礼服")
        f_new = self._fact("f3", "age", "四十岁", 22, offset_in_ch=15, quote="已是四十岁中年")
        facts = [f_old, f_reappear, f_new]
        cuts = infer_life_stage_cuts(facts, [], chs)
        self.assertEqual(len(cuts), 1)
        cut = cuts[0]
        # 主导断档 ch2 -> ch20 (Δ=18 >= 10 且 18 >= 2*(22-20)=4)，切界前移至 ch20 首条事实
        self.assertEqual(cut["cut_start"], f_reappear["start"])
        self.assertEqual(cut["cut_chapter_id"], "ch20")
        # 触发事件仍严格保留 f_new 的逐字原文地址
        self.assertEqual((cut["start"], cut["end"], cut["quote"]), (f_new["start"], f_new["end"], "已是四十岁中年"))
        self.assertEqual(cut["chapter_id"], "ch22")

        stages = build_entity_stages("person", facts, [], chs, (0, 3000))
        self.assertEqual(len(stages), 2)
        self.assertEqual([f["id"] for f in stages[0]["facts"]], ["f1"])
        self.assertEqual([f["id"] for f in stages[1]["facts"]], ["f2", "f3"])
        self.assertEqual(stages[1]["start"], f_reappear["start"])
        _assert_stage_invariants(self, {"kind": "person", "facts": facts, "stages": stages}, (0, 3000))

    def test_22_dominant_long_gap_not_triggered_when_gap_insufficient(self):
        chs = self._chapters(30)
        f1 = self._fact("f1", "age", "十九岁", 2)
        f2 = self._fact("f2", "costume", "旧衣", 10)
        f3 = self._fact("f3", "costume", "新衣", 18)
        f4 = self._fact("f4", "age", "四十岁", 24)
        cuts = infer_life_stage_cuts([f1, f2, f3, f4], [], chs)
        self.assertEqual(len(cuts), 1)
        # 最大断档仅为 8 < 10，切界留在 f4
        self.assertEqual(cuts[0]["cut_start"], f4["start"])

    def test_23_build_entity_stages_person_only_and_trigger_schema(self):
        chs = self._chapters(10)
        facts = [
            self._fact("f1", "age", "十岁", 1),
            self._fact("f2", "costume", "白袍", 3),
            self._fact("f3", "age", "二十岁", 5),
        ]
        ev_normal = {
            "id": "ev1", "event_type": "costume_change", "stage_split": True,
            "value": "换白袍", "quote": "白袍", "start": facts[1]["start"],
            "end": facts[1]["end"], "chapter_id": "ch3",
        }
        stages = build_entity_stages("person", facts, [ev_normal], chs, (0, 1000))
        # ch3 已有显式切点处于 f1 与 f3 之间，不重复合成人生阶段切点，且普通事件保持原 7 键
        self.assertEqual(len(stages), 2)
        self.assertEqual(set(stages[1]["trigger_events"][0].keys()),
                         {"id", "event_type", "value", "quote", "start", "end", "chapter_id"})
        # 无既有事件时，人生阶段切点带 11 键
        stages_ls = build_entity_stages("person", facts, [], chs, (0, 1000))
        self.assertEqual(len(stages_ls), 2)
        self.assertEqual(
            set(stages_ls[1]["trigger_events"][0].keys()),
            {
                "id", "event_type", "value", "quote", "start", "end", "chapter_id",
                "trigger_source", "from_stage_bucket", "to_stage_bucket", "source_fact_id",
            },
        )
        # 非人物粗类不触发人生阶段切分
        place_facts = [
            self._fact("p1", "scale", "十岁", 1),
            self._fact("p2", "scale", "四十岁", 5),
        ]
        self.assertEqual(len(build_entity_stages("place", place_facts, [], chs, (0, 1000))), 1)

    def test_24_infer_life_stages_false_disables_synthetic_cuts(self):
        chs = self._chapters(10)
        facts = [
            self._fact("f1", "age", "十岁", 1),
            self._fact("f2", "age", "二十五岁", 6),
        ]
        stages = build_entity_stages("person", facts, [], chs, (0, 1000), infer_life_stages=False)
        self.assertEqual(len(stages), 1)


class DescribeRebindAndCliTests(unittest.TestCase):
    """更正后描述零模型请求投影重绑、SQLite 零改表可逆性与 CLI --rebind 契约测试。"""

    def _build_novel_fixture(self, root: Path) -> tuple[Path, Path, str, dict[str, str], str]:
        # 构造 15 章迷你小说：
        # ch1: 少年唐太斯（19岁，佩带短剑）；误把“老仆”与“沈公子”归到一起（用于 split-surface / assign）；
        #      另有一个误抽非实体“石阶”（用于 whole-entity not-entity）和一个局部非实体“寒风”
        # ch14: 中年伯爵（40岁，一袭黑袍，持短剑）——与唐太斯是同一人两马甲（用于 merge + 人生阶段切分）
        # ch15: 伯爵（四十岁，黑袍，铁面）；山庄（高墙，用于跨粗类 merge 测试）
        chapters_text = [
            "第一章 少年出航\n唐太斯年方十九岁，身着水手短衫。唐太斯腰佩短剑。沈公子面如冠玉，老仆白发苍苍。石阶青灰冰冷。\n",
        ]
        for idx in range(2, 14):
            chapters_text.append(f"第{idx}章 岁月流转\n岁月静静流逝。\n")
        chapters_text.append(
            "第十四章 归来重逢\n伯爵身穿一袭黑袍，神色冷峻。伯爵手执短剑。\n"
        )
        chapters_text.append(
            "第十五章 终局对峙\n伯爵已是四十岁中年，铁面冷峻。山庄高墙耸立。\n"
        )
        text = "".join(chapters_text)
        book = root / "novel.txt"
        book.write_bytes(text.encode("utf-8"))
        db_path = root / "rebind_test.sqlite3"
        out_root = root / "outputs"
        store = Store(db_path)
        try:
            source = import_text(store, book)
            headings = []
            for idx in range(1, 16):
                title = (
                    "第一章 少年出航" if idx == 1
                    else "第十四章 归来重逢" if idx == 14
                    else "第十五章 终局对峙" if idx == 15
                    else f"第{idx}章 岁月流转"
                )
                pos = text.index(title)
                headings.append({"start": pos, "end": pos + len(title), "title": title, "kind": "chapter"})
            ver, chapters = build_chapters(source, headings)
            run_id = "run_rebind_021a_0000000000000001"
            cfg = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {"complete": True, "chapter_set_version": ver, "issues": 0, "path_counts": {"adopted": len(chapters)}},
                "discovery": {"complete": True, "tasks": len(chapters), "completed": len(chapters)},
            }
            store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], dumps(cfg), dumps(state), "now", "now"))
            for ch in chapters:
                store.put(run_id, "chapter", ch)
            tasks = plan_tasks(source, chapters, 9000)

            def _m(mid: str, surf: str, kind: str, ch_idx: int, start: int):
                ch = chapters[ch_idx]
                t = tasks[ch_idx]
                u0 = t["units"][0]
                return {
                    "id": mid, "text": surf, "start": start, "end": start + len(surf),
                    "address": address(source["id"], start, start + len(surf)),
                    "chapter_id": ch["id"], "task_id": t["id"],
                    "provisional_kinds": [kind], "observation_ids": [f"obs_{mid}"],
                    "evidence": [address(source["id"], u0["start"], u0["end"])],
                }

            ch1_pos = chapters[0]["start"]
            ch14_pos = chapters[13]["start"]
            ch15_pos = chapters[14]["start"]
            t1_first = text.index("唐太斯", ch1_pos)
            t1_second = text.index("唐太斯", t1_first + 3)
            m_by_ch: dict[int, list[dict]] = {
                0: [
                    _m("m_tang_1", "唐太斯", "person", 0, t1_first),
                    _m("m_tang_2", "唐太斯", "person", 0, t1_second),
                    _m("m_sword_1", "短剑", "object", 0, text.index("短剑", ch1_pos)),
                    _m("m_shen_1", "沈公子", "person", 0, text.index("沈公子", ch1_pos)),
                    _m("m_laopu_1", "老仆", "person", 0, text.index("老仆", ch1_pos)),
                    _m("m_shijie_1", "石阶", "place", 0, text.index("石阶", ch1_pos)),
                ],
                13: [
                    _m("m_count_1", "伯爵", "person", 13, text.index("伯爵", ch14_pos)),
                    _m("m_count_2", "伯爵", "person", 13, text.index("伯爵", text.index("伯爵", ch14_pos) + 2)),
                    _m("m_sword_2", "短剑", "object", 13, text.index("短剑", ch14_pos)),
                ],
                14: [
                    _m("m_count_3", "伯爵", "person", 14, text.index("伯爵", ch15_pos)),
                    _m("m_villa_1", "山庄", "place", 14, text.index("山庄", ch15_pos)),
                ],
            }
            for idx, task in enumerate(tasks):
                ms = m_by_ch.get(idx, [])
                store.finish_task(run_id, task, {"request_id": f"req_d_{idx}"}, [("mention", m) for m in ms])

            def handle_c1(st):
                uid = st["units"][0]["id"]
                title = st["chapter_title"]
                if title == "第一章 少年出航":
                    return {
                        "entities": [
                            {"local_id": "e1", "name": "唐太斯", "kind": "person",
                             "surfaces": [{"surface": "唐太斯", "surface_type": "proper_name"}], "evidence_ref": uid},
                            {"local_id": "e2", "name": "短剑", "kind": "object",
                             "surfaces": [{"surface": "短剑", "surface_type": "proper_name"}], "evidence_ref": uid},
                            {"local_id": "e3", "name": "沈公子", "kind": "person",
                             "surfaces": [
                                 {"surface": "沈公子", "surface_type": "proper_name"},
                                 {"surface": "老仆", "surface_type": "proper_name"},
                             ], "evidence_ref": uid},
                            {"local_id": "e4", "name": "石阶", "kind": "place",
                             "surfaces": [{"surface": "石阶", "surface_type": "proper_name"}], "evidence_ref": uid},
                        ],
                        "assignments": [],
                        "not_entity": [],
                    }, "req_c1_1"
                if title == "第十四章 归来重逢":
                    return {
                        "entities": [
                            {"local_id": "e1", "name": "伯爵", "kind": "person",
                             "surfaces": [{"surface": "伯爵", "surface_type": "proper_name"}], "evidence_ref": uid},
                            {"local_id": "e2", "name": "短剑", "kind": "object",
                             "surfaces": [{"surface": "短剑", "surface_type": "proper_name"}], "evidence_ref": uid},
                        ],
                        "assignments": [],
                        "not_entity": [],
                    }, "req_c1_14"
                if title == "第十五章 终局对峙":
                    return {
                        "entities": [
                            {"local_id": "e1", "name": "伯爵", "kind": "person",
                             "surfaces": [{"surface": "伯爵", "surface_type": "proper_name"}], "evidence_ref": uid},
                            {"local_id": "e2", "name": "山庄", "kind": "place",
                             "surfaces": [{"surface": "山庄", "surface_type": "proper_name"}], "evidence_ref": uid},
                        ],
                        "assignments": [],
                        "not_entity": [],
                    }, "req_c1_15"
                return {"entities": [], "assignments": [], "not_entity": []}, f"req_c1_empty_{title}"

            def handle_c2_review(st):
                return {
                    "groups": [{
                        "group_id": "g1",
                        "name": st["rows"][0]["name"],
                        "kind": st["rows"][0]["kind"],
                        "row_ids": [r["row_id"] for r in st["rows"]],
                        "reason": "同体",
                    }],
                    "unresolved_row_ids": [],
                }, "req_c2_rev"

            def handle_c2_propose(st):
                return {"proposals": []}, "req_c2_prop"

            rmodels = _RosterModels({
                "roster-c1": handle_c1,
                "roster-c2-review": handle_c2_review,
                "roster-c2-propose": handle_c2_propose,
            })
            out = resolve_roster(store, rmodels, source, run_id, concurrency=1, log=lambda *_: None, task_chars=9000)
            self.assertTrue(out["complete"])
            store.state(run_id, identity=out)
            bundle = build_export_bundle(store, run_id)
            by_name = {e["name"]: e["id"] for e in bundle["result"]["entities"]}

            # 跑一次 describe (views=1, verify=1, persistence=0) 落盘初始描述事实与事件
            def desc_handler(stage, st, _call_no):
                if stage == "describe-d1":
                    chunk = st.get("text", "")
                    eids_here = {e["entity_id"] for e in st["entities"]}
                    facts, events = [], []
                    if by_name["唐太斯"] in eids_here and "年方十九岁" in chunk:
                        facts.append({"entity_id": by_name["唐太斯"], "facet": "age", "value": "十九岁", "quote": "唐太斯年方十九岁", "source": "narration"})
                        facts.append({"entity_id": by_name["唐太斯"], "facet": "costume", "value": "水手短衫", "quote": "身着水手短衫", "source": "narration"})
                    if by_name["短剑"] in eids_here and "唐太斯腰佩短剑" in chunk:
                        facts.append({"entity_id": by_name["短剑"], "facet": "holder", "value": "唐太斯", "quote": "唐太斯腰佩短剑", "source": "narration", "holder_entity_id": by_name["唐太斯"]})
                    if by_name["沈公子"] in eids_here and "面如冠玉" in chunk:
                        facts.append({"entity_id": by_name["沈公子"], "facet": "face", "value": "面如冠玉", "quote": "沈公子面如冠玉", "source": "narration"})
                        facts.append({"entity_id": by_name["沈公子"], "facet": "hair", "value": "白发苍苍", "quote": "老仆白发苍苍", "source": "narration"})
                    if by_name["石阶"] in eids_here and "石阶青灰冰冷" in chunk:
                        facts.append({"entity_id": by_name["石阶"], "facet": "architecture", "value": "青灰石阶", "quote": "石阶青灰冰冷", "source": "narration"})
                    if by_name["伯爵"] in eids_here and "一袭黑袍" in chunk:
                        facts.append({"entity_id": by_name["伯爵"], "facet": "costume", "value": "黑袍", "quote": "伯爵身穿一袭黑袍", "source": "narration"})
                    if by_name["短剑"] in eids_here and "伯爵手执短剑" in chunk:
                        facts.append({"entity_id": by_name["短剑"], "facet": "holder", "value": "伯爵", "quote": "伯爵手执短剑", "source": "narration", "holder_entity_id": by_name["伯爵"]})
                    if by_name["伯爵"] in eids_here and "四十岁中年" in chunk:
                        facts.append({"entity_id": by_name["伯爵"], "facet": "age", "value": "四十岁", "quote": "伯爵已是四十岁中年", "source": "narration"})
                    if by_name["山庄"] in eids_here and "山庄高墙耸立" in chunk:
                        facts.append({"entity_id": by_name["山庄"], "facet": "architecture", "value": "高墙耸立", "quote": "山庄高墙耸立", "source": "narration"})
                    return {"facts": facts, "events": events}
                if stage == "describe-verify":
                    return {"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None} for it in st["items"]]}
                if stage == "describe-event-verify":
                    return {"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None, "stage_split": False} for it in st["items"]]}
                raise AssertionError(stage)

            fmodels = FakeModels(SERVICES, desc_handler)
            res = run_describe(
                store, run_id, services=SERVICES, models=fmodels,
                top_k=20, kinds="person,place,object,ability", views=1, entities_per_request=0, verify=1,
            )
            return db_path, out_root, run_id, by_name, res["describe_id"]
        finally:
            store.close()

    def test_25_no_corrections_zero_rebind_and_state_records_revision(self):
        with tempfile.TemporaryDirectory() as td:
            db_path, _, run_id, _, did = self._build_novel_fixture(Path(td))
            store = Store(db_path)
            try:
                row = store.rows("SELECT state FROM describe_runs WHERE id=?", (did,))[0]
                st = json.loads(row["state"])
                self.assertEqual(st["status"], "completed")
                self.assertIn("result_revision", st)
                self.assertEqual(st["active_correction_ids"], [])
                self.assertEqual(set(st["stats"].keys()), {"requests", "prompt_tokens", "completion_tokens", "wall_seconds"})

                doc = export_describe_run(store, did)
                self.assertNotIn("rebound_from_describe_id", doc)
                self.assertNotIn("rebind_summary", doc)
                self.assertEqual(set(doc["stats"].keys()), {"requests", "prompt_tokens", "completion_tokens", "wall_seconds"})
            finally:
                store.close()

    def test_26_merge_rebind_unites_facts_triggers_life_stage_and_redirects_holder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                apply_correction(
                    store, run_id, "merge",
                    {"target_entity_id": by_name["伯爵"], "merged_entity_id": by_name["唐太斯"]},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                doc = export_describe_run(store, did)
                self.assertEqual(doc["rebound_from_describe_id"], did)
                self.assertEqual(doc["rebind_summary"]["merged_entities"], 1)
                self.assertEqual(doc["rebind_summary"]["redirected_holders"], 1)

                eids = [e["entity_id"] for e in doc["entities"]]
                self.assertIn(by_name["伯爵"], eids)
                self.assertNotIn(by_name["唐太斯"], eids)

                count_ent = next(e for e in doc["entities"] if e["entity_id"] == by_name["伯爵"])
                self.assertEqual(len(count_ent["facts"]), 4)
                # 合并后 ch1 (19岁) + ch14 (黑袍) + ch15 (40岁) 经主导断档对齐切在 ch14！
                self.assertEqual(len(count_ent["stages"]), 2)
                self.assertEqual((count_ent["stages"][0]["from_chapter"], count_ent["stages"][0]["to_chapter"]), (1, 1))
                self.assertEqual((count_ent["stages"][1]["from_chapter"], count_ent["stages"][1]["to_chapter"]), (14, 15))
                trig = count_ent["stages"][1]["trigger_events"][0]
                self.assertEqual(trig["trigger_source"], TRIGGER_SOURCE_LIFE_STAGE)
                self.assertEqual((trig["from_stage_bucket"], trig["to_stage_bucket"]), ("young_adult", "middle"))
                _assert_stage_invariants(self, count_ent, (0, len(store.source(store.run(run_id)["source_id"])["text"])))

                # 短剑在 ch1 的 holder_entity_id 已由 唐太斯 重定向至 伯爵
                sword_ent = next(e for e in doc["entities"] if e["entity_id"] == by_name["短剑"])
                self.assertTrue(all(f["holder_entity_id"] == by_name["伯爵"] for f in sword_ent["facts"]))
            finally:
                store.close()

    def test_27_split_surface_rebind_separates_by_overlap_and_nearest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                rc = apply_correction(
                    store, run_id, "split-surface",
                    {"entity_id": by_name["沈公子"], "surface": "老仆"},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                doc = export_describe_run(store, did)
                self.assertEqual(doc["rebind_summary"]["split_reassigned_facts"], 1)
                shen = next(e for e in doc["entities"] if e["entity_id"] == by_name["沈公子"])
                laopu = next(e for e in doc["entities"] if e["name"] == "老仆")
                self.assertEqual([f["value"] for f in shen["facts"]], ["面如冠玉"])
                self.assertEqual([f["value"] for f in laopu["facts"]], ["白发苍苍"])
                self.assertIn(laopu["entity_id"], rc["affected_entity_ids"])
            finally:
                store.close()

    def test_28_assign_rebind_to_existing_and_new(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                r1 = apply_correction(
                    store, run_id, "assign",
                    {"mention_id": "m_laopu_1", "target": "new"},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                doc = export_describe_run(store, did)
                self.assertEqual(doc["rebind_summary"]["split_reassigned_facts"], 1)
                new_ent = next(e for e in doc["entities"] if e["name"] == "老仆")
                self.assertEqual([f["value"] for f in new_ent["facts"]], ["白发苍苍"])
                self.assertIsNotNone(r1["result_revision"])
            finally:
                store.close()

    def test_29_not_entity_whole_and_partial_drop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                # 1. 整实体非实体（石阶）
                r1 = apply_correction(
                    store, run_id, "not-entity", {"mention_id": "m_shijie_1"},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                # 2. 局部非实体（沈公子中的“老仆”提及）
                apply_correction(
                    store, run_id, "not-entity", {"mention_id": "m_laopu_1"},
                    source="user", expect_revision=r1["result_revision"], output_root=out_root,
                )
                doc = export_describe_run(store, did)
                self.assertEqual(doc["rebind_summary"]["dropped_entities"], 1)
                self.assertEqual(doc["rebind_summary"]["dropped_not_entity_facts"], 2)
                eids = {e["entity_id"] for e in doc["entities"]}
                self.assertNotIn(by_name["石阶"], eids)
                shen = next(e for e in doc["entities"] if e["entity_id"] == by_name["沈公子"])
                self.assertEqual([f["value"] for f in shen["facts"]], ["面如冠玉"])
            finally:
                store.close()

    def test_30_cross_kind_merge_drops_incompatible_facets_no_keyerror(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                # 将 place(山庄) 并入 person(伯爵)，architecture 不在 FACETS["person"]
                apply_correction(
                    store, run_id, "merge",
                    {"target_entity_id": by_name["伯爵"], "merged_entity_id": by_name["山庄"]},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                doc = export_describe_run(store, did)
                self.assertEqual(doc["rebind_summary"]["dropped_incompatible_facet_facts"], 1)
                count_ent = next(e for e in doc["entities"] if e["entity_id"] == by_name["伯爵"])
                self.assertTrue(all(f["facet"] in FACETS["person"] for f in count_ent["facts"]))
            finally:
                store.close()

    def test_31_split_entity_unknown_kind_falls_back_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                # 将老仆提及的 provisional_kinds 清空，使 split-surface 产生 kind='unknown' 新实体
                rec = store.records(run_id, "roster_output")[0]
                for m in rec["mentions"]:
                    if m["id"] == "m_laopu_1":
                        m["provisional_kinds"] = []
                store.put(run_id, "roster_output", rec)
                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                apply_correction(
                    store, run_id, "split-surface",
                    {"entity_id": by_name["沈公子"], "surface": "老仆"},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                doc = export_describe_run(store, did)
                laopu = next(e for e in doc["entities"] if e["name"] == "老仆")
                self.assertEqual(laopu["kind"], "person")
                self.assertEqual([f["value"] for f in laopu["facts"]], ["白发苍苍"])
            finally:
                store.close()

    def test_32_dedup_facts_and_events_after_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                # 构造一条同坐标同值重复事实与一对重复事件（一条 stage_split=True）分别挂在 唐太斯 与 伯爵
                f_ref = store.rows("SELECT * FROM facts WHERE describe_id=? AND entity_id=? LIMIT 1", (did, by_name["伯爵"]))[0]
                store.execute(
                    "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (did, "dup_fact_tang", by_name["唐太斯"], f_ref["facet"], "  " + f_ref["value"] + " ",
                     f_ref["quote"], f_ref["start"], f_ref["end"], f_ref["chapter_id"], f_ref["source"],
                     None, f_ref["task_id"]),
                )
                store.execute(
                    "INSERT INTO fact_verdicts VALUES(?,?,?,?,?)",
                    (did, "dup_fact_tang", "keep", None, "vt_dup"),
                )
                store.execute(
                    "INSERT INTO state_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (did, "ev_dup_1", by_name["伯爵"], "costume_change", 0, "换黑袍",
                     f_ref["quote"], f_ref["start"], f_ref["end"], f_ref["chapter_id"], f_ref["task_id"]),
                )
                store.execute(
                    "INSERT INTO state_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (did, "ev_dup_2", by_name["唐太斯"], "costume_change", 1, "换黑袍",
                     f_ref["quote"], f_ref["start"], f_ref["end"], f_ref["chapter_id"], f_ref["task_id"]),
                )
                store.execute("INSERT INTO event_verdicts VALUES(?,?,?,?,?,?)", (did, "ev_dup_1", "keep", None, 0, "evt1"))
                store.execute("INSERT INTO event_verdicts VALUES(?,?,?,?,?,?)", (did, "ev_dup_2", "keep", None, 1, "evt2"))

                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                apply_correction(
                    store, run_id, "merge",
                    {"target_entity_id": by_name["伯爵"], "merged_entity_id": by_name["唐太斯"]},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                doc = export_describe_run(store, did)
                self.assertEqual(doc["rebind_summary"]["deduped_facts"], 1)
                self.assertEqual(doc["rebind_summary"]["deduped_events"], 1)
                count_ent = next(e for e in doc["entities"] if e["entity_id"] == by_name["伯爵"])
                self.assertEqual(len(count_ent["state_events"]), 1)
                self.assertTrue(count_ent["state_events"][0]["stage_split"])
            finally:
                store.close()

    def test_33_equidistant_tie_stays_on_orig_and_counts_ambiguous(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                # 在第一章中点制造一条与 沈公子 和 老仆 等距的事实
                text = store.source(store.run(run_id)["source_id"])["text"]
                p_shen = text.index("沈公子")
                p_laopu = text.index("老仆")
                # “沈公子面如冠玉，老仆” -> 沈公子[s..s+3], 老仆[l..l+2]，中间逗号前 1 字与两者等距
                # 取精确对称单字坐标
                mid_start = (p_shen + 3 + p_laopu) // 2
                # 调整至两侧码点距离严格相等（若间隔为偶数，取单字；若为奇数，取 2 字）
                gap = p_laopu - (p_shen + 3)
                if gap % 2 == 0:
                    f_start = p_shen + 3 + gap // 2 - 1
                    f_end = f_start + 2
                else:
                    f_start = p_shen + 3 + gap // 2
                    f_end = f_start + 1
                self.assertEqual(f_start - (p_shen + 3), p_laopu - f_end)
                ch1_id = store.records(run_id, "chapter")[0]["id"]
                t0_id = store.rows("SELECT task_id FROM facts WHERE describe_id=? LIMIT 1", (did,))[0]["task_id"]
                store.execute(
                    "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (did, "f_tie_1", by_name["沈公子"], "voice", "沉稳", text[f_start:f_end],
                     f_start, f_end, ch1_id, "narration", None, t0_id),
                )
                store.execute("INSERT INTO fact_verdicts VALUES(?,?,?,?,?)", (did, "f_tie_1", "keep", None, "vt_tie"))

                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                apply_correction(
                    store, run_id, "split-surface",
                    {"entity_id": by_name["沈公子"], "surface": "老仆"},
                    source="user", expect_revision=rev0, output_root=out_root,
                )
                doc = export_describe_run(store, did)
                self.assertEqual(doc["rebind_summary"]["ambiguous_facts"], 1)
                shen = next(e for e in doc["entities"] if e["entity_id"] == by_name["沈公子"])
                self.assertIn("f_tie_1", {f["id"] for f in shen["facts"]})
            finally:
                store.close()

    def test_34_sql_tables_untouched_and_revoke_restores_exact_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                out_before = root / "before.json"
                export_describe_run(store, did, out=out_before)
                sha_before = _sha256_bytes(out_before.read_bytes())
                raw_facts_before = store.rows("SELECT * FROM facts WHERE describe_id=? ORDER BY id", (did,))
                raw_events_before = store.rows("SELECT * FROM state_events WHERE describe_id=? ORDER BY id", (did,))

                rev = build_export_bundle(store, run_id)["result"]["result_revision"]
                cids = []
                for op, args in [
                    ("merge", {"target_entity_id": by_name["伯爵"], "merged_entity_id": by_name["唐太斯"]}),
                    ("split-surface", {"entity_id": by_name["沈公子"], "surface": "老仆"}),
                    ("not-entity", {"mention_id": "m_shijie_1"}),
                ]:
                    r = apply_correction(
                        store, run_id, op, args,
                        source="user", expect_revision=rev, output_root=out_root,
                    )
                    cids.append(r["correction_id"])
                    rev = r["result_revision"]

                out_r1 = root / "rebind1.json"
                out_r2 = root / "rebind2.json"
                export_describe_run(store, did, out=out_r1)
                export_describe_run(store, did, out=out_r2)
                self.assertEqual(_sha256_bytes(out_r1.read_bytes()), _sha256_bytes(out_r2.read_bytes()))
                self.assertNotEqual(_sha256_bytes(out_r1.read_bytes()), sha_before)

                # 原 SQL 表零改动
                self.assertEqual(store.rows("SELECT * FROM facts WHERE describe_id=? ORDER BY id", (did,)), raw_facts_before)
                self.assertEqual(store.rows("SELECT * FROM state_events WHERE describe_id=? ORDER BY id", (did,)), raw_events_before)

                # 逆序 revoke 全部更正后逐字节 SHA-256 100% 复原
                for cid in reversed(cids):
                    r = apply_correction(
                        store, run_id, "revoke", {"correction_id": cid},
                        source="user", expect_revision=rev, output_root=out_root,
                    )
                    rev = r["result_revision"]
                out_revoked = root / "revoked.json"
                export_describe_run(store, did, out=out_revoked)
                self.assertEqual(_sha256_bytes(out_revoked.read_bytes()), sha_before)
            finally:
                store.close()

    def test_35_describe_created_after_c1_rebinds_only_delta_c2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, _ = self._build_novel_fixture(root)
            store = Store(db_path)
            try:
                rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                r1 = apply_correction(
                    store, run_id, "not-entity", {"mention_id": "m_shijie_1"},
                    source="user", expect_revision=rev0, output_root=out_root,
                )

                def desc_handler(stage, st, _n):
                    if stage == "describe-d1":
                        chunk = st.get("text", "")
                        eids_here = {e["entity_id"] for e in st["entities"]}
                        facts = []
                        if by_name["唐太斯"] in eids_here and "年方十九岁" in chunk:
                            facts.append({"entity_id": by_name["唐太斯"], "facet": "age", "value": "十九岁", "quote": "唐太斯年方十九岁", "source": "narration"})
                        if by_name["伯爵"] in eids_here and "四十岁中年" in chunk:
                            facts.append({"entity_id": by_name["伯爵"], "facet": "age", "value": "四十岁", "quote": "伯爵已是四十岁中年", "source": "narration"})
                        return {"facts": facts, "events": []}
                    return {"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None, **({"stage_split": False} if stage == "describe-event-verify" else {})} for it in st["items"]]}

                fmodels = FakeModels(SERVICES, desc_handler)
                res = run_describe(
                    store, run_id, services=SERVICES, models=fmodels,
                    top_k=20, kinds="person,place,object,ability", views=1, entities_per_request=0, verify=1,
                )
                did2 = res["describe_id"]
                # 创建后无新更正 -> 不触发重绑
                doc_same = export_describe_run(store, did2)
                self.assertNotIn("rebind_summary", doc_same)
                # 再执行 c2 (merge 伯爵 <- 唐太斯) -> 仅重绑增量 c2
                apply_correction(
                    store, run_id, "merge",
                    {"target_entity_id": by_name["伯爵"], "merged_entity_id": by_name["唐太斯"]},
                    source="user", expect_revision=r1["result_revision"], output_root=out_root,
                )
                doc_delta = export_describe_run(store, did2)
                self.assertEqual(doc_delta["rebind_summary"]["merged_entities"], 1)
                self.assertEqual(doc_delta["rebind_summary"]["dropped_entities"], 0)
            finally:
                store.close()

    def test_36_cli_describe_rebind_zero_credentials_and_receipt_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did = self._build_novel_fixture(root)
            out_zero = root / "rebind_zero.json"
            out_merged = root / "rebind_merged.json"
            # 清空所有模型凭证环境变量并指向不存在的 --config，验证 100% 无需凭证
            clean_env = {
                k: v for k, v in os.environ.items()
                if not k.startswith(("DEEPSEEK_", "TYPESAFE_", "ENTITYTRACE_OPENAI_", "OPENAI_", "GENERATOR_"))
            }
            with patch.dict(os.environ, clean_env, clear=True):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = cli_main([
                        "describe", run_id, "--db", str(db_path),
                        "--config", str(root / "missing.env"),
                        "--rebind", did, "--out", str(out_zero),
                    ])
                self.assertEqual(rc, 0)
                receipt0 = json.loads(buf.getvalue().strip())
                self.assertTrue(receipt0["ok"])
                self.assertTrue(receipt0["rebound"])
                self.assertEqual(receipt0["describe_id"], did)
                self.assertEqual(receipt0["counts"]["tasks"], 0)
                self.assertEqual(receipt0["rebind_summary"]["active_corrections"], 0)
                self.assertTrue(receipt0["result_revision"])

                # 执行 merge 更正后再跑 CLI --rebind
                store = Store(db_path)
                try:
                    apply_correction(
                        store, run_id, "merge",
                        {"target_entity_id": by_name["伯爵"], "merged_entity_id": by_name["唐太斯"]},
                        source="user", expect_revision=receipt0["result_revision"], output_root=out_root,
                    )
                finally:
                    store.close()

                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    rc2 = cli_main([
                        "describe", run_id, "--db", str(db_path),
                        "--config", str(root / "missing.env"),
                        "--rebind", did, "--out", str(out_merged),
                    ])
                self.assertEqual(rc2, 0)
                receipt1 = json.loads(buf2.getvalue().strip())
                self.assertTrue(receipt1["ok"])
                self.assertTrue(receipt1["rebound"])
                self.assertNotEqual(receipt1["result_revision"], receipt0["result_revision"])
                self.assertEqual(receipt1["rebind_summary"]["merged_entities"], 1)
                self.assertEqual(receipt1["counts"]["multi_stage_entities"], 1)
                exported = json.loads(out_merged.read_text(encoding="utf-8"))
                self.assertEqual(exported["rebound_from_describe_id"], did)

    def test_37_cli_describe_rebind_rejects_missing_or_mismatched_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, _, run_id, _, did = self._build_novel_fixture(root)
            for bad_argv in [
                ["describe", run_id, "--db", str(db_path), "--rebind", "nonexistent-describe-id"],
                ["describe", "wrong-run-id", "--db", str(db_path), "--rebind", did],
            ]:
                out_buf, err_buf = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                    rc = cli_main(bad_argv)
                self.assertEqual(rc, 1)
                payload = json.loads(out_buf.getvalue().strip())
                self.assertEqual(payload["error"]["category"], "input")
                self.assertFalse(payload["error"]["retryable"])

    def test_38_build_describe_id_and_prompts_invariant(self):
        # build_describe_id 哈希完全保持不变
        d_id = build_describe_id(
            20, ["person", "place", "object", "ability"], ["e1", "e2"],
            "run_fixed_001", "deepseek", "deepseek-chat",
            views=3, entities_per_request=4, verify=1, persistence=0,
        )
        self.assertEqual(len(d_id), 64)
        self.assertEqual(
            d_id,
            build_describe_id(
                20, ["person", "place", "object", "ability"], ["e1", "e2"],
                "run_fixed_001", "deepseek", "deepseek-chat",
                views=3, entities_per_request=4, verify=1, persistence=0,
            ),
        )
        # 提示词常量存在且非空
        for prompt in (D1_PROMPT, VERIFY_PROMPT, EVENT_VERIFY_PROMPT, PERSISTENCE_PROMPT):
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 50)


if __name__ == "__main__":
    unittest.main()
