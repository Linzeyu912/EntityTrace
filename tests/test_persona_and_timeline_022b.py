"""跨马甲同人待审提示、正文跨年阶段切分与时间序场景造型快照专项回归测试。"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import entitytrace.describe as describe_mod
import entitytrace.life_stages as life_stages_mod
from entitytrace.chapters import build_chapters
from entitytrace.cli import main as cli_main
from entitytrace.corrections import apply_correction
from entitytrace.describe import (
    D1_PROMPT,
    DESCRIBE_VERSION,
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
    TRIGGER_SOURCE_LIFE_STAGE,
    TRIGGER_SOURCE_TIME_JUMP,
    detect_narrative_time_jumps,
)
from entitytrace.persona_hints import propose_persona_link_hints
from entitytrace.pipeline import build_export_bundle
from entitytrace.roster import VERSION as ROSTER_VERSION, resolve_roster
from entitytrace.storage import Store
from entitytrace.text import address, import_text
from entitytrace.timeline_profiles import (
    build_entity_timeline_profiles,
    costumes_conflict,
    count_costume_conflict_pairs,
    parse_costume_signature,
)
from entitytrace.util import dumps
from tests.test_describe import FakeModels, SERVICES
from tests.test_describe_v6 import _assert_stage_invariants


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mk_chapters(n: int, stride: int = 200) -> list[dict]:
    return [
        {
            "id": f"ch{i}",
            "kind": "chapter",
            "ordinal": i,
            "start": (i - 1) * stride,
            "body_start": (i - 1) * stride,
            "end": i * stride,
        }
        for i in range(1, n + 1)
    ]


def _mk_fact(
    fid: str,
    facet: str,
    value: str,
    quote: str,
    start: int,
    chapter_id: str,
    *,
    entity_id: str = "e1",
    persistence: str | None = None,
    holder_entity_id: str | None = None,
) -> dict:
    item = {
        "id": fid,
        "entity_id": entity_id,
        "facet": facet,
        "value": value,
        "quote": quote,
        "start": start,
        "end": start + len(quote),
        "chapter_id": chapter_id,
        "verdict": "keep",
        "persistence": persistence,
    }
    if holder_entity_id is not None:
        item["holder_entity_id"] = holder_entity_id
    return item


def _mk_event(
    eid: str,
    event_type: str,
    value: str,
    quote: str,
    start: int,
    chapter_id: str,
    *,
    stage_split: bool = False,
    entity_id: str = "e1",
) -> dict:
    return {
        "id": eid,
        "entity_id": entity_id,
        "event_type": event_type,
        "value": value,
        "quote": quote,
        "start": start,
        "end": start + len(quote),
        "chapter_id": chapter_id,
        "stage_split": stage_split,
    }


def _assert_timeline_invariants(case: unittest.TestCase, ent: dict, text: str | None = None) -> None:
    base_ids = {str(f["id"]) for f in ent["base_traits"]}
    total_apps = 0
    for m, st in enumerate(ent["stages"]):
        if m == 0:
            case.assertEqual(st["inherited_traits"], [])
        st_fact_ids = {str(f["id"]) for f in st["facts"]}
        own_trait_ids = st_fact_ids & base_ids
        seen_scene_ids: set[str] = set()
        seen_own_in_active: set[str] = set()
        apps = st["scene_appearances"]
        total_apps += len(apps)
        prev_app = None
        for r_idx, app in enumerate(apps, start=1):
            case.assertEqual(
                set(app),
                {
                    "appearance_index",
                    "stage_index",
                    "appearance_code",
                    "from_chapter",
                    "to_chapter",
                    "start",
                    "end",
                    "trigger_kind",
                    "trigger_summary",
                    "scene_facts",
                    "active_traits",
                    "scene_events",
                },
            )
            case.assertEqual(app["appearance_index"], r_idx)
            case.assertEqual(app["stage_index"], st["stage_index"])
            case.assertEqual(app["appearance_code"], f"S{st['stage_index']}-L{r_idx}")
            case.assertLessEqual(app["start"], app["end"])
            if prev_app is not None:
                case.assertLessEqual(prev_app["end"], app["start"])
                if prev_app["from_chapter"] is not None and app["from_chapter"] is not None:
                    case.assertLessEqual(prev_app["from_chapter"], app["from_chapter"])
            prev_app = app
            for sf in app["scene_facts"]:
                sfid = str(sf["id"])
                case.assertNotIn(sfid, seen_scene_ids)
                case.assertNotIn(sfid, own_trait_ids)
                seen_scene_ids.add(sfid)
            for at in app["active_traits"]:
                atid = str(at["id"])
                if atid in own_trait_ids:
                    seen_own_in_active.add(atid)
            if text is not None:
                for item in [*app["scene_facts"], *app["active_traits"], *app["scene_events"]]:
                    s, e = int(item["start"]), int(item["end"])
                    case.assertEqual(text[s:e], item["quote"])
        case.assertEqual(seen_scene_ids | own_trait_ids, st_fact_ids)
        case.assertEqual(seen_own_in_active, own_trait_ids)
    case.assertEqual(ent["total_scene_appearances"], total_apps)


class _RosterModels:
    def __init__(self, handlers):
        self.handlers = handlers

    def choices(self, *args, **kwargs):
        raise AssertionError("离线测试不应调用 choices")

    def generate(self, system, state, stage, validate, *args, **kwargs):
        val, req_id = self.handlers[stage](state)
        validate(val)
        return val, req_id


class PersonaHintsTests(unittest.TestCase):
    """跨马甲同人待审提示三路信号、否决守卫与更正感知测试。"""

    def test_01_csig1_reviews_conflict_block_hit(self):
        txt = "林墨与青云道人曾在北山修行，后以青云道人行走江湖。"
        chs = [{"id": "ch1", "kind": "chapter", "start": 0, "end": len(txt)}]
        snap = {
            "entities": [
                {"id": "p1", "name": "林墨", "names": ["林墨"], "kind": "person", "mention_count": 20},
                {"id": "p2", "name": "青云道人", "names": ["青云道人"], "kind": "person", "mention_count": 10},
            ]
        }
        ro = {
            "entities": snap["entities"],
            "mentions": [
                {"id": "m1", "entity_id": "p1", "text": "林墨", "start": 0, "end": 2, "chapter_id": "ch1"},
                {"id": "m2", "entity_id": "p2", "text": "青云道人", "start": 3, "end": 7, "chapter_id": "ch1"},
            ],
            "reviews": [{
                "stage": "roster-c2-verify",
                "reason": "C2 名册补合并连通块内存在冲突，整组暂不合并",
                "left_name": "林墨",
                "right_name": "青云道人",
                "member_names": ["林墨", "青云道人"],
                "evidence": [{"start": 0, "end": 12}],
            }],
        }
        hints = propose_persona_link_hints(snap, ro, None, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["target_entity_id"], "p1")
        self.assertEqual(hints[0]["candidate_entity_id"], "p2")
        self.assertIn("c2_blocked_same", hints[0]["signals"])
        self.assertEqual(hints[0]["confidence_tier"], "high")
        self.assertEqual(txt[hints[0]["evidence_start"]:hints[0]["evidence_end"]], hints[0]["evidence_quote"])

    def test_02_csig1_explicit_c2_blocked_pairs_hit(self):
        txt = "第一章。陆羽初入师门。\n第五章。无极客独行天下。"
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 12},
            {"id": "ch2", "kind": "chapter", "start": 12, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "a", "name": "陆羽", "names": ["陆羽"], "kind": "person", "mention_count": 12},
                {"id": "b", "name": "无极客", "names": ["无极客"], "kind": "person", "mention_count": 8},
            ]
        }
        ro = {
            "mentions": [
                {"id": "m1", "entity_id": "a", "text": "陆羽", "start": 4, "end": 6, "chapter_id": "ch1"},
                {"id": "m2", "entity_id": "b", "text": "无极客", "start": 16, "end": 19, "chapter_id": "ch2"},
            ],
            "c2_blocked_pairs": [
                {"left_entity_id": "a", "right_entity_id": "b", "evidence_start": 4, "evidence_end": 11},
            ],
        }
        hints = propose_persona_link_hints(snap, ro, None, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["signals"], ["c2_blocked_same"])
        self.assertEqual(hints[0]["target_chapter_span"], [1, 1])
        self.assertEqual(hints[0]["candidate_chapter_span"], [2, 2])
        self.assertEqual(txt[hints[0]["evidence_start"]:hints[0]["evidence_end"]], hints[0]["evidence_quote"])

    def test_03_csig2_zh_forward_alias_reveal(self):
        txt = "第一回。张铁在山村长大。\n第十回。那位剑客厉飞，原名正是张铁，后来改名厉飞走南闯北。"
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 13},
            {"id": "ch2", "kind": "chapter", "start": 13, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "p1", "name": "张铁", "names": ["张铁"], "kind": "person", "mention_count": 12},
                {"id": "p2", "name": "厉飞", "names": ["厉飞"], "kind": "person", "mention_count": 8},
            ]
        }
        ro = {
            "mentions": [
                {"id": "m1", "entity_id": "p1", "text": "张铁", "start": 4, "end": 6, "chapter_id": "ch1"},
                {"id": "m2", "entity_id": "p2", "text": "厉飞", "start": 21, "end": 23, "chapter_id": "ch2"},
            ]
        }
        hints = propose_persona_link_hints(snap, ro, None, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["signals"], ["explicit_alias_text"])
        self.assertEqual(hints[0]["confidence_tier"], "high")
        self.assertEqual(hints[0]["chapter_overlap_ratio"], 0.0)
        self.assertEqual(txt[hints[0]["evidence_start"]:hints[0]["evidence_end"]], hints[0]["evidence_quote"])

    def test_04_csig2_zh_modifier_alias_reveal(self):
        txt = "第一回。韩明在北地避祸。\n第八回。那位化名“白龙”的韩明悄然归来。"
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 13},
            {"id": "ch2", "kind": "chapter", "start": 13, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "p1", "name": "韩明", "names": ["韩明"], "kind": "person", "mention_count": 10},
                {"id": "p2", "name": "白龙", "names": ["白龙"], "kind": "person", "mention_count": 6},
            ]
        }
        hints = propose_persona_link_hints(snap, None, None, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["signals"], ["explicit_alias_text"])
        self.assertIn("化名", hints[0]["evidence_quote"])

    def test_05_csig2_en_alias_reveal(self):
        txt = (
            "Chapter 1.\nArthur Pendelton left the town.\n"
            "Chapter 5.\nThe stranger living under the name of Marcus Vance was formerly known as Arthur Pendelton."
        )
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 43},
            {"id": "ch2", "kind": "chapter", "start": 43, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "e1", "name": "Arthur Pendelton", "names": ["Arthur Pendelton"], "kind": "person", "mention_count": 15},
                {"id": "e2", "name": "Marcus Vance", "names": ["Marcus Vance"], "kind": "person", "mention_count": 9},
            ]
        }
        hints = propose_persona_link_hints(snap, None, None, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["signals"], ["explicit_alias_text"])
        self.assertEqual(txt[hints[0]["evidence_start"]:hints[0]["evidence_end"]], hints[0]["evidence_quote"])

    def test_06_csig2_kinship_veto_blocks_false_positive(self):
        txt = "梁朝大之子名叫梁贵，表字仲文；凌宗客就是梁朝大的妻舅，表字祈安。"
        chs = [{"id": "ch1", "kind": "chapter", "start": 0, "end": len(txt)}]
        snap = {
            "entities": [
                {"id": "a", "name": "梁朝大", "names": ["梁朝大"], "kind": "person", "mention_count": 20},
                {"id": "b", "name": "梁贵", "names": ["梁贵"], "kind": "person", "mention_count": 15},
                {"id": "c", "name": "凌宗客", "names": ["凌宗客"], "kind": "person", "mention_count": 14},
                {"id": "d", "name": "祈安", "names": ["祈安"], "kind": "person", "mention_count": 5},
            ]
        }
        self.assertEqual(propose_persona_link_hints(snap, None, None, txt, chs), [])

    def test_07_csig2_masked_name_with_kinship_char_not_vetoed(self):
        txt = "孙景行早年在家读书，表字祈伯，后游学四方。"
        chs = [{"id": "ch1", "kind": "chapter", "start": 0, "end": len(txt)}]
        snap = {
            "entities": [
                {"id": "a", "name": "孙景行", "names": ["孙景行"], "kind": "person", "mention_count": 20},
                {"id": "b", "name": "祈伯", "names": ["祈伯"], "kind": "person", "mention_count": 8},
            ]
        }
        hints = propose_persona_link_hints(snap, None, None, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["target_name"], "孙景行")
        self.assertEqual(hints[0]["candidate_name"], "祈伯")

    def test_08_csig2_master_disciple_rival_and_jishi_concessive_veto(self):
        txt = (
            "柳玉拜在韩立门下当弟子，即便改名柳眉仍是韩立徒弟。"
            "韩立冷笑，即是南陇侯不说，他也定要击杀宿敌南陇侯。"
        )
        chs = [{"id": "ch1", "kind": "chapter", "start": 0, "end": len(txt)}]
        snap = {
            "entities": [
                {"id": "h", "name": "韩立", "names": ["韩立"], "kind": "person", "mention_count": 50},
                {"id": "ly1", "name": "柳玉", "names": ["柳玉"], "kind": "person", "mention_count": 10},
                {"id": "ly2", "name": "柳眉", "names": ["柳眉"], "kind": "person", "mention_count": 10},
                {"id": "n", "name": "南陇侯", "names": ["南陇侯"], "kind": "person", "mention_count": 15},
            ]
        }
        self.assertEqual(propose_persona_link_hints(snap, None, None, txt, chs), [])

    def test_09_csig2_between_interaction_and_right_connector_veto(self):
        txt = "陈风对改名为赤霄的赵雷说道。吴用道号加亮先生，人称智多星宋江。"
        chs = [{"id": "ch1", "kind": "chapter", "start": 0, "end": len(txt)}]
        snap = {
            "entities": [
                {"id": "a", "name": "陈风", "names": ["陈风"], "kind": "person", "mention_count": 20},
                {"id": "b", "name": "赤霄", "names": ["赤霄"], "kind": "person", "mention_count": 10},
                {"id": "c", "name": "吴用", "names": ["吴用"], "kind": "person", "mention_count": 15},
                {"id": "d", "name": "宋江", "names": ["宋江"], "kind": "person", "mention_count": 30},
            ]
        }
        self.assertEqual(propose_persona_link_hints(snap, None, None, txt, chs), [])

    def test_10_csig3_shared_object_holder_complementary_chapters(self):
        txt = "ch1陆沉手持青冥剑巡山。ch2陆沉腰悬青冥剑。ch5无名客携青冥剑隐居。ch6无名客抚摸青冥剑。"
        chs = [
            {"id": "c1", "kind": "chapter", "start": 0, "end": 13},
            {"id": "c2", "kind": "chapter", "start": 13, "end": 24},
            {"id": "c3", "kind": "chapter", "start": 24, "end": 24},
            {"id": "c4", "kind": "chapter", "start": 24, "end": 24},
            {"id": "c5", "kind": "chapter", "start": 24, "end": 37},
            {"id": "c6", "kind": "chapter", "start": 37, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "p1", "name": "陆沉", "names": ["陆沉"], "kind": "person", "mention_count": 10},
                {"id": "p2", "name": "无名客", "names": ["无名客"], "kind": "person", "mention_count": 8},
            ]
        }
        ro = {
            "mentions": [
                {"id": "m1", "entity_id": "p1", "text": "陆沉", "start": 3, "end": 5, "chapter_id": "c1"},
                {"id": "m2", "entity_id": "p1", "text": "陆沉", "start": 16, "end": 18, "chapter_id": "c2"},
                {"id": "m3", "entity_id": "p2", "text": "无名客", "start": 27, "end": 30, "chapter_id": "c5"},
                {"id": "m4", "entity_id": "p2", "text": "无名客", "start": 40, "end": 43, "chapter_id": "c6"},
            ]
        }
        q1 = "陆沉手持青冥剑"
        q2 = "无名客携青冥剑"
        desc = {
            "entities": [{
                "entity_id": "obj1",
                "kind": "object",
                "name": "青冥剑",
                "state_events": [],
                "facts": [
                    _mk_fact("f1", "holder", "陆沉", q1, txt.index(q1), "c1", entity_id="obj1", holder_entity_id="p1"),
                    _mk_fact("f2", "holder", "无名客", q2, txt.index(q2), "c5", entity_id="obj1", holder_entity_id="p2"),
                ],
            }]
        }
        hints = propose_persona_link_hints(snap, ro, desc, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["signals"], ["complementary_chapters_shared_anchor"])
        self.assertEqual(hints[0]["confidence_tier"], "medium")
        self.assertEqual(txt[hints[0]["evidence_start"]:hints[0]["evidence_end"]], hints[0]["evidence_quote"])

    def test_11_csig3_object_owner_change_and_transfer_verb_veto(self):
        txt = "ch1陆沉手持青冥剑。ch2陆沉携剑。ch5敌将夺走青冥剑。ch6敌将持剑。"
        chs = [
            {"id": "c1", "kind": "chapter", "start": 0, "end": 11},
            {"id": "c2", "kind": "chapter", "start": 11, "end": 19},
            {"id": "c5", "kind": "chapter", "start": 19, "end": 30},
            {"id": "c6", "kind": "chapter", "start": 30, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "p1", "name": "陆沉", "names": ["陆沉"], "kind": "person", "mention_count": 10},
                {"id": "p2", "name": "敌将", "names": ["敌将"], "kind": "person", "mention_count": 8},
            ]
        }
        ro = {
            "mentions": [
                {"id": "m1", "entity_id": "p1", "text": "陆沉", "start": 3, "end": 5, "chapter_id": "c1"},
                {"id": "m2", "entity_id": "p1", "text": "陆沉", "start": 14, "end": 16, "chapter_id": "c2"},
                {"id": "m3", "entity_id": "p2", "text": "敌将", "start": 22, "end": 24, "chapter_id": "c5"},
                {"id": "m4", "entity_id": "p2", "text": "敌将", "start": 33, "end": 35, "chapter_id": "c6"},
            ]
        }
        q1, q2 = "陆沉手持青冥剑", "敌将夺走青冥剑"
        desc_owner_change = {
            "entities": [{
                "entity_id": "obj1", "kind": "object", "name": "青冥剑",
                "state_events": [_mk_event("ev1", "owner_change", "易主", q2, txt.index(q2), "c5", stage_split=True)],
                "facts": [
                    _mk_fact("f1", "holder", "陆沉", q1, txt.index(q1), "c1", entity_id="obj1", holder_entity_id="p1"),
                    _mk_fact("f2", "holder", "敌将", q2, txt.index(q2), "c5", entity_id="obj1", holder_entity_id="p2"),
                ],
            }]
        }
        self.assertEqual(propose_persona_link_hints(snap, ro, desc_owner_change, txt, chs), [])
        # 即使无 owner_change 事件，引文含“夺”字同样被转移词否决
        desc_transfer_verb = copy.deepcopy(desc_owner_change)
        desc_transfer_verb["entities"][0]["state_events"] = []
        self.assertEqual(propose_persona_link_hints(snap, ro, desc_transfer_verb, txt, chs), [])

    def test_12_csig3_rare_feature_complementary_and_non_unique_veto(self):
        txt = "c1方寒左眉红痣。c2方寒练剑。c5幽冥客左眉红痣。c6幽冥客饮茶。c8路人左眉红痣。"
        chs = [
            {"id": "c1", "kind": "chapter", "start": 0, "end": 9},
            {"id": "c2", "kind": "chapter", "start": 9, "end": 16},
            {"id": "c5", "kind": "chapter", "start": 16, "end": 26},
            {"id": "c6", "kind": "chapter", "start": 26, "end": 34},
            {"id": "c8", "kind": "chapter", "start": 34, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "p1", "name": "方寒", "names": ["方寒"], "kind": "person", "mention_count": 10},
                {"id": "p2", "name": "幽冥客", "names": ["幽冥客"], "kind": "person", "mention_count": 8},
                {"id": "p3", "name": "路人丙", "names": ["路人丙"], "kind": "person", "mention_count": 6},
            ]
        }
        ro = {
            "mentions": [
                {"id": "m1", "entity_id": "p1", "text": "方寒", "start": 2, "end": 4, "chapter_id": "c1"},
                {"id": "m2", "entity_id": "p1", "text": "方寒", "start": 11, "end": 13, "chapter_id": "c2"},
                {"id": "m3", "entity_id": "p2", "text": "幽冥客", "start": 18, "end": 21, "chapter_id": "c5"},
                {"id": "m4", "entity_id": "p2", "text": "幽冥客", "start": 28, "end": 31, "chapter_id": "c6"},
            ]
        }
        q1, q2, q3 = "方寒左眉红痣", "幽冥客左眉红痣", "路人左眉红痣"
        desc_two = {
            "entities": [
                {"entity_id": "p1", "kind": "person", "name": "方寒",
                 "facts": [_mk_fact("f1", "feature", "左眉红痣", q1, txt.index(q1), "c1", entity_id="p1")]},
                {"entity_id": "p2", "kind": "person", "name": "幽冥客",
                 "facts": [_mk_fact("f2", "feature", "左眉红痣", q2, txt.index(q2), "c5", entity_id="p2")]},
            ]
        }
        hints = propose_persona_link_hints(snap, ro, desc_two, txt, chs)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["signals"], ["complementary_chapters_shared_anchor"])

        # 若全书 3 人均具该特征则不再罕见 -> 拦截
        desc_three = copy.deepcopy(desc_two)
        desc_three["entities"].append({
            "entity_id": "p3", "kind": "person", "name": "路人丙",
            "facts": [_mk_fact("f3", "feature", "左眉红痣", q3, txt.index(q3), "c8", entity_id="p3")],
        })
        self.assertEqual(propose_persona_link_hints(snap, ro, desc_three, txt, chs), [])

    def test_13_persona_hints_empty_inputs_and_max_hints_cap(self):
        self.assertEqual(propose_persona_link_hints(None, None, None, "abc", []), [])
        self.assertEqual(propose_persona_link_hints({"entities": []}, None, None, "", []), [])
        self.assertEqual(
            propose_persona_link_hints(
                {"entities": [{"id": "p1", "name": "甲乙", "kind": "person", "mention_count": 5}]},
                None, None, "甲乙在此", [],
            ),
            [],
        )
        # 多组提示受 max_hints 截断
        pairs = [("甲一", "乙一"), ("甲二", "乙二"), ("甲三", "乙三"), ("甲四", "乙四")]
        txt = "\n".join(f"{a}原名{b}。" for a, b in pairs)
        chs = [{"id": "c1", "kind": "chapter", "start": 0, "end": len(txt)}]
        ents = []
        for idx, (a, b) in enumerate(pairs):
            ents.append({"id": f"a{idx}", "name": a, "names": [a], "kind": "person", "mention_count": 20 - idx})
            ents.append({"id": f"b{idx}", "name": b, "names": [b], "kind": "person", "mention_count": 5})
        hints = propose_persona_link_hints({"entities": ents}, None, None, txt, chs, max_hints=2)
        self.assertEqual(len(hints), 2)

    def test_14_persona_hints_correction_aware_disappears_after_merge(self):
        txt = "第一回。张铁出场。\n第五回。厉飞原名张铁。"
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 10},
            {"id": "ch2", "kind": "chapter", "start": 10, "end": len(txt)},
        ]
        snap_before = {
            "entities": [
                {"id": "p1", "name": "张铁", "names": ["张铁"], "kind": "person", "mention_count": 12},
                {"id": "p2", "name": "厉飞", "names": ["厉飞"], "kind": "person", "mention_count": 8},
            ]
        }
        ro = {
            "mentions": [
                {"id": "m1", "entity_id": "p1", "text": "张铁", "start": 4, "end": 6, "chapter_id": "ch1"},
                {"id": "m2", "entity_id": "p2", "text": "厉飞", "start": 14, "end": 16, "chapter_id": "ch2"},
            ]
        }
        self.assertEqual(len(propose_persona_link_hints(snap_before, ro, None, txt, chs)), 1)

        # 合体后 p2 不再活跃，且 mentions_override 将 m2 归入 p1 -> 提示自动消隐
        snap_after = {
            "entities": [
                {"id": "p1", "name": "张铁", "names": ["张铁", "厉飞"], "kind": "person", "mention_count": 20},
            ]
        }
        merged_mentions = [
            {"id": "m1", "entity_id": "p1", "text": "张铁", "start": 4, "end": 6, "chapter_id": "ch1"},
            {"id": "m2", "entity_id": "p1", "text": "厉飞", "start": 14, "end": 16, "chapter_id": "ch2"},
        ]
        self.assertEqual(
            propose_persona_link_hints(snap_after, ro, None, txt, chs, mentions_override=merged_mentions),
            [],
        )

    def test_15_persona_hints_fixed_14_keys_and_verbatim_invariant(self):
        txt = "第一回。周显在京城。\n第十回。周显表字仲明，游历江南。"
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 11},
            {"id": "ch2", "kind": "chapter", "start": 11, "end": len(txt)},
        ]
        snap = {
            "entities": [
                {"id": "p1", "name": "周显", "names": ["周显"], "kind": "person", "mention_count": 10},
                {"id": "p2", "name": "仲明", "names": ["仲明"], "kind": "person", "mention_count": 5},
            ]
        }
        hints = propose_persona_link_hints(snap, None, None, txt, chs)
        self.assertEqual(len(hints), 1)
        expected_keys = {
            "hint_id", "target_entity_id", "target_name", "candidate_entity_id",
            "candidate_name", "signals", "confidence_tier", "target_chapter_span",
            "candidate_chapter_span", "chapter_overlap_ratio", "evidence_quote",
            "evidence_start", "evidence_end", "suggested_cli",
        }
        self.assertEqual(set(hints[0]), expected_keys)
        s, e = hints[0]["evidence_start"], hints[0]["evidence_end"]
        self.assertLessEqual(len(hints[0]["evidence_quote"]), 160)
        self.assertEqual(txt[s:e], hints[0]["evidence_quote"])


class NarrativeTimeJumpTests(unittest.TestCase):
    """正文显式多年时间跨度跳跃切分与守卫回归测试。"""

    def test_16_time_jump_zh_chapter_head_window(self):
        c1 = "第一章\n黄天化身穿青布道袍，在山中习武。\n"
        c2 = "第三十二章\n光阴似箭，不觉又是十有三年。黄天化如今头戴攒顶盔，身披金锁甲，下山助父。\n"
        txt = c1 + c2
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "body_start": 4, "end": len(c1)},
            {"id": "ch32", "kind": "chapter", "start": len(c1), "body_start": len(c1) + 6, "end": len(txt)},
        ]
        q1, q2 = "身穿青布道袍", "身披金锁甲"
        facts = [
            _mk_fact("f1", "costume", "青布道袍", q1, txt.index(q1), "ch1"),
            _mk_fact("f2", "costume", "金锁甲", q2, txt.index(q2), "ch32"),
        ]
        cuts = detect_narrative_time_jumps(facts, [], chs, txt)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["trigger_source"], TRIGGER_SOURCE_TIME_JUMP)
        self.assertEqual(cuts[0]["jump_years"], 13.0)
        self.assertEqual(txt[cuts[0]["start"]:cuts[0]["end"]], cuts[0]["quote"])

    def test_17_time_jump_zh_local_paragraph_window(self):
        pad = "背景铺垫文字。" * 70 + "\n"
        c1 = "第一章\n韩立身着灰布短打。\n"
        c2 = "第五百三十四章\n" + pad + "他静坐洞府，闭关了二十多年才出关，如今换上了一身青色长袍。\n"
        txt = c1 + c2
        chs = [
            {"id": "ch1", "kind": "chapter", "start": 0, "body_start": 4, "end": len(c1)},
            {"id": "ch534", "kind": "chapter", "start": len(c1), "body_start": len(c1) + 8, "end": len(txt)},
        ]
        q1, q2 = "身着灰布短打", "一身青色长袍"
        facts = [
            _mk_fact("f1", "costume", "灰布短打", q1, txt.index(q1), "ch1"),
            _mk_fact("f2", "costume", "青色长袍", q2, txt.index(q2), "ch534"),
        ]
        cuts = detect_narrative_time_jumps(facts, [], chs, txt)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["jump_years"], 20.0)
        self.assertIn("闭关了二十多年才出关", cuts[0]["quote"])
        self.assertEqual(txt[cuts[0]["start"]:cuts[0]["end"]], cuts[0]["quote"])

    def test_18_time_jump_en_years_later_and_after_years(self):
        c1 = "Chapter 1\nEdmund wore a rough sailor jacket.\n"
        c2 = "Chapter 14\nFourteen years later, Edmund appeared in a black velvet coat.\n"
        txt = c1 + c2
        chs = [
            {"id": "c1", "kind": "chapter", "start": 0, "body_start": 10, "end": len(c1)},
            {"id": "c14", "kind": "chapter", "start": len(c1), "body_start": len(c1) + 11, "end": len(txt)},
        ]
        q1, q2 = "rough sailor jacket", "black velvet coat"
        facts = [
            _mk_fact("f1", "costume", "水手夹克", q1, txt.index(q1), "c1"),
            _mk_fact("f2", "costume", "黑丝绒外套", q2, txt.index(q2), "c14"),
        ]
        cuts = detect_narrative_time_jumps(facts, [], chs, txt)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["jump_years"], 14.0)
        self.assertEqual(txt[cuts[0]["start"]:cuts[0]["end"]], cuts[0]["quote"])

    def test_19_time_jump_under_5_years_and_over_120_rejected(self):
        c1 = "第一章\n林宇身穿青衫。\n"
        c2 = "第二章\n转眼过了三年，林宇仍穿青衫。\n"
        c3 = "第三章\n闭关了二百年后，林宇身穿白袍。\n"
        txt = c1 + c2 + c3
        chs = [
            {"id": "c1", "kind": "chapter", "start": 0, "body_start": 4, "end": len(c1)},
            {"id": "c2", "kind": "chapter", "start": len(c1), "body_start": len(c1) + 4, "end": len(c1) + len(c2)},
            {"id": "c3", "kind": "chapter", "start": len(c1) + len(c2), "body_start": len(c1) + len(c2) + 4, "end": len(txt)},
        ]
        q1, q2, q3 = "身穿青衫", "仍穿青衫", "身穿白袍"
        facts = [
            _mk_fact("f1", "costume", "青衫", q1, txt.index(q1), "c1"),
            _mk_fact("f2", "costume", "青衫", q2, txt.index(q2), "c2"),
            _mk_fact("f3", "costume", "白袍", q3, txt.index(q3), "c3"),
        ]
        self.assertEqual(detect_narrative_time_jumps(facts, [], chs, txt), [])

    def test_20_time_jump_dialogue_quotes_zh_and_en_rejected(self):
        c1 = "第一章\n林凡穿青袍。\n"
        c2 = '第二章\n他叹道：“转眼过了十年，我也老了。”林凡穿白袍。\n'
        c3 = 'Chapter 3\nHe said, "Ten years later we shall meet again." He wore a black coat.\n'
        txt = c1 + c2 + c3
        chs = [
            {"id": "c1", "kind": "chapter", "start": 0, "body_start": 4, "end": len(c1)},
            {"id": "c2", "kind": "chapter", "start": len(c1), "body_start": len(c1) + 4, "end": len(c1) + len(c2)},
            {"id": "c3", "kind": "chapter", "start": len(c1) + len(c2), "body_start": len(c1) + len(c2) + 10, "end": len(txt)},
        ]
        q1, q2, q3 = "穿青袍", "穿白袍", "black coat"
        facts = [
            _mk_fact("f1", "costume", "青袍", q1, txt.index(q1), "c1"),
            _mk_fact("f2", "costume", "白袍", q2, txt.index(q2), "c2"),
            _mk_fact("f3", "costume", "黑外套", q3, txt.index(q3), "c3"),
        ]
        self.assertEqual(detect_narrative_time_jumps(facts, [], chs, txt), [])

    def test_21_time_jump_flashback_hypothetical_regnal_rejected(self):
        c1 = "第一章\n张昭穿青衫。\n"
        c2 = "第二章\n他回想起十年前之事，张昭穿白袍。\n"
        c3 = "第三章\n他打算闭关二十年再出来，张昭穿黑袍。\n"
        c4 = "第四章\n维大周十三年后春，张昭穿红袍。\n"
        txt = c1 + c2 + c3 + c4
        offs = [0, len(c1), len(c1) + len(c2), len(c1) + len(c2) + len(c3), len(txt)]
        chs = [
            {"id": f"c{i+1}", "kind": "chapter", "start": offs[i], "body_start": offs[i] + 4, "end": offs[i + 1]}
            for i in range(4)
        ]
        quotes = ["穿青衫", "穿白袍", "穿黑袍", "穿红袍"]
        facts = [
            _mk_fact(f"f{i+1}", "costume", q, q, txt.index(q), f"c{i+1}")
            for i, q in enumerate(quotes)
        ]
        self.assertEqual(detect_narrative_time_jumps(facts, [], chs, txt), [])

    def test_22_time_jump_skipped_when_already_covered_or_same_chapter(self):
        c1 = "第一章\n韩立穿灰袍。转眼过了十年，韩立穿青袍。\n"
        c2 = "第二章\n一晃过了十年，韩立换上金甲。\n"
        txt = c1 + c2
        chs = [
            {"id": "c1", "kind": "chapter", "start": 0, "body_start": 4, "end": len(c1)},
            {"id": "c2", "kind": "chapter", "start": len(c1), "body_start": len(c1) + 4, "end": len(txt)},
        ]
        q1, q2, q3 = "穿灰袍", "穿青袍", "换上金甲"
        facts = [
            _mk_fact("f1", "costume", "灰袍", q1, txt.index(q1), "c1"),
            _mk_fact("f2", "costume", "青袍", q2, txt.index(q2), "c1"),
            _mk_fact("f3", "costume", "金甲", q3, txt.index(q3), "c2"),
        ]
        existing = [_mk_event("ev1", "costume_change", "换金甲", q3, txt.index(q3), "c2", stage_split=True)]
        # c1 内部同章不触发，c2 已被 existing_splits 覆盖 -> 0 切点
        self.assertEqual(detect_narrative_time_jumps(facts, existing, chs, txt), [])

    def test_23_build_entity_stages_integrates_time_jump_with_12_keys_and_preserves_life_stage_11_keys(self):
        c1 = "第一章\n林轩年方十岁，身穿灰衫。\n"
        c2 = "第二章\n林轩二十五岁，身穿青袍。\n"
        c3 = "第三章\n闭关二十年后，林轩身穿黑袍。\n"
        txt = c1 + c2 + c3
        offs = [0, len(c1), len(c1) + len(c2), len(txt)]
        chs = [
            {"id": f"c{i+1}", "kind": "chapter", "start": offs[i], "body_start": offs[i] + 4, "end": offs[i + 1]}
            for i in range(3)
        ]
        facts = [
            _mk_fact("f1", "age", "十岁", "年方十岁", txt.index("年方十岁"), "c1"),
            _mk_fact("f2", "costume", "灰衫", "身穿灰衫", txt.index("身穿灰衫"), "c1"),
            _mk_fact("f3", "age", "二十五岁", "二十五岁", txt.index("二十五岁"), "c2"),
            _mk_fact("f4", "costume", "青袍", "身穿青袍", txt.index("身穿青袍"), "c2"),
            _mk_fact("f5", "costume", "黑袍", "身穿黑袍", txt.index("身穿黑袍"), "c3"),
        ]
        events: list[dict] = []
        stages = build_entity_stages("person", facts, events, chs, (0, len(txt)), text=txt)
        self.assertEqual(len(stages), 3)
        self.assertEqual(events, [])
        t_age = stages[1]["trigger_events"][0]
        self.assertEqual(t_age["trigger_source"], TRIGGER_SOURCE_LIFE_STAGE)
        self.assertEqual(
            set(t_age),
            {
                "id", "event_type", "value", "quote", "start", "end", "chapter_id",
                "trigger_source", "from_stage_bucket", "to_stage_bucket", "source_fact_id",
            },
        )
        t_jump = stages[2]["trigger_events"][0]
        self.assertEqual(t_jump["trigger_source"], TRIGGER_SOURCE_TIME_JUMP)
        self.assertEqual(t_jump["jump_years"], 20.0)
        self.assertEqual(
            set(t_jump),
            {
                "id", "event_type", "value", "quote", "start", "end", "chapter_id",
                "trigger_source", "from_stage_bucket", "to_stage_bucket", "source_fact_id",
                "jump_years",
            },
        )
        self.assertEqual(txt[t_jump["start"]:t_jump["end"]], t_jump["quote"])

    def test_24_build_entity_stages_infer_time_jumps_false_disables_only_time_jumps(self):
        c1 = "第一章\n林轩年方十岁。\n"
        c2 = "第二章\n林轩二十五岁。\n"
        c3 = "第三章\n转眼过了十年，林轩穿黑袍。\n"
        txt = c1 + c2 + c3
        offs = [0, len(c1), len(c1) + len(c2), len(txt)]
        chs = [
            {"id": f"c{i+1}", "kind": "chapter", "start": offs[i], "body_start": offs[i] + 4, "end": offs[i + 1]}
            for i in range(3)
        ]
        facts = [
            _mk_fact("f1", "age", "十岁", "年方十岁", txt.index("年方十岁"), "c1"),
            _mk_fact("f2", "age", "二十五岁", "二十五岁", txt.index("二十五岁"), "c2"),
            _mk_fact("f3", "costume", "黑袍", "穿黑袍", txt.index("穿黑袍"), "c3"),
        ]
        stages = build_entity_stages(
            "person", facts, [], chs, (0, len(txt)), text=txt,
            infer_life_stages=True, infer_time_jumps=False,
        )
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[1]["trigger_events"][0]["trigger_source"], TRIGGER_SOURCE_LIFE_STAGE)


class TimelineProfilesTests(unittest.TestCase):
    """恒定底色继承、时间序场景造型快照及三项通用打磨回归测试。"""

    def test_25_costume_signature_winter_fur_vs_casual_calibrated(self):
        sig_hua = parse_costume_signature("穿紫绣团胸绣花袍")
        sig_fur = parse_costume_signature("穿一领貂鼠皮袄")
        sig_cot = parse_costume_signature("旧棉袄")
        sig_gen = parse_costume_signature("整整齐齐的衣服")
        sig_en = parse_costume_signature("dark formal attire")

        self.assertEqual(sig_hua["categories"], {"casual"})
        self.assertEqual(sig_fur["categories"], {"winter_fur"})
        self.assertEqual(sig_cot["categories"], {"winter_fur"})
        self.assertNotIn("casual", sig_fur["categories"])
        self.assertNotIn("casual", sig_cot["categories"])
        self.assertEqual(sig_gen["categories"], {"casual"})
        self.assertEqual(sig_en["categories"], {"casual"})
        self.assertTrue(costumes_conflict(sig_hua, sig_fur))

    def test_26_costumes_conflict_strong_state_and_color_and_portrait_exemption(self):
        sig_casual = parse_costume_signature("整整齐齐的衣服")
        sig_armor_robe = parse_costume_signature("锦袍金甲")
        sig_fur = parse_costume_signature("貂鼠皮袄")
        # 跨章单侧含强状态品类 armor -> 判互斥
        self.assertTrue(costumes_conflict(sig_casual, sig_armor_robe))

        # 同章 <= 260 码点全身肖像叠穿豁免（含 winter_fur）
        same_ch_facts = [
            _mk_fact("f1", "costume", "紫绣花袍", "穿紫绣花袍", 10, "ch1"),
            _mk_fact("f2", "costume", "貂鼠皮袄", "外罩貂鼠皮袄", 30, "ch1"),
            _mk_fact("f3", "costume", "金甲", "披挂金甲", 50, "ch1"),
        ]
        self.assertEqual(count_costume_conflict_pairs(same_ch_facts), 0)

        # 跨章则计为互斥冲突
        cross_ch_facts = [
            _mk_fact("f1", "costume", "整整齐齐的衣服", "穿着整整齐齐的衣服", 10, "ch1"),
            _mk_fact("f2", "costume", "锦袍金甲", "身披锦袍金甲", 250, "ch2"),
        ]
        self.assertEqual(count_costume_conflict_pairs(cross_ch_facts), 1)

        # build_entity_timeline_profiles 将跨章常服与锦袍金甲拆为两个快照，单快照内冲突 == 0
        chs = _mk_chapters(3)
        stages = build_entity_stages("person", cross_ch_facts, [], chs, (0, 600))
        ent = {
            "entity_id": "e1", "kind": "person", "kind_rank": 1, "name": "柴进",
            "facts": cross_ch_facts, "state_events": [], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        apps = prof["stages"][0]["scene_appearances"]
        self.assertEqual(len(apps), 2)
        self.assertEqual(apps[1]["trigger_kind"], "costume_switch")
        self.assertEqual(sum(count_costume_conflict_pairs(a["scene_facts"]) for a in apps), 0)

    def test_27_base_traits_persistence_label_priority(self):
        chs = _mk_chapters(3)
        f1 = _mk_fact("f1", "costume", "祖传青袍", "穿祖传青袍", 10, "ch1", persistence="trait")
        f2 = _mk_fact("f2", "face", "俊朗面容", "面容俊朗", 30, "ch1", persistence="scene")
        stages = build_entity_stages("person", [f1, f2], [], chs, (0, 600))
        ent = {
            "entity_id": "e1", "kind": "person", "kind_rank": 1, "name": "甲",
            "facts": [f1, f2], "state_events": [], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        self.assertEqual([f["id"] for f in prof["base_traits"]], ["f1"])

    def test_28_rb2_1_object_transform_earlier_chapter_shifts_cutoff_to_chapter_start(self):
        chs = _mk_chapters(3)
        # ch1 已有初始形态底色；ch2 在 transform 事件前 10 码点已描写蜕皮后雪白身躯
        f_orig = _mk_fact("f1", "color_material", "通体漆黑", "墨蛟通体漆黑", 20, "ch1")
        f_molt_pre = _mk_fact("f2", "color_material", "蜕皮雪白", "浑身雪白无鳞", 210, "ch2")
        ev_tr = _mk_event("ev1", "transform", "蜕皮化形", "轰然蜕去旧皮", 230, "ch2", stage_split=True)
        f_after = _mk_fact("f3", "size", "庞大如山", "身躯庞大如山", 260, "ch2")
        stages = build_entity_stages("object", [f_orig, f_molt_pre, f_after], [ev_tr], chs, (0, 600))
        ent = {
            "entity_id": "obj1", "kind": "object", "kind_rank": 1, "name": "墨蛟",
            "facts": [f_orig, f_molt_pre, f_after], "state_events": [ev_tr], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        self.assertEqual([f["id"] for f in prof["base_traits"]], ["f1"])
        _assert_timeline_invariants(self, prof)

    def test_29_rb2_1_object_single_chapter_keeps_pre_event_traits(self):
        chs = _mk_chapters(2)
        # 单章登场即损毁（更早章节无固有事实）：仍以 ev["start"] 为界保留章首完好底色
        f_intact = _mk_fact("f1", "color_material", "玄铁铸就", "宝剑玄铁铸就", 10, "ch1")
        ev_dmg = _mk_event("ev1", "damage", "折断", "咔嚓折断成两截", 60, "ch1", stage_split=True)
        f_broken = _mk_fact("f2", "shape", "断成两截", "断剑仅余半尺", 100, "ch1")
        stages = build_entity_stages("object", [f_intact, f_broken], [ev_dmg], chs, (0, 400))
        ent = {
            "entity_id": "obj2", "kind": "object", "kind_rank": 1, "name": "玄铁剑",
            "facts": [f_intact, f_broken], "state_events": [ev_dmg], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        self.assertEqual([f["id"] for f in prof["base_traits"]], ["f1"])

    def test_30_rb2_2_en_transient_states_excluded_from_person_base_traits(self):
        chs = _mk_chapters(2)
        f_ok = _mk_fact("f1", "face", "dark expressive eyes", "She had dark expressive eyes", 10, "ch1")
        f_blush = _mk_fact("f2", "face", "blushed deeply", "She blushed deeply at the remark", 50, "ch1")
        f_pale = _mk_fact("f3", "face", "turned pale", "Her face turned pale", 90, "ch1")
        f_trem = _mk_fact("f4", "voice", "trembling voice", "spoke in a trembling voice", 130, "ch1")
        stages = build_entity_stages("person", [f_ok, f_blush, f_pale, f_trem], [], chs, (0, 400))
        ent = {
            "entity_id": "p1", "kind": "person", "kind_rank": 1, "name": "Elizabeth",
            "facts": [f_ok, f_blush, f_pale, f_trem], "state_events": [], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        self.assertEqual([f["id"] for f in prof["base_traits"]], ["f1"])

    def test_31_rb2_3_pure_age_cluster_skips_reappearance_gap_split(self):
        chs = _mk_chapters(30)
        f1 = _mk_fact("f1", "costume", "素色长裙", "穿着素色长裙", 10, "ch1")
        # 隔 15 章仅有纯 age 事实 -> 不应切出空壳重登场快照
        f2 = _mk_fact("f2", "age", "二十岁", "刚满二十岁", 15 * 200 + 10, "ch16")
        # 再隔 12 章出现具体视觉发式 -> 正常切出重登场快照
        f3 = _mk_fact("f3", "hair", "乌发高挽", "乌发高挽", 27 * 200 + 10, "ch28")
        stages = build_entity_stages("person", [f1, f2, f3], [], chs, (0, 30 * 200))
        ent = {
            "entity_id": "p1", "kind": "person", "kind_rank": 1, "name": "Charlotte",
            "facts": [f1, f2, f3], "state_events": [], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        apps = prof["stages"][0]["scene_appearances"]
        self.assertEqual(len(apps), 2)
        self.assertEqual([f["id"] for f in apps[0]["scene_facts"]], ["f1", "f2"])
        self.assertEqual([f["id"] for f in apps[1]["scene_facts"]], ["f3"])
        self.assertEqual(apps[1]["trigger_kind"], "reappearance_gap")

    def test_32_inherited_traits_and_effective_gaps_across_stages(self):
        chs = _mk_chapters(5)
        f_gen = _mk_fact("f1", "gender", "男", "男子", 10, "ch1", persistence="trait")
        f_voice = _mk_fact("f2", "voice", "声如洪钟", "声如洪钟", 20, "ch1", persistence="trait")
        f_c1 = _mk_fact("f3", "costume", "青衫", "穿青衫", 30, "ch1", persistence="stage")
        ev = _mk_event("ev1", "costume_change", "换金甲", "披上金甲", 410, "ch3", stage_split=True)
        f_c2 = _mk_fact("f4", "costume", "金甲", "穿金甲", 430, "ch3", persistence="stage")
        stages = build_entity_stages("person", [f_gen, f_voice, f_c1, f_c2], [ev], chs, (0, 1000))
        ent = {
            "entity_id": "p1", "kind": "person", "kind_rank": 1, "name": "将军",
            "facts": [f_gen, f_voice, f_c1, f_c2], "state_events": [ev], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        self.assertEqual(prof["stages"][0]["inherited_traits"], [])
        self.assertEqual({f["facet"] for f in prof["stages"][1]["inherited_traits"]}, {"gender", "voice"})
        self.assertIn("gender", prof["stages"][1]["gaps"])
        self.assertNotIn("gender", prof["stages"][1]["effective_gaps"])
        self.assertNotIn("voice", prof["stages"][1]["effective_gaps"])
        _assert_timeline_invariants(self, prof)

    def test_33_scene_appearances_event_and_same_chapter_repeat_injury_merge(self):
        chs = _mk_chapters(3)
        f1 = _mk_fact("f1", "costume", "白衣", "一身白衣", 10, "ch1")
        ev_inj1 = _mk_event("e1", "injury", "左臂中剑", "左臂中了一剑", 50, "ch1", stage_split=False)
        ev_inj2 = _mk_event("e2", "injury", "肩头又伤", "肩头又挨一刀", 80, "ch1", stage_split=False)
        ev_disg = _mk_event("e3", "disguise", "乔装货郎", "乔装成货郎模样", 220, "ch2", stage_split=False)
        stages = build_entity_stages("person", [f1], [ev_inj1, ev_inj2, ev_disg], chs, (0, 600))
        ent = {
            "entity_id": "p1", "kind": "person", "kind_rank": 1, "name": "侠客",
            "facts": [f1], "state_events": [ev_inj1, ev_inj2, ev_disg], "stages": stages, "gaps": [],
        }
        prof = build_entity_timeline_profiles(ent, chs)
        apps = prof["stages"][0]["scene_appearances"]
        # 首快照(白衣) -> 第二快照(两处同章受伤合并) -> 第三快照(乔装货郎)
        self.assertEqual(len(apps), 3)
        self.assertEqual(len(apps[1]["scene_events"]), 2)
        self.assertEqual(apps[1]["trigger_kind"], "scene_event:injury")
        self.assertEqual(apps[2]["trigger_kind"], "scene_event:disguise")
        _assert_timeline_invariants(self, prof)

    def test_34_scene_appearances_holder_switch_and_place_atmosphere_switch(self):
        chs = _mk_chapters(4)
        # object 跨章持有者切换
        of1 = _mk_fact("o1", "holder", "甲", "甲持宝刀", 10, "ch1", holder_entity_id="pa")
        of2 = _mk_fact("o2", "holder", "乙", "乙持宝刀", 210, "ch2", holder_entity_id="pb")
        ostages = build_entity_stages("object", [of1, of2], [], chs, (0, 800))
        oent = {
            "entity_id": "obj", "kind": "object", "kind_rank": 1, "name": "宝刀",
            "facts": [of1, of2], "state_events": [], "stages": ostages, "gaps": [],
        }
        oprof = build_entity_timeline_profiles(oent, chs)
        self.assertEqual(len(oprof["stages"][0]["scene_appearances"]), 2)
        self.assertEqual(oprof["stages"][0]["scene_appearances"][1]["trigger_kind"], "holder_switch")

        # place 昼夜气象切换
        pf1 = _mk_fact("p1", "atmosphere", "阳光晴朗", "白昼阳光晴朗", 10, "ch1")
        pf2 = _mk_fact("p2", "atmosphere", "月色漆黑", "深夜月色漆黑", 210, "ch2")
        pstages = build_entity_stages("place", [pf1, pf2], [], chs, (0, 800))
        pent = {
            "entity_id": "pl", "kind": "place", "kind_rank": 1, "name": "山谷",
            "facts": [pf1, pf2], "state_events": [], "stages": pstages, "gaps": [],
        }
        pprof = build_entity_timeline_profiles(pent, chs)
        self.assertEqual(len(pprof["stages"][0]["scene_appearances"]), 2)
        self.assertEqual(pprof["stages"][0]["scene_appearances"][1]["trigger_kind"], "atmosphere_switch")

    def test_35_timeline_profiles_invariants_conservation_monotonicity_non_mutation(self):
        txt = (
            "第一章。男主面如冠玉，穿青布长衫，腰悬古玉。\n"
            "第二章。男主披挂红锦战袍与金锁甲出征。\n"
            "第三章。男主戴上木枷身穿囚服。\n"
        )
        offs = [0, txt.index("第二章"), txt.index("第三章"), len(txt)]
        chs = [
            {"id": f"c{i+1}", "kind": "chapter", "start": offs[i], "body_start": offs[i] + 4, "end": offs[i + 1]}
            for i in range(3)
        ]
        facts = [
            _mk_fact("f1", "face", "面如冠玉", "面如冠玉", txt.index("面如冠玉"), "c1"),
            _mk_fact("f2", "costume", "青布长衫", "穿青布长衫", txt.index("穿青布长衫"), "c1"),
            _mk_fact("f3", "costume", "红锦战袍与金锁甲", "红锦战袍与金锁甲", txt.index("红锦战袍与金锁甲"), "c2"),
            _mk_fact("f4", "costume", "木枷囚服", "戴上木枷身穿囚服", txt.index("戴上木枷身穿囚服"), "c3"),
        ]
        stages = build_entity_stages("person", facts, [], chs, (0, len(txt)), text=txt)
        ent = {
            "entity_id": "p1", "kind": "person", "kind_rank": 1, "name": "男主",
            "facts": facts, "state_events": [], "stages": stages, "gaps": [],
        }
        orig_snapshot = copy.deepcopy(ent)
        prof = build_entity_timeline_profiles(ent, chs, text=txt)
        # 输入零副作用
        self.assertEqual(ent, orig_snapshot)
        # 既有字段零改写
        for k in ("entity_id", "kind", "kind_rank", "name", "facts", "state_events", "gaps"):
            self.assertEqual(prof[k], orig_snapshot[k])
        _assert_timeline_invariants(self, prof, text=txt)
        self.assertEqual(prof["total_scene_appearances"], 3)

    def test_36_timeline_profiles_pure_trait_stage_and_empty_stage(self):
        chs = _mk_chapters(2)
        f_gen = _mk_fact("f1", "gender", "女", "女子", 10, "ch1", persistence="trait")
        stages_pure = build_entity_stages("person", [f_gen], [], chs, (0, 400))
        ent_pure = {
            "entity_id": "p1", "kind": "person", "kind_rank": 1, "name": "乙",
            "facts": [f_gen], "state_events": [], "stages": stages_pure, "gaps": [],
        }
        prof_pure = build_entity_timeline_profiles(ent_pure, chs)
        self.assertEqual(prof_pure["total_scene_appearances"], 1)
        self.assertEqual(prof_pure["stages"][0]["scene_appearances"][0]["scene_facts"], [])
        self.assertEqual(len(prof_pure["stages"][0]["scene_appearances"][0]["active_traits"]), 1)
        _assert_timeline_invariants(self, prof_pure)

        stages_empty = build_entity_stages("person", [], [], chs, (0, 400))
        ent_empty = {
            "entity_id": "p2", "kind": "person", "kind_rank": 2, "name": "丙",
            "facts": [], "state_events": [], "stages": stages_empty, "gaps": list(FACETS["person"]),
        }
        prof_empty = build_entity_timeline_profiles(ent_empty, chs)
        self.assertEqual(prof_empty["total_scene_appearances"], 0)
        self.assertEqual(prof_empty["stages"][0]["scene_appearances"], [])


class EndToEndAndFrozenGuardTests(unittest.TestCase):
    """端到端 SQLite/CLI 重绑闭环、撤回原字节复原与冻结提示词守卫测试。"""

    def _build_fixture(self, root: Path) -> tuple[Path, Path, str, dict[str, str], str, str]:
        c1 = "第一章 少年出航\n林海年方十六岁，面如冠玉，身着青布短衫。\n"
        c2 = "第十四章 化名归来\n转眼过了十年，那位黑衣剑客苍狼原名林海，如今身披玄铁铠甲。\n"
        text = c1 + c2
        book = root / "novel.txt"
        book.write_bytes(text.encode("utf-8"))
        db_path = root / "test_022b.sqlite3"
        out_root = root / "outputs"
        store = Store(db_path)
        try:
            source = import_text(store, book)
            headings = []
            for title in ("第一章 少年出航", "第十四章 化名归来"):
                pos = text.index(title)
                headings.append({"start": pos, "end": pos + len(title), "title": title, "kind": "chapter"})
            ver, chapters = build_chapters(source, headings)
            run_id = "run_022b_e2e_00000000000000000001"
            cfg = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {"complete": True, "chapter_set_version": ver, "issues": 0, "path_counts": {"adopted": len(chapters)}},
                "discovery": {"complete": True, "tasks": len(chapters), "completed": len(chapters)},
            }
            store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], dumps(cfg), dumps(state), "now", "now"))
            for ch in chapters:
                store.put(run_id, "chapter", ch)
            tasks = plan_tasks(source, chapters, 9000)

            def _m(mid: str, surf: str, ch_idx: int, start: int):
                ch, t = chapters[ch_idx], tasks[ch_idx]
                u0 = t["units"][0]
                return {
                    "id": mid, "text": surf, "start": start, "end": start + len(surf),
                    "address": address(source["id"], start, start + len(surf)),
                    "chapter_id": ch["id"], "task_id": t["id"],
                    "provisional_kinds": ["person"], "observation_ids": [f"obs_{mid}"],
                    "evidence": [address(source["id"], u0["start"], u0["end"])],
                }

            c1_pos = chapters[0]["start"]
            c2_pos = chapters[1]["start"]
            m_by_ch = {
                0: [_m("m_lh_1", "林海", 0, text.index("林海", c1_pos))],
                1: [
                    _m("m_cl_1", "苍狼", 1, text.index("苍狼", c2_pos)),
                    _m("m_lh_2", "林海", 1, text.index("林海", c2_pos)),
                ],
            }
            for idx, t in enumerate(tasks):
                store.finish_task(run_id, t, {"request_id": f"req_d_{idx}"}, [("mention", m) for m in m_by_ch[idx]])

            def handle_c1(st):
                uid = st["units"][0]["id"]
                if st["chapter_title"] == "第一章 少年出航":
                    return {
                        "entities": [{
                            "local_id": "e1", "name": "林海", "kind": "person",
                            "surfaces": [{"surface": "林海", "surface_type": "proper_name"}], "evidence_ref": uid,
                        }],
                        "assignments": [], "not_entity": [],
                    }, "req_c1_1"
                return {
                    "entities": [
                        {
                            "local_id": "e1", "name": "苍狼", "kind": "person",
                            "surfaces": [{"surface": "苍狼", "surface_type": "proper_name"}], "evidence_ref": uid,
                        },
                        {
                            "local_id": "e2", "name": "林海", "kind": "person",
                            "surfaces": [{"surface": "林海", "surface_type": "proper_name"}], "evidence_ref": uid,
                        },
                    ],
                    "assignments": [], "not_entity": [],
                }, "req_c1_2"

            rmodels = _RosterModels({
                "roster-c1": handle_c1,
                "roster-c2-review": lambda st: ({"groups": [], "unresolved_row_ids": []}, "r2"),
                "roster-c2-propose": lambda st: ({"proposals": []}, "r3"),
            })
            out = resolve_roster(store, rmodels, source, run_id, concurrency=1, log=lambda *_: None, task_chars=9000)
            self.assertTrue(out["complete"])
            store.state(run_id, identity=out)
            bundle = build_export_bundle(store, run_id)
            by_name = {e["name"]: e["id"] for e in bundle["result"]["entities"]}

            def desc_handler(stage, st, _call_no):
                if stage == "describe-d1":
                    chunk = st.get("text", "")
                    eids_here = {e["entity_id"] for e in st["entities"]}
                    facts: list[dict] = []
                    if by_name["林海"] in eids_here and "青布短衫" in chunk:
                        facts.append({"entity_id": by_name["林海"], "facet": "face", "value": "面如冠玉", "quote": "面如冠玉", "source": "narration"})
                        facts.append({"entity_id": by_name["林海"], "facet": "costume", "value": "青布短衫", "quote": "身着青布短衫", "source": "narration"})
                    if by_name["苍狼"] in eids_here and "玄铁铠甲" in chunk:
                        facts.append({"entity_id": by_name["苍狼"], "facet": "costume", "value": "玄铁铠甲", "quote": "身披玄铁铠甲", "source": "narration"})
                    return {"facts": facts, "events": []}
                if stage == "describe-verify":
                    return {"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None} for it in st["items"]]}
                if stage == "describe-event-verify":
                    return {"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None, "stage_split": False} for it in st["items"]]}
                raise AssertionError(stage)

            fmodels = FakeModels(SERVICES, desc_handler)
            res = run_describe(
                store, run_id, services=SERVICES, models=fmodels,
                top_k=10, kinds="person", views=1, entities_per_request=0, verify=1,
            )
            return db_path, out_root, run_id, by_name, res["describe_id"], text
        finally:
            store.close()

    def test_37_run_describe_and_export_describe_run_aligned_with_8_key_counts(self):
        with tempfile.TemporaryDirectory() as td:
            db_path, _, run_id, by_name, did, text = self._build_fixture(Path(td))
            store = Store(db_path)
            try:
                doc_exp = export_describe_run(store, did)
                self.assertEqual(len(doc_exp["persona_alias_hints"]), 1)
                hint = doc_exp["persona_alias_hints"][0]
                self.assertEqual({hint["target_entity_id"], hint["candidate_entity_id"]}, {by_name["林海"], by_name["苍狼"]})
                for ent in doc_exp["entities"]:
                    self.assertIn("base_traits", ent)
                    self.assertIn("total_scene_appearances", ent)
                    _assert_timeline_invariants(self, ent, text=text)
                # 再次零新任务 run_describe 与 export_describe_run 导出文档完全对齐
                res2 = run_describe(
                    store, run_id, services=SERVICES,
                    models=FakeModels(SERVICES, lambda *_: (_ for _ in ()).throw(AssertionError("不应调用模型"))),
                    top_k=10, kinds="person", views=1, entities_per_request=0, verify=1,
                )
                self.assertEqual(
                    set(res2["counts"]),
                    {
                        "tasks", "facts", "events", "stages", "multi_stage_entities",
                        "reviews", "scene_appearances", "persona_alias_hints",
                    },
                )
                self.assertEqual(res2["counts"]["persona_alias_hints"], 1)
                self.assertGreaterEqual(res2["counts"]["scene_appearances"], 2)
                self.assertEqual(res2["document"], doc_exp)
            finally:
                store.close()

    def test_38_cli_describe_rebind_zero_credentials_and_double_run_sha256_identical(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, _, run_id, _, did, _ = self._build_fixture(root)
            out1 = root / "rebind1.json"
            out2 = root / "rebind2.json"
            clean_env = {
                k: v for k, v in os.environ.items()
                if not k.startswith(("DEEPSEEK_", "TYPESAFE_", "OPENAI_", "ENTITYTRACE_"))
                and k != "GENERATOR_PROVIDER"
            }
            with patch.dict(os.environ, clean_env, clear=True):
                buf1, err1 = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf1), contextlib.redirect_stderr(err1):
                    rc1 = cli_main(["describe", run_id, "--db", str(db_path), "--rebind", did, "--out", str(out1)])
                buf2, err2 = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf2), contextlib.redirect_stderr(err2):
                    rc2 = cli_main(["describe", run_id, "--db", str(db_path), "--rebind", did, "--out", str(out2)])
            self.assertEqual(rc1, 0, err1.getvalue())
            self.assertEqual(rc2, 0, err2.getvalue())
            receipt = json.loads(buf1.getvalue().strip())
            self.assertEqual(
                set(receipt["counts"]),
                {
                    "tasks", "facts", "events", "stages", "multi_stage_entities",
                    "reviews", "scene_appearances", "persona_alias_hints",
                },
            )
            self.assertEqual(_sha256_bytes(out1.read_bytes()), _sha256_bytes(out2.read_bytes()))

    def test_39_cli_merge_eliminates_persona_hint_and_revoke_restores_exact_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path, out_root, run_id, by_name, did, text = self._build_fixture(root)
            before_out = root / "before.json"
            merged_out = root / "merged.json"
            revoked_out = root / "revoked.json"
            clean_env = {
                k: v for k, v in os.environ.items()
                if not k.startswith(("DEEPSEEK_", "TYPESAFE_", "OPENAI_", "ENTITYTRACE_"))
                and k != "GENERATOR_PROVIDER"
            }
            with patch.dict(os.environ, clean_env, clear=True):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc0 = cli_main(["describe", run_id, "--db", str(db_path), "--rebind", did, "--out", str(before_out)])
                self.assertEqual(rc0, 0)
                before_sha = _sha256_bytes(before_out.read_bytes())

                store = Store(db_path)
                try:
                    rev0 = build_export_bundle(store, run_id)["result"]["result_revision"]
                    c_merge = apply_correction(
                        store, run_id, "merge",
                        {"target_entity_id": by_name["林海"], "merged_entity_id": by_name["苍狼"]},
                        source="user", expect_revision=rev0, output_root=out_root,
                    )
                    rev1 = c_merge["result_revision"]
                finally:
                    store.close()

                buf_m = io.StringIO()
                with contextlib.redirect_stdout(buf_m):
                    rc_m = cli_main(["describe", run_id, "--db", str(db_path), "--rebind", did, "--out", str(merged_out)])
                self.assertEqual(rc_m, 0)
                rec_m = json.loads(buf_m.getvalue().strip())
                doc_m = json.loads(merged_out.read_text(encoding="utf-8"))
                # 合体后跨马甲提示消隐为 0，且合并后的林海触发跨年时间跳跃切分为 2 阶段并继承面如冠玉底色
                self.assertEqual(rec_m["counts"]["persona_alias_hints"], 0)
                self.assertEqual(doc_m["persona_alias_hints"], [])
                self.assertEqual(len(doc_m["entities"]), 1)
                ent_m = doc_m["entities"][0]
                self.assertEqual(len(ent_m["stages"]), 2)
                self.assertEqual(ent_m["stages"][1]["trigger_events"][0]["trigger_source"], TRIGGER_SOURCE_TIME_JUMP)
                self.assertEqual(ent_m["stages"][1]["trigger_events"][0]["jump_years"], 10.0)
                self.assertEqual([f["facet"] for f in ent_m["stages"][1]["inherited_traits"]], ["face"])
                _assert_stage_invariants(self, ent_m, (0, len(text)))
                _assert_timeline_invariants(self, ent_m, text=text)

                store = Store(db_path)
                try:
                    apply_correction(
                        store, run_id, "revoke",
                        {"correction_id": c_merge["correction_id"]},
                        source="user", expect_revision=rev1, output_root=out_root,
                    )
                finally:
                    store.close()

                buf_r = io.StringIO()
                with contextlib.redirect_stdout(buf_r):
                    rc_r = cli_main(["describe", run_id, "--db", str(db_path), "--rebind", did, "--out", str(revoked_out)])
                self.assertEqual(rc_r, 0)
                self.assertEqual(_sha256_bytes(revoked_out.read_bytes()), before_sha)

    def test_40_prompts_describe_version_and_build_describe_id_frozen(self):
        self.assertEqual(DESCRIBE_VERSION, "describe-v7")
        self.assertEqual(
            _sha256_text(D1_PROMPT),
            "496a3481aadc5676684259326055f3584673686b0de68f4605defd0762fd8d13",
        )
        self.assertEqual(
            _sha256_text(VERIFY_PROMPT),
            "113e3f8e0c6bbc804f658ae56cb917064492505b6d63b53321605e9bb80fbbe0",
        )
        self.assertEqual(
            _sha256_text(EVENT_VERIFY_PROMPT),
            "1569553a1bb50b001e773c20493a24269ec30a6b5c9721b40915e4e80c96cdef",
        )
        self.assertEqual(
            _sha256_text(PERSISTENCE_PROMPT),
            "ec8020277055f07c777a39fb86e6b389fea3fbd0f9f5f8efaf88290ac16f2a47",
        )
        self.assertEqual(
            _sha256_text(inspect.getsource(build_describe_id)),
            "46b51c15b8e8ce1ea842be131fcf8f316cf216a51c9ff9b561cd4f188671fdf4",
        )
        self.assertEqual(
            build_describe_id(
                20, ["person"], ["e1"], "r1",
                provider="deepseek", model="deepseek-flash",
                views=3, entities_per_request=4, verify=1, persistence=0,
            ),
            "09837b05ab1f28d4762886af976afc9ef891cf48fa775f8a1e36788af10d6a60",
        )
        self.assertEqual(
            build_describe_id(
                20, ["person"], ["e1"], "r1",
                provider="deepseek", model="deepseek-flash",
                views=3, entities_per_request=4, verify=1, persistence=1,
            ),
            "b9a9e5e1724885c64286c8bdea73e65e8e7b9dbb60dfc4140cabd1f4f94470b5",
        )

    def test_41_track_b1_visual_anchor_not_in_production_package(self):
        self.assertFalse(hasattr(life_stages_mod, "parse_visual_age_anchor"))
        self.assertFalse(hasattr(life_stages_mod, "TRIGGER_SOURCE_VISUAL_ANCHOR"))
        self.assertFalse(hasattr(describe_mod, "parse_visual_age_anchor"))
        pkg_root = Path(describe_mod.__file__).resolve().parent
        for pyf in sorted(pkg_root.glob("*.py")):
            src = pyf.read_text(encoding="utf-8")
            self.assertNotIn("parse_visual_age_anchor", src, pyf.name)

    def test_42_persona_hints_and_timeline_profiles_can_be_toggled_off_via_api(self):
        with tempfile.TemporaryDirectory() as td:
            db_path, _, _, _, did, _ = self._build_fixture(Path(td))
            store = Store(db_path)
            try:
                doc = export_describe_run(
                    store, did,
                    timeline_profiles=False,
                    persona_hints=False,
                    infer_time_jumps=False,
                )
                self.assertNotIn("persona_alias_hints", doc)
                for ent in doc["entities"]:
                    self.assertNotIn("base_traits", ent)
                    self.assertNotIn("total_scene_appearances", ent)
                    for st in ent["stages"]:
                        self.assertNotIn("scene_appearances", st)
                        self.assertNotIn("inherited_traits", st)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
