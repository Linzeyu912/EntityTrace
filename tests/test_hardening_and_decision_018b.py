from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import (
    classify_non_heading,
    collect_candidates,
    detect_chapters,
    filter_heading_candidates,
    is_exempt_heading_syntax,
)
from entitytrace.cli import main as cli_main
from entitytrace.config import (
    DECISION_MODES,
    Service,
    is_typesafe_decision_service,
    load_services,
    resolve_decision_mode,
)
from entitytrace.describe import (
    D1_PROMPT,
    DESC_VERIFY_DECISION_TAU,
    DESCRIBE_VERSION,
    EVENT_VERIFY_PROMPT,
    EVENT_VERIFY_PROMPT_VERSION,
    VERIFY_PROMPT,
    apply_stage_split_guards,
    build_describe_id,
    build_event_verify_decision_batches,
    build_verify_decision_batches,
    run_describe,
)
from entitytrace.chapters import build_chapters
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.models import BudgetError, ModelError, Models, RequestBudget, check_choices
from entitytrace.pipeline import run_pipeline
from entitytrace.roster import (
    C2_VERIFY_DECISION_BATCH,
    C2_VERIFY_DECISION_MIN_PROB,
    C3_DECISION_SUBBATCH,
    C3_DECISION_TAU,
    C3_LINK_PROMPT,
    build_c2_verify_decision_batches,
    build_c3_decision_subbatches,
    resolve_roster,
)
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import dumps


_HEX_ID_RE = re.compile(r"\b[0-9a-f]{32,64}\b")


class _ScriptedRosterModels:
    def __init__(self, handlers, *, choices_handler=None, services=None, budget=None):
        self.handlers = handlers
        self.choices_handler = choices_handler
        self.services = services if services is not None else {
            "discovery": Service("discovery", "https://api.deepseek.com", "deepseek-flash", "k-ds"),
            "decision": Service("decision", "https://api.typesafe.ai/v1", "jev-latest", "k-ts"),
        }
        self.budget = budget
        self.calls = []
        self.choice_calls = []

    def generate(self, system, state, stage, validate, *args, **kwargs):
        self.calls.append((stage, state))
        handler = self.handlers[stage]
        value, req_id = handler(state, len(self.calls))
        validate(value)
        return value, req_id

    def choices(self, state, questions, stage):
        self.choice_calls.append((stage, state, questions))
        if self.choices_handler is None:
            raise AssertionError(f"Unexpected choices() call for stage={stage}")
        return self.choices_handler(state, questions, stage, len(self.choice_calls))


class _FakeHttpResponse:
    def __init__(self, body: dict):
        self._raw = dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _seed_describe_snapshot(
    store: Store,
    root: Path,
    run_id: str,
    text: str,
    chapter_titles: list[str],
    entity_specs: list[tuple[str, str, str]],
) -> tuple[dict, list[dict]]:
    """构造完整 query_snapshot 供 run_describe 无外部名册运行依赖地直跑。"""
    txt_path = root / f"{run_id}.txt"
    txt_path.write_bytes(text.encode("utf-8"))
    source = import_text(store, txt_path)
    spans = []
    cursor = 0
    for title in chapter_titles:
        pos = text.index(title, cursor)
        spans.append({"start": pos, "end": pos + len(title), "title": title, "kind": "chapter"})
        cursor = pos + len(title)
    _, chapters = build_chapters(source, spans)
    config = {"identity_algorithm": "entity-roster-v1", "task_chars": 9000}
    state = {
        "chapters": {"complete": True, "chapter_set_version": chapters[0]["chapter_set_version"]},
        "discovery": {"complete": True},
    }
    store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], dumps(config), dumps(state), "now", "now"))
    for ch in chapters:
        store.put(run_id, "chapter", ch)

    entities, rows = [], []
    for entity_id, surface, kind in entity_specs:
        mentions, chapter_stats, surface_counts = [], {}, {}
        pos_cursor = chapters[0]["body_start"] if chapters else 0
        while True:
            at = text.find(surface, pos_cursor)
            if at < 0:
                break
            ch = next(c for c in chapters if c["start"] <= at < c["end"])
            mid = f"m_{entity_id}_{at}"
            mentions.append({
                "id": mid, "chapter_id": ch["id"], "start": at, "end": at + len(surface),
                "text": surface, "link_method": "chapter_local",
            })
            st = chapter_stats.setdefault(ch["id"], {"count": 0, "min_start": at})
            st["count"] += 1
            st["min_start"] = min(st["min_start"], at)
            surface_counts[surface] = surface_counts.get(surface, 0) + 1
            pos_cursor = at + 1
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
        rows.append((entity_id, dumps(header), dumps(mentions)))
    meta = {
        "result_revision": "rev-1", "status": "completed", "source_id": source["id"],
        "chapters": chapters, "counts": {}, "review_counts": {}, "snapshot_version": 2,
    }
    store.save_query_snapshot(run_id, store.query_fingerprint(run_id), meta, entities, [], rows)
    return source, chapters


class StageSplitGuardH1aTests(unittest.TestCase):
    """H-1a 事件空侧整数时间线守卫（>= 6 项）。"""

    def test_h1a_first_chapter_empty_before_side_cut_blocked(self):
        facts = [
            {"id": "f1", "entity_id": "e1", "chapter_index": 1, "start": 120, "end": 130, "verdict": "keep"},
            {"id": "f2", "entity_id": "e1", "chapter_index": 3, "start": 500, "end": 520, "verdict": "keep"},
        ]
        events = [
            {
                "id": "ev1", "entity_id": "e1", "event_type": "costume_change",
                "chapter_index": 1, "start": 50, "end": 65, "verdict": "keep", "stage_split": True,
            },
        ]
        out = apply_stage_split_guards(facts, events)
        self.assertFalse(out[0]["stage_split"])
        self.assertEqual(out[0]["verdict"], "keep")

    def test_h1a_last_chapter_empty_after_side_cut_blocked_fanren_pattern(self):
        facts = [
            {"id": "f1", "entity_id": "e_place", "chapter_index": 180, "start": 1000, "end": 1020, "verdict": "keep"},
            {"id": "f2", "entity_id": "e_place", "chapter_index": 206, "start": 4000, "end": 4020, "verdict": "keep"},
        ]
        events = [
            {
                "id": "ev_end", "entity_id": "e_place", "event_type": "damage",
                "chapter_index": 206, "start": 4500, "end": 4530, "verdict": "keep", "stage_split": True,
            },
        ]
        out = apply_stage_split_guards(facts, events)
        self.assertFalse(out[0]["stage_split"])
        self.assertEqual(out[0]["verdict"], "keep")

    def test_h1a_boundary_chapter_with_both_side_facts_preserved(self):
        facts = [
            {"id": "f1", "entity_id": "e1", "chapter_index": 1, "start": 100, "end": 110, "verdict": "keep"},
            {"id": "f2", "entity_id": "e1", "chapter_index": 5, "start": 900, "end": 920, "verdict": "keep"},
            {"id": "f3", "entity_id": "e1", "chapter_index": 5, "start": 980, "end": 995, "verdict": "keep"},
        ]
        events = [
            {
                "id": "ev1", "entity_id": "e1", "event_type": "costume_change",
                "chapter_index": 5, "start": 940, "end": 955, "verdict": "keep", "stage_split": True,
            },
        ]
        out = apply_stage_split_guards(facts, events)
        self.assertTrue(out[0]["stage_split"])

    def test_h1a_same_quote_same_start_costume_change_preserved(self):
        facts = [
            {"id": "f1", "entity_id": "e_shen", "chapter_index": 1, "start": 20, "end": 30, "verdict": "keep"},
            {"id": "f2", "entity_id": "e_shen", "chapter_index": 2, "start": 80, "end": 85, "verdict": "keep"},
        ]
        events = [
            {
                "id": "ev_same_pos", "entity_id": "e_shen", "event_type": "costume_change",
                "chapter_index": 2, "start": 80, "end": 85, "verdict": "keep", "stage_split": True,
            },
        ]
        out = apply_stage_split_guards(facts, events)
        self.assertTrue(out[0]["stage_split"])

    def test_h1a_owner_change_transform_upgrade_boundary_and_outside_bounds(self):
        facts = [
            {"id": "f1", "entity_id": "e_obj", "chapter_index": 2, "start": 200, "end": 220, "verdict": "keep"},
            {"id": "f2", "entity_id": "e_obj", "chapter_index": 4, "start": 400, "end": 420, "verdict": "keep"},
        ]
        events = [
            {
                "id": "ev_own_in", "entity_id": "e_obj", "event_type": "owner_change",
                "chapter_index": 2, "start": 150, "end": 165, "verdict": "keep", "stage_split": True,
            },
            {
                "id": "ev_out", "entity_id": "e_obj", "event_type": "owner_change",
                "chapter_index": 5, "start": 500, "end": 520, "verdict": "keep", "stage_split": True,
            },
        ]
        out = apply_stage_split_guards(facts, events)
        self.assertTrue(out[0]["stage_split"])
        self.assertFalse(out[1]["stage_split"])

    def test_h1a_middle_chapter_zero_facts_entity_and_non_mutation(self):
        facts = [
            {"id": "f1", "entity_id": "e1", "chapter_index": 1, "start": 100, "end": 110, "verdict": "keep"},
            {"id": "f2", "entity_id": "e1", "chapter_index": 5, "start": 500, "end": 510, "verdict": "keep"},
            {"id": "f_drop", "entity_id": "e2", "chapter_index": 1, "start": 10, "end": 20, "verdict": "drop"},
        ]
        events = [
            {
                "id": "ev_mid", "entity_id": "e1", "event_type": "injury",
                "chapter_index": 3, "start": 300, "end": 310, "verdict": "keep", "stage_split": True,
            },
            {
                "id": "ev_empty", "entity_id": "e2", "event_type": "owner_change",
                "chapter_index": 1, "start": 15, "end": 25, "verdict": "keep", "stage_split": True,
            },
            {
                "id": "ev_dropped", "entity_id": "e1", "event_type": "damage",
                "chapter_index": 1, "start": 10, "end": 20, "verdict": "drop", "stage_split": False,
            },
        ]
        out = apply_stage_split_guards(facts, events)
        self.assertTrue(out[0]["stage_split"])
        self.assertFalse(out[1]["stage_split"])
        self.assertTrue(events[1]["stage_split"])
        self.assertEqual(out[2]["verdict"], "drop")

    def test_h1a_run_describe_verify0_untouched_vs_verify1_guarded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = (
                "第一章 初见\n宝塔高耸十丈。\n"
                "第二章 终末\n宝塔矗立山巅。忽然塔顶坍塌。\n"
            )
            store = Store(root / "db.sqlite3")
            try:
                run_id = "run_h1a_e2e"
                _seed_describe_snapshot(store, root, run_id, text, ["第一章 初见", "第二章 终末"], [("e_tower", "宝塔", "place")])
                svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "k-ds")
                services = {"discovery": svc}

                def fake_opener(req, timeout=None):
                    payload = json.loads(req.data.decode("utf-8"))
                    sys_prompt = payload["messages"][0]["content"]
                    user_state = json.loads(payload["messages"][1]["content"])
                    if sys_prompt == D1_PROMPT:
                        t = user_state.get("text", "")
                        facts, events = [], []
                        if "宝塔高耸十丈" in t:
                            facts.append({"entity_id": "e_tower", "facet": "scale", "value": "十丈",
                                          "quote": "宝塔高耸十丈", "source": "narration"})
                        if "忽然塔顶坍塌" in t:
                            events.append({"entity_id": "e_tower", "event_type": "damage",
                                           "splits_entity": True, "value": "塔顶坍塌", "quote": "忽然塔顶坍塌"})
                        content = dumps({"facts": facts, "events": events})
                    elif sys_prompt == VERIFY_PROMPT:
                        content = dumps({"verdicts": [
                            {"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None}
                            for it in user_state["items"]
                        ]})
                    elif sys_prompt == EVENT_VERIFY_PROMPT:
                        content = dumps({"verdicts": [
                            {"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None, "stage_split": True}
                            for it in user_state["items"]
                        ]})
                    else:
                        raise AssertionError("Unexpected prompt")
                    return _FakeHttpResponse({
                        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    })

                m0 = Models(store, run_id, "inv-v0", services, opener=fake_opener)
                res0 = run_describe(store, run_id, services=services, models=m0, kinds="place", verify=0, concurrency=1)
                ev0 = res0["document"]["entities"][0]["state_events"][0]
                self.assertTrue(ev0["stage_split"])

                m1 = Models(store, run_id, "inv-v1", services, opener=fake_opener)
                res1 = run_describe(store, run_id, services=services, models=m1, kinds="place", verify=1, concurrency=1)
                ev1 = res1["document"]["entities"][0]["state_events"][0]
                self.assertFalse(ev1["stage_split"])
                self.assertEqual(ev1["verdict"] if "verdict" in ev1 else "keep", "keep")
            finally:
                store.close()


class ChapterSyntaxPrefilterH3Tests(unittest.TestCase):
    """H-3 章节候选行非标题语法预筛（>= 6 项）。"""

    def test_h3_filters_chinese_and_english_leading_dialogue_quotes(self):
        for line in ["“你不要走！", "「发生了什么？", "『快看那边", "‘为什么", '"What happened?']:
            self.assertIsNotNone(classify_non_heading(line), line)

    def test_h3_filters_paired_quote_short_lines(self):
        for line in ["“对啊。”", "\"Exactly.\"", "「知道了！」", "『小心』", "‘走吧’"]:
            self.assertIsNotNone(classify_non_heading(line), line)

    def test_h3_filters_mid_punct_with_sentence_end_stop(self):
        for line in ["他叹了口气，转身离去。", "众人闻言大惊；片刻后沉默！", "Well, he thought; never mind!"]:
            self.assertIsNotNone(classify_non_heading(line), line)

    def test_h3_exempts_numbered_chapters_with_quotes_or_punctuation(self):
        for line in [
            "第十二章 风雨，夜归人！",
            "第15回 “真假猴王”",
            "Chapter 3: Darkness, Falls.",
            "卷三 初入灵界",
            "12. 归途",
        ]:
            self.assertTrue(is_exempt_heading_syntax(line), line)
            self.assertIsNone(classify_non_heading(line), line)

    def test_h3_exempts_special_front_and_back_matter_titles(self):
        for line in [
            "楔子",
            "序章 天地初开",
            "序言",
            "尾声",
            "番外 昔日故人",
            "后记",
            "Prologue",
            "Epilogue",
            "Part II",
            "Appendix",
        ]:
            self.assertTrue(is_exempt_heading_syntax(line), line)
            self.assertIsNone(classify_non_heading(line), line)

    def test_h3_gap_fill_recovers_from_raw_candidates_and_path_counts_7_keys(self):
        text = (
            "第一章 启程\n少年背上行囊离开故乡，一路向东行走百里。\n"
            "“不要回头。”老者说完便转身。\n"
            "第二章 迷林\n林中雾气弥漫，鸟兽绝迹。\n"
            "第三章 出山\n终于穿过密林看到炊烟。\n"
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "book.txt"
            p.write_bytes(text.encode("utf-8"))
            store = Store(root / "db.sqlite3")
            try:
                source = import_text(store, p)
                raw = collect_candidates(text)
                filtered = filter_heading_candidates(raw)
                self.assertGreater(len(raw), len(filtered))

                seen_batch_titles = []

                class _FakeChapModels:
                    def generate(self, prompt, state, stage, validate, *args, **kwargs):
                        cands = state["candidates"]
                        seen_batch_titles.extend(c["text"] for c in cands)
                        headings = [
                            {"id": c["id"], "kind": "chapter"}
                            for c in cands
                            if c["text"] in ("第一章 启程", "第三章 出山")
                        ]
                        val = {"headings": headings}
                        validate(val)
                        return val, "req-ch"

                with patch("entitytrace.chapters._fill_chapter_gaps", wraps=__import__("entitytrace.chapters", fromlist=["_fill_chapter_gaps"])._fill_chapter_gaps) as spy_fill:
                    store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("r-gap", source["id"], "{}", "{}", "now", "now"))
                    outcome = detect_chapters(store, _FakeChapModels(), source, "r-gap", lambda _: None, concurrency=1)
                    self.assertEqual(len(spy_fill.call_args[0][1]), len(raw))

                self.assertEqual(seen_batch_titles.count("“不要回头。”老者说完便转身。"), 0)
                self.assertEqual([c["title"] for c in outcome["chapters"]], ["第一章 启程", "第二章 迷林", "第三章 出山"])
                self.assertEqual(
                    set(outcome["path_counts"]),
                    {"candidates", "labeled", "adopted", "dropped_no_body", "unnumbered", "numbering_issues", "label_requests"},
                )
                self.assertEqual(outcome["path_counts"]["candidates"], len(filtered))
            finally:
                store.close()

    def test_h3_detect_chapters_preserves_mixed_headings_with_fewer_candidates(self):
        text = (
            "楔子\n古剑沉眠千年。\n"
            "“这就是传说中的灵剑吗？”他喃喃道。\n"
            "他停下脚步，望着远方。\n"
            "第一章 重逢，故人！\n两人在渡口相遇，长谈一夜。\n"
            "第二章 别离\n天亮后各奔东西。\n"
        )
        raw = collect_candidates(text)
        filtered = filter_heading_candidates(raw)
        titles = [c["title"] for c in filtered]
        self.assertIn("楔子", titles)
        self.assertIn("第一章 重逢，故人！", titles)
        self.assertIn("第二章 别离", titles)
        self.assertNotIn("“这就是传说中的灵剑吗？”他喃喃道。", titles)
        self.assertNotIn("他停下脚步，望着远方。", titles)


class DecisionSwitchAndConfigTests(unittest.TestCase):
    """models.py 兼容与 --decision 开关/无 Key 静默回退契约（>= 7 项）。"""

    def test_check_choices_allows_jev_latest_to_specific_jev_version(self):
        questions = {"q1": {"type": "choice", "instructions": "x", "criteria": {"keep": "k", "drop": "d"}}}
        raw = {
            "model": "jev-1.14.0",
            "answers": {
                "q1": {"type": "choice", "choice": "keep", "probabilities": {"keep": 0.91, "drop": 0.09}, "confidence": 0.91},
            },
        }
        ans = check_choices(raw, questions, "jev-latest")
        self.assertEqual(ans["q1"]["choice"], "keep")

    def test_check_choices_strict_equality_for_explicit_version(self):
        questions = {"q1": {"type": "choice", "instructions": "x", "criteria": {"keep": "k", "drop": "d"}}}
        raw = {
            "model": "jev-1.14.0",
            "answers": {
                "q1": {"type": "choice", "choice": "keep", "probabilities": {"keep": 0.9, "drop": 0.1}, "confidence": 0.9},
            },
        }
        with self.assertRaises(ModelError):
            check_choices(raw, questions, "jev-1.13.0")

    def test_check_choices_rejects_invalid_jev_prefix_or_non_jev_model(self):
        questions = {"q1": {"type": "choice", "instructions": "x", "criteria": {"keep": "k", "drop": "d"}}}
        for bad_model in ("jev-", "deepseek-flash", "", None):
            raw = {
                "model": bad_model,
                "answers": {
                    "q1": {"type": "choice", "choice": "keep", "probabilities": {"keep": 0.9, "drop": 0.1}, "confidence": 0.9},
                },
            }
            with self.assertRaises(ModelError):
                check_choices(raw, questions, "jev-latest")

    def test_resolve_decision_mode_precedence_cli_env_file_default_and_case(self):
        self.assertEqual(DECISION_MODES, ("auto", "on", "off"))
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / ".env"
            cfg.write_text("ENTITYTRACE_DECISION_MODE=OFF\n", encoding="utf-8")
            clean_env = {k: v for k, v in os.environ.items() if k != "ENTITYTRACE_DECISION_MODE"}
            with patch.dict(os.environ, clean_env, clear=True):
                self.assertEqual(resolve_decision_mode(None, None), "auto")
                self.assertEqual(resolve_decision_mode(None, cfg), "off")
                with patch.dict(os.environ, {"ENTITYTRACE_DECISION_MODE": "On"}, clear=False):
                    self.assertEqual(resolve_decision_mode(None, cfg), "on")
                    self.assertEqual(resolve_decision_mode(" AUTO ", cfg), "auto")

    def test_resolve_decision_mode_invalid_raises_config_error(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_decision_mode("bogus")
        self.assertEqual(getattr(ctx.exception, "et_category", None), "config")

    def test_load_services_auto_no_key_silent_vs_on_missing_key(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / ".env"
            cfg.write_text("DEEPSEEK_API_KEY=ds_key\n", encoding="utf-8")
            clean_env = {
                k: v for k, v in os.environ.items()
                if not k.startswith("TYPESAFE_") and not k.startswith("DEEPSEEK_") and k != "ENTITYTRACE_DECISION_MODE"
            }
            with patch.dict(os.environ, clean_env, clear=True):
                svcs = load_services(cfg, require_decision=False, decision_mode="auto")
                self.assertEqual(set(svcs), {"discovery"})
                with self.assertRaises(ValueError) as ctx:
                    load_services(cfg, require_decision=False, decision_mode="on")
                self.assertEqual(getattr(ctx.exception, "et_category", None), "config")

    def test_load_services_off_ignores_key_and_groups_v2_stays_jev_1_13_0(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / ".env"
            cfg.write_text("DEEPSEEK_API_KEY=ds_key\nTYPESAFE_API_KEY=ts_key\n", encoding="utf-8")
            clean_env = {
                k: v for k, v in os.environ.items()
                if not k.startswith("TYPESAFE_") and not k.startswith("DEEPSEEK_") and k != "ENTITYTRACE_DECISION_MODE"
            }
            with patch.dict(os.environ, clean_env, clear=True):
                svcs_off = load_services(cfg, require_decision=False, decision_mode="off")
                self.assertEqual(set(svcs_off), {"discovery"})

                svcs_auto = load_services(cfg, require_decision=False, decision_mode="auto")
                self.assertEqual(set(svcs_auto), {"discovery", "decision"})
                self.assertEqual(svcs_auto["decision"].model, "jev-latest")

                svcs_gv2 = load_services(cfg, require_decision=True, decision_mode="off")
                self.assertEqual(svcs_gv2["decision"].model, "jev-1.13.0")

    def test_is_typesafe_decision_service_distinguishes_mock_discovery(self):
        ds = Service("discovery", "https://api.deepseek.com/v1", "deepseek-flash", "k")
        ts_real = Service("decision", "https://api.typesafe.ai/v1", "jev-latest", "ts-k")
        ts_mock = Service("decision", "http://127.0.0.1:9/v1", "jev-1.14.0", "ts-k", provider="typesafe")
        self.assertFalse(is_typesafe_decision_service(None))
        self.assertFalse(is_typesafe_decision_service(ds))
        self.assertTrue(is_typesafe_decision_service(ts_real))
        self.assertTrue(is_typesafe_decision_service(ts_mock))
        self.assertFalse(is_typesafe_decision_service(Service("decision", "https://api.typesafe.ai/v1", "jev-latest", "")))

    def test_cli_run_and_describe_no_key_auto_vs_off_byte_identical_ids(self):
        did_none = build_describe_id(10, ["person"], ["e1"], "run_1", "deepseek", "deepseek-flash")
        did_no_dec = build_describe_id(10, ["person"], ["e1"], "run_1", "deepseek", "deepseek-flash", decision_service=None)
        self.assertEqual(did_none, did_no_dec)
        did_with_dec = build_describe_id(
            10, ["person"], ["e1"], "run_1", "deepseek", "deepseek-flash",
            decision_service={"name": "decision", "url": "https://api.typesafe.ai/v1", "model": "jev-latest", "provider": "deepseek"},
        )
        self.assertNotEqual(did_none, did_with_dec)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            novel = root / "novel.txt"
            novel.write_text("第一章 相遇\n沈墨看着林川。\n", encoding="utf-8")
            cfg_no_ts = root / "no_ts.env"
            cfg_no_ts.write_text("DEEPSEEK_API_KEY=ds_key\n", encoding="utf-8")

            clean_env = {
                k: v for k, v in os.environ.items()
                if not k.startswith("TYPESAFE_") and not k.startswith("DEEPSEEK_") and k != "ENTITYTRACE_DECISION_MODE"
            }
            with patch.dict(os.environ, clean_env, clear=True), patch("entitytrace.pipeline.detect_chapters") as m_ch, patch("entitytrace.pipeline.discover") as m_di, patch("entitytrace.pipeline.resolve_roster") as m_ro:
                m_ch.return_value = {"complete": True, "chapters": []}
                m_di.return_value = {"complete": True}
                m_ro.return_value = {"complete": True, "entities": 0}
                rc_auto = cli_main([
                    "run", str(novel), "--db", str(root / "a.sqlite3"),
                    "--output", str(root / "out_a"), "--config", str(cfg_no_ts),
                    "--decision", "auto", "--stop-after", "chapters", "--json",
                ])
                rc_off = cli_main([
                    "run", str(novel), "--db", str(root / "b.sqlite3"),
                    "--output", str(root / "out_b"), "--config", str(cfg_no_ts),
                    "--decision", "off", "--stop-after", "chapters", "--json",
                ])
                self.assertEqual(rc_auto, 2)
                self.assertEqual(rc_off, 2)
                man_a = next((root / "out_a").glob("*/manifest.json"))
                man_b = next((root / "out_b").glob("*/manifest.json"))
                da = json.loads(man_a.read_text(encoding="utf-8"))
                db = json.loads(man_b.read_text(encoding="utf-8"))
                self.assertEqual(da["run_id"], db["run_id"])
                self.assertEqual(da["config"], db["config"])
                self.assertEqual(da["config"]["decision_concurrency"], 3)
                svcs_auto = load_services(cfg_no_ts, require_decision=False, decision_mode="auto")
                svcs_off = load_services(cfg_no_ts, require_decision=False, decision_mode="off")
                self.assertEqual(
                    build_describe_id(20, ["person"], ["e1"], da["run_id"], "deepseek", "deepseek-flash", decision_service=svcs_auto.get("decision")),
                    build_describe_id(20, ["person"], ["e1"], db["run_id"], "deepseek", "deepseek-flash", decision_service=svcs_off.get("decision")),
                )


class DecisionRuntimeRoutingAndGuardsTests(unittest.TestCase):
    """D-1/D-2/D-3 运行时分流、软熔断、BudgetError 与硬守卫不绕过（>= 9 项）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "rt.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def _seed_roster_book(self, text: str, ch1_disc: dict, ch2_disc: dict) -> tuple[dict, str, list[dict]]:
        path = self.root / "book.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1 = text.index("第一章")
        idx2 = text.index("第二章")
        _, chapters = build_chapters(source, [
            {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
            {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
        ])
        run_id = "run_dec_rt"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        for task, disc in zip(tasks, [ch1_disc, ch2_disc]):
            grounded = ground_result(source, task, disc, "req_" + task["id"])
            recs = [
                (k, r)
                for k, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                for r in grounded[key]
            ]
            self.store.finish_task(run_id, task, {"request_id": "req_" + task["id"]}, recs)
        return source, run_id, tasks

    def test_d2_c2_verify_decision_batching_cutoff_and_cannot_merge_guard(self):
        self.assertEqual(C2_VERIFY_DECISION_BATCH, 5)
        self.assertEqual(C2_VERIFY_DECISION_MIN_PROB, 0.50)
        fake_tasks = [
            {
                "id": f"vt{i}",
                "prop": {"left": f"L{i}", "right": f"R{i}", "reason": "r"},
                "state": {
                    "left": {"name": f"甲{i}", "kind": "person", "strong_surfaces": [f"甲{i}"], "excerpts": []},
                    "right": {"name": f"乙{i}", "kind": "person", "strong_surfaces": [f"乙{i}"], "excerpts": []},
                },
            }
            for i in range(7)
        ]
        batches = build_c2_verify_decision_batches(fake_tasks)
        self.assertEqual([len(b["pid_to_vt"]) for b in batches], [5, 2])
        for b in batches:
            self.assertNotRegex(dumps({"state": b["state"], "questions": b["questions"]}), _HEX_ID_RE)

        text = "第一章\n李青石在东街。王玄在东街。\n第二章\n青石兄在西街。\n"
        p_path = self.root / "b2.txt"
        p_path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, p_path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(source, [
            {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
            {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
        ])
        run_id = "r_d2_guard"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        for task, disc in zip(tasks, [
            {"mentions": [{"text": "李青石", "kind": "person", "refs": [u1]}, {"text": "王玄", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "青石兄", "kind": "person", "refs": [u2]}], "links": []},
        ]):
            g = ground_result(source, task, disc, "rq_" + task["id"])
            recs = [(k, r) for k, key in [("mention", "mentions"), ("link", "links")] for r in g[key]]
            self.store.finish_task(run_id, task, {"request_id": "rq_" + task["id"]}, recs)

        def h_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "李青石", "kind": "person", "surfaces": ["李青石"], "evidence_ref": uid},
                        {"local_id": "e2", "name": "王玄", "kind": "person", "surfaces": ["王玄"], "evidence_ref": uid},
                    ],
                    "assignments": [], "not_entity": [],
                }, f"c1_{n}"
            return {
                "entities": [{"local_id": "e1", "name": "青石兄", "kind": "person", "surfaces": ["青石兄"], "evidence_ref": uid}],
                "assignments": [], "not_entity": [],
            }, f"c1_{n}"

        def h_c2_propose(state, n):
            ents = state["entries"]
            li = next(e["entry_id"] for e in ents if e["name"] == "李青石")
            qi = next(e["entry_id"] for e in ents if e["name"] == "青石兄")
            wx = next(e["entry_id"] for e in ents if e["name"] == "王玄")
            return {
                "proposals": [
                    {"left": li, "right": qi, "reason": "字号接近"},
                    {"left": qi, "right": wx, "reason": "强行测试不可并守卫"},
                ]
            }, f"prop_{n}"

        def h_choices(state, questions, stage, n):
            self.assertEqual(stage, "roster-c2-verify")
            answers = {}
            pairs = {p["pair_id"]: p for p in state["pairs"]}
            for qid, q in questions.items():
                pair = pairs[qid]
                names = {pair["left"]["name"], pair["right"]["name"]}
                if names == {"李青石", "青石兄"}:
                    answers[qid] = {"type": "choice", "choice": "same", "probabilities": {"same": 0.90, "different": 0.05, "insufficient_evidence": 0.05}, "confidence": 0.90}
                else:
                    # p_max = 0.45 (< 0.50 -> insufficient_evidence), 另一分支测 same 仍受 cannot_merge 拦截
                    answers[qid] = {"type": "choice", "choice": "same", "probabilities": {"same": 0.45, "different": 0.35, "insufficient_evidence": 0.20}, "confidence": 0.45}
            return answers, f"dec_c2v_{n}"

        models = _ScriptedRosterModels(
            {"roster-c1": h_c1, "roster-c2-propose": h_c2_propose},
            choices_handler=h_choices,
        )
        out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(out["complete"])
        self.assertEqual(len(models.choice_calls), 1)
        ro = self.store.records(run_id, "roster_output")[0]
        # 李青石+青石兄合并，但 (青石兄, 王玄) p_max=0.45 转 insufficient_evidence（不合并且进入待审）
        self.assertEqual(len(ro["entities"]), 2)
        self.assertTrue(any(r["stage"] == "roster-c2-verify" for r in ro["reviews"]))

        # 再测 transitive cannot_merge: (李青石, 王玄)=different -> 写入 cannot_merge_entries，拦截 (青石兄, 王玄)=same (0.95) 传递并入
        self.store.db.execute("DELETE FROM tasks WHERE run_id=? AND stage LIKE 'roster%'", (run_id,))
        self.store.db.execute("DELETE FROM records WHERE run_id=? AND kind='roster_output'", (run_id,))
        self.store.db.commit()

        def h_c2_propose_with_conflict(state, n):
            ents = state["entries"]
            li = next(e["entry_id"] for e in ents if e["name"] == "李青石")
            qi = next(e["entry_id"] for e in ents if e["name"] == "青石兄")
            wx = next(e["entry_id"] for e in ents if e["name"] == "王玄")
            return {
                "proposals": [
                    {"left": li, "right": wx, "reason": "建立不可并约束"},
                    {"left": li, "right": qi, "reason": "字号接近"},
                    {"left": qi, "right": wx, "reason": "测试传递不可并拦截"},
                ]
            }, f"prop2_{n}"

        def h_choices_both_same(state, questions, stage, n):
            pairs = {p["pair_id"]: p for p in state["pairs"]}
            answers = {}
            for qid in questions:
                names = {pairs[qid]["left"]["name"], pairs[qid]["right"]["name"]}
                if names == {"李青石", "王玄"}:
                    answers[qid] = {"type": "choice", "choice": "different", "probabilities": {"same": 0.02, "different": 0.95, "insufficient_evidence": 0.03}, "confidence": 0.95}
                else:
                    answers[qid] = {"type": "choice", "choice": "same", "probabilities": {"same": 0.95, "different": 0.03, "insufficient_evidence": 0.02}, "confidence": 0.95}
            return answers, f"dec_c2v_s_{n}"

        models2 = _ScriptedRosterModels(
            {"roster-c1": h_c1, "roster-c2-propose": h_c2_propose_with_conflict},
            choices_handler=h_choices_both_same,
        )
        out2 = resolve_roster(self.store, models2, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(out2["complete"])
        ro2 = self.store.records(run_id, "roster_output")[0]
        self.assertEqual(len(ro2["entities"]), 2)

    def test_d1_c3_decision_all_high_conf_zero_generative_calls(self):
        self.assertEqual(C3_DECISION_SUBBATCH, 15)
        self.assertEqual(C3_DECISION_TAU, 0.90)
        c3_task = {
            "id": "ct_1",
            "allowed_choices": {"e1", "none", "uncertain"},
            "state": {
                "candidates": [{"entity_id": "e1", "name": "沈墨", "kind": "person", "strong_surfaces": ["沈墨"], "evidence": "片段"}],
                "items": [
                    {"mention_id": f"m{i}", "surface": "他", "context": "ctx", "candidate_entity_ids": ["e1"]}
                    for i in range(1, 19)
                ],
            },
        }
        subs = build_c3_decision_subbatches(c3_task)
        self.assertEqual([len(s["chunk_items"]) for s in subs], [15, 3])
        self.assertEqual(set(subs[0]["questions"]["m1"]["criteria"]), {"e1", "none", "uncertain"})

        text = "第一章\n沈墨在厅内。\n第二章\n沈墨与这位少年走进来。\n"
        path = self.root / "d1.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(source, [
            {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
            {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
        ])
        run_id = "r_d1_all_high"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        for tk, disc in zip(tasks, [
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u2]}, {"text": "这位少年", "kind": "person", "refs": [u2]}], "links": []},
        ]):
            g = ground_result(source, tk, disc, "rq_" + tk["id"])
            self.store.finish_task(run_id, tk, {"request_id": "rq_" + tk["id"]}, [("mention", m) for m in g["mentions"]])

        def h_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {"entities": [{"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid}], "assignments": [], "not_entity": []}, f"c1_{n}"
            return {
                "entities": [
                    {"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid},
                    {"local_id": "e2", "name": "这位少年", "kind": "person", "surfaces": [{"surface": "这位少年", "surface_type": "description"}], "evidence_ref": uid},
                ],
                "assignments": [], "not_entity": [],
            }, f"c1_{n}"

        gen_c3_calls = []

        def h_c3_fail_if_called(state, n):
            gen_c3_calls.append(state)
            raise AssertionError("All >= 0.90 should skip generative C3")

        def h_choices(state, questions, stage, n):
            self.assertEqual(stage, "roster-c3-decision")
            ans = {}
            for qid, q in questions.items():
                e_choice = next(k for k in q["criteria"] if k not in ("none", "uncertain"))
                ans[qid] = {"type": "choice", "choice": e_choice, "probabilities": {e_choice: 0.95, "none": 0.03, "uncertain": 0.02}, "confidence": 0.95}
            return ans, f"dec_c3_{n}"

        models = _ScriptedRosterModels(
            {"roster-c1": h_c1, "roster-c2-propose": lambda s, n: ({"proposals": []}, f"p_{n}"), "roster-c3-link": h_c3_fail_if_called},
            choices_handler=h_choices,
        )
        out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(out["complete"])
        self.assertEqual(len(gen_c3_calls), 0)
        self.assertEqual(len(models.choice_calls), 1)
        ro = self.store.records(run_id, "roster_output")[0]
        self.assertEqual(len(ro["entities"]), 1)
        self.assertEqual(ro["entities"][0]["mention_count"], 3)

    def test_d1_c3_hybrid_shrinks_generative_payload_to_low_conf_subset(self):
        text = "第一章\n沈墨与白鹤在此。\n第二章\n沈墨与白鹤看见黑衣人和灰衣人。\n"
        source, run_id, tasks = self._prepare_c3_two_items_fixture(text)

        def h_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid},
                        {"local_id": "e2", "name": "白鹤", "kind": "person", "surfaces": ["白鹤"], "evidence_ref": uid},
                    ],
                    "assignments": [], "not_entity": [],
                }, f"c1_{n}"
            return {
                "entities": [
                    {"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid},
                    {"local_id": "e2", "name": "白鹤", "kind": "person", "surfaces": ["白鹤"], "evidence_ref": uid},
                    {"local_id": "e3", "name": "黑衣人", "kind": "person", "surfaces": [{"surface": "黑衣人", "surface_type": "description"}], "evidence_ref": uid},
                    {"local_id": "e4", "name": "灰衣人", "kind": "person", "surfaces": [{"surface": "灰衣人", "surface_type": "description"}], "evidence_ref": uid},
                ],
                "assignments": [], "not_entity": [],
            }, f"c1_{n}"

        shrunk_seen = []

        def h_c3_gen(state, n):
            shrunk_seen.append(state)
            shen_id = next(c["entity_id"] for c in state["candidates"] if c["name"] == "白鹤")
            return {"links": [{"mention_id": state["items"][0]["mention_id"], "selected_entity_id": shen_id, "reason": "gen"}]}, f"c3g_{n}"

        def h_choices(state, questions, stage, n):
            ans = {}
            by_mid = {it["mention_id"]: it for it in state["items"]}
            c_by_id = {c["entity_id"]: c for c in state["candidates"]}
            for qid, q in questions.items():
                c_keys = [k for k in q["criteria"] if k not in ("none", "uncertain")]
                if by_mid[qid]["surface"] == "黑衣人":
                    shen_cid = next(k for k in c_keys if c_by_id[k]["name"] == "沈墨")
                    other_cid = next(k for k in c_keys if k != shen_cid)
                    ans[qid] = {"type": "choice", "choice": shen_cid, "probabilities": {shen_cid: 0.94, other_cid: 0.02, "none": 0.02, "uncertain": 0.02}, "confidence": 0.94}
                else:
                    ans[qid] = {"type": "choice", "choice": c_keys[0], "probabilities": {c_keys[0]: 0.72, c_keys[1]: 0.20, "none": 0.04, "uncertain": 0.04}, "confidence": 0.72}
            return ans, f"dec_{n}"

        models = _ScriptedRosterModels(
            {"roster-c1": h_c1, "roster-c2-propose": lambda s, n: ({"proposals": []}, f"p_{n}"), "roster-c3-link": h_c3_gen},
            choices_handler=h_choices,
        )
        out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(out["complete"])
        self.assertEqual(len(shrunk_seen), 1)
        self.assertEqual([it["surface"] for it in shrunk_seen[0]["items"]], ["灰衣人"])

    def _prepare_c3_two_items_fixture(self, text: str):
        path = self.root / "c3mix.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(source, [
            {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
            {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
        ])
        run_id = "r_c3mix"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        for tk, disc in zip(tasks, [
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u1]}, {"text": "白鹤", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [
                {"text": "沈墨", "kind": "person", "refs": [u2]}, {"text": "白鹤", "kind": "person", "refs": [u2]},
                {"text": "黑衣人", "kind": "person", "refs": [u2]}, {"text": "灰衣人", "kind": "person", "refs": [u2]},
            ], "links": []},
        ]):
            g = ground_result(source, tk, disc, "rq_" + tk["id"])
            self.store.finish_task(run_id, tk, {"request_id": "rq_" + tk["id"]}, [("mention", m) for m in g["mentions"]])
        return source, run_id, tasks

    def test_d1_c3_high_conf_proper_mismatch_still_blocked_by_d12(self):
        text = "第一章\n沈知远与沈东家说话。\n第二章\n沈知远与沈西家说话。\n第三章\n沈知远、韩子敬、督办大人、老管家在堂中。\n"
        path = self.root / "d12.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        i1, i2, i3 = text.index("第一章"), text.index("第二章"), text.index("第三章")
        _, chapters = build_chapters(source, [
            {"start": i1, "end": i1 + 3, "title": "第一章", "kind": "chapter"},
            {"start": i2, "end": i2 + 3, "title": "第二章", "kind": "chapter"},
            {"start": i3, "end": i3 + 3, "title": "第三章", "kind": "chapter"},
        ])
        run_id = "run_d12"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chapters:
            self.store.put(run_id, "chapter", ch)
        tasks = plan_tasks(source, chapters, 9000)
        u1, u2, u3 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"], tasks[2]["units"][0]["id"]
        for tk, disc in zip(tasks, [
            {"mentions": [{"text": "沈知远", "kind": "person", "refs": [u1]}, {"text": "沈东家", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "沈知远", "kind": "person", "refs": [u2]}, {"text": "沈西家", "kind": "person", "refs": [u2]}], "links": []},
            {"mentions": [
                {"text": "沈知远", "kind": "person", "refs": [u3]},
                {"text": "韩子敬", "kind": "person", "refs": [u3]},
                {"text": "督办大人", "kind": "person", "refs": [u3]},
                {"text": "老管家", "kind": "person", "refs": [u3]},
            ], "links": []},
        ]):
            g = ground_result(source, tk, disc, "rq_" + tk["id"])
            self.store.finish_task(run_id, tk, {"request_id": "rq_" + tk["id"]}, [("mention", m) for m in g["mentions"]])

        def h_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "沈知远", "kind": "person", "surfaces": [{"surface": "沈知远", "surface_type": "proper_name"}, {"surface": "沈东家", "surface_type": "proper_name"}], "evidence_ref": uid},
                    ],
                    "assignments": [], "not_entity": [],
                }, f"c1_{n}"
            if state["chapter_title"] == "第二章":
                return {
                    "entities": [
                        {"local_id": "e1", "name": "沈知远", "kind": "person", "surfaces": [{"surface": "沈知远", "surface_type": "proper_name"}, {"surface": "沈西家", "surface_type": "proper_name"}], "evidence_ref": uid},
                    ],
                    "assignments": [], "not_entity": [],
                }, f"c1_{n}"
            m_shen = next(r["mentions"][0]["mention_id"] for r in state["surfaces"] if r["surface"] == "沈知远")
            return {
                "entities": [
                    {"local_id": "e1", "name": "韩子敬", "kind": "person", "surfaces": [{"surface": "韩子敬", "surface_type": "proper_name"}], "evidence_ref": uid},
                    {"local_id": "e2", "name": "督办大人", "kind": "person", "surfaces": [{"surface": "督办大人", "surface_type": "title_or_kinship"}], "evidence_ref": uid},
                    {"local_id": "e3", "name": "老管家", "kind": "person", "surfaces": [{"surface": "老管家", "surface_type": "title_or_kinship"}], "evidence_ref": uid},
                ],
                "assignments": [{"mention_id": m_shen, "surface": "沈知远", "ref": uid, "local_id": None, "reason": "未决"}],
                "not_entity": [],
            }, f"c1_{n}"

        def h_c2_review(state, n):
            groups = []
            for idx, rw in enumerate(state["rows"], start=1):
                groups.append({"group_id": f"g{idx}", "name": rw["name"], "kind": "person", "row_ids": [rw["row_id"]]})
            return {"groups": groups, "unresolved_row_ids": [], "unresolved_member_ids": []}, f"rev_{n}"

        def h_choices(state, questions, stage, n):
            by_mid = {it["mention_id"]: it for it in state["items"]}
            c_by_id = {c["entity_id"]: c for c in state["candidates"]}
            ans = {}
            for qid, q in questions.items():
                c_keys = [k for k in q["criteria"] if k not in ("none", "uncertain")]
                if by_mid[qid]["surface"] == "督办大人":
                    han_id = next(k for k in c_keys if c_by_id[k]["name"] == "韩子敬")
                    probs = {k: (0.96 if k == han_id else 0.0) for k in q["criteria"]}
                    probs["none"] = 0.04
                    ans[qid] = {"type": "choice", "choice": han_id, "probabilities": probs, "confidence": 0.96}
                else:
                    probs = {k: (0.60 if k == c_keys[0] else 0.0) for k in q["criteria"]}
                    probs["none"] = 0.40
                    ans[qid] = {"type": "choice", "choice": c_keys[0], "probabilities": probs, "confidence": 0.60}
            return ans, f"dec_d12_{n}"

        def h_c3_hijack(state, n):
            han_id = next(c["entity_id"] for c in state["candidates"] if c["name"] == "韩子敬")
            return {"links": [{"mention_id": it["mention_id"], "selected_entity_id": han_id, "reason": "hijack"} for it in state["items"]]}, f"c3_{n}"

        models = _ScriptedRosterModels(
            {"roster-c1": h_c1, "roster-c2-review": h_c2_review, "roster-c2-propose": lambda s, n: ({"proposals": []}, f"p_{n}"), "roster-c3-link": h_c3_hijack},
            choices_handler=h_choices,
        )
        out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None)
        self.assertTrue(out["complete"])
        ro = self.store.records(run_id, "roster_output")[0]
        self.assertGreaterEqual(ro["roster_summary"].get("c3_rejected_name_mismatch", 0), 1)
        self.assertTrue(any(e.get("link_methods", {}).get("c3_rejected_name_mismatch", 0) > 0 for e in ro["entities"]))
        han = next(e for e in ro["entities"] if "韩子敬" in e.get("aliases", []))
        self.assertGreaterEqual(han["link_methods"].get("c3_model", 0), 2)

    def test_d3_verify_decision_batches_le_8_no_hex_leak_and_hybrid_fallback(self):
        self.assertEqual(DESC_VERIFY_DECISION_TAU, 0.80)
        facts = [
            {"id": f"{i:064x}", "entity_id": f"{i+100:064x}", "chapter_id": "ch1",
             "start": i * 20, "end": i * 20 + 4, "facet": "costume", "value": f"v{i}", "quote": "q"}
            for i in range(10)
        ]
        e_kinds = {f["entity_id"]: "person" for f in facts}
        e_names = {f["entity_id"]: ["沈墨"] for f in facts}
        chapters = [{"id": "ch1", "kind": "chapter", "start": 0, "end": 1000}]
        batches = build_verify_decision_batches("desc1", facts, e_kinds, e_names, "q" * 1000, chapters)
        self.assertEqual([len(b["items_with_facts"]) for b in batches], [8, 2])
        for b in batches:
            self.assertRegex(",".join(b["questions"]), r"^(F[1-8],?)+$")
            self.assertNotRegex(dumps({"state": b["state"], "questions": b["questions"]}), _HEX_ID_RE)

        # Run full run_describe with verify=3 & mixed Decision probabilities -> fallback only for < 0.80
        text = "第一章\n沈墨穿着青衫。沈墨身长八尺。\n"
        run_id = "r_d3_f"
        _seed_describe_snapshot(self.store, self.root, run_id, text, ["第一章"], [("e_shen", "沈墨", "person")])
        svc_ds = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "k-ds")
        svc_ts = Service("decision", "https://api.typesafe.ai/v1", "jev-latest", "k-ts")
        services = {"discovery": svc_ds, "decision": svc_ts}
        gen_verify_items = []
        dec_payloads = []

        def opener(req, timeout=None):
            url = req.full_url
            payload = json.loads(req.data.decode("utf-8"))
            if url.endswith("/systemone"):
                self.assertNotRegex(dumps({"state": payload["state"], "questions": payload["questions"]}), _HEX_ID_RE)
                dec_payloads.append(payload)
                ans = {}
                by_id = {it["id"]: it for it in payload["state"]["items"]}
                for qid in payload["questions"]:
                    if by_id[qid]["value"] == "青衫":
                        ans[qid] = {"type": "choice", "choice": "keep", "probabilities": {"keep": 0.90, "drop": 0.10}, "confidence": 0.90}
                    else:
                        ans[qid] = {"type": "choice", "choice": "keep", "probabilities": {"keep": 0.65, "drop": 0.35}, "confidence": 0.65}
                return _FakeHttpResponse({"model": "jev-1.14.0", "answers": ans})
            sys_prompt = payload["messages"][0]["content"]
            u = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                content = dumps({"facts": [
                    {"entity_id": "e_shen", "facet": "costume", "value": "青衫", "quote": "穿着青衫", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "build", "value": "八尺", "quote": "身长八尺", "source": "narration"},
                ], "events": []})
            elif sys_prompt == VERIFY_PROMPT:
                gen_verify_items.append([it["value"] for it in u["items"]])
                content = dumps({"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "drop", "reason": "not_description"} for it in u["items"]]})
            else:
                raise AssertionError(sys_prompt)
            return _FakeHttpResponse({"choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        m = Models(self.store, run_id, "d3_mix", services, discovery_concurrency=1, decision_concurrency=2, opener=opener)
        res = run_describe(self.store, run_id, services=services, models=m, kinds="person", verify=3, concurrency=1, decision_concurrency=2)
        self.assertEqual(len(dec_payloads), 1)
        self.assertEqual(len(gen_verify_items), 1)
        self.assertTrue(all(vals == ["八尺"] for vals in gen_verify_items))
        votes = self.store.rows("SELECT pass, verdict FROM fact_verify_votes WHERE describe_id=?", (res["describe_id"],))
        self.assertEqual(len(votes), 6)
        kept_vals = [f["value"] for f in res["document"]["entities"][0]["facts"]]
        self.assertEqual(kept_vals, ["青衫"])

    def test_d3_event_verify_decision_batches_le_6_disguise_and_h1a_enforced(self):
        events_sample = [
            {"id": f"{i:064x}", "entity_id": "0" * 64, "chapter_id": "ch1",
             "start": i * 20, "end": i * 20 + 5, "event_type": "costume_change", "value": f"v{i}", "quote": "q"}
            for i in range(8)
        ]
        chs = [{"id": "ch1", "kind": "chapter", "start": 0, "end": 1000}]
        ebatches = build_event_verify_decision_batches(
            "d1", events_sample, {"0" * 64: "person"}, {"0" * 64: ["沈墨"]}, "q" * 1000, chs,
        )
        self.assertEqual([len(b["items_with_events"]) for b in ebatches], [6, 2])
        for b in ebatches:
            self.assertNotRegex(dumps({"state": b["state"], "questions": b["questions"]}), _HEX_ID_RE)

        text = "第一章 初见\n沈墨穿着青衫。\n第二章 尾声\n沈墨易容改扮。\n"
        run_id = "r_d3_ev"
        _seed_describe_snapshot(self.store, self.root, run_id, text, ["第一章 初见", "第二章 尾声"], [("e_shen", "沈墨", "person")])
        svc_ds = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "k-ds")
        svc_ts = Service("decision", "https://api.typesafe.ai/v1", "jev-latest", "k-ts")
        services = {"discovery": svc_ds, "decision": svc_ts}

        def opener(req, timeout=None):
            payload = json.loads(req.data.decode("utf-8"))
            if req.full_url.endswith("/systemone"):
                ans = {}
                for qid in payload["questions"]:
                    if qid.startswith("F"):
                        ans[qid] = {"type": "choice", "choice": "keep", "probabilities": {"keep": 0.95, "drop": 0.05}, "confidence": 0.95}
                    else:
                        ans[qid] = {"type": "choice", "choice": "keep_split", "probabilities": {"keep_split": 0.96, "keep_nosplit": 0.02, "drop": 0.02}, "confidence": 0.96}
                return _FakeHttpResponse({"model": "jev-1.14.0", "answers": ans})
            sys_prompt = payload["messages"][0]["content"]
            t = json.loads(payload["messages"][1]["content"]).get("text", "")
            if sys_prompt == D1_PROMPT:
                facts, evs = [], []
                if "穿着青衫" in t:
                    facts.append({"entity_id": "e_shen", "facet": "costume", "value": "青衫", "quote": "穿着青衫", "source": "narration"})
                if "易容改扮" in t:
                    evs.append({"entity_id": "e_shen", "event_type": "disguise", "splits_entity": True, "value": "易容", "quote": "易容改扮"})
                    evs.append({"entity_id": "e_shen", "event_type": "injury", "splits_entity": True, "value": "受伤", "quote": "沈墨易容改扮"})
                content = dumps({"facts": facts, "events": evs})
            else:
                raise AssertionError(f"Unexpected gen verify {sys_prompt}")
            return _FakeHttpResponse({"choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        m = Models(self.store, run_id, "d3_ev", services, opener=opener)
        res = run_describe(self.store, run_id, services=services, models=m, kinds="person", verify=1, concurrency=1, decision_concurrency=2)
        evs_out = res["document"]["entities"][0]["state_events"]
        self.assertEqual(len(evs_out), 2)
        self.assertTrue(all(ev["stage_split"] is False for ev in evs_out))

    def test_auto_mode_c2_and_c3_decision_model_error_soft_circuit_breaker(self):
        text = "第一章\n沈墨来了，周玄也来了。\n第二章\n沈墨看着青衫客。\n"
        p = self.root / "auto_fb.txt"
        p.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, p)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chs = build_chapters(source, [
            {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
            {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
        ])
        run_id = "r_auto_fb"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], "{}", "{}", "now", "now"))
        for ch in chs:
            self.store.put(run_id, "chapter", ch)
        tks = plan_tasks(source, chs, 9000)
        u1, u2 = tks[0]["units"][0]["id"], tks[1]["units"][0]["id"]
        for tk, disc in zip(tks, [
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u2]}, {"text": "青衫客", "kind": "person", "refs": [u2]}], "links": []},
        ]):
            g = ground_result(source, tk, disc, "rq_" + tk["id"])
            self.store.finish_task(run_id, tk, {"request_id": "rq_" + tk["id"]}, [("mention", m) for m in g["mentions"]])

        budget = RequestBudget(max_network_failures=9)
        budget.consecutive_network_failures = 1

        def h_choices_fail(state, questions, stage, n):
            budget.consecutive_network_failures += 2
            raise ModelError("TypeSafe 503", status=503)

        def h_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {"entities": [{"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid}], "assignments": [], "not_entity": []}, f"c1_{n}"
            return {"entities": [
                {"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid},
                {"local_id": "e2", "name": "青衫客", "kind": "person", "surfaces": [{"surface": "青衫客", "surface_type": "description"}], "evidence_ref": uid},
            ], "assignments": [], "not_entity": []}, f"c1_{n}"

        c3_gen_called = []

        def h_c3(state, n):
            c3_gen_called.append(1)
            return {"links": [{"mention_id": it["mention_id"], "selected_entity_id": it["candidate_entity_ids"][0], "reason": "fb"} for it in state["items"]]}, f"c3_{n}"

        models = _ScriptedRosterModels(
            {
                "roster-c1": h_c1,
                "roster-c2-propose": lambda s, n: ({"proposals": []}, f"p_{n}"),
                "roster-c3-link": h_c3,
            },
            choices_handler=h_choices_fail,
            budget=budget,
        )
        out = resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None, decision_mode="auto")
        self.assertTrue(out["complete"])
        self.assertEqual(len(models.choice_calls), 1)
        self.assertEqual(len(c3_gen_called), 1)
        self.assertEqual(budget.consecutive_network_failures, 1)

    def test_auto_mode_describe_decision_error_soft_fallback_without_tripping_budget(self):
        text = "第一章\n沈墨穿着青衫。沈墨换了白衣。\n"
        run_id = "r_desc_auto_fb"
        _seed_describe_snapshot(self.store, self.root, run_id, text, ["第一章"], [("e_shen", "沈墨", "person")])
        svc_ds = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "k-ds")
        svc_ts = Service("decision", "https://api.typesafe.ai/v1", "jev-latest", "k-ts")
        services = {"discovery": svc_ds, "decision": svc_ts}
        budget = RequestBudget(max_network_failures=9)
        ts_attempts = []
        gen_verify_calls = []

        def opener(req, timeout=None):
            if req.full_url.endswith("/systemone"):
                ts_attempts.append(1)
                raise __import__("urllib").error.HTTPError(req.full_url, 503, "unavailable", {}, None)
            payload = json.loads(req.data.decode("utf-8"))
            sys_prompt = payload["messages"][0]["content"]
            u = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                content = dumps({
                    "facts": [{"entity_id": "e_shen", "facet": "costume", "value": "青衫", "quote": "穿着青衫", "source": "narration"}],
                    "events": [{"entity_id": "e_shen", "event_type": "costume_change", "splits_entity": False, "value": "白衣", "quote": "换了白衣"}],
                })
            elif sys_prompt == VERIFY_PROMPT:
                gen_verify_calls.append("fact")
                content = dumps({"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None} for it in u["items"]]})
            elif sys_prompt == EVENT_VERIFY_PROMPT:
                gen_verify_calls.append("event")
                content = dumps({"verdicts": [{"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None, "stage_split": False} for it in u["items"]]})
            else:
                raise AssertionError(sys_prompt)
            return _FakeHttpResponse({"choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        m = Models(self.store, run_id, "d_auto", services, attempts=1, opener=opener, budget=budget)
        res = run_describe(self.store, run_id, services=services, models=m, kinds="person", verify=1, concurrency=1, decision_mode="auto")
        self.assertEqual(len(ts_attempts), 1)
        self.assertEqual(gen_verify_calls, ["fact", "event"])
        self.assertEqual(budget.consecutive_network_failures, 0)
        self.assertEqual(res["counts"]["facts"], 1)

    def test_on_mode_decision_model_error_raises_without_fallback(self):
        text = "第一章\n沈墨在厅内。\n第二章\n沈墨看着青衫客。\n"
        source, run_id, tasks = self._seed_roster_book(text, {"mentions": [], "links": []}, {"mentions": [], "links": []})
        self.store.db.execute("DELETE FROM records WHERE run_id=? AND kind='mention'", (run_id,))
        self.store.db.execute("DELETE FROM tasks WHERE run_id=?", (run_id,))
        self.store.db.commit()
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        for tk, disc in zip(tasks, [
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u2]}, {"text": "青衫客", "kind": "person", "refs": [u2]}], "links": []},
        ]):
            g = ground_result(source, tk, disc, "rq_" + tk["id"])
            self.store.finish_task(run_id, tk, {"request_id": "rq_" + tk["id"]}, [("mention", m) for m in g["mentions"]])

        def h_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {"entities": [{"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid}], "assignments": [], "not_entity": []}, f"c1_{n}"
            return {"entities": [
                {"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid},
                {"local_id": "e2", "name": "青衫客", "kind": "person", "surfaces": [{"surface": "青衫客", "surface_type": "description"}], "evidence_ref": uid},
            ], "assignments": [], "not_entity": []}, f"c1_{n}"

        def h_choices_err(state, questions, stage, n):
            raise ModelError("quota exhausted", status=402)

        gen_called = []
        models = _ScriptedRosterModels(
            {
                "roster-c1": h_c1,
                "roster-c2-propose": lambda s, n: ({"proposals": []}, f"p_{n}"),
                "roster-c3-link": lambda s, n: gen_called.append(1) or ({"links": []}, "x"),
            },
            choices_handler=h_choices_err,
        )
        with self.assertRaises(ModelError) as ctx:
            resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None, decision_mode="on")
        self.assertEqual(ctx.exception.status, 402)
        self.assertEqual(gen_called, [])

    def test_budget_error_not_swallowed_in_auto_or_on_mode(self):
        text = "第一章\n沈墨在厅内。\n第二章\n沈墨看着青衫客。\n"
        source, run_id, tasks = self._seed_roster_book(text, {"mentions": [], "links": []}, {"mentions": [], "links": []})
        self.store.db.execute("DELETE FROM records WHERE run_id=? AND kind='mention'", (run_id,))
        self.store.db.execute("DELETE FROM tasks WHERE run_id=?", (run_id,))
        self.store.db.commit()
        u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
        for tk, disc in zip(tasks, [
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u1]}], "links": []},
            {"mentions": [{"text": "沈墨", "kind": "person", "refs": [u2]}, {"text": "青衫客", "kind": "person", "refs": [u2]}], "links": []},
        ]):
            g = ground_result(source, tk, disc, "rq_" + tk["id"])
            self.store.finish_task(run_id, tk, {"request_id": "rq_" + tk["id"]}, [("mention", m) for m in g["mentions"]])

        def h_c1(state, n):
            uid = state["units"][0]["id"]
            if state["chapter_title"] == "第一章":
                return {"entities": [{"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid}], "assignments": [], "not_entity": []}, f"c1_{n}"
            return {"entities": [
                {"local_id": "e1", "name": "沈墨", "kind": "person", "surfaces": ["沈墨"], "evidence_ref": uid},
                {"local_id": "e2", "name": "青衫客", "kind": "person", "surfaces": [{"surface": "青衫客", "surface_type": "description"}], "evidence_ref": uid},
            ], "assignments": [], "not_entity": []}, f"c1_{n}"

        def h_choices_budget(state, questions, stage, n):
            raise BudgetError("request budget exhausted", stop_reason="request_budget")

        for mode in ("auto", "on"):
            self.store.db.execute("DELETE FROM tasks WHERE run_id=? AND stage LIKE 'roster%'", (run_id,))
            self.store.db.execute("DELETE FROM records WHERE run_id=? AND kind LIKE 'roster%'", (run_id,))
            self.store.db.commit()
            models = _ScriptedRosterModels(
                {
                    "roster-c1": h_c1,
                    "roster-c2-propose": lambda s, n: ({"proposals": []}, f"p_{n}"),
                    "roster-c3-link": lambda s, n: ({"links": []}, f"g_{n}"),
                },
                choices_handler=h_choices_budget,
            )
            with self.assertRaises(BudgetError):
                resolve_roster(self.store, models, source, run_id, concurrency=1, log=lambda _: None, decision_mode=mode)


if __name__ == "__main__":
    unittest.main()
