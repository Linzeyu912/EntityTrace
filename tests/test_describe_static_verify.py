from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.config import Service
from entitytrace.describe import (
    D1_PROMPT,
    DESC_VERIFY_DECISION_TAU,
    DESCRIBE_VERSION,
    EVENT_VERIFY_PROMPT,
    EVENT_VERIFY_PROMPT_VERSION,
    FACET_DEFINITIONS,
    FACETS,
    VERIFY_BATCH,
    VERIFY_DECISION_BATCH,
    VERIFY_PROMPT,
    VERIFY_PROMPT_VERSION,
    apply_stage_split_guards,
    build_d1_tasks,
    build_describe_id,
    build_entity_stages,
    build_verify_decision_batches,
    build_verify_tasks,
    check_verify_output,
    decide_verdict,
    export_describe_run,
    run_describe,
)
from entitytrace.discovery import plan_tasks
from entitytrace.models import Models
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import dumps

_HEX64_RE = re.compile(r"\b[0-9a-f]{64}\b")

_D1_PROMPT_SHA256 = "496a3481aadc5676684259326055f3584673686b0de68f4605defd0762fd8d13"
_EVENT_VERIFY_PROMPT_SHA256 = "1569553a1bb50b001e773c20493a24269ec30a6b5c9721b40915e4e80c96cdef"
_D1_PAYLOAD_FINGERPRINT = "16efd6ccbaa3e6ff70cf4277bb825fbaa3229ba6dc0cada58076e4c98ceb1e24"

_EXPECTED_STATIC_PERSON_FACETS = {
    "face": "固有面容、常驻肤色气色或标志性容貌特征（不含单次场景一时的喜怒惊惧等瞬时表情神态）",
    "costume": "具体可见的衣冠、甲胄、饰物或阶段性穿戴造型（不含无具体外观的泛化穿脱更衣或伪装动词）",
    "voice": "固有或标志性的嗓音音色与音量（不含因奔波干渴、受伤喘息等单次情景引起的临时失声嘶哑）",
}

_UNCHANGED_FACET_DEFINITIONS = {
    "person": {
        "age": "年龄",
        "gender": "性别",
        "build": "体形身量",
        "hair": "头发胡须",
        "feature": "五官或身体特征",
    },
    "place": {
        "terrain": "地形地貌、山水",
        "architecture": "建筑与营造",
        "scale": "规模、方圆",
        "atmosphere": "该地本身的景色、光线、声响、气味与气象",
    },
    "object": {
        "shape": "形状",
        "size": "大小或重量",
        "color_material": "颜色或材质",
        "visual_effect": "外观或施展时的光色形象",
        "holder": "持有者或乘骑者",
    },
    "ability": {
        "visual_effect": "施展时的光色形象",
        "sound_effect": "施展时的声音",
    },
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _FakeHttpResponse:
    def __init__(self, body: dict):
        self._raw = dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _seed_snapshot(
    store: Store,
    root: Path,
    run_id: str,
    text: str,
    chapter_titles: list[str],
    entity_specs: list[tuple[str, str, str]],
) -> tuple[dict, list[dict]]:
    """构造完整只读查询快照，供描述阶段零外部依赖测试运行。"""
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
    store.execute(
        "INSERT INTO runs VALUES(?,?,?,?,?,?)",
        (run_id, source["id"], dumps(config), dumps(state), "now", "now"),
    )
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
            mentions.append({
                "id": f"m_{entity_id}_{at}",
                "chapter_id": ch["id"],
                "start": at,
                "end": at + len(surface),
                "text": surface,
                "link_method": "chapter_local",
            })
            st = chapter_stats.setdefault(ch["id"], {"count": 0, "min_start": at})
            st["count"] += 1
            st["min_start"] = min(st["min_start"], at)
            surface_counts[surface] = surface_counts.get(surface, 0) + 1
            pos_cursor = at + 1
        entities.append({
            "id": entity_id,
            "name": surface,
            "kind": kind,
            "mention_count": len(mentions),
            "first_chapter_id": mentions[0]["chapter_id"] if mentions else None,
            "identity_status": "linked_by_roster",
            "surface_counts": surface_counts,
            "chapter_stats": chapter_stats,
        })
        header = {
            "id": entity_id,
            "name": surface,
            "names": [],
            "aliases": [],
            "kind": kind,
            "identity_status": "linked_by_roster",
            "mention_count": len(mentions),
            "first_chapter_id": mentions[0]["chapter_id"] if mentions else None,
            "link_methods": {"chapter_local": len(mentions)},
        }
        rows.append((entity_id, dumps(header), dumps(mentions)))
    meta = {
        "result_revision": "rev-static-1",
        "status": "completed",
        "source_id": source["id"],
        "chapters": chapters,
        "counts": {},
        "review_counts": {},
        "snapshot_version": 2,
    }
    store.save_query_snapshot(run_id, store.query_fingerprint(run_id), meta, entities, [], rows)
    return source, chapters


class StaticVerifyPromptAndFreezeTests(unittest.TestCase):
    """静态设定边界提示词、维度定义与抽取负载冻结校验。"""

    def test_d1_prompt_and_event_prompt_byte_frozen(self):
        self.assertEqual(DESCRIBE_VERSION, "describe-v6")
        self.assertEqual(EVENT_VERIFY_PROMPT_VERSION, "event-verify-v2")
        self.assertEqual(_sha256(D1_PROMPT), _D1_PROMPT_SHA256)
        self.assertEqual(_sha256(EVENT_VERIFY_PROMPT), _EVENT_VERIFY_PROMPT_SHA256)
        text = "第一章 山门\n沈墨身长八尺。"
        chapters = [{
            "id": "ch1", "title": "第一章 山门", "kind": "chapter",
            "start": 0, "end": len(text), "body_start": 7, "chapter_set_version": "chs-1",
        }]
        tasks = build_d1_tasks(
            {"id": "src-d1", "text": text},
            chapters,
            [{"entity_id": "e1", "kind": "person", "score": 1.0}],
            {"e1": [{"start": 8, "end": 10, "text": "沈墨", "chapter_id": "ch1"}]},
            "did-freeze",
            task_chars=9000,
            views=1,
            entities_per_request=0,
        )
        self.assertEqual(len(tasks), 1)
        fp = _sha256(dumps({"prompt": tasks[0]["prompt"], "state": tasks[0]["state"]}))
        self.assertEqual(fp, _D1_PAYLOAD_FINGERPRINT)

    def test_verify_v3_version_and_person_facet_definitions(self):
        self.assertEqual(VERIFY_PROMPT_VERSION, "verify-v3")
        for facet, expected in _EXPECTED_STATIC_PERSON_FACETS.items():
            self.assertEqual(FACET_DEFINITIONS["person"][facet], expected)
        for kind, mapping in _UNCHANGED_FACET_DEFINITIONS.items():
            for facet, expected in mapping.items():
                self.assertEqual(FACET_DEFINITIONS[kind][facet], expected)
        self.assertEqual(
            sum(len(mapping) for mapping in _UNCHANGED_FACET_DEFINITIONS.values()),
            16,
        )

    def test_verify_prompt_contains_static_boundary_clauses_and_drops_v2_transient_clause(self):
        required_fragments = (
            "固有、常驻或阶段性外观与声响特征",
            "娇美",
            "赤膊负荆",
            "面如重枣",
            "披甲",
            "扮作虞候",
            "临时声音嘶哑",
            "途经异地风雪山路",
            "value 必须原样抄写",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, VERIFY_PROMPT)
        self.assertNotIn(
            "因惊惧、愤怒、羞愧、喜悦等一时出现的面色与表情，也是该人物可见的样子",
            VERIFY_PROMPT,
        )
        for code in ("not_description", "wrong_entity", "wrong_facet", "unsupported_value"):
            self.assertIn(code, VERIFY_PROMPT)

    def test_build_verify_tasks_injects_updated_facet_definitions_without_hex_leak(self):
        text = "第一章 聚义\n沈墨面如冠玉，着青衫，声如洪钟，身长八尺。\n"
        eid = "a" * 64
        chapters = [{"id": "ch1", "kind": "chapter", "start": 0, "end": len(text)}]
        facts = [
            {"id": f"{idx:064x}", "entity_id": eid, "facet": facet, "value": val,
             "quote": val, "start": text.index(val), "end": text.index(val) + len(val),
             "chapter_id": "ch1"}
            for idx, (facet, val) in enumerate(
                [("face", "面如冠玉"), ("costume", "青衫"), ("voice", "声如洪钟"), ("build", "身长八尺")],
                start=1,
            )
        ]
        tasks = build_verify_tasks(
            "d" * 64, facts, {eid: "person"}, {eid: ["沈墨"]}, text, chapters, passes=1,
        )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["prompt"], VERIFY_PROMPT)
        serialized = dumps(tasks[0]["state"])
        self.assertIsNone(_HEX64_RE.search(serialized))
        by_facet = {item["facet"]: item for item in tasks[0]["state"]["items"]}
        self.assertEqual(by_facet["face"]["facet_definition"], _EXPECTED_STATIC_PERSON_FACETS["face"])
        self.assertEqual(by_facet["costume"]["facet_definition"], _EXPECTED_STATIC_PERSON_FACETS["costume"])
        self.assertEqual(by_facet["voice"]["facet_definition"], _EXPECTED_STATIC_PERSON_FACETS["voice"])
        self.assertEqual(by_facet["build"]["facet_definition"], "体形身量")

    def test_build_verify_decision_batches_syncs_static_criteria_without_hex_leak(self):
        text = "第一章 聚义\n沈墨面如冠玉，着青衫。\n"
        eid = "b" * 64
        chapters = [{"id": "ch1", "kind": "chapter", "start": 0, "end": len(text)}]
        facts = [
            {"id": f"{idx:064x}", "entity_id": eid, "facet": facet, "value": f"v{idx}",
             "quote": "沈墨面如冠玉", "start": 7, "end": 13, "chapter_id": "ch1"}
            for idx, facet in enumerate(("face", "costume", "voice", "build") * 3)
        ]
        batches = build_verify_decision_batches(
            "c" * 64, facts, {eid: "person"}, {eid: ["沈墨"]}, text, chapters,
        )
        self.assertEqual([len(b["items_with_facts"]) for b in batches], [VERIFY_DECISION_BATCH, 4])
        for batch in batches:
            serialized = dumps({"state": batch["state"], "questions": batch["questions"]})
            self.assertIsNone(_HEX64_RE.search(serialized))
            for q in batch["questions"].values():
                self.assertIn("固有、常驻或阶段性具体特征", q["criteria"]["keep"])
                self.assertIn("瞬时表情变色", q["criteria"]["drop"])
                self.assertIn("泛化穿脱伪装动作", q["criteria"]["drop"])
                self.assertIn("临时失声嘶哑", q["criteria"]["drop"])

    def test_describe_id_changes_between_v2_and_v3_while_d1_request_id_stays_identical(self):
        did_v3 = build_describe_id(20, ["person"], ["e1"], "run-1", "openai", "example/model", verify=1)
        with patch("entitytrace.describe.VERIFY_PROMPT_VERSION", "verify-v2"):
            did_v2 = build_describe_id(20, ["person"], ["e1"], "run-1", "openai", "example/model", verify=1)
        self.assertNotEqual(did_v3, did_v2)

        text = "第一章 山门\n沈墨身长八尺。"
        chapters = [{"id": "ch1", "title": "第一章 山门", "kind": "chapter",
                     "start": 0, "end": len(text), "body_start": 7, "chapter_set_version": "chs-1"}]
        t_v2 = build_d1_tasks({"id": "s1", "text": text}, chapters,
                              [{"entity_id": "e1", "kind": "person", "score": 1.0}],
                              {"e1": [{"start": 8, "end": 10, "text": "沈墨", "chapter_id": "ch1"}]},
                              did_v2, views=3, entities_per_request=4)
        t_v3 = build_d1_tasks({"id": "s1", "text": text}, chapters,
                              [{"entity_id": "e1", "kind": "person", "score": 1.0}],
                              {"e1": [{"start": 8, "end": 10, "text": "沈墨", "chapter_id": "ch1"}]},
                              did_v3, views=3, entities_per_request=4)
        self.assertEqual(len(t_v2), len(t_v3))
        for a, b in zip(t_v2, t_v3):
            self.assertEqual(a["prompt"], b["prompt"])
            self.assertEqual(a["state"], b["state"])


class SameDbCacheReplayAndHybridDecisionTests(unittest.TestCase):
    """同库双版本零 D1 请求缓存重放、混合决策同步过滤与边界守卫端到端测试。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "replay.sqlite3")
        self.svc_gen = Service(
            "discovery", "http://127.0.0.1:9/v1", "example/model", "k-test",
            provider="openai", extra_body={"reasoning_effort": "none"},
        )
        self.svc_dec = Service(
            "decision", "https://api.typesafe.ai/v1", "jev-latest", "k-dec",
            provider="typesafe",
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_e2e_same_db_v2_to_v3_replay_zero_d1_network_calls_and_old_export_lossless(self):
        text = (
            "第一章 赴约\n"
            "沈墨面如冠玉，着一袭青衫，听闻变故顿时脸色阴沉，披甲出门。\n"
            "他急赶十里后嘴唇干裂，声音嘶哑地开口。\n"
        )
        run_id = "run_replay_v2_v3"
        _seed_snapshot(self.store, self.root, run_id, text, ["第一章 赴约"], [("e_shen", "沈墨", "person")])
        services = {"discovery": self.svc_gen}
        old_prompt_v2 = "OLD_VERIFY_V2_PROMPT"
        stage_http_calls = {"d1": 0, "verify_v2": 0, "verify_v3": 0}

        def opener(req, timeout=None):
            payload = json.loads(req.data.decode("utf-8"))
            sys_prompt = payload["messages"][0]["content"]
            user_state = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                stage_http_calls["d1"] += 1
                facts = [
                    {"entity_id": "e_shen", "facet": "face", "value": "面如冠玉", "quote": "面如冠玉", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "costume", "value": "青衫", "quote": "着一袭青衫", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "face", "value": "脸色阴沉", "quote": "顿时脸色阴沉", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "costume", "value": "披甲", "quote": "披甲出门", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "voice", "value": "声音嘶哑", "quote": "嘴唇干裂，声音嘶哑", "source": "narration"},
                ]
                content = dumps({"facts": facts, "events": []})
            elif sys_prompt == old_prompt_v2:
                stage_http_calls["verify_v2"] += 1
                content = dumps({"verdicts": [
                    {"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None}
                    for it in user_state["items"]
                ]})
            elif sys_prompt == VERIFY_PROMPT:
                stage_http_calls["verify_v3"] += 1
                verdicts = []
                for it in user_state["items"]:
                    if it["value"] in ("脸色阴沉", "披甲", "声音嘶哑"):
                        verdicts.append({"id": it["id"], "value": it["value"], "verdict": "drop", "reason": "not_description"})
                    else:
                        verdicts.append({"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None})
                content = dumps({"verdicts": verdicts})
            else:
                raise AssertionError(f"Unexpected prompt: {sys_prompt[:40]}")
            return _FakeHttpResponse({
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        with patch("urllib.request.urlopen", opener):
            with patch("entitytrace.describe.VERIFY_PROMPT_VERSION", "verify-v2"), \
                 patch("entitytrace.describe.VERIFY_PROMPT", old_prompt_v2):
                res_v2 = run_describe(self.store, run_id, services=services,
                                      kinds="person", verify=1, views=1, concurrency=1)

            old_describe_id = res_v2["describe_id"]
            self.assertGreater(stage_http_calls["d1"], 0)
            self.assertEqual(stage_http_calls["verify_v2"], 1)
            old_values = {f["value"] for f in res_v2["document"]["entities"][0]["facts"]}
            self.assertEqual(old_values, {"面如冠玉", "青衫", "脸色阴沉", "披甲", "声音嘶哑"})

            d1_calls_before_v3 = stage_http_calls["d1"]
            res_v3 = run_describe(self.store, run_id, services=services,
                                  kinds="person", verify=1, views=1, concurrency=1)
        new_describe_id = res_v3["describe_id"]

        self.assertNotEqual(new_describe_id, old_describe_id)
        self.assertEqual(stage_http_calls["d1"], d1_calls_before_v3)
        self.assertEqual(stage_http_calls["verify_v3"], 1)

        d1_rows = self.store.rows(
            "SELECT status FROM calls WHERE invocation=? AND stage='describe-d1'",
            (new_describe_id,),
        )
        self.assertGreater(len(d1_rows), 0)
        self.assertTrue(all(r["status"] == "cached" for r in d1_rows))

        new_values = {f["value"] for f in res_v3["document"]["entities"][0]["facts"]}
        self.assertEqual(new_values, {"面如冠玉", "青衫"})
        self.assertEqual(res_v3["document"]["verification"]["dropped"]["not_description"], 3)

        reexported_old = export_describe_run(self.store, old_describe_id)
        self.assertEqual(
            {f["value"] for f in reexported_old["entities"][0]["facts"]},
            {"面如冠玉", "青衫", "脸色阴沉", "披甲", "声音嘶哑"},
        )

    def test_e2e_same_db_multiview_replay_zero_d1_network_calls(self):
        text = "第一章 赴约\n沈墨面如冠玉，着青衫，微微一笑。\n"
        run_id = "run_mv_replay"
        _seed_snapshot(self.store, self.root, run_id, text, ["第一章 赴约"], [("e_shen", "沈墨", "person")])
        services = {"discovery": self.svc_gen}
        d1_http = {"count": 0}

        def opener(req, timeout=None):
            payload = json.loads(req.data.decode("utf-8"))
            sys_prompt = payload["messages"][0]["content"]
            u = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                d1_http["count"] += 1
                content = dumps({"facts": [
                    {"entity_id": "e_shen", "facet": "face", "value": "面如冠玉", "quote": "面如冠玉", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "face", "value": "微微一笑", "quote": "微微一笑", "source": "narration"},
                ], "events": []})
            elif sys_prompt == VERIFY_PROMPT or sys_prompt == "OLD_MV_V2":
                keep_all = sys_prompt == "OLD_MV_V2"
                verdicts = []
                for it in u["items"]:
                    if not keep_all and it["value"] == "微微一笑":
                        verdicts.append({"id": it["id"], "value": it["value"], "verdict": "drop", "reason": "not_description"})
                    else:
                        verdicts.append({"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None})
                content = dumps({"verdicts": verdicts})
            else:
                raise AssertionError(sys_prompt)
            return _FakeHttpResponse({
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        with patch("entitytrace.describe.VERIFY_PROMPT_VERSION", "verify-v2"), \
             patch("entitytrace.describe.VERIFY_PROMPT", "OLD_MV_V2"):
            run_describe(self.store, run_id, services=services,
                         models=Models(self.store, run_id, "mv2", services, opener=opener),
                         kinds="person", views=3, entities_per_request=4, verify=1, concurrency=1)
        self.assertEqual(d1_http["count"], 3)

        res_v3 = run_describe(self.store, run_id, services=services,
                              models=Models(self.store, run_id, "mv3", services, opener=opener),
                              kinds="person", views=3, entities_per_request=4, verify=1, concurrency=1)
        self.assertEqual(d1_http["count"], 3)
        self.assertEqual([f["value"] for f in res_v3["document"]["entities"][0]["facts"]], ["面如冠玉"])

    def test_e2e_typesafe_hybrid_080_static_filtering_and_fallback(self):
        self.assertEqual(DESC_VERIFY_DECISION_TAU, 0.80)
        text = "第一章 赴约\n沈墨面如冠玉，着青衫，见客大惊失色，说话时略有迟疑。\n"
        run_id = "run_hybrid_080"
        _seed_snapshot(self.store, self.root, run_id, text, ["第一章 赴约"], [("e_shen", "沈墨", "person")])
        services = {"discovery": self.svc_gen, "decision": self.svc_dec}
        dec_payloads = []
        gen_verify_items = []

        def opener(req, timeout=None):
            payload = json.loads(req.data.decode("utf-8"))
            if req.full_url.endswith("/systemone"):
                dec_payloads.append(payload)
                by_id = {it["id"]: it for it in payload["state"]["items"]}
                answers = {}
                for qid, q in payload["questions"].items():
                    self.assertIn("固有、常驻或阶段性具体特征", q["criteria"]["keep"])
                    self.assertIn("瞬时表情变色", q["criteria"]["drop"])
                    val = by_id[qid]["value"]
                    if val == "大惊失色":
                        answers[qid] = {"type": "choice", "choice": "drop",
                                        "probabilities": {"keep": 0.08, "drop": 0.92}, "confidence": 0.92}
                    elif val == "面如冠玉":
                        answers[qid] = {"type": "choice", "choice": "keep",
                                        "probabilities": {"keep": 0.94, "drop": 0.06}, "confidence": 0.94}
                    else:
                        answers[qid] = {"type": "choice", "choice": "keep",
                                        "probabilities": {"keep": 0.65, "drop": 0.35}, "confidence": 0.65}
                return _FakeHttpResponse({"model": "jev-1.14.0", "answers": answers})

            sys_prompt = payload["messages"][0]["content"]
            u = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                content = dumps({"facts": [
                    {"entity_id": "e_shen", "facet": "face", "value": "面如冠玉", "quote": "面如冠玉", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "costume", "value": "青衫", "quote": "着青衫", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "face", "value": "大惊失色", "quote": "大惊失色", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "face", "value": "略有迟疑", "quote": "略有迟疑", "source": "narration"},
                ], "events": []})
            elif sys_prompt == VERIFY_PROMPT:
                gen_verify_items.append([(it["value"], it["facet_definition"]) for it in u["items"]])
                verdicts = []
                for it in u["items"]:
                    if it["value"] == "略有迟疑":
                        verdicts.append({"id": it["id"], "value": it["value"], "verdict": "drop", "reason": "not_description"})
                    else:
                        verdicts.append({"id": it["id"], "value": it["value"], "verdict": "keep", "reason": None})
                content = dumps({"verdicts": verdicts})
            else:
                raise AssertionError(sys_prompt)
            return _FakeHttpResponse({
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        models = Models(self.store, run_id, "d3_static", services, discovery_concurrency=1, decision_concurrency=2, opener=opener)
        res = run_describe(self.store, run_id, services=services, models=models,
                           kinds="person", verify=1, concurrency=1, decision_concurrency=2, decision_mode="auto")

        self.assertEqual(len(dec_payloads), 1)
        self.assertEqual(len(gen_verify_items), 1)
        self.assertEqual([pair[0] for pair in gen_verify_items[0]], ["青衫", "略有迟疑"])
        self.assertEqual(gen_verify_items[0][0][1], _EXPECTED_STATIC_PERSON_FACETS["costume"])
        self.assertEqual(gen_verify_items[0][1][1], _EXPECTED_STATIC_PERSON_FACETS["face"])

        verdict_rows = self.store.rows(
            "SELECT fact_id, verdict, reason FROM fact_verdicts WHERE describe_id=?",
            (res["describe_id"],),
        )
        dropped_reasons = [r["reason"] for r in verdict_rows if r["verdict"] == "drop"]
        self.assertEqual(dropped_reasons, ["not_description", "not_description"])
        kept_vals = {f["value"] for f in res["document"]["entities"][0]["facts"]}
        self.assertEqual(kept_vals, {"面如冠玉", "青衫"})

    def test_decision_off_skips_decision_endpoint_and_uses_verify_v3(self):
        text = "第一章 赴约\n沈墨着青衫，面露喜色。\n"
        run_id = "run_dec_off"
        _seed_snapshot(self.store, self.root, run_id, text, ["第一章 赴约"], [("e_shen", "沈墨", "person")])
        services = {"discovery": self.svc_gen, "decision": self.svc_dec}
        urls_called = []

        def opener(req, timeout=None):
            urls_called.append(req.full_url)
            payload = json.loads(req.data.decode("utf-8"))
            sys_prompt = payload["messages"][0]["content"]
            u = json.loads(payload["messages"][1]["content"])
            if sys_prompt == D1_PROMPT:
                content = dumps({"facts": [
                    {"entity_id": "e_shen", "facet": "costume", "value": "青衫", "quote": "着青衫", "source": "narration"},
                    {"entity_id": "e_shen", "facet": "face", "value": "面露喜色", "quote": "面露喜色", "source": "narration"},
                ], "events": []})
            else:
                self.assertEqual(sys_prompt, VERIFY_PROMPT)
                content = dumps({"verdicts": [
                    {"id": it["id"], "value": it["value"],
                     "verdict": "drop" if it["value"] == "面露喜色" else "keep",
                     "reason": "not_description" if it["value"] == "面露喜色" else None}
                    for it in u["items"]
                ]})
            return _FakeHttpResponse({
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        res = run_describe(self.store, run_id, services=services,
                           models=Models(self.store, run_id, "dec_off", services, opener=opener),
                           kinds="person", verify=1, concurrency=1, decision_mode="off")
        self.assertFalse(any(url.endswith("/systemone") for url in urls_called))
        self.assertEqual([f["value"] for f in res["document"]["entities"][0]["facts"]], ["青衫"])

    def test_dropping_transient_face_triggers_empty_side_stage_guard_and_keeps_nonempty_stages(self):
        chapters = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": 100},
            {"id": "ch2", "kind": "chapter", "start": 100, "end": 200},
        ]
        # Only a transient face fact existed before the event; once verify-v3 drops it, left side is empty.
        kept_facts = [
            {"id": "f_after", "entity_id": "e1", "facet": "costume", "value": "僧袍",
             "quote": "着僧袍", "start": 140, "end": 143, "chapter_id": "ch2", "verdict": "keep"},
        ]
        events = [
            {"id": "ev1", "entity_id": "e1", "event_type": "costume_change", "value": "剃发换袍",
             "quote": "剃发换袍", "start": 120, "end": 124, "chapter_id": "ch2",
             "verdict": "keep", "stage_split": True},
        ]
        guarded = apply_stage_split_guards(kept_facts, events, chapters=chapters)
        self.assertFalse(guarded[0]["stage_split"])
        stages = build_entity_stages("person", kept_facts, guarded, chapters, (0, 200))
        self.assertEqual(len(stages), 1)
        self.assertEqual(len(stages[0]["facts"]), 1)

    def test_verify_tasks_preserve_before_after_within_chapter_bounds(self):
        text = "第一章 前文\n甲乙丙。\n第二章 人物\n前句在此。沈墨赤膊负荆，声如巨雷。后句在此。\n第三章 尾声\n后续无关。"
        ch2_start = text.index("第二章 人物")
        ch2_end = text.index("第三章 尾声")
        chapters = [
            {"id": "ch1", "kind": "chapter", "start": 0, "end": ch2_start},
            {"id": "ch2", "kind": "chapter", "start": ch2_start, "end": ch2_end},
            {"id": "ch3", "kind": "chapter", "start": ch2_end, "end": len(text)},
        ]
        q1 = "赤膊负荆"
        q2 = "声如巨雷"
        facts = [
            {"id": "f1", "entity_id": "e1", "facet": "costume", "value": q1, "quote": q1,
             "start": text.index(q1), "end": text.index(q1) + len(q1), "chapter_id": "ch2"},
            {"id": "f2", "entity_id": "e1", "facet": "voice", "value": q2, "quote": q2,
             "start": text.index(q2), "end": text.index(q2) + len(q2), "chapter_id": "ch2"},
        ]
        tasks = build_verify_tasks("did", facts, {"e1": "person"}, {"e1": ["沈墨"]}, text, chapters, passes=1)
        items = tasks[0]["state"]["items"]
        self.assertNotIn("甲乙丙", items[0]["before"])
        self.assertNotIn("后续无关", items[1]["after"])
        self.assertIn("前句在此", items[0]["before"])
        self.assertIn("后句在此", items[1]["after"])

    def test_legacy_verify_v2_run_exports_without_regression(self):
        store = Store(self.root / "legacy_v2.sqlite3")
        try:
            store.execute(
                "INSERT INTO describe_runs VALUES(?,?,?,?,?,?)",
                (
                    "did-v2-old",
                    "run-old",
                    dumps({
                        "describe_version": "describe-v6",
                        "top_k": 20,
                        "kinds": ["person"],
                        "entity_ids": ["e1"],
                        "run_id": "run-old",
                        "views": 1,
                        "entities_per_request": 4,
                        "verify": 1,
                        "verify_batch": VERIFY_BATCH,
                        "verify_prompt_version": "verify-v2",
                        "event_verify_prompt_version": EVENT_VERIFY_PROMPT_VERSION,
                        "generator": {"provider": "openai", "model": "example/model"},
                    }),
                    dumps({"status": "completed"}),
                    "t0",
                    "t0",
                ),
            )
            store.execute(
                "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                ("did-v2-old", "f1", "e1", "face", "面带忧容", "面带忧容", 0, 4, "ch1", "narration", None, "t1"),
            )
            store.execute(
                "INSERT INTO fact_verdicts VALUES(?,?,?,?,?)",
                ("did-v2-old", "f1", "keep", None, "t1"),
            )
            doc = export_describe_run(store, "did-v2-old")
        finally:
            store.close()
        self.assertTrue(doc["verification"]["enabled"])
        self.assertEqual([f["value"] for f in doc["entities"][0]["facts"]], ["面带忧容"])

    def test_verify_output_and_three_pass_voting_reject_static_violations(self):
        expected = {"F1": "面色阴沉", "F2": "披甲", "F3": "面如重枣"}
        check_verify_output({
            "verdicts": [
                {"id": "F1", "value": "面色阴沉", "verdict": "drop", "reason": "not_description"},
                {"id": "F2", "value": "披甲", "verdict": "drop", "reason": "not_description"},
                {"id": "F3", "value": "面如重枣", "verdict": "keep", "reason": None},
            ]
        }, expected)
        self.assertEqual(
            decide_verdict([(0, "drop", "not_description"), (1, "keep", None), (2, "drop", "not_description")], 3),
            ("drop", "not_description"),
        )
        self.assertEqual(
            decide_verdict([(0, "keep", None), (1, "drop", "not_description"), (2, "keep", None)], 3),
            ("keep", None),
        )


if __name__ == "__main__":
    unittest.main()
