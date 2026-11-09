"""0.16a 描述事实金标准评分工具。

输入为描述导出 JSON（`entitytrace describe --out`）与上游运行数据库；复用 0.15f
规格二工具的三层金标准提及匹配与无歧义名册平票/兜底规则，优先以六回内的提及匹配
做多数表决把系统 top-K 实体映射到金标准 `gold_id`，计算严格／宽松／全量召回、
六回精确率、事件召回与精确率，并导出盲审包与全书抽样包。

本工具只读金标准与数据库，不改动金标准文件，也不调用任何模型。

清单格式（`--manifest`）：
    {"runs": [{"label": "e1-1", "arm": "e1", "method": "d1", "db": "臂数据库副本",
               "run_id": "上游运行编号", "describe": "describe --out 导出 JSON"}]}

`arm` 可省略：标签形如 `e1-1` 时取前缀，否则等于 `method`。聚合既按 `method` 也按 `arm` 计算。
`--reuse-adjudication` 指向旧裁定结果 JSONL，按键复用已裁定未匹配项，只把新键导出到 `--adjudication`；
`--blind-labels` 指定全书盲审抽样的运行标签（默认优先 `e2-1,e3-1`）。

盲审回填格式（`--blind-result`，JSONL）：每行
    {"sample_id": "...", "wrong_entity": false, "wrong_facet": true,
     "unsupported_value": false}
裁定回填格式（`--adjudication-result`，JSONL）：每行
    {"adj_id": "...", "verdict": "gold_missed|system_wrong_entity|system_wrong_facet|unsupported_value|not_description"}
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entitytrace.roster import normalize_surface
from entitytrace.storage import Store
from entitytrace.util import digest, write_json
from tools.evaluate_identity_gold import fraction, infer_predicted_entity_gold_id
from tools.evaluate_type_gold import match_gold_to_pred

ENTITY_TYPES = ("person", "place", "object", "ability")
FACET_ENUMS = {
    "person": ("age", "gender", "build", "face", "hair", "feature", "costume", "demeanor", "voice"),
    "place": ("terrain", "architecture", "scale", "atmosphere"),
    "object": ("shape", "size", "color_material", "visual_effect", "holder"),
    "ability": ("visual_effect", "sound_effect"),
}
EVENT_TYPES = {
    "costume_change", "injury", "disguise", "transform", "age_jump",
    "upgrade", "damage", "owner_change", "death", "other",
}
ADJUDICATION_VERDICTS = {
    "gold_missed", "system_wrong_entity", "system_wrong_facet", "unsupported_value", "not_description",
}
DEFAULT_SEED = 20261109
BLIND_PER_ARM = 60
CONTEXT_CHARS = 120


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
                    encoding="utf-8")


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) > 0


def _mean(values: list) -> float | None:
    usable = [value for value in values if isinstance(value, (int, float))]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 6)


def _in_sampled(start: int, sampled: list[dict]) -> bool:
    return any(ch["start"] <= start < ch["end"] for ch in sampled)


def _chapter_ordinal(start: int, chapters: list[dict]) -> int | None:
    for ch in chapters:
        if ch["start"] <= start < ch["end"]:
            return ch.get("ordinal")
    return None


def _gold_items(doc: dict, key: str) -> list[dict]:
    items = []
    for item in doc.get(key, []):
        gold_id = item.get("gold_id")
        if not gold_id or gold_id == "uncertain":
            continue
        if item.get("entity_type") not in ENTITY_TYPES:
            continue
        items.append(item)
    return items


def _build_alias_map(roster_doc: dict) -> dict[str, str]:
    """只保留映射到唯一 gold_id 的规范化别名，供平票裁决与无歧义名册回退。"""
    candidates: dict[str, set[str]] = collections.defaultdict(set)
    for ent in roster_doc.get("entities", []):
        gold_id = ent.get("gold_id")
        if not gold_id:
            continue
        for name in [ent.get("canonical_name") or ent.get("canonical") or ""] + list(ent.get("aliases", [])):
            if name:
                candidates[normalize_surface(name)].add(gold_id)
    return {key: next(iter(ids)) for key, ids in candidates.items() if len(ids) == 1}


def _build_combined_gold_mentions(
    roster_doc: dict,
    mentions_doc: dict,
    hard_cases_doc: dict | None = None,
) -> list[dict]:
    """复用 0.15f 规格二（evaluate_type_gold.py）的三层合并口径：
    roster representative_mentions + hard_cases mentions + exhaustive mentions，
    按 (start, end) 去重，保留 gold_id 或 refers_to 指向的有效金标准实体提及。
    """
    roster_rep: list[dict] = []
    for ent in roster_doc.get("entities", []):
        gid = ent.get("gold_id")
        if not gid:
            continue
        for idx, rm in enumerate(ent.get("representative_mentions", [])):
            roster_rep.append({
                "start": int(rm["start"]),
                "end": int(rm["end"]),
                "text": rm.get("text", ""),
                "entity_type": ent.get("entity_type"),
                "gold_id": gid,
                "_eval_key": f"ro:{gid}:{idx}",
            })

    hc_items: list[dict] = []
    if hard_cases_doc:
        for case in hard_cases_doc.get("cases", []):
            cid = case.get("case_id", "")
            for hm in case.get("mentions", []):
                eff_gid = hm.get("gold_id") or hm.get("refers_to")
                if not eff_gid or eff_gid in ("not_entity", "uncertain") or hm.get("entity_type") == "not_entity":
                    continue
                hc_items.append({
                    "start": int(hm["start"]),
                    "end": int(hm["end"]),
                    "text": hm.get("text", ""),
                    "entity_type": hm.get("entity_type"),
                    "gold_id": eff_gid,
                    "_eval_key": f"hc:{cid}:{hm.get('mention_id', '')}",
                })

    ex_items: list[dict] = []
    for idx, m in enumerate(mentions_doc.get("mentions", [])):
        eff_gid = m.get("gold_id") or m.get("refers_to")
        if not eff_gid or eff_gid in ("not_entity", "uncertain") or m.get("entity_type") == "not_entity":
            continue
        mid = m.get("mention_id") or m.get("sample_id") or str(idx)
        ex_items.append({
            "start": int(m["start"]),
            "end": int(m["end"]),
            "text": m.get("text", ""),
            "entity_type": m.get("entity_type"),
            "gold_id": eff_gid,
            "_eval_key": f"ex:{mid}",
        })

    by_span: dict[tuple[int, int], dict] = {}
    for g in roster_rep + hc_items + ex_items:
        sk = (int(g["start"]), int(g["end"]))
        if sk not in by_span:
            by_span[sk] = {**g, "_eval_key": f"all:{sk[0]}_{sk[1]}"}
    return list(by_span.values())


def _load_mentions_and_headers(
    store: Store, run_id: str, entity_ids: list[str]
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    mentions: dict[str, list[dict]] = {}
    headers: dict[str, dict] = {}
    for entity_id in entity_ids:
        rows = store.rows(
            "SELECT entity_json, mentions_json FROM query_entity_mentions WHERE run_id=? AND entity_id=?",
            (run_id, entity_id),
        )
        if not rows:
            raise ValueError(f"运行快照缺少实体提及：{entity_id}")
        headers[entity_id] = json.loads(rows[0]["entity_json"])
        mentions[entity_id] = json.loads(rows[0]["mentions_json"])
    return mentions, headers


def _normalize_gold_for_matching(items: list[dict]) -> list[dict]:
    normalized = []
    for idx, item in enumerate(items):
        eff_gid = item.get("gold_id") or item.get("refers_to")
        if not eff_gid or eff_gid in ("not_entity", "uncertain") or item.get("entity_type") == "not_entity":
            continue
        key = item.get("_eval_key") or item.get("mention_id") or item.get("sample_id") or f"g:{idx}:{item['start']}:{item['end']}"
        normalized.append({
            "start": int(item["start"]),
            "end": int(item["end"]),
            "gold_id": eff_gid,
            "_eval_key": str(key),
        })
    return normalized


def map_entities(
    describe_doc: dict,
    gold_mentions: list[dict],
    entity_mentions: dict[str, list[dict]],
    roster_doc: dict,
    full_gold_mentions: list[dict] | None = None,
    entity_headers: dict[str, dict] | None = None,
) -> tuple[dict[str, str], dict[str, dict]]:
    """复用 0.15f 规格二（match_gold_to_pred + infer_predicted_entity_gold_id）映射系统实体：
    1. 优先以六回内的提及匹配做多数表决（平票按无歧义名册别名裁决）；
    2. 六回金标准层无命中时（如人物仅在名册留代表提及），回退全书金标准层提及多数表决与无歧义名册别名。
    """
    alias_map = _build_alias_map(roster_doc)
    gold_6ch = _normalize_gold_for_matching(gold_mentions)
    gold_all = _normalize_gold_for_matching(full_gold_mentions) if full_gold_mentions is not None else gold_6ch

    pred_mentions: list[dict] = []
    for entity in describe_doc.get("entities", []):
        eid = entity["entity_id"]
        for m in entity_mentions.get(eid, []):
            if m.get("identity_status") == "not_entity_suggested" or m.get("link_method") == "not_entity_suggested":
                continue
            mid = m.get("id") or f"{eid}:{m['start']}:{m['end']}"
            pred_mentions.append({
                "id": f"{eid}:{mid}",
                "start": int(m["start"]),
                "end": int(m["end"]),
                "entity_id": eid,
            })

    matched_6ch = match_gold_to_pred(gold_6ch, pred_mentions, strict_only=False) if gold_6ch and pred_mentions else {}
    matched_all = match_gold_to_pred(gold_all, pred_mentions, strict_only=False) if gold_all and pred_mentions else {}

    by_pred_6ch: dict[str, list[str]] = collections.defaultdict(list)
    for g in gold_6ch:
        pred = matched_6ch.get(g["_eval_key"])
        if pred and pred.get("entity_id"):
            by_pred_6ch[pred["entity_id"]].append(g["gold_id"])

    by_pred_all: dict[str, list[str]] = collections.defaultdict(list)
    for g in gold_all:
        pred = matched_all.get(g["_eval_key"])
        if pred and pred.get("entity_id"):
            by_pred_all[pred["entity_id"]].append(g["gold_id"])

    mapping: dict[str, str] = {}
    details: dict[str, dict] = {}
    for entity in describe_doc.get("entities", []):
        eid = entity["entity_id"]
        header = dict((entity_headers or {}).get(eid) or {})
        pred_ent_obj = {
            "name": entity.get("name") or header.get("name", ""),
            "aliases": list(header.get("aliases") or []),
            "names": list(header.get("names") or []),
        }
        counts_6ch = collections.Counter(by_pred_6ch.get(eid, []))
        counts_all = collections.Counter(by_pred_all.get(eid, []))
        if counts_6ch:
            gold_id = infer_predicted_entity_gold_id(eid, by_pred_6ch, pred_ent_obj, alias_map)
            source = "sampled_6ch_mentions"
            active_counts = counts_6ch
        elif counts_all:
            gold_id = infer_predicted_entity_gold_id(eid, by_pred_all, pred_ent_obj, alias_map)
            source = "full_book_mentions"
            active_counts = counts_all
        else:
            gold_id = infer_predicted_entity_gold_id(eid, {}, pred_ent_obj, alias_map)
            source = "roster_unique_alias" if gold_id else "unmapped"
            active_counts = counts_6ch

        if gold_id:
            mapping[eid] = gold_id
        details[eid] = {
            "gold_id": gold_id,
            "source": source,
            "matched_mentions": sum(counts_6ch.values()),
            "matched_mentions_full_book": sum(counts_all.values()),
            "candidates": dict(sorted(active_counts.items())),
        }
    return mapping, details


def _top_k_entities(describe_doc: dict, k: int) -> list[str]:
    return [
        entity["entity_id"]
        for entity in describe_doc.get("entities", [])
        if isinstance(entity.get("kind_rank"), int) and entity["kind_rank"] <= k
    ]


def _entity_facts(describe_doc: dict) -> list[dict]:
    facts = []
    for entity in describe_doc.get("entities", []):
        for fact in entity.get("facts", []):
            facts.append({**fact, "entity_id": fact.get("entity_id", entity["entity_id"])})
    return facts


def _entity_events(describe_doc: dict) -> list[dict]:
    events = []
    for entity in describe_doc.get("entities", []):
        for event in entity.get("state_events", []):
            events.append({**event, "entity_id": event.get("entity_id", entity["entity_id"])})
    return events


def _index_by_gold(items: list[dict], mapping: dict[str, str]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = collections.defaultdict(list)
    for item in items:
        gold_id = mapping.get(item.get("entity_id"))
        if gold_id:
            index[gold_id].append(item)
    return index


def _same_field(kind: str, left: dict, right: dict) -> bool:
    field = "facet" if kind == "fact" else "event_type"
    return left.get(field) == right.get(field)


def recall_metrics(
    gold_items: list[dict],
    system_items: list[dict],
    mapping: dict[str, str],
    gold_ids: set[str],
    kind: str,
    loose: bool = False,
) -> dict:
    index = _index_by_gold(system_items, mapping)
    matched_ids: list[str] = []
    matched_gold_objs: list[dict] = []
    for gold in gold_items:
        if gold["gold_id"] not in gold_ids:
            continue
        for item in index.get(gold["gold_id"], []):
            if not _overlap(gold["start"], gold["end"], item["start"], item["end"]):
                continue
            if not loose and not _same_field(kind, gold, item):
                continue
            matched_ids.append(_gold_key(gold))
            matched_gold_objs.append(gold)
            break
    denominator = [gold for gold in gold_items if gold["gold_id"] in gold_ids]
    field = "facet" if kind == "fact" else "event_type"
    by_kind: dict[str, dict] = {}
    for et in ENTITY_TYPES:
        den_k = [g for g in denominator if g.get("entity_type") == et]
        num_k = [g for g in matched_gold_objs if g.get("entity_type") == et]
        by_kind[et] = fraction(len(num_k), len(den_k))
    facets = sorted({g.get(field) for g in denominator if g.get(field)})
    by_facet: dict[str, dict] = {}
    for fc in facets:
        den_f = [g for g in denominator if g.get(field) == fc]
        num_f = [g for g in matched_gold_objs if g.get(field) == fc]
        by_facet[fc] = fraction(len(num_f), len(den_f))
    return {
        "fraction": fraction(len(matched_ids), len(denominator)),
        "matched_ids": sorted(set(matched_ids)),
        "gold_ids": sorted({_gold_key(gold) for gold in denominator}),
        "by_kind": by_kind,
        "by_facet": by_facet,
    }


def _gold_key(item: dict) -> str:
    return item.get("fact_id") or item.get("event_id") or ""


def _adj_id(kind: str, item: dict) -> str:
    return digest([
        "description-adjudication-v1", kind, item.get("entity_id"), item.get("facet"),
        item.get("event_type"), item.get("value"), item.get("quote"),
        int(item["start"]), int(item["end"]),
    ])[:24]


def _int_span(item: dict) -> tuple[int, int] | None:
    try:
        return int(item["start"]), int(item["end"])
    except (KeyError, TypeError, ValueError):
        return None


def _reuse_key(kind: str, item: dict) -> tuple | None:
    span = _int_span(item)
    if span is None:
        return None
    start, end = span
    if kind == "fact":
        return (item.get("entity_id"), item.get("facet"), start, end, item.get("value"))
    return (item.get("entity_id"), item.get("event_type"), start, end, item.get("value"))


def _load_reuse_table(path: Path | None) -> dict:
    """读取旧裁定结果，建立事实/事件键与 adj_id 的复用查找表。"""
    table = {
        "by_adj_id": {},
        "facts": {},
        "events": {},
        "rows": 0,
        "path": str(path) if path is not None else None,
    }
    if path is None or not Path(path).is_file():
        return table
    for row in _read_jsonl(Path(path)):
        verdict = row.get("verdict")
        if verdict not in ADJUDICATION_VERDICTS:
            continue
        table["rows"] += 1
        adj_id = row.get("adj_id")
        if adj_id:
            table["by_adj_id"][adj_id] = verdict
        span = _int_span(row)
        if span is None:
            continue
        start, end = span
        kind = row.get("kind")
        if kind == "event" or (kind != "fact" and row.get("event_type")):
            entity_id, event_type, value = row.get("entity_id"), row.get("event_type"), row.get("value")
            table["events"][(entity_id, event_type, start, end, value)] = verdict
            table["events"][(entity_id, None, start, end, value)] = verdict
        elif kind == "fact" or row.get("facet"):
            table["facts"][
                (row.get("entity_id"), row.get("facet"), start, end, row.get("value"))
            ] = verdict
    return table


def _reuse_verdict(reuse: dict | None, kind: str, item: dict) -> str | None:
    if not reuse:
        return None
    verdict = reuse["by_adj_id"].get(_adj_id(kind, item))
    if verdict is not None:
        return verdict
    key = _reuse_key(kind, item)
    if key is None:
        return None
    if kind == "fact":
        return reuse["facts"].get(key)
    verdict = reuse["events"].get(key)
    if verdict is not None:
        return verdict
    return reuse["events"].get((key[0], None, key[2], key[3], key[4]))


def _arm_for(spec: dict) -> str:
    arm = spec.get("arm")
    if isinstance(arm, str) and arm:
        return arm
    label = str(spec.get("label") or "")
    matched = re.match(r"^([A-Za-z]+\d+)-\d+$", label)
    if matched:
        return matched.group(1)
    return str(spec.get("method"))


def _match_system_item(gold_items: list[dict], item: dict, kind: str) -> dict | None:
    for gold in gold_items:
        if gold["gold_id"] != item.get("_gold_id"):
            continue
        if not _overlap(gold["start"], gold["end"], item["start"], item["end"]):
            continue
        if not _same_field(kind, gold, item):
            continue
        return gold
    return None


def _precision(
    gold_items: list[dict],
    system_items: list[dict],
    mapping: dict[str, str],
    sampled: list[dict],
    kind: str,
    adjudication: dict[str, str] | None,
    entity_by_id: dict[str, dict] | None = None,
    reuse: dict | None = None,
) -> tuple[dict, list[dict]]:
    items = [item for item in system_items if _in_sampled(int(item["start"]), sampled)]
    for item in items:
        item["_gold_id"] = mapping.get(item.get("entity_id"))
    auto_matched = [item for item in items if item["_gold_id"] and _match_system_item(gold_items, item, kind)]
    matched_object_ids = {id(item) for item in auto_matched}
    unmatched = [item for item in items if id(item) not in matched_object_ids]
    verdict_counts: collections.Counter[str] = collections.Counter()
    adjudicated_correct = 0
    unresolved = 0
    adj_correct_ids: set[int] = set()
    reused_items = 0
    new_items = 0
    reused_keys: set[tuple] = set()
    new_keys: set[tuple] = set()
    for item in unmatched:
        verdict = (adjudication or {}).get(_adj_id(kind, item))
        reused = False
        if verdict is None:
            verdict = _reuse_verdict(reuse, kind, item)
            reused = verdict is not None
        key = _reuse_key(kind, item)
        if verdict is None:
            unresolved += 1
            new_items += 1
            if key is not None:
                new_keys.add(key)
            continue
        if reused:
            reused_items += 1
            if key is not None:
                reused_keys.add(key)
        else:
            new_items += 1
            if key is not None:
                new_keys.add(key)
        verdict_counts[verdict] += 1
        if verdict == "gold_missed":
            adjudicated_correct += 1
            adj_correct_ids.add(id(item))

    judged = adjudication is not None or bool(reuse)
    field = "facet" if kind == "fact" else "event_type"
    by_kind: dict[str, dict] = {}
    for et in ENTITY_TYPES:
        et_items = [
            it for it in items
            if ((entity_by_id or {}).get(it.get("entity_id"), {}).get("kind") or it.get("kind")) == et
        ]
        et_raw = [it for it in et_items if id(it) in matched_object_ids]
        et_adj = [it for it in et_items if id(it) in matched_object_ids or id(it) in adj_correct_ids]
        by_kind[et] = {
            "raw": fraction(len(et_raw), len(et_items)),
            "adjudicated": fraction(len(et_adj), len(et_items)) if judged else None,
        }
    facets = sorted({it.get(field) for it in items if it.get(field)})
    by_facet: dict[str, dict] = {}
    for fc in facets:
        fc_items = [it for it in items if it.get(field) == fc]
        fc_raw = [it for it in fc_items if id(it) in matched_object_ids]
        fc_adj = [it for it in fc_items if id(it) in matched_object_ids or id(it) in adj_correct_ids]
        by_facet[fc] = {
            "raw": fraction(len(fc_raw), len(fc_items)),
            "adjudicated": fraction(len(fc_adj), len(fc_items)) if judged else None,
        }

    result = {
        "raw": fraction(len(auto_matched), len(items)),
        "adjudicated": (
            fraction(len(auto_matched) + adjudicated_correct, len(items))
            if judged else None
        ),
        "total": len(items),
        "auto_matched": len(auto_matched),
        "unmatched": len(unmatched),
        "unresolved_adjudication": unresolved,
        "reused_adjudication_count": reused_items,
        "reused_adjudication_keys": len(reused_keys),
        "new_adjudication_count": new_items,
        "new_adjudication_keys": len(new_keys),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "by_kind": by_kind,
        "by_facet": by_facet,
    }
    return result, unmatched


def _adjudication_entry(kind: str, item: dict, entity_by_id: dict[str, dict],
                        sampled: list[dict], text: str) -> dict:
    start, end = int(item["start"]), int(item["end"])
    entry = {
        "adj_id": _adj_id(kind, item),
        "kind": kind,
        "entity_id": item.get("entity_id"),
        "entity_name": entity_by_id.get(item.get("entity_id"), {}).get("name"),
        "entity_kind": entity_by_id.get(item.get("entity_id"), {}).get("kind"),
        "facet": item.get("facet"),
        "event_type": item.get("event_type"),
        "value": item.get("value"),
        "quote": item.get("quote"),
        "start": start,
        "end": end,
        "chapter_ordinal": _chapter_ordinal(start, sampled),
        "context_before": text[max(0, start - CONTEXT_CHARS):start],
        "context_after": text[end:end + CONTEXT_CHARS],
    }
    return entry


def _source_text_and_chapters(store: Store, run_id: str) -> tuple[str, list[dict]]:
    rows = store.rows(
        "SELECT s.text FROM runs r JOIN sources s ON r.source_id=s.id WHERE r.id=?",
        (run_id,),
    )
    if not rows:
        raise ValueError(f"运行不存在：{run_id}")
    meta_rows = store.rows("SELECT meta_json FROM query_snapshots WHERE run_id=?", (run_id,))
    chapters: list[dict] = []
    if meta_rows:
        meta = json.loads(meta_rows[0]["meta_json"])
        pos = 0
        for ch in meta.get("chapters", []):
            if ch.get("kind") == "chapter":
                pos += 1
                chapters.append({"ordinal": pos, "start": int(ch["start"]), "end": int(ch["end"])})
    return rows[0]["text"], chapters


def _load_manifest(path: Path) -> list[dict]:
    doc = _read_json(path)
    runs = doc.get("runs") if isinstance(doc, dict) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest 缺少非空 runs 数组")
    seen: set[str] = set()
    for run in runs:
        label = run.get("label")
        if not isinstance(label, str) or not label or label in seen:
            raise ValueError("manifest 的 label 必须为非空且唯一")
        seen.add(label)
        if run.get("method") not in ("d1", "d2"):
            raise ValueError(f"manifest 运行 {label} 的 method 必须为 d1 或 d2")
        arm = run.get("arm")
        if arm is not None and (not isinstance(arm, str) or not arm):
            raise ValueError(f"manifest 运行 {label} 的 arm 必须为非空字符串")
        for key in ("db", "run_id", "describe"):
            if not run.get(key):
                raise ValueError(f"manifest 运行 {label} 缺少 {key}")
    return runs


def _arm_means(run_results: dict, arm: str) -> tuple[float | None, float | None]:
    runs = [run for run in run_results.values() if run.get("arm") == arm]
    recalls = [run["recall"]["strict"]["20"]["fraction"]["value"] for run in runs]
    precisions = []
    for run in runs:
        sampled_p = run["precision_sampled"]
        chosen = sampled_p.get("adjudicated") or sampled_p.get("raw") or {}
        precisions.append(chosen.get("value"))
    return _mean(recalls), _mean(precisions)


def _select_union_labels(
    run_context: dict, run_results: dict
) -> tuple[str, str | None, str | None, dict]:
    """按预登记规则确定主方法 W 与 W-1、E3-1 两个运行标签。"""
    labels = set(run_context)
    e1_label = "e1-1" if "e1-1" in labels else next(
        (label for label, run in run_results.items() if run["method"] == "d1"), None
    )
    e2_label = "e2-1" if "e2-1" in labels else next(
        (label for label, run in run_results.items() if run["arm"] == "e2"), None
    )
    e3_label = "e3-1" if "e3-1" in labels else next(
        (label for label, run in run_results.items() if run["method"] == "d2"), None
    )
    e1_rec, e1_prec = _arm_means(run_results, "e1")
    e2_rec, e2_prec = _arm_means(run_results, "e2")
    rule = {
        "e1_recall_strict_K20_mean": e1_rec,
        "e1_precision_adjudicated_mean": e1_prec,
        "e2_recall_strict_K20_mean": e2_rec,
        "e2_precision_adjudicated_mean": e2_prec,
        "recall_margin": 0.03,
        "precision_margin": 0.03,
    }
    winner = "e1"
    if (
        e1_rec is not None and e2_rec is not None
        and e1_prec is not None and e2_prec is not None
        and e2_rec >= e1_rec + 0.03 and e2_prec >= e1_prec - 0.03
    ):
        winner = "e2"
    w_label = (e2_label if winner == "e2" else e1_label) or e2_label or e1_label
    return winner, w_label, e3_label, rule


def _union_result(
    left_label: str,
    right_label: str,
    run_context: dict,
    run_results: dict,
    gold_facts: list[dict],
    all_gold_fact_ids: set[str],
    sampled: list[dict],
    adjudication: dict[str, str] | None,
    reuse: dict | None,
) -> dict:
    left, right = run_context[left_label], run_context[right_label]
    merged: list[dict] = []
    seen: set[tuple] = set()
    for source in (left, right):
        for fact in source["facts"]:
            key = (fact.get("entity_id"), fact.get("facet"), int(fact["start"]), int(fact["end"]))
            if key in seen:
                continue
            seen.add(key)
            merged.append(fact)
    mapping = dict(right["mapping"])
    mapping.update(left["mapping"])
    entity_by_id = dict(right["entity_by_id"])
    entity_by_id.update(left["entity_by_id"])

    strict: dict[str, dict] = {}
    loose: dict[str, dict] = {}
    full_strict: dict[str, dict] = {}
    full_loose: dict[str, dict] = {}
    for k in (5, 10, 20):
        top_eids = set(_top_k_entities(left["doc"], k)) | set(_top_k_entities(right["doc"], k))
        top_ids = {mapping[eid] for eid in top_eids if eid in mapping}
        facts_k = [item for item in merged if item.get("entity_id") in top_eids]
        strict[str(k)] = recall_metrics(gold_facts, facts_k, mapping, top_ids, "fact")
        loose[str(k)] = recall_metrics(gold_facts, facts_k, mapping, top_ids, "fact", loose=True)
        full_strict[str(k)] = recall_metrics(gold_facts, facts_k, mapping, all_gold_fact_ids, "fact")
        full_loose[str(k)] = recall_metrics(
            gold_facts, facts_k, mapping, all_gold_fact_ids, "fact", loose=True,
        )

    precision, _unmatched = _precision(
        gold_facts, merged, mapping, sampled, "fact", adjudication, entity_by_id, reuse,
    )
    left_precision = run_results[left_label]["precision_sampled"]
    left_precision_value = left_precision.get("adjudicated") or left_precision["raw"]
    union_precision_value = precision.get("adjudicated") or precision["raw"]
    left_strict_20 = run_results[left_label]["recall"]["strict"]["20"]["fraction"]["value"]
    union_strict_20 = strict["20"]["fraction"]["value"]
    recall_gain = (
        round(union_strict_20 - left_strict_20, 6)
        if union_strict_20 is not None and left_strict_20 is not None else None
    )
    precision_diff = (
        round(union_precision_value["value"] - left_precision_value["value"], 6)
        if union_precision_value["value"] is not None and left_precision_value["value"] is not None
        else None
    )
    full_all = recall_metrics(gold_facts, merged, mapping, all_gold_fact_ids, "fact")
    return {
        "left": left_label,
        "right": right_label,
        "facts_total": len(merged),
        "facts_added": len(merged) - len(left["facts"]),
        "recall": {
            "strict": strict,
            "loose": loose,
            "full": {
                **full_all,
                "strict_by_k": full_strict,
                "loose_by_k": full_loose,
                "loose_all": recall_metrics(
                    gold_facts, merged, mapping, all_gold_fact_ids, "fact", loose=True,
                ),
            },
        },
        "precision_sampled": precision,
        "recall_strict_K20": union_strict_20,
        "precision_adjudicated": union_precision_value["value"],
        "recall_gain": recall_gain,
        "precision_diff": precision_diff,
    }


def _build_union(
    run_context: dict,
    run_results: dict,
    gold_facts: list[dict],
    all_gold_fact_ids: set[str],
    sampled: list[dict],
    adjudication: dict[str, str] | None,
    reuse: dict | None,
) -> dict:
    """W-1 ∪ E3-1 并集评估；存在 e1-1/e2-1/e3-1 时同时给出两组对照。"""
    winner, w_label, e3_label, rule = _select_union_labels(run_context, run_results)
    if w_label is None or e3_label is None or w_label == e3_label:
        return {
            "available": False,
            "reason": "缺少可比较的 W-1 或 E3-1 运行",
            "winner_method": winner,
            "rule": rule,
        }
    labels = set(run_context)
    pair_specs: list[tuple[str, str]] = []
    if "e1-1" in labels and "e3-1" in labels:
        pair_specs.append(("e1-1", "e3-1"))
    if "e2-1" in labels and "e3-1" in labels:
        pair_specs.append(("e2-1", "e3-1"))
    selected_key = f"{w_label}∪{e3_label}"
    if all(f"{left}∪{right}" != selected_key for left, right in pair_specs):
        pair_specs.append((w_label, e3_label))
    pairs: dict[str, dict] = {}
    for left_label, right_label in pair_specs:
        pairs[f"{left_label}∪{right_label}"] = _union_result(
            left_label, right_label, run_context, run_results,
            gold_facts, all_gold_fact_ids, sampled, adjudication, reuse,
        )
    selected = pairs[selected_key]
    return {
        "available": True,
        "winner_method": winner,
        "selected_pair": selected_key,
        "rule": rule,
        **selected,
        "pairs": pairs,
    }


def evaluate(gold_path: Path, manifest: list[dict], adjudication_path: Path,
             adjudication_result_path: Path, blind_sample_path: Path,
             blind_result_path: Path | None, seed: int,
             reuse_adjudication_path: Path | None = None,
             blind_labels: str | None = None) -> dict:
    gold_doc = _read_json(gold_path)
    gold_dir = gold_path.parent
    roster_doc = _read_json(gold_dir / "roster.json")
    mentions_doc = _read_json(gold_dir / "mentions.json")
    hard_cases_path = gold_dir / "hard_cases.json"
    hard_cases_doc = _read_json(hard_cases_path) if hard_cases_path.is_file() else None
    sampled = gold_doc.get("sampled_chapters") or mentions_doc.get("sampled_chapters") or []
    gold_facts = _gold_items(gold_doc, "facts")
    gold_events = _gold_items(gold_doc, "events")
    all_gold_fact_ids = {g["gold_id"] for g in gold_facts}
    all_gold_event_ids = {g["gold_id"] for g in gold_events}

    full_gold_mentions = _build_combined_gold_mentions(roster_doc, mentions_doc, hard_cases_doc)
    gold_mentions_6ch = [
        item for item in full_gold_mentions
        if _in_sampled(int(item["start"]), sampled)
    ]

    adjudication = None
    adjudication_rows = _read_jsonl(adjudication_result_path)
    if adjudication_rows:
        adjudication = {}
        for row in adjudication_rows:
            verdict = row.get("verdict")
            if verdict not in ADJUDICATION_VERDICTS:
                raise ValueError(f"未知裁定结果：{verdict}")
            adjudication[row["adj_id"]] = verdict
    reuse = _load_reuse_table(reuse_adjudication_path)

    run_results: dict[str, dict] = {}
    adjudication_entries: dict[str, dict] = {}
    first_by_method: dict[str, str] = {}
    describe_docs: dict[str, dict] = {}
    run_context: dict[str, dict] = {}
    ref_text = ""
    ref_chapters: list[dict] = []
    for spec in manifest:
        label = spec["label"]
        method = spec["method"]
        arm = _arm_for(spec)
        if method not in first_by_method:
            first_by_method[method] = label
        describe_doc = _read_json(Path(spec["describe"]))
        describe_docs[label] = describe_doc
        entity_ids = [entity["entity_id"] for entity in describe_doc.get("entities", [])]
        store = Store(Path(spec["db"]))
        try:
            mentions, headers = _load_mentions_and_headers(store, spec["run_id"], entity_ids)
            text, book_chapters = _source_text_and_chapters(store, spec["run_id"])
            if not ref_text:
                ref_text = text
                ref_chapters = book_chapters
        finally:
            store.close()
        mapping, mapping_details = map_entities(
            describe_doc, gold_mentions_6ch, mentions, roster_doc,
            full_gold_mentions=full_gold_mentions, entity_headers=headers,
        )
        sampled_gold_ids = set(mapping.values())
        facts = _entity_facts(describe_doc)
        events = _entity_events(describe_doc)

        recall_strict: dict[str, dict] = {}
        recall_loose: dict[str, dict] = {}
        full_strict_by_k: dict[str, dict] = {}
        full_loose_by_k: dict[str, dict] = {}
        for k in (5, 10, 20):
            top_eids_k = set(_top_k_entities(describe_doc, k))
            facts_k = [f for f in facts if f.get("entity_id") in top_eids_k]
            top_ids = {mapping[eid] for eid in top_eids_k if eid in mapping}
            recall_strict[str(k)] = recall_metrics(gold_facts, facts_k, mapping, top_ids, "fact")
            recall_loose[str(k)] = recall_metrics(gold_facts, facts_k, mapping, top_ids, "fact", loose=True)
            full_strict_by_k[str(k)] = recall_metrics(gold_facts, facts_k, mapping, all_gold_fact_ids, "fact")
            full_loose_by_k[str(k)] = recall_metrics(
                gold_facts, facts_k, mapping, all_gold_fact_ids, "fact", loose=True
            )

        full_strict_all = recall_metrics(gold_facts, facts, mapping, all_gold_fact_ids, "fact")
        full_loose_all = recall_metrics(gold_facts, facts, mapping, all_gold_fact_ids, "fact", loose=True)
        recall = {
            "strict": recall_strict,
            "loose": recall_loose,
            "full": {
                **full_strict_all,
                "strict_by_k": full_strict_by_k,
                "loose_by_k": full_loose_by_k,
                "loose_all": full_loose_all,
            },
        }

        entity_by_id = {entity["entity_id"]: entity for entity in describe_doc.get("entities", [])}
        precision, unmatched_facts = _precision(
            gold_facts, facts, mapping, sampled, "fact", adjudication, entity_by_id, reuse,
        )
        event_precision, unmatched_events = _precision(
            gold_events, events, mapping, sampled, "event", adjudication, entity_by_id, reuse,
        )
        event_recall = recall_metrics(gold_events, events, mapping, sampled_gold_ids, "event")
        event_recall_full = recall_metrics(gold_events, events, mapping, all_gold_event_ids, "event")

        for kind, items in (("fact", unmatched_facts), ("event", unmatched_events)):
            for item in items:
                if _reuse_verdict(reuse, kind, item) is not None:
                    continue
                entry = _adjudication_entry(kind, item, entity_by_id, sampled, text)
                adjudication_entries.setdefault(entry["adj_id"], entry)

        unmapped = [
            {
                "entity_id": entity["entity_id"],
                "name": entity.get("name"),
                "kind": entity.get("kind"),
                "kind_rank": entity.get("kind_rank"),
            }
            for entity in describe_doc.get("entities", [])
            if entity["entity_id"] not in mapping
        ]
        run_results[label] = {
            "method": method,
            "arm": arm,
            "first": first_by_method[method] == label,
            "describe_id": describe_doc.get("describe_id"),
            "run_id": describe_doc.get("run_id"),
            "entity_mapping": {
                "mapped": dict(sorted(mapping.items())),
                "details": mapping_details,
                "unmapped": unmapped,
            },
            "recall": recall,
            "precision_sampled": precision,
            "events": {
                "recall": event_recall,
                "recall_full": event_recall_full,
                "precision_sampled": event_precision,
            },
            "facts_total": len(facts),
            "events_total": len(events),
        }
        run_context[label] = {
            "method": method,
            "arm": arm,
            "doc": describe_doc,
            "facts": facts,
            "mapping": mapping,
            "entity_by_id": entity_by_id,
            "unmatched_facts": unmatched_facts,
            "unmatched_events": unmatched_events,
        }

    adjudication_list = sorted(adjudication_entries.values(), key=lambda item: item["adj_id"])
    _write_jsonl(adjudication_path, adjudication_list)

    blind_label_list = _blind_sample_labels(blind_labels, run_results, first_by_method)
    arm_by_label = {label: run_results[label]["arm"] for label in run_results}
    blind_items, blind_meta = _build_blind_sample(
        describe_docs, blind_label_list, ref_chapters or sampled, ref_text, seed, arm_by_label,
    )
    _write_jsonl(blind_sample_path, blind_items)
    blind_summary = (
        _summarize_blind(blind_result_path, blind_items, blind_meta)
        if blind_result_path else None
    )

    def _aggregate(labels: list[str]) -> dict:
        strict_20 = [run_results[label]["recall"]["strict"]["20"]["fraction"]["value"] for label in labels]
        loose_20 = [run_results[label]["recall"]["loose"]["20"]["fraction"]["value"] for label in labels]
        full_vals = [run_results[label]["recall"]["full"]["fraction"]["value"] for label in labels]
        precisions = []
        raw_precisions = []
        for label in labels:
            sampled_p = run_results[label]["precision_sampled"]
            raw_precisions.append(sampled_p["raw"]["value"])
            chosen = sampled_p.get("adjudicated")
            if chosen is None:
                chosen = sampled_p.get("raw")
            precisions.append(chosen["value"])
        event_values = [run_results[label]["events"]["recall"]["fraction"]["value"] for label in labels]
        event_full_values = [run_results[label]["events"]["recall_full"]["fraction"]["value"] for label in labels]
        return {
            "runs": labels,
            "recall_strict_K20_mean": _mean(strict_20),
            "recall_strict_K20_values": strict_20,
            "recall_loose_K20_mean": _mean(loose_20),
            "recall_loose_K20_values": loose_20,
            "recall_full_mean": _mean(full_vals),
            "recall_full_values": full_vals,
            "precision_raw_mean": _mean(raw_precisions),
            "precision_raw_values": raw_precisions,
            "precision_adjudicated_mean": _mean(precisions),
            "precision_values": precisions,
            "event_recall_mean": _mean(event_values),
            "event_recall_full_mean": _mean(event_full_values),
        }

    aggregate: dict[str, dict] = {}
    for method in ("d1", "d2"):
        labels = [label for label, run in run_results.items() if run["method"] == method]
        if labels:
            aggregate[method] = _aggregate(labels)
    aggregate_by_arm: dict[str, dict] = {}
    for arm in sorted({run["arm"] for run in run_results.values()}):
        labels = [label for label, run in run_results.items() if run["arm"] == arm]
        if not labels:
            continue
        aggregate_by_arm[arm] = _aggregate(labels)
        aggregate[arm] = aggregate_by_arm[arm]

    union = _build_union(
        run_context, run_results, gold_facts, all_gold_fact_ids, sampled,
        adjudication, reuse,
    )

    reused_key_set: set[tuple] = set()
    new_key_set: set[tuple] = set()
    reused_item_total = 0
    new_item_total = 0
    for label, run in run_results.items():
        for kind in ("fact", "event"):
            sampled_p = run["precision_sampled"] if kind == "fact" else run["events"]["precision_sampled"]
            reused_item_total += int(sampled_p.get("reused_adjudication_count") or 0)
            new_item_total += int(sampled_p.get("new_adjudication_count") or 0)
        for kind, unmatched in (("fact", run_context[label]["unmatched_facts"]),
                                ("event", run_context[label]["unmatched_events"])):
            for item in unmatched:
                key = _reuse_key(kind, item)
                if key is None:
                    continue
                if _reuse_verdict(reuse, kind, item) is not None:
                    reused_key_set.add((kind, key))
                else:
                    new_key_set.add((kind, key))

    return {
        "gold": {
            "path": str(gold_path),
            "sha256": _sha256(gold_path),
            "facts": len(gold_facts),
            "events": len(gold_events),
            "sampled_chapters": sampled,
        },
        "adjudication": {
            "package_path": str(adjudication_path),
            "package_items": len(adjudication_list),
            "result_path": str(adjudication_result_path),
            "result_items": len(adjudication_rows),
            "verdicts": dict(sorted(collections.Counter(
                row["verdict"] for row in adjudication_rows
            ).items())),
            "reuse_source_path": reuse["path"],
            "reuse_source_rows": reuse["rows"],
            "reused_adjudication_count": len(reused_key_set),
            "reused_adjudication_items": reused_item_total,
            "new_adjudication_count": len(new_key_set),
            "new_adjudication_items": new_item_total,
        },
        "blind_sample": {
            "path": str(blind_sample_path),
            "count": len(blind_items),
            "per_arm": BLIND_PER_ARM,
            "labels": blind_label_list,
            "seed": seed,
        },
        "blind_summary": blind_summary,
        "runs": run_results,
        "aggregate": aggregate,
        "aggregate_by_arm": aggregate_by_arm,
        "union": union,
    }


def _blind_sample_labels(
    raw: str | None, run_results: dict, first_by_method: dict[str, str]
) -> list[str]:
    """默认优先 e2-1 与 e3-1，缺失时回退到各方法首个运行。"""
    labels: list[str] = []
    if raw:
        labels = [part.strip() for part in raw.split(",") if part.strip()]
    elif "e2-1" in run_results and "e3-1" in run_results:
        labels = ["e2-1", "e3-1"]
    else:
        labels = [
            first_by_method[method]
            for method in ("d1", "d2")
            if method in first_by_method
        ]
    ordered: list[str] = []
    for label in labels:
        if label not in run_results:
            raise ValueError(f"盲审抽样运行不存在：{label}")
        if label not in ordered:
            ordered.append(label)
    return ordered


def _build_blind_sample(
    describe_docs: dict[str, dict],
    labels: list[str],
    chapters: list[dict],
    text: str,
    seed: int,
    arm_by_label: dict[str, str] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    entries: list[dict] = []
    meta_by_sample_id: dict[str, dict] = {}
    for index, label in enumerate(labels):
        doc = describe_docs[label]
        entity_by_id = {entity["entity_id"]: entity for entity in doc.get("entities", [])}
        facts = _entity_facts(doc)
        rng = random.Random(seed + index)
        picked = rng.sample(facts, min(BLIND_PER_ARM, len(facts)))
        for idx, fact in enumerate(picked):
            entity = entity_by_id.get(fact.get("entity_id"), {})
            start, end = int(fact["start"]), int(fact["end"])
            sample_id = digest([
                "description-blind-v1", label, idx, fact.get("entity_id"),
                fact.get("facet"), fact.get("value"), fact.get("quote"),
                start, end,
            ])[:24]
            meta_by_sample_id[sample_id] = {
                "label": label,
                "arm": (arm_by_label or {}).get(label, label.split("-")[0]),
                "method": doc.get("method"),
            }
            entries.append({
                "sample_id": sample_id,
                "entity_name": entity.get("name"),
                "entity_kind": entity.get("kind"),
                "facet": fact.get("facet"),
                "value": fact.get("value"),
                "quote": fact.get("quote"),
                "start": start,
                "end": end,
                "chapter_ordinal": _chapter_ordinal(start, chapters),
                "context_before": text[max(0, start - CONTEXT_CHARS):start] if text else "",
                "context_after": text[end:end + CONTEXT_CHARS] if text else "",
            })
    random.Random(seed + 2).shuffle(entries)
    return entries, meta_by_sample_id


def _summarize_blind(
    blind_result_path: Path,
    blind_items: list[dict],
    meta_by_sample_id: dict[str, dict],
) -> dict:
    rows = _read_jsonl(blind_result_path)
    by_id = {item["sample_id"]: item for item in blind_items}
    fields = ("wrong_entity", "wrong_facet", "unsupported_value")
    group_keys = ("method", "label", "arm")
    counts = {field: 0 for field in fields}
    group_counts: dict[str, dict[str, dict[str, int]]] = {key: {} for key in group_keys}
    group_totals: dict[str, collections.Counter] = {key: collections.Counter() for key in group_keys}
    group_judged: dict[str, collections.Counter] = {key: collections.Counter() for key in group_keys}
    for meta in meta_by_sample_id.values():
        for key in group_keys:
            value = meta.get(key)
            if value:
                group_totals[key][value] += 1
    judged = 0
    seen_ids: set[str] = set()
    for row in rows:
        sid = row.get("sample_id")
        if sid not in by_id or sid in seen_ids:
            continue
        seen_ids.add(sid)
        judged += 1
        meta = meta_by_sample_id.get(sid) or {}
        for key in group_keys:
            value = meta.get(key)
            if value:
                group_judged[key][value] += 1
        for field in fields:
            if bool(row.get(field)):
                counts[field] += 1
                for key in group_keys:
                    value = meta.get(key)
                    if not value:
                        continue
                    bucket = group_counts[key].setdefault(
                        value, {name: 0 for name in fields}
                    )
                    bucket[field] += 1

    def _group_summary(key: str) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for value, total in sorted(group_totals[key].items()):
            judged_value = group_judged[key][value]
            value_counts = group_counts[key].get(value, {name: 0 for name in fields})
            summary[value] = {
                "judged": judged_value,
                "total": total,
                "counts": dict(value_counts),
                "rates": {
                    field: fraction(value_counts[field], judged_value) for field in fields
                },
            }
        return summary

    return {
        "result_path": str(blind_result_path),
        "judged": judged,
        "total": len(blind_items),
        "counts": counts,
        "rates": {field: fraction(counts[field], judged) for field in fields},
        "by_method": _group_summary("method"),
        "by_label": _group_summary("label"),
        "by_arm": _group_summary("arm"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="描述事实金标准评分与盲审包导出（零模型调用）")
    parser.add_argument("--gold", type=Path,
                        default=ROOT / "test/entity-gold/fengshen/reference/descriptions.json")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="评分清单 JSON：{\"runs\":[{\"label\",\"arm\",\"method\",\"db\",\"run_id\",\"describe\"}]}")
    parser.add_argument("--out", type=Path, required=True, help="评分结果 JSON 路径")
    parser.add_argument("--adjudication", type=Path, default=ROOT / ".local/016a/adjudication.jsonl")
    parser.add_argument("--adjudication-result", type=Path,
                        default=ROOT / ".local/016a/adjudication_result.jsonl")
    parser.add_argument("--reuse-adjudication", type=Path, default=None,
                        help="旧裁定结果 JSONL；命中的未匹配项直接复用，只把新键导出到 --adjudication")
    parser.add_argument("--blind-sample", type=Path, default=ROOT / ".local/016a/blind_sample.jsonl")
    parser.add_argument("--blind-result", type=Path, default=None)
    parser.add_argument("--blind-labels", default=None,
                        help="全书盲审抽样运行标签，逗号分隔；默认优先 e2-1,e3-1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        manifest = _load_manifest(args.manifest)
        result = evaluate(
            args.gold, manifest, args.adjudication, args.adjudication_result,
            args.blind_sample, args.blind_result, args.seed,
            reuse_adjudication_path=args.reuse_adjudication,
            blind_labels=args.blind_labels,
        )
        write_json(args.out, result)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        print(f"评分失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "ok": True,
        "out": str(args.out),
        "runs": len(result["runs"]),
        "aggregate": result["aggregate"],
        "adjudication_items": result["adjudication"]["package_items"],
        "reused_adjudication_count": result["adjudication"]["reused_adjudication_count"],
        "blind_sample_count": result["blind_sample"]["count"],
        "union_available": result["union"]["available"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
