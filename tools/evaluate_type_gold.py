"""0.14a 分类别评分工具：评估《封神演义》多类型金标准三层（roster, mentions, hard_cases）与多次运行噪声。
可导入 tools/evaluate_identity_gold.py 的辅助函数，保持 tools/evaluate_identity_gold.py 不变。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entitytrace.roster import normalize_surface
from tools.evaluate_identity_gold import fraction, infer_predicted_entity_gold_id

GOLD_TYPES = ["person", "place", "org", "item", "generic", "not_entity"]
PRED_TYPES = ["person", "place", "org", "item", "unmatched"]
PROPER_TYPES = ["person", "place", "org", "item"]
SEVEN_KINDS = ["person", "place", "object", "organization", "ability", "concept", "unknown"]
NON_PERSON_4CLASS = ["place", "org", "item"]
NON_PERSON_7CLASS = ["place", "object", "organization", "ability", "concept", "unknown"]


def _map_raw_kind_to_4class(raw_kind: str) -> str:
    """将发现/名册阶段的 7 类粗类映射为 4 类（person, place, org, item）。"""
    if raw_kind == "person":
        return "person"
    if raw_kind == "place":
        return "place"
    if raw_kind in {"organization", "org"}:
        return "org"
    if raw_kind in {"object", "ability", "concept", "item", "unknown"}:
        return "item"
    return "person"


def _enrich_roster_entity(ent: dict) -> dict:
    """为名册实体补齐 7 类 kind、alt_kinds 与 policy（T1/T2），兼容已有显式字段。"""
    etype = ent.get("entity_type") or _map_raw_kind_to_4class(ent.get("kind", "person"))
    stype = ent.get("sub_type") or ""
    cname = ent.get("canonical_name") or ent.get("canonical") or ""

    if "kind" in ent and ent["kind"] in SEVEN_KINDS:
        kind_7 = ent["kind"]
        alt_kinds = list(ent.get("alt_kinds", []))
        policy = ent.get("policy", "T1")
    else:
        if etype == "person":
            kind_7 = "person"
            alt_kinds = []
            policy = "T1"
        elif etype == "place":
            kind_7 = "place"
            alt_kinds = ["organization"] if stype in {"palace", "building"} else []
            policy = "T1"
        elif etype == "org":
            kind_7 = "organization"
            alt_kinds = ["place"] if stype == "dynasty" else (["person"] if stype == "troop_group" else [])
            policy = "T1"
        else:  # item
            if stype == "mount":
                kind_7 = "object"
                alt_kinds = [] if cname in {"风火轮", "九龙沉香辇"} else ["person"]
                policy = "T1"
            elif stype == "formation":
                kind_7 = "ability"
                alt_kinds = ["place", "object"]
                policy = "T1"
            elif stype == "spell":
                kind_7 = "ability"
                alt_kinds = ["concept", "object"]
                policy = "T2"
            else:
                kind_7 = "object"
                if cname in {"黄庭", "六韬", "封神榜"}:
                    alt_kinds = ["concept"]
                elif cname == "花狐貂":
                    alt_kinds = ["person"]
                else:
                    alt_kinds = ["ability"]
                policy = "T1"

    return {
        **ent,
        "entity_type": etype,
        "kind_7": kind_7,
        "alt_kinds": alt_kinds,
        "policy": ent.get("policy", policy),
        "canonical": cname,
    }


def _extract_pred_mentions_and_entities(result_doc: dict) -> tuple[list[dict], dict[str, dict], dict[str, int], list[tuple[int, int, int]]]:
    """提取活跃预测实体与预测提及，同时保留 7 类 raw_kind 与 4 类 kind，并建立码点到章节 ordinal 的映射。"""
    chapters = result_doc.get("chapters", [])
    ch_id_to_ord: dict[str, int] = {}
    ch_ranges: list[tuple[int, int, int]] = []
    for idx, ch in enumerate(chapters):
        ord_no = ch.get("ordinal", idx + 1)
        if "id" in ch:
            ch_id_to_ord[ch["id"]] = ord_no
        s = ch.get("start", 0)
        e = ch.get("end", 0)
        ch_ranges.append((s, e, ord_no))

    pred_entities: dict[str, dict] = {}
    mid_to_eid: dict[str, str] = {}
    for ent in result_doc.get("entities", []):
        eid = ent["id"]
        raw_kind = ent.get("kind") or ent.get("entity_type") or ent.get("type") or "person"
        kind_4 = _map_raw_kind_to_4class(raw_kind)
        pred_entities[eid] = {
            **ent,
            "raw_kind": raw_kind if raw_kind in SEVEN_KINDS else "unknown",
            "kind": kind_4,
        }
        for mid in ent.get("mention_ids", []):
            mid_to_eid[mid] = eid

    pred_mentions: list[dict] = []
    for m in result_doc.get("mentions", []):
        if m.get("identity_status") == "not_entity_suggested" or m.get("link_method") == "not_entity_suggested":
            continue
        mid = m["id"]
        eid = m.get("entity_id") or mid_to_eid.get(mid)
        if eid and eid in pred_entities and pred_entities[eid].get("identity_status") == "not_entity_suggested":
            continue
        span = m.get("span") if isinstance(m.get("span"), dict) else m
        start = int(span["start"])
        end = int(span["end"])
        surf = m.get("text") or m.get("surface") or m.get("quote") or ""
        if eid and eid in pred_entities:
            raw_kind = pred_entities[eid]["raw_kind"]
            p_kind = pred_entities[eid]["kind"]
        else:
            p_kinds = m.get("provisional_kinds") or []
            raw_kind = m.get("kind") or m.get("entity_type") or (p_kinds[0] if p_kinds else "person")
            p_kind = _map_raw_kind_to_4class(raw_kind)

        ch_ord = ch_id_to_ord.get(m.get("chapter_id", ""))
        if ch_ord is None:
            for cs, ce, cord in ch_ranges:
                if cs <= start < ce:
                    ch_ord = cord
                    break

        pred_mentions.append({
            "id": mid,
            "start": start,
            "end": end,
            "text": surf,
            "entity_id": eid,
            "raw_kind": raw_kind if raw_kind in SEVEN_KINDS else "unknown",
            "kind": p_kind if p_kind in PROPER_TYPES else "person",
            "chapter_ordinal": ch_ord,
        })

    # 为每个预测实体预计算其活跃提及覆盖的金标准回目集合
    eid_to_chapters: dict[str, set[int]] = collections.defaultdict(set)
    eid_to_active_mids: dict[str, list[str]] = collections.defaultdict(list)
    for pm in pred_mentions:
        peid = pm.get("entity_id")
        if peid:
            eid_to_active_mids[peid].append(pm["id"])
            if pm.get("chapter_ordinal") is not None:
                eid_to_chapters[peid].add(int(pm["chapter_ordinal"]))

    for eid, ent in pred_entities.items():
        ent["active_chapters"] = sorted(eid_to_chapters.get(eid, set()))
        ent["active_mention_count"] = len(eid_to_active_mids.get(eid, ent.get("mention_ids", [])))

    return pred_mentions, pred_entities, ch_id_to_ord, ch_ranges


def match_gold_to_pred(
    gold_items: list[dict],
    pred_mentions: list[dict],
    strict_only: bool = True,
) -> dict[str, dict]:
    """一对一匹配金标准条目与预测提及（与 evaluate_identity_gold.match_mentions 对齐）。
    strict_only=True: pred.start == gold.start and pred.end == gold.end
    strict_only=False: 重叠 >= 50% 金标准长度 (overlap * 2 >= glen)
    """
    candidates: list[tuple[int, int, int, int, str, dict]] = []
    for g in gold_items:
        gid_key = g["_eval_key"]
        gs, ge = int(g["start"]), int(g["end"])
        glen = ge - gs
        if glen <= 0:
            continue
        for p in pred_mentions:
            ps, pe = p["start"], p["end"]
            overlap = min(ge, pe) - max(gs, ps)
            if overlap <= 0:
                continue
            exact = int(gs == ps and ge == pe)
            if strict_only and not exact:
                continue
            if not exact and overlap * 2 < glen:
                continue
            len_diff = abs((pe - ps) - glen)
            candidates.append((exact, overlap, -len_diff, -ps, gid_key, p))

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    matched: dict[str, dict] = {}
    used_pred: set[str] = set()
    for _, _, _, _, gid_key, p in candidates:
        if gid_key in matched or p["id"] in used_pred:
            continue
        matched[gid_key] = p
        used_pred.add(p["id"])
    return matched


def _build_confusion_matrix(gold_items: list[dict], matched_map: dict[str, dict]) -> dict[str, dict[str, int]]:
    matrix = {gt: {pt: 0 for pt in PRED_TYPES} for gt in GOLD_TYPES}
    for g in gold_items:
        gt = g["entity_type"]
        if gt not in matrix:
            continue
        p = matched_map.get(g["_eval_key"])
        pt = p["kind"] if p is not None and p["kind"] in PROPER_TYPES else "unmatched"
        matrix[gt][pt] += 1
    return matrix


def _compute_recall_by_type(gold_items: list[dict], matched_map: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for pt in PROPER_TYPES:
        sub = [g for g in gold_items if g["entity_type"] == pt]
        denom = len(sub)
        hit_any = sum(1 for g in sub if g["_eval_key"] in matched_map)
        hit_correct = sum(
            1 for g in sub if g["_eval_key"] in matched_map and matched_map[g["_eval_key"]]["kind"] == pt
        )
        out[pt] = {
            "type_correct_recall": fraction(hit_correct, denom),
            "span_any_type_recall": fraction(hit_any, denom),
        }
    return out


def _compute_exhaustive_precision(
    exhaustive_gold: list[dict],
    matched_map: dict[str, dict],
    pred_mentions: list[dict],
    sampled_ords: set[int],
) -> dict:
    pred_id_to_gold: dict[str, dict] = {}
    for g in exhaustive_gold:
        p = matched_map.get(g["_eval_key"])
        if p is not None:
            pred_id_to_gold[p["id"]] = g

    by_type_interval: dict[str, dict] = {}
    for pt in PROPER_TYPES:
        matched_preds_of_pt = [
            (p_id, g) for p_id, g in pred_id_to_gold.items()
            if matched_map[g["_eval_key"]]["kind"] == pt
        ]
        denom = len(matched_preds_of_pt)
        num = sum(1 for _, g in matched_preds_of_pt if g["entity_type"] == pt)
        by_type_interval[pt] = fraction(num, denom)

    non_person_matched = [
        (p_id, g) for p_id, g in pred_id_to_gold.items()
        if matched_map[g["_eval_key"]]["kind"] in {"place", "org", "item"}
    ]
    np_interval_num = sum(
        1 for _, g in non_person_matched if g["entity_type"] == matched_map[g["_eval_key"]]["kind"]
    )
    by_type_interval["non_person_overall"] = fraction(np_interval_num, len(non_person_matched))

    preds_in_sampled = [p for p in pred_mentions if p.get("chapter_ordinal") in sampled_ords]
    by_type_chapter: dict[str, dict] = {}
    for pt in ("place", "org", "item"):
        pt_preds = [p for p in preds_in_sampled if p["kind"] == pt]
        num = sum(1 for p in pt_preds if p["id"] in pred_id_to_gold and pred_id_to_gold[p["id"]]["entity_type"] == pt)
        by_type_chapter[pt] = fraction(num, len(pt_preds))
    np_preds = [p for p in preds_in_sampled if p["kind"] in {"place", "org", "item"}]
    np_ch_num = sum(
        1 for p in np_preds if p["id"] in pred_id_to_gold and pred_id_to_gold[p["id"]]["entity_type"] == p["kind"]
    )
    by_type_chapter["non_person_overall"] = fraction(np_ch_num, len(np_preds))

    return {
        "annotated_interval_precision": by_type_interval,
        "sampled_chapter_non_person_precision": by_type_chapter,
    }


def _compute_pollution(
    exhaustive_gold: list[dict],
    matched_map: dict[str, dict],
    pred_entities: dict[str, dict],
    pred_mentions: list[dict],
) -> dict:
    generic_items = [g for g in exhaustive_gold if g["entity_type"] == "generic"]
    not_entity_items = [g for g in exhaustive_gold if g["entity_type"] == "not_entity"]

    def analyze_group(items: list[dict]) -> dict:
        polluted = []
        by_pred_kind = collections.Counter()
        surf_counter = collections.Counter()
        surf_kinds: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for g in items:
            p = matched_map.get(g["_eval_key"])
            if p is not None and p["kind"] in PROPER_TYPES:
                polluted.append((g, p))
                by_pred_kind[p["kind"]] += 1
                surf_counter[g["text"]] += 1
                surf_kinds[g["text"]][p["kind"]] += 1
        top_surfaces = [
            {"surface": s, "count": cnt, "pred_types": dict(surf_kinds[s])}
            for s, cnt in surf_counter.most_common(15)
        ]
        return {
            "pollution_rate": fraction(len(polluted), len(items)),
            "by_predicted_type": {k: by_pred_kind.get(k, 0) for k in PROPER_TYPES},
            "top_polluted_surfaces": top_surfaces,
        }

    gen_report = analyze_group(generic_items)
    ne_report = analyze_group(not_entity_items)
    combined_report = analyze_group(generic_items + not_entity_items)

    gen_with_ref = [g for g in generic_items if g.get("refers_to") is not None]
    gen_pure = [g for g in generic_items if g.get("refers_to") is None]
    gen_report["with_refers_to_rate"] = analyze_group(gen_with_ref)["pollution_rate"]
    gen_report["pure_generic_rate"] = analyze_group(gen_pure)["pollution_rate"]

    pure_pollution_surfaces = {
        g["text"] for g in (generic_items + not_entity_items)
    }
    proper_gold_surfaces = {
        g["text"] for g in exhaustive_gold if g["entity_type"] in PROPER_TYPES
    }
    strictly_non_proper_surfaces = pure_pollution_surfaces - proper_gold_surfaces

    mid_to_surf = {m["id"]: m["text"] for m in pred_mentions}
    active_entities = [
        e for e in pred_entities.values() if e.get("identity_status") != "not_entity_suggested"
    ]
    polluted_entities = []
    polluted_by_kind = collections.Counter()
    polluted_by_raw_kind = collections.Counter()
    for ent in active_entities:
        surfs = set(ent.get("names") or []) | set(ent.get("aliases") or [])
        if not surfs and ent.get("name"):
            surfs.add(ent["name"])
        for mid in ent.get("mention_ids", []):
            if mid in mid_to_surf:
                surfs.add(mid_to_surf[mid])
        surfs = {s for s in surfs if s}
        if surfs and surfs.issubset(strictly_non_proper_surfaces):
            polluted_entities.append({
                "entity_id": ent["id"],
                "name": ent.get("name", ""),
                "kind": ent.get("kind", "person"),
                "raw_kind": ent.get("raw_kind", "person"),
                "mention_count": ent.get("mention_count", len(ent.get("mention_ids", []))),
                "active_chapters": ent.get("active_chapters", []),
                "surfaces": sorted(surfs),
            })
            polluted_by_kind[ent.get("kind", "person")] += 1
            polluted_by_raw_kind[ent.get("raw_kind", "person")] += 1

    polluted_entities.sort(key=lambda x: (-x["mention_count"], x["name"]))
    return {
        "generic": gen_report,
        "not_entity": ne_report,
        "combined_generic_and_not_entity": combined_report,
        "entity_table_pollution": {
            "polluted_entity_rate": fraction(len(polluted_entities), len(active_entities)),
            "by_kind": {k: polluted_by_kind.get(k, 0) for k in PROPER_TYPES},
            "by_raw_kind": {k: polluted_by_raw_kind.get(k, 0) for k in SEVEN_KINDS},
            "top_polluted_entities": polluted_entities[:20],
        },
    }


def _compute_b3_and_split_merge(
    positive_gold_items: list[dict],
    matched_map: dict[str, dict],
    roster_map: dict[str, dict],
    pred_entities: dict[str, dict],
) -> dict:
    def eval_subset(items: list[dict]) -> dict:
        if not items:
            return {
                "gold_mention_count": 0,
                "matched_mention_count": 0,
                "b3_precision": None,
                "b3_recall": None,
                "b3_f1": None,
                "b3_recall_matched_only": None,
                "b3_f1_matched_only": None,
                "over_split": fraction(0, 0),
                "over_split_details": [],
                "over_merge": fraction(0, 0),
                "over_merge_details": [],
            }
        gold_by_gid: dict[str, list[dict]] = collections.defaultdict(list)
        matched_gold_by_gid: dict[str, list[dict]] = collections.defaultdict(list)
        pred_cluster_gids: dict[str, list[str]] = collections.defaultdict(list)
        gid_to_pred_eids: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

        matched_pairs: list[tuple[dict, str]] = []
        for g in items:
            gid = g.get("gold_id") or g.get("refers_to")
            if not gid:
                continue
            gold_by_gid[gid].append(g)
            p = matched_map.get(g["_eval_key"])
            if p is not None and p.get("entity_id"):
                peid = p["entity_id"]
                matched_pairs.append((g, peid))
                matched_gold_by_gid[gid].append(g)
                pred_cluster_gids[peid].append(gid)
                gid_to_pred_eids[gid][peid] += 1

        n_total = len(items)
        n_matched = len(matched_pairs)

        if n_matched == 0:
            b3_p = 0.0
            b3_r_all = 0.0
            b3_r_matched = 0.0
            b3_f1 = 0.0
            b3_f1_matched = 0.0
        else:
            p_sum = 0.0
            r_all_sum = 0.0
            r_matched_sum = 0.0
            for g, peid in matched_pairs:
                gid = g.get("gold_id") or g.get("refers_to")
                cluster_list = pred_cluster_gids[peid]
                same_both = sum(1 for x in cluster_list if x == gid)
                p_i = same_both / len(cluster_list)
                r_all_i = same_both / len(gold_by_gid[gid])
                r_matched_i = same_both / len(matched_gold_by_gid[gid])
                p_sum += p_i
                r_all_sum += r_all_i
                r_matched_sum += r_matched_i
            b3_p = p_sum / n_matched
            b3_r_all = r_all_sum / n_total
            b3_r_matched = r_matched_sum / n_matched
            b3_f1 = (2 * b3_p * b3_r_all / (b3_p + b3_r_all)) if (b3_p + b3_r_all) > 0 else 0.0
            b3_f1_matched = (2 * b3_p * b3_r_matched / (b3_p + b3_r_matched)) if (b3_p + b3_r_matched) > 0 else 0.0

        multi_mention_gids = [gid for gid, eids in gid_to_pred_eids.items() if sum(eids.values()) >= 2]
        over_split_details = []
        for gid in sorted(multi_mention_gids):
            eids = gid_to_pred_eids[gid]
            if len(eids) > 1:
                over_split_details.append({
                    "gold_id": gid,
                    "canonical_name": roster_map.get(gid, {}).get("canonical_name", gid),
                    "split_into_entity_count": len(eids),
                    "predicted_entities": [
                        {
                            "entity_id": peid,
                            "name": pred_entities.get(peid, {}).get("name", ""),
                            "kind": pred_entities.get(peid, {}).get("kind", ""),
                            "raw_kind": pred_entities.get(peid, {}).get("raw_kind", ""),
                            "matched_mentions": cnt,
                        }
                        for peid, cnt in eids.most_common()
                    ],
                })

        over_merge_details = []
        for peid, gids_list in sorted(pred_cluster_gids.items()):
            distinct_gids = sorted(set(gids_list))
            if len(distinct_gids) > 1:
                over_merge_details.append({
                    "predicted_entity_id": peid,
                    "predicted_name": pred_entities.get(peid, {}).get("name", ""),
                    "predicted_kind": pred_entities.get(peid, {}).get("kind", ""),
                    "predicted_raw_kind": pred_entities.get(peid, {}).get("raw_kind", ""),
                    "merged_gold_count": len(distinct_gids),
                    "gold_entities": [
                        {
                            "gold_id": g_id,
                            "canonical_name": roster_map.get(g_id, {}).get("canonical_name", g_id),
                            "entity_type": roster_map.get(g_id, {}).get("entity_type", ""),
                            "count": gids_list.count(g_id),
                        }
                        for g_id in distinct_gids
                    ],
                })

        return {
            "gold_mention_count": n_total,
            "matched_mention_count": n_matched,
            "b3_precision": round(b3_p, 6),
            "b3_recall": round(b3_r_all, 6),
            "b3_f1": round(b3_f1, 6),
            "b3_recall_matched_only": round(b3_r_matched, 6),
            "b3_f1_matched_only": round(b3_f1_matched, 6),
            "over_split": fraction(len(over_split_details), len(multi_mention_gids)),
            "over_split_details": over_split_details,
            "over_merge": fraction(len(over_merge_details), len(pred_cluster_gids)),
            "over_merge_details": over_merge_details,
        }

    by_type = {}
    for pt in PROPER_TYPES:
        sub = [
            g for g in positive_gold_items
            if roster_map.get(g.get("gold_id") or g.get("refers_to"), {}).get("entity_type", g.get("entity_type")) == pt
        ]
        by_type[pt] = eval_subset(sub)
    by_type["overall"] = eval_subset(positive_gold_items)
    return by_type


def _classify_hard_case_type(case: dict) -> str:
    """将难例映射为 0.14a 规格二第 6 项规定的四类：same | different | generic_no_merge | policy_pending。"""
    if case.get("type") in {"same", "different", "generic_no_merge", "policy_pending"}:
        return case["type"]
    cid = case.get("case_id", "")
    title = case.get("title", "")
    if cid == "HC021" or "一气化三清" in title:
        return "policy_pending"
    c_mentions = case.get("mentions", [])
    if any(m.get("entity_type") == "generic" for m in c_mentions):
        return "generic_no_merge"
    gids = {m.get("gold_id") for m in c_mentions if m.get("gold_id")}
    has_ne = any(m.get("entity_type") == "not_entity" for m in c_mentions)
    if len(gids) == 1 and not has_ne:
        return "same"
    return "different"


def _evaluate_hard_cases(
    cases: list[dict],
    matched_map: dict[str, dict],
    pred_entities: dict[str, dict] | None = None,
) -> dict:
    """逐条评估 hard_cases.json：
    1) 保持原有 overall_pass_rate / by_category（严格检查召回+类型+归组）；
    2) 同时按 0.14a 规格二第 6 项输出 by_spec2_type（same / different / generic_no_merge / policy_pending，其中 policy_pending 只列出不计分）。
    """
    pred_entities = pred_entities or {}
    case_results = []
    by_cat_pass: dict[str, int] = collections.Counter()
    by_cat_total: dict[str, int] = collections.Counter()
    passed_ids: list[str] = []
    failed_ids: list[str] = []

    spec2_type_pass: dict[str, list[str]] = collections.defaultdict(list)
    spec2_type_fail: dict[str, list[str]] = collections.defaultdict(list)
    policy_pending_list: list[dict] = []

    for case in cases:
        cid = case["case_id"]
        cat = case["category"]
        hc_type = _classify_hard_case_type(case)
        by_cat_total[cat] += 1
        c_mentions = case["mentions"]

        reasons: list[str] = []
        spec2_reasons: list[str] = []
        gid_to_peids: dict[str, set[str]] = collections.defaultdict(set)
        peid_to_gids: dict[str, set[str]] = collections.defaultdict(set)
        matched_peids_all: list[str] = []

        for hm in c_mentions:
            ekey = f"hc:{cid}:{hm['mention_id']}"
            gt = hm["entity_type"]
            gid = hm.get("gold_id") or hm.get("refers_to")
            p = matched_map.get(ekey)

            if gt in PROPER_TYPES:
                if p is None:
                    reasons.append(f"{hm['mention_id']}({hm['text']}) 未被召回")
                    spec2_reasons.append(f"{hm['mention_id']}({hm['text']}) 未被召回")
                else:
                    if p["kind"] != gt:
                        reasons.append(f"{hm['mention_id']}({hm['text']}) 类型误判为 {p['kind']}（期望 {gt}）")
                    peid = p.get("entity_id")
                    if peid:
                        matched_peids_all.append(peid)
                        if gid:
                            gid_to_peids[gid].add(peid)
                            peid_to_gids[peid].add(gid)
            elif gt == "not_entity":
                if p is not None:
                    reasons.append(f"{hm['mention_id']}({hm['text']}) 非实体被误提取为 {p['kind']}")
                    peid = p.get("entity_id")
                    if peid:
                        matched_peids_all.append(peid)
                        peid_to_gids[peid].add(f"__NE_{hm['mention_id']}")
                        if pred_entities.get(peid, {}).get("active_mention_count", 2) >= 2:
                            spec2_reasons.append(f"{hm['mention_id']}({hm['text']}) 非实体被并入多成员实体 {peid}")
            elif gt == "generic":
                if p is not None:
                    reasons.append(f"{hm['mention_id']}({hm['text']}) 泛指词被提取为专名实体({p['kind']})")
                    peid = p.get("entity_id")
                    if peid:
                        matched_peids_all.append(peid)
                        tag = gid if gid else f"__GEN_{hm['chapter_ordinal']}_{hm['mention_id']}"
                        peid_to_gids[peid].add(tag)
                        if hm.get("refers_to") is None:
                            chs = pred_entities.get(peid, {}).get("active_chapters", [])
                            if len(chs) >= 2:
                                spec2_reasons.append(
                                    f"{hm['mention_id']}({hm['text']}) 无指代泛称所在预测实体 {peid} 跨 {len(chs)} 回({chs})"
                                )

        for gid, peids in gid_to_peids.items():
            if len(peids) > 1:
                msg = f"同指实体 {gid} 被拆成 {len(peids)} 个预测实体"
                reasons.append(msg)
                if hc_type in {"same", "policy_pending"}:
                    spec2_reasons.append(msg)

        for peid, gids in peid_to_gids.items():
            if len(gids) > 1:
                msg = f"不同对象 {sorted(gids)} 被错并入同一预测实体 {peid}"
                reasons.append(msg)
                if hc_type in {"different", "generic_no_merge"}:
                    spec2_reasons.append(msg)

        # 对 generic_no_merge：检查不同回或不同所指的泛称是否落入同一预测实体
        if hc_type == "generic_no_merge":
            peid_to_ch_or_ref: dict[str, set[str]] = collections.defaultdict(set)
            for hm in c_mentions:
                ekey = f"hc:{cid}:{hm['mention_id']}"
                p = matched_map.get(ekey)
                if p is not None and p.get("entity_id"):
                    peid = p["entity_id"]
                    ref_key = hm.get("refers_to") or f"null_ch{hm['chapter_ordinal']}_{hm['mention_id']}"
                    peid_to_ch_or_ref[peid].add(ref_key)
            for peid, ref_keys in peid_to_ch_or_ref.items():
                if len(ref_keys) > 1:
                    msg = f"不同回/不同所指的泛称 {sorted(ref_keys)} 落入同一预测实体 {peid}"
                    if msg not in spec2_reasons:
                        spec2_reasons.append(msg)

        passed = len(reasons) == 0
        spec2_passed = len(spec2_reasons) == 0

        if passed:
            by_cat_pass[cat] += 1
            passed_ids.append(cid)
        else:
            failed_ids.append(cid)

        if hc_type == "policy_pending":
            policy_pending_list.append({
                "case_id": cid,
                "title": case["title"],
                "grouped_as_same_in_run": spec2_passed,
                "notes": spec2_reasons,
            })
        else:
            if spec2_passed:
                spec2_type_pass[hc_type].append(cid)
            else:
                spec2_type_fail[hc_type].append(cid)

        case_results.append({
            "case_id": cid,
            "category": cat,
            "spec2_type": hc_type,
            "title": case["title"],
            "passed": passed,
            "spec2_passed": spec2_passed,
            "failure_reasons": reasons,
            "spec2_failure_reasons": spec2_reasons,
        })

    by_category = {
        cat: {
            **fraction(by_cat_pass[cat], total),
            "failed_case_ids": [
                r["case_id"] for r in case_results if r["category"] == cat and not r["passed"]
            ],
        }
        for cat, total in sorted(by_cat_total.items())
    }

    by_spec2_type = {}
    scored_pass_all: list[str] = []
    scored_fail_all: list[str] = []
    for t in ("same", "different", "generic_no_merge"):
        p_list = spec2_type_pass.get(t, [])
        f_list = spec2_type_fail.get(t, [])
        scored_pass_all.extend(p_list)
        scored_fail_all.extend(f_list)
        by_spec2_type[t] = {
            **fraction(len(p_list), len(p_list) + len(f_list)),
            "passed_case_ids": p_list,
            "failed_case_ids": f_list,
        }
    by_spec2_type["policy_pending"] = {
        "status": "listed_only_not_scored",
        "count": len(policy_pending_list),
        "cases": policy_pending_list,
    }
    by_spec2_type["scored_total"] = {
        **fraction(len(scored_pass_all), len(scored_pass_all) + len(scored_fail_all)),
        "passed_case_ids": scored_pass_all,
        "failed_case_ids": scored_fail_all,
    }

    return {
        "overall_pass_rate": fraction(len(passed_ids), len(cases)),
        "passed_case_ids": passed_ids,
        "failed_case_ids": failed_ids,
        "by_category": by_category,
        "by_spec2_type": by_spec2_type,
        "details": case_results,
    }


def _compute_spec2_metrics(
    exhaustive_gold: list[dict],
    combined_gold: list[dict],
    m_ex: dict[str, dict],
    m_all: dict[str, dict],
    m_hc: dict[str, dict],
    roster_map: dict[str, dict],
    pred_mentions: list[dict],
    pred_entities: dict[str, dict],
    hard_cases_eval: dict,
    sampled_ords: set[int],
    run_meta: dict,
) -> dict:
    """严格按 0.14a 规格二第 1—8 项口径计算全套分类指标（附分子、分母与样本 ID）。"""
    # 构造无歧义别名映射，供 infer_predicted_entity_gold_id 平票裁决
    alias_owners: dict[str, set[str]] = collections.defaultdict(set)
    for gid, gent in roster_map.items():
        for al in [gent.get("canonical_name", "")] + list(gent.get("aliases", [])):
            if al:
                alias_owners[normalize_surface(al)].add(gid)
    alias_norm_to_unique_gid = {
        k: next(iter(v)) for k, v in alias_owners.items() if len(v) == 1
    }

    def sample_id_of(g: dict) -> str:
        return g.get("mention_id") or g.get("skip_id") or g.get("_eval_key", "")

    # ------------------------------------------------------------------
    # 1. 分类发现召回：穷尽层中标为 gold_id、或 generic 且 refers_to 非空的提及，按粗类分列并合计
    # ------------------------------------------------------------------
    def compute_item1_recall(gold_list: list[dict], matched_map: dict[str, dict], include_person: bool = False) -> dict:
        target_items = [
            g for g in gold_list
            if (g.get("gold_id") is not None or (g.get("entity_type") == "generic" and g.get("refers_to") is not None))
            and g.get("entity_type") != "not_entity"
        ]
        by_4class: dict[str, dict] = {}
        classes_4 = PROPER_TYPES if include_person else NON_PERSON_4CLASS
        for c4 in classes_4:
            sub = [
                g for g in target_items
                if roster_map.get(g.get("gold_id") or g.get("refers_to"), {}).get("entity_type", g.get("entity_type")) == c4
            ]
            hits = [sample_id_of(g) for g in sub if g["_eval_key"] in matched_map]
            misses = [sample_id_of(g) for g in sub if g["_eval_key"] not in matched_map]
            by_4class[c4] = {
                **fraction(len(hits), len(sub)),
                "hit_ids": hits,
                "miss_ids": misses,
            }

        by_7class: dict[str, dict] = {}
        classes_7 = SEVEN_KINDS if include_person else NON_PERSON_7CLASS
        for c7 in classes_7:
            sub = [
                g for g in target_items
                if roster_map.get(g.get("gold_id") or g.get("refers_to"), {}).get("kind_7") == c7
            ]
            if not sub and c7 in {"concept", "unknown"}:
                continue
            hits = [sample_id_of(g) for g in sub if g["_eval_key"] in matched_map]
            misses = [sample_id_of(g) for g in sub if g["_eval_key"] not in matched_map]
            by_7class[c7] = {
                **fraction(len(hits), len(sub)),
                "hit_ids": hits,
                "miss_ids": misses,
            }

        # 单独统计 generic 且 refers_to 非空的召回
        gen_ref_sub = [g for g in target_items if g.get("entity_type") == "generic" and g.get("refers_to") is not None]
        gen_ref_hits = [sample_id_of(g) for g in gen_ref_sub if g["_eval_key"] in matched_map]
        gen_ref_misses = [sample_id_of(g) for g in gen_ref_sub if g["_eval_key"] not in matched_map]

        total_hits = [sample_id_of(g) for g in target_items if g["_eval_key"] in matched_map]
        total_misses = [sample_id_of(g) for g in target_items if g["_eval_key"] not in matched_map]
        return {
            "by_4class": by_4class,
            "by_7class": by_7class,
            "generic_with_refers_to": {
                **fraction(len(gen_ref_hits), len(gen_ref_sub)),
                "hit_ids": gen_ref_hits,
                "miss_ids": gen_ref_misses,
            },
            "total": {
                **fraction(len(total_hits), len(target_items)),
                "hit_ids": total_hits,
                "miss_ids": total_misses,
            },
        }

    item1_exhaustive = compute_item1_recall(exhaustive_gold, m_ex, include_person=False)
    item1_all_layers = compute_item1_recall(combined_gold, m_all, include_person=True)

    # ------------------------------------------------------------------
    # 预先基于全层与穷尽层构建每个预测实体的主导 gold_id
    # ------------------------------------------------------------------
    by_pred_all_gids: dict[str, list[str]] = collections.defaultdict(list)
    for g in combined_gold:
        eff_gid = g.get("gold_id") or g.get("refers_to")
        if eff_gid and g["_eval_key"] in m_all:
            peid = m_all[g["_eval_key"]].get("entity_id")
            if peid:
                by_pred_all_gids[peid].append(eff_gid)

    by_pred_ex_gids: dict[str, list[str]] = collections.defaultdict(list)
    for g in exhaustive_gold:
        eff_gid = g.get("gold_id") or g.get("refers_to")
        if eff_gid and g["_eval_key"] in m_ex:
            peid = m_ex[g["_eval_key"]].get("entity_id")
            if peid:
                by_pred_ex_gids[peid].append(eff_gid)

    dom_gid_ex: dict[str, str | None] = {}
    dom_gid_all: dict[str, str | None] = {}
    for peid, pent in pred_entities.items():
        dom_gid_all[peid] = infer_predicted_entity_gold_id(peid, by_pred_all_gids, pent, alias_norm_to_unique_gid)
        if peid in by_pred_ex_gids:
            dom_gid_ex[peid] = infer_predicted_entity_gold_id(peid, by_pred_ex_gids, pent, alias_norm_to_unique_gid)
        else:
            dom_gid_ex[peid] = dom_gid_all[peid]

    # ------------------------------------------------------------------
    # 2. 分类链接精确率：已匹配、且被系统放进多成员实体的提及中，所属实体正确的比例。
    #    实体的金标准身份取其已匹配成员中占多数的 gold_id；
    #    not_entity 进入多成员实体、refers_to=null 的泛称所在实体跨 >=2 回，都算错并计入分母。
    # ------------------------------------------------------------------
    def compute_item2_link_precision(
        gold_list: list[dict],
        matched_map: dict[str, dict],
        dom_gid_map: dict[str, str | None],
        include_person: bool = False,
    ) -> dict:
        records = []
        for g in gold_list:
            p = matched_map.get(g["_eval_key"])
            if p is None or not p.get("entity_id"):
                continue
            peid = p["entity_id"]
            pent = pred_entities.get(peid, {})
            active_cnt = pent.get("active_mention_count", len(pent.get("mention_ids", [])))
            if active_cnt < 2:
                continue

            sid = sample_id_of(g)
            eff_gid = g.get("gold_id") or g.get("refers_to")
            gt = g.get("entity_type")
            pred_c4 = pent.get("kind", "item")
            pred_c7 = pent.get("raw_kind", "object")

            if eff_gid is not None and gt != "not_entity":
                gold_ent = roster_map.get(eff_gid, {})
                c4 = gold_ent.get("entity_type", gt if gt in PROPER_TYPES else pred_c4)
                c7 = gold_ent.get("kind_7", pred_c7)
                dom = dom_gid_map.get(peid)
                is_ok = (dom == eff_gid)
                records.append({
                    "sample_id": sid,
                    "surface": g.get("text", ""),
                    "chapter_ordinal": g.get("chapter_ordinal"),
                    "class_4": c4,
                    "class_7": c7,
                    "effective_gold_id": eff_gid,
                    "dominant_gold_id": dom,
                    "predicted_entity_id": peid,
                    "predicted_entity_name": pent.get("name", ""),
                    "correct": is_ok,
                    "error_type": None if is_ok else "wrong_entity_identity",
                })
            elif gt == "not_entity":
                records.append({
                    "sample_id": sid,
                    "surface": g.get("text", ""),
                    "chapter_ordinal": g.get("chapter_ordinal"),
                    "class_4": pred_c4,
                    "class_7": pred_c7,
                    "effective_gold_id": None,
                    "dominant_gold_id": dom_gid_map.get(peid),
                    "predicted_entity_id": peid,
                    "predicted_entity_name": pent.get("name", ""),
                    "correct": False,
                    "error_type": "not_entity_in_multi_member_entity",
                })
            elif gt == "generic" and g.get("refers_to") is None:
                chs = pent.get("active_chapters", [])
                if len(chs) >= 2:
                    records.append({
                        "sample_id": sid,
                        "surface": g.get("text", ""),
                        "chapter_ordinal": g.get("chapter_ordinal"),
                        "class_4": pred_c4,
                        "class_7": pred_c7,
                        "effective_gold_id": None,
                        "dominant_gold_id": dom_gid_map.get(peid),
                        "predicted_entity_id": peid,
                        "predicted_entity_name": pent.get("name", ""),
                        "correct": False,
                        "error_type": "generic_null_cross_chapter_overmerge",
                    })

        def summarize_records(sub_recs: list[dict]) -> dict:
            ok_ids = [r["sample_id"] for r in sub_recs if r["correct"]]
            err_ids = [r["sample_id"] for r in sub_recs if not r["correct"]]
            err_ne = [r["sample_id"] for r in sub_recs if r["error_type"] == "not_entity_in_multi_member_entity"]
            err_gen = [r["sample_id"] for r in sub_recs if r["error_type"] == "generic_null_cross_chapter_overmerge"]
            err_id = [r["sample_id"] for r in sub_recs if r["error_type"] == "wrong_entity_identity"]
            return {
                **fraction(len(ok_ids), len(sub_recs)),
                "correct_ids": ok_ids,
                "wrong_ids": err_ids,
                "wrong_entity_identity_ids": err_id,
                "wrong_not_entity_ids": err_ne,
                "wrong_cross_chapter_generic_ids": err_gen,
            }

        classes_4 = PROPER_TYPES if include_person else NON_PERSON_4CLASS
        by_4class = {
            c4: summarize_records([r for r in records if r["class_4"] == c4])
            for c4 in classes_4
        }
        # 若穷尽层中有非实体/泛称被归入 person 多成员实体，也单独列出但全部计入 total
        if not include_person:
            per_recs = [r for r in records if r["class_4"] == "person"]
            if per_recs:
                by_4class["person_polluted_in_exhaustive"] = summarize_records(per_recs)

        by_7class = {}
        for c7 in SEVEN_KINDS:
            sub_7 = [r for r in records if r["class_7"] == c7]
            if sub_7 or c7 in {"place", "object", "organization", "ability"}:
                by_7class[c7] = summarize_records(sub_7)

        # 同时提供仅在 gold_id 提及（不含泛称/not_entity 惩罚项）上的纯净链接精确率作对照
        gold_only_recs = [r for r in records if r["effective_gold_id"] is not None]
        return {
            "by_4class": by_4class,
            "by_7class": by_7class,
            "gold_id_only_total": summarize_records(gold_only_recs),
            "total": summarize_records(records),
        }

    item2_exhaustive = compute_item2_link_precision(exhaustive_gold, m_ex, dom_gid_ex, include_person=False)
    item2_all_layers = compute_item2_link_precision(combined_gold, m_all, dom_gid_all, include_person=True)

    # ------------------------------------------------------------------
    # 3. 分类 B³ 精确率/召回率/F1：只在已匹配的 gold_id 提及（含 refers_to 非空的泛称）上计算
    # ------------------------------------------------------------------
    def compute_item3_b3_matched_only(gold_list: list[dict], matched_map: dict[str, dict], include_person: bool = False) -> dict:
        matched_items = []
        for g in gold_list:
            eff_gid = g.get("gold_id") or g.get("refers_to")
            if eff_gid is None or g.get("entity_type") == "not_entity":
                continue
            p = matched_map.get(g["_eval_key"])
            if p is None or not p.get("entity_id"):
                continue
            gent = roster_map.get(eff_gid, {})
            matched_items.append({
                "sample_id": sample_id_of(g),
                "gold_id": eff_gid,
                "pred_eid": p["entity_id"],
                "class_4": gent.get("entity_type", g.get("entity_type", "item")),
                "class_7": gent.get("kind_7", "object"),
            })

        def eval_b3_sub(sub: list[dict]) -> dict:
            n = len(sub)
            if n == 0:
                return {
                    "matched_mention_count": 0,
                    "gold_entity_count": 0,
                    "pred_entity_count": 0,
                    "b3_precision": None,
                    "b3_recall": None,
                    "b3_f1": None,
                    "sample_ids": [],
                }
            by_g: dict[str, list[str]] = collections.defaultdict(list)
            by_p: dict[str, list[str]] = collections.defaultdict(list)
            for it in sub:
                by_g[it["gold_id"]].append(it["pred_eid"])
                by_p[it["pred_eid"]].append(it["gold_id"])
            p_sum = 0.0
            r_sum = 0.0
            for it in sub:
                gid = it["gold_id"]
                peid = it["pred_eid"]
                p_list = by_p[peid]
                g_list = by_g[gid]
                same = sum(1 for x in p_list if x == gid)
                p_sum += same / len(p_list)
                r_sum += same / len(g_list)
            bp = p_sum / n
            br = r_sum / n
            bf1 = (2 * bp * br / (bp + br)) if (bp + br) > 0 else 0.0
            return {
                "matched_mention_count": n,
                "gold_entity_count": len(by_g),
                "pred_entity_count": len(by_p),
                "b3_precision": round(bp, 6),
                "b3_recall": round(br, 6),
                "b3_f1": round(bf1, 6),
                "sample_ids": [it["sample_id"] for it in sub],
            }

        classes_4 = PROPER_TYPES if include_person else NON_PERSON_4CLASS
        by_4 = {c4: eval_b3_sub([it for it in matched_items if it["class_4"] == c4]) for c4 in classes_4}
        by_7 = {}
        for c7 in (SEVEN_KINDS if include_person else NON_PERSON_7CLASS):
            sub_7 = [it for it in matched_items if it["class_7"] == c7]
            if sub_7 or c7 in {"place", "object", "organization", "ability"}:
                by_7[c7] = eval_b3_sub(sub_7)
        return {
            "by_4class": by_4,
            "by_7class": by_7,
            "total": eval_b3_sub(matched_items),
        }

    item3_exhaustive = compute_item3_b3_matched_only(exhaustive_gold, m_ex, include_person=False)
    item3_all_layers = compute_item3_b3_matched_only(combined_gold, m_all, include_person=True)

    # ------------------------------------------------------------------
    # 4. 泛称跨章误并：refers_to=null 的泛称中，所属预测实体跨 >=2 回的条数 / 已匹配条数
    # ------------------------------------------------------------------
    def compute_item4_generic_overmerge(gold_list: list[dict], matched_map: dict[str, dict]) -> dict:
        pure_generics = [
            g for g in gold_list
            if g.get("entity_type") == "generic" and g.get("refers_to") is None
        ]
        matched_gens = []
        unmatched_ids = []
        for g in pure_generics:
            sid = sample_id_of(g)
            p = matched_map.get(g["_eval_key"])
            if p is not None and p.get("entity_id"):
                matched_gens.append((g, p, p["entity_id"]))
            else:
                unmatched_ids.append(sid)

        # 统计每个预测实体在已匹配金标准提及中覆盖的回目集合
        peid_gold_chs: dict[str, set[int]] = collections.defaultdict(set)
        for g in gold_list:
            p = matched_map.get(g["_eval_key"])
            if p is not None and p.get("entity_id") and g.get("chapter_ordinal") is not None:
                peid_gold_chs[p["entity_id"]].add(int(g["chapter_ordinal"]))

        overmerged_all_ch_ids = []
        clean_all_ch_ids = []
        overmerged_gold_ch_ids = []
        entity_breakdown: dict[str, dict] = {}

        for g, p, peid in matched_gens:
            sid = sample_id_of(g)
            pent = pred_entities.get(peid, {})
            all_chs = pent.get("active_chapters", [])
            gold_chs = sorted(peid_gold_chs.get(peid, set()))
            if len(all_chs) >= 2:
                overmerged_all_ch_ids.append(sid)
                info = entity_breakdown.setdefault(peid, {
                    "entity_id": peid,
                    "name": pent.get("name", ""),
                    "raw_kind": pent.get("raw_kind", ""),
                    "kind_4class": pent.get("kind", ""),
                    "active_mention_count": pent.get("active_mention_count", 0),
                    "chapters_spanned": all_chs,
                    "sampled_chapters_spanned": gold_chs,
                    "matched_generic_sample_ids": [],
                })
                info["matched_generic_sample_ids"].append(sid)
            else:
                clean_all_ch_ids.append(sid)

            if len(gold_chs) >= 2:
                overmerged_gold_ch_ids.append(sid)

        overmerged_entities_list = sorted(
            entity_breakdown.values(),
            key=lambda x: (-len(x["matched_generic_sample_ids"]), -x["active_mention_count"], x["name"]),
        )
        return {
            "overmerge_rate_among_matched": fraction(len(overmerged_all_ch_ids), len(matched_gens)),
            "overmerge_rate_sampled_chapters_only": fraction(len(overmerged_gold_ch_ids), len(matched_gens)),
            "overmerge_rate_among_all_pure_generic": fraction(len(overmerged_all_ch_ids), len(pure_generics)),
            "total_pure_generic_count": len(pure_generics),
            "matched_pure_generic_count": len(matched_gens),
            "unmatched_pure_generic_count": len(unmatched_ids),
            "overmerged_sample_ids": overmerged_all_ch_ids,
            "overmerged_sampled_chapters_sample_ids": overmerged_gold_ch_ids,
            "clean_matched_sample_ids": clean_all_ch_ids,
            "unmatched_sample_ids": unmatched_ids,
            "overmerged_entities": overmerged_entities_list,
        }

    item4_exhaustive = compute_item4_generic_overmerge(exhaustive_gold, m_ex)
    item4_all_layers = compute_item4_generic_overmerge(combined_gold, m_all)

    # ------------------------------------------------------------------
    # 5. 粗类正确率：预测实体粗类属于 kind 或 alt_kinds 的比例（分实体级与提及级）
    # ------------------------------------------------------------------
    def compute_item5_coarse_type_accuracy(
        gold_list: list[dict],
        matched_map: dict[str, dict],
        dom_gid_map: dict[str, str | None],
        include_person: bool = False,
    ) -> dict:
        # 5a) 预测实体级：所有在本层匹配到至少一个有效 gold_id 的预测实体
        matched_peids: set[str] = set()
        mention_recs = []
        for g in gold_list:
            eff_gid = g.get("gold_id") or g.get("refers_to")
            if eff_gid is None or g.get("entity_type") == "not_entity":
                continue
            p = matched_map.get(g["_eval_key"])
            if p is None or not p.get("entity_id"):
                continue
            peid = p["entity_id"]
            matched_peids.add(peid)
            gent = roster_map.get(eff_gid, {})
            allowed_7 = {gent.get("kind_7", "object")} | set(gent.get("alt_kinds", []))
            pred_raw = p.get("raw_kind", "person")
            pred_c4 = p.get("kind", "person")
            gold_c4 = gent.get("entity_type", g.get("entity_type", "item"))
            mention_recs.append({
                "sample_id": sample_id_of(g),
                "gold_id": eff_gid,
                "gold_c4": gold_c4,
                "gold_c7": gent.get("kind_7", "object"),
                "allowed_7": sorted(allowed_7),
                "pred_raw": pred_raw,
                "pred_c4": pred_c4,
                "ok_kind_or_alt": pred_raw in allowed_7,
                "ok_strict_7": pred_raw == gent.get("kind_7", "object"),
                "ok_4class": pred_c4 == gold_c4,
            })

        entity_recs = []
        for peid in sorted(matched_peids):
            dom_gid = dom_gid_map.get(peid)
            if not dom_gid or dom_gid not in roster_map:
                continue
            gent = roster_map[dom_gid]
            pent = pred_entities.get(peid, {})
            allowed_7 = {gent.get("kind_7", "object")} | set(gent.get("alt_kinds", []))
            pred_raw = pent.get("raw_kind", "person")
            pred_c4 = pent.get("kind", "person")
            gold_c4 = gent.get("entity_type", "item")
            entity_recs.append({
                "entity_id": peid,
                "entity_name": pent.get("name", ""),
                "dominant_gold_id": dom_gid,
                "gold_canonical": gent.get("canonical_name", dom_gid),
                "gold_c4": gold_c4,
                "gold_c7": gent.get("kind_7", "object"),
                "allowed_7": sorted(allowed_7),
                "pred_raw": pred_raw,
                "pred_c4": pred_c4,
                "ok_kind_or_alt": pred_raw in allowed_7,
                "ok_strict_7": pred_raw == gent.get("kind_7", "object"),
                "ok_4class": pred_c4 == gold_c4,
            })

        def summarize_acc(recs: list[dict], id_key: str) -> dict:
            ok_alt = [r[id_key] for r in recs if r["ok_kind_or_alt"]]
            ng_alt = [r[id_key] for r in recs if not r["ok_kind_or_alt"]]
            ok_s7 = [r[id_key] for r in recs if r["ok_strict_7"]]
            ok_c4 = [r[id_key] for r in recs if r["ok_4class"]]
            return {
                "kind_or_alt_kinds": {
                    **fraction(len(ok_alt), len(recs)),
                    "correct_ids": ok_alt,
                    "wrong_ids": ng_alt,
                },
                "strict_7class": fraction(len(ok_s7), len(recs)),
                "mapped_4class": fraction(len(ok_c4), len(recs)),
            }

        classes_4 = PROPER_TYPES if include_person else NON_PERSON_4CLASS
        ent_by_4 = {
            c4: summarize_acc([r for r in entity_recs if r["gold_c4"] == c4], "entity_id")
            for c4 in classes_4
        }
        men_by_4 = {
            c4: summarize_acc([r for r in mention_recs if r["gold_c4"] == c4], "sample_id")
            for c4 in classes_4
        }
        ent_by_7 = {}
        men_by_7 = {}
        for c7 in (SEVEN_KINDS if include_person else NON_PERSON_7CLASS):
            e_sub = [r for r in entity_recs if r["gold_c7"] == c7]
            m_sub = [r for r in mention_recs if r["gold_c7"] == c7]
            if e_sub or m_sub or c7 in {"place", "object", "organization", "ability"}:
                ent_by_7[c7] = summarize_acc(e_sub, "entity_id")
                men_by_7[c7] = summarize_acc(m_sub, "sample_id")

        return {
            "entity_level": {
                "by_4class": ent_by_4,
                "by_7class": ent_by_7,
                "total": summarize_acc(entity_recs, "entity_id"),
                "wrong_entity_details": [r for r in entity_recs if not r["ok_kind_or_alt"]],
            },
            "mention_level": {
                "by_4class": men_by_4,
                "by_7class": men_by_7,
                "total": summarize_acc(mention_recs, "sample_id"),
            },
        }

    item5_exhaustive = compute_item5_coarse_type_accuracy(exhaustive_gold, m_ex, dom_gid_ex, include_person=False)
    item5_all_layers = compute_item5_coarse_type_accuracy(combined_gold, m_all, dom_gid_all, include_person=True)

    # ------------------------------------------------------------------
    # 7. 抽样回内未匹配任何金标准区间的预测非人物提及条数
    # ------------------------------------------------------------------
    matched_pred_ids_ex = {p["id"] for p in m_ex.values()}
    matched_pred_ids_all = {p["id"] for p in m_all.values()}
    preds_non_person_in_sampled = [
        p for p in pred_mentions
        if p.get("chapter_ordinal") in sampled_ords and p.get("raw_kind") != "person"
    ]
    unmatched_np_ex = [p for p in preds_non_person_in_sampled if p["id"] not in matched_pred_ids_ex]
    unmatched_np_all = [p for p in preds_non_person_in_sampled if p["id"] not in matched_pred_ids_all]

    by_raw_kind_unmatched = collections.Counter(p["raw_kind"] for p in unmatched_np_ex)
    by_ch_unmatched = collections.Counter(str(p["chapter_ordinal"]) for p in unmatched_np_ex)
    surf_unmatched = collections.Counter(p["text"] for p in unmatched_np_ex)

    item7_unmatched = {
        "total_pred_non_person_in_sampled_chapters": len(preds_non_person_in_sampled),
        "unmatched_against_exhaustive_count": len(unmatched_np_ex),
        "unmatched_against_all_layers_count": len(unmatched_np_all),
        "unmatched_rate": fraction(len(unmatched_np_ex), len(preds_non_person_in_sampled)),
        "by_raw_kind": {k: by_raw_kind_unmatched.get(k, 0) for k in NON_PERSON_7CLASS},
        "by_chapter": {str(ch): by_ch_unmatched.get(str(ch), 0) for ch in sorted(sampled_ords)},
        "top_unmatched_surfaces": [
            {"surface": s, "count": cnt} for s, cnt in surf_unmatched.most_common(30)
        ],
        "sample_unmatched_mention_ids": [p["id"] for p in unmatched_np_ex[:100]],
    }

    return {
        "item1_discovery_recall": {
            "exhaustive_layer": item1_exhaustive,
            "all_layers": item1_all_layers,
        },
        "item2_link_precision": {
            "exhaustive_layer": item2_exhaustive,
            "all_layers": item2_all_layers,
        },
        "item3_b3_matched_gold_id": {
            "exhaustive_layer": item3_exhaustive,
            "all_layers": item3_all_layers,
        },
        "item4_generic_cross_chapter_overmerge": {
            "exhaustive_layer": item4_exhaustive,
            "all_layers": item4_all_layers,
        },
        "item5_coarse_type_accuracy": {
            "exhaustive_layer": item5_exhaustive,
            "all_layers": item5_all_layers,
        },
        "item6_hard_cases_by_type": hard_cases_eval["by_spec2_type"],
        "item7_unmatched_pred_non_person_in_sampled_chapters": item7_unmatched,
        "item8_global_summary": {
            "entity_count_active": run_meta["entity_count_active"],
            "singleton_count": run_meta["singleton_count"],
            "entities_by_kind_7class": run_meta["entities_by_kind_7class"],
            "singletons_by_kind_7class": run_meta["singletons_by_kind_7class"],
            "entities_by_kind_4class": run_meta["entities_by_kind"],
            "singletons_by_kind_4class": run_meta["singletons_by_kind_4class"],
            "mentions_by_kind_7class": run_meta["mentions_by_kind_7class"],
            "mentions_by_kind_4class": run_meta["mentions_by_kind"],
            "review_counts": run_meta["review_counts"],
            "wall_seconds": run_meta["wall_seconds"],
            "stage_seconds": run_meta["stage_seconds"],
            "total_model_metrics": run_meta["total_model_metrics"],
        },
    }


def _extract_run_meta(result_path: Path, result_doc: dict, pred_mentions: list[dict], pred_entities: dict[str, dict]) -> dict:
    out_dir = result_path.parent
    manifest_path = out_dir / "manifest.json"
    review_path = out_dir / "review.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}

    active_entities = [e for e in pred_entities.values() if e.get("identity_status") != "not_entity_suggested"]
    singletons = [e for e in active_entities if e.get("active_mention_count", len(e.get("mention_ids", []))) == 1]

    by_kind_4 = collections.Counter(e["kind"] for e in active_entities)
    by_kind_7 = collections.Counter(e["raw_kind"] for e in active_entities)
    sing_by_kind_4 = collections.Counter(e["kind"] for e in singletons)
    sing_by_kind_7 = collections.Counter(e["raw_kind"] for e in singletons)

    mention_by_kind_4 = collections.Counter(m["kind"] for m in pred_mentions)
    mention_by_kind_7 = collections.Counter(m["raw_kind"] for m in pred_mentions)

    invocations = manifest.get("invocations", [])
    wall_seconds = round(sum(float(inv.get("wall_seconds", 0.0)) for inv in invocations), 4) if invocations else None
    stage_seconds: dict[str, float] = collections.defaultdict(float)
    for inv in invocations:
        for st_name, st_info in inv.get("stages", {}).items():
            if isinstance(st_info, dict) and isinstance(st_info.get("seconds"), (int, float)):
                stage_seconds[st_name] = round(stage_seconds[st_name] + float(st_info["seconds"]), 4)

    chapters = result_doc.get("chapters", [])
    chapters_version = chapters[0].get("chapter_set_version") if chapters else None

    return {
        "run_id": result_doc.get("run_id"),
        "result_path": str(result_path),
        "status": result_doc.get("status"),
        "chapters_version": chapters_version,
        "chapter_count": len(chapters),
        "invocation_count": len(invocations),
        "invocations": invocations,
        "wall_seconds": wall_seconds,
        "identity_seconds": manifest.get("identity_seconds"),
        "stage_seconds": dict(stage_seconds),
        "total_model_metrics": manifest.get("total_model_metrics", {}),
        "entity_count_total": len(pred_entities),
        "entity_count_active": len(active_entities),
        "singleton_count": len(singletons),
        "entities_by_kind": {k: by_kind_4.get(k, 0) for k in PROPER_TYPES},
        "singletons_by_kind_4class": {k: sing_by_kind_4.get(k, 0) for k in PROPER_TYPES},
        "entities_by_kind_7class": {k: by_kind_7.get(k, 0) for k in SEVEN_KINDS},
        "singletons_by_kind_7class": {k: sing_by_kind_7.get(k, 0) for k in SEVEN_KINDS},
        "mention_count_active": len(pred_mentions),
        "mentions_by_kind": {k: mention_by_kind_4.get(k, 0) for k in PROPER_TYPES},
        "mentions_by_kind_7class": {k: mention_by_kind_7.get(k, 0) for k in SEVEN_KINDS},
        "review_counts": review.get("counts", {}),
    }


def evaluate_single_result(
    result_path: Path,
    result_doc: dict,
    roster_doc: dict,
    mentions_doc: dict,
    hard_cases_doc: dict,
) -> dict:
    pred_mentions, pred_entities, _, _ = _extract_pred_mentions_and_entities(result_doc)
    roster_map = {e["gold_id"]: _enrich_roster_entity(e) for e in roster_doc.get("entities", [])}
    sampled_ords = {sc["ordinal"] for sc in mentions_doc.get("sampled_chapters", [])}

    exhaustive_gold: list[dict] = []
    for m in mentions_doc.get("mentions", []):
        exhaustive_gold.append({**m, "_eval_key": f"ex:{m['mention_id']}"})
    for sk in mentions_doc.get("skips", []):
        exhaustive_gold.append({**sk, "_eval_key": f"ex:{sk['skip_id']}"})

    hc_gold: list[dict] = []
    for case in hard_cases_doc.get("cases", []):
        cid = case["case_id"]
        for hm in case.get("mentions", []):
            hc_gold.append({**hm, "_eval_key": f"hc:{cid}:{hm['mention_id']}"})

    roster_rep_gold: list[dict] = []
    for ent in roster_doc.get("entities", []):
        gid = ent["gold_id"]
        etype = ent["entity_type"]
        for idx, rm in enumerate(ent.get("representative_mentions", [])):
            roster_rep_gold.append({
                "mention_id": f"RO_{gid}_{idx}",
                "chapter_ordinal": rm["chapter_ordinal"],
                "start": rm["start"],
                "end": rm["end"],
                "text": rm["text"],
                "entity_type": etype,
                "gold_id": gid,
                "refers_to": None,
                "_eval_key": f"ro:{gid}:{idx}",
            })

    combined_by_span: dict[tuple[int, int], dict] = {}
    for g in roster_rep_gold + hc_gold + exhaustive_gold:
        span_key = (int(g["start"]), int(g["end"]))
        if span_key not in combined_by_span:
            combined_by_span[span_key] = {**g, "_eval_key": f"all:{span_key[0]}_{span_key[1]}"}
    combined_gold = list(combined_by_span.values())

    rh_by_span: dict[tuple[int, int], dict] = {}
    for g in roster_rep_gold + hc_gold:
        if g["entity_type"] in PROPER_TYPES and g.get("gold_id"):
            span_key = (int(g["start"]), int(g["end"]))
            if span_key not in rh_by_span:
                rh_by_span[span_key] = {**g, "_eval_key": f"rh:{span_key[0]}_{span_key[1]}"}
    rh_positive_gold = list(rh_by_span.values())

    all_pos_gold = [
        g for g in combined_gold
        if (g.get("gold_id") is not None or (g.get("entity_type") == "generic" and g.get("refers_to") is not None))
        and g.get("entity_type") != "not_entity"
    ]

    run_meta = _extract_run_meta(result_path, result_doc, pred_mentions, pred_entities)

    def compute_under_mode(strict_only: bool) -> dict:
        m_ex = match_gold_to_pred(exhaustive_gold, pred_mentions, strict_only=strict_only)
        m_hc = match_gold_to_pred(hc_gold, pred_mentions, strict_only=strict_only)
        m_ro = match_gold_to_pred(roster_rep_gold, pred_mentions, strict_only=strict_only)
        m_all = match_gold_to_pred(combined_gold, pred_mentions, strict_only=strict_only)
        m_rh = match_gold_to_pred(rh_positive_gold, pred_mentions, strict_only=strict_only)

        per_chapter_cm = {}
        for ch_ord in sorted(sampled_ords):
            ch_sub = [g for g in exhaustive_gold if g.get("chapter_ordinal") == ch_ord]
            per_chapter_cm[str(ch_ord)] = _build_confusion_matrix(ch_sub, m_ex)

        hc_eval = _evaluate_hard_cases(hard_cases_doc.get("cases", []), m_hc, pred_entities)
        spec2 = _compute_spec2_metrics(
            exhaustive_gold,
            combined_gold,
            m_ex,
            m_all,
            m_hc,
            roster_map,
            pred_mentions,
            pred_entities,
            hc_eval,
            sampled_ords,
            run_meta,
        )

        return {
            "confusion_matrix": {
                "exhaustive_layer": _build_confusion_matrix(exhaustive_gold, m_ex),
                "hard_cases_layer": _build_confusion_matrix(hc_gold, m_hc),
                "roster_layer": _build_confusion_matrix(roster_rep_gold, m_ro),
                "combined_all_layers": _build_confusion_matrix(combined_gold, m_all),
                "exhaustive_by_chapter": per_chapter_cm,
            },
            "recall_by_type": {
                "exhaustive_layer": _compute_recall_by_type(exhaustive_gold, m_ex),
                "hard_cases_layer": _compute_recall_by_type(hc_gold, m_hc),
                "roster_layer": _compute_recall_by_type(roster_rep_gold, m_ro),
                "combined_all_layers": _compute_recall_by_type(combined_gold, m_all),
            },
            "exhaustive_precision": _compute_exhaustive_precision(
                exhaustive_gold, m_ex, pred_mentions, sampled_ords
            ),
            "pollution": _compute_pollution(
                exhaustive_gold, m_ex, pred_entities, pred_mentions
            ),
            "clustering_roster_and_hard_cases": _compute_b3_and_split_merge(
                rh_positive_gold, m_rh, roster_map, pred_entities
            ),
            "clustering_all_three_layers": _compute_b3_and_split_merge(
                all_pos_gold, m_all, roster_map, pred_entities
            ),
            "hard_cases": hc_eval,
            "spec2_metrics": spec2,
        }

    return {
        "run_meta": run_meta,
        "strict": compute_under_mode(strict_only=True),
        "overlap": compute_under_mode(strict_only=False),
    }


def _compute_co_matched_b3(
    gold_items: list[dict],
    m1_map: dict[str, dict],
    m2_map: dict[str, dict],
    roster_map: dict[str, dict],
    only_entity_bearing: bool = False,
) -> dict:
    """在两次运行共同匹配的金标准提及集合上，以第一次运行的聚类划分为参照计算 B³ P/R/F1。"""
    co_items = []
    for g in gold_items:
        eff_gid = g.get("gold_id") or g.get("refers_to")
        if only_entity_bearing and (eff_gid is None or g.get("entity_type") == "not_entity"):
            continue
        k = g["_eval_key"]
        p1 = m1_map.get(k)
        p2 = m2_map.get(k)
        if p1 is not None and p2 is not None and p1.get("entity_id") and p2.get("entity_id"):
            gent = roster_map.get(eff_gid, {}) if eff_gid else {}
            co_items.append({
                "key": k,
                "sample_id": g.get("mention_id") or g.get("skip_id") or k,
                "e1": p1["entity_id"],
                "e2": p2["entity_id"],
                "class_4": gent.get("entity_type", g.get("entity_type", "item")),
                "class_7": gent.get("kind_7", p1.get("raw_kind", "object")),
            })

    def calc_partition_b3(sub: list[dict]) -> dict:
        n = len(sub)
        if n == 0:
            return {
                "co_matched_count": 0,
                "run1_cluster_count": 0,
                "run2_cluster_count": 0,
                "b3_precision_vs_run1": None,
                "b3_recall_vs_run1": None,
                "b3_f1_vs_run1": None,
            }
        by_e1: dict[str, list[int]] = collections.defaultdict(list)
        by_e2: dict[str, list[int]] = collections.defaultdict(list)
        for idx, it in enumerate(sub):
            by_e1[it["e1"]].append(idx)
            by_e2[it["e2"]].append(idx)

        p_sum = 0.0
        r_sum = 0.0
        for idx, it in enumerate(sub):
            c1 = set(by_e1[it["e1"]])
            c2 = set(by_e2[it["e2"]])
            inter = len(c1 & c2)
            p_sum += inter / len(c2)
            r_sum += inter / len(c1)
        bp = p_sum / n
        br = r_sum / n
        bf1 = (2 * bp * br / (bp + br)) if (bp + br) > 0 else 0.0
        return {
            "co_matched_count": n,
            "run1_cluster_count": len(by_e1),
            "run2_cluster_count": len(by_e2),
            "b3_precision_vs_run1": round(bp, 6),
            "b3_recall_vs_run1": round(br, 6),
            "b3_f1_vs_run1": round(bf1, 6),
        }

    by_4 = {
        c4: calc_partition_b3([it for it in co_items if it["class_4"] == c4])
        for c4 in PROPER_TYPES
    }
    by_7 = {
        c7: calc_partition_b3([it for it in co_items if it["class_7"] == c7])
        for c7 in SEVEN_KINDS
    }
    return {
        "total": calc_partition_b3(co_items),
        "by_4class": by_4,
        "by_7class": by_7,
    }


def _diff_stat(v1: float | int | None, v2: float | int | None) -> dict:
    if v1 is None or v2 is None:
        return {
            "run1": v1,
            "run2": v2,
            "mean": None,
            "delta_run2_minus_run1": None,
            "abs_diff": None,
            "threshold_max_2x_or_2pp": None,
        }
    d = round(float(v2) - float(v1), 6)
    ad = round(abs(float(v1) - float(v2)), 6)
    mean_v = round((float(v1) + float(v2)) / 2.0, 6)
    return {
        "run1": v1,
        "run2": v2,
        "mean": mean_v,
        "delta_run2_minus_run1": d,
        "abs_diff": ad,
        "threshold_max_2x_or_2pp": round(max(2.0 * ad, 0.02), 6),
    }


def compare_multiple_runs(
    run_evals: list[dict],
    result_docs: list[dict],
    roster_doc: dict,
    mentions_doc: dict,
    hard_cases_doc: dict,
) -> dict:
    if len(run_evals) < 2:
        return {}
    r1, r2 = run_evals[0], run_evals[1]
    m1, m2 = r1["run_meta"], r2["run_meta"]
    s1 = r1["strict"]["spec2_metrics"]
    s2 = r2["strict"]["spec2_metrics"]

    pm1, pe1, _, _ = _extract_pred_mentions_and_entities(result_docs[0])
    pm2, pe2, _, _ = _extract_pred_mentions_and_entities(result_docs[1])
    roster_map = {e["gold_id"]: _enrich_roster_entity(e) for e in roster_doc.get("entities", [])}

    exhaustive_gold: list[dict] = []
    for m in mentions_doc.get("mentions", []):
        exhaustive_gold.append({**m, "_eval_key": f"ex:{m['mention_id']}"})
    for sk in mentions_doc.get("skips", []):
        exhaustive_gold.append({**sk, "_eval_key": f"ex:{sk['skip_id']}"})

    hc_gold: list[dict] = []
    for case in hard_cases_doc.get("cases", []):
        cid = case["case_id"]
        for hm in case.get("mentions", []):
            hc_gold.append({**hm, "_eval_key": f"hc:{cid}:{hm['mention_id']}"})

    roster_rep_gold: list[dict] = []
    for ent in roster_doc.get("entities", []):
        gid = ent["gold_id"]
        etype = ent["entity_type"]
        for idx, rm in enumerate(ent.get("representative_mentions", [])):
            roster_rep_gold.append({
                "chapter_ordinal": rm["chapter_ordinal"],
                "start": rm["start"],
                "end": rm["end"],
                "text": rm["text"],
                "entity_type": etype,
                "gold_id": gid,
                "refers_to": None,
                "_eval_key": f"ro:{gid}:{idx}",
            })

    combined_by_span: dict[tuple[int, int], dict] = {}
    for g in roster_rep_gold + hc_gold + exhaustive_gold:
        span_key = (int(g["start"]), int(g["end"]))
        if span_key not in combined_by_span:
            combined_by_span[span_key] = {**g, "_eval_key": f"all:{span_key[0]}_{span_key[1]}"}
    combined_gold = list(combined_by_span.values())

    m1_ex_strict = match_gold_to_pred(exhaustive_gold, pm1, strict_only=True)
    m2_ex_strict = match_gold_to_pred(exhaustive_gold, pm2, strict_only=True)
    m1_all_strict = match_gold_to_pred(combined_gold, pm1, strict_only=True)
    m2_all_strict = match_gold_to_pred(combined_gold, pm2, strict_only=True)

    m1_ex_ov = match_gold_to_pred(exhaustive_gold, pm1, strict_only=False)
    m2_ex_ov = match_gold_to_pred(exhaustive_gold, pm2, strict_only=False)

    co_b3_ex_all_strict = _compute_co_matched_b3(exhaustive_gold, m1_ex_strict, m2_ex_strict, roster_map, only_entity_bearing=False)
    co_b3_ex_entity_strict = _compute_co_matched_b3(exhaustive_gold, m1_ex_strict, m2_ex_strict, roster_map, only_entity_bearing=True)
    co_b3_all_layers_strict = _compute_co_matched_b3(combined_gold, m1_all_strict, m2_all_strict, roster_map, only_entity_bearing=True)
    co_b3_ex_all_overlap = _compute_co_matched_b3(exhaustive_gold, m1_ex_ov, m2_ex_ov, roster_map, only_entity_bearing=False)

    surfs1 = {m["text"] for m in pm1 if m["text"]}
    surfs2 = {m["text"] for m in pm2 if m["text"]}
    surf_inter = len(surfs1 & surfs2)
    surf_union = len(surfs1 | surfs2)

    spans1 = {(m["start"], m["end"]) for m in pm1}
    spans2 = {(m["start"], m["end"]) for m in pm2}
    span_inter = len(spans1 & spans2)
    span_union = len(spans1 | spans2)

    enames1 = {e.get("name", "") for e in pe1.values() if e.get("identity_status") != "not_entity_suggested" and e.get("name")}
    enames2 = {e.get("name", "") for e in pe2.values() if e.get("identity_status") != "not_entity_suggested" and e.get("name")}
    ename_inter = len(enames1 & enames2)
    ename_union = len(enames1 | enames2)

    b3_diff_rh = {}
    for k in PROPER_TYPES + ["overall"]:
        f1_1 = r1["strict"]["clustering_roster_and_hard_cases"][k]["b3_f1"]
        f1_2 = r2["strict"]["clustering_roster_and_hard_cases"][k]["b3_f1"]
        b3_diff_rh[k] = {
            "run1_b3_f1": f1_1,
            "run2_b3_f1": f1_2,
            "delta_run2_minus_run1": round(f1_2 - f1_1, 6) if (f1_1 is not None and f1_2 is not None) else None,
            "abs_diff": round(abs(f1_2 - f1_1), 6) if (f1_1 is not None and f1_2 is not None) else None,
        }

    b3_diff_all = {}
    for k in PROPER_TYPES + ["overall"]:
        f1_1 = r1["strict"]["clustering_all_three_layers"][k]["b3_f1"]
        f1_2 = r2["strict"]["clustering_all_three_layers"][k]["b3_f1"]
        b3_diff_all[k] = {
            "run1_b3_f1": f1_1,
            "run2_b3_f1": f1_2,
            "delta_run2_minus_run1": round(f1_2 - f1_1, 6) if (f1_1 is not None and f1_2 is not None) else None,
            "abs_diff": round(abs(f1_2 - f1_1), 6) if (f1_1 is not None and f1_2 is not None) else None,
        }

    # 汇总 0.14a 规格二各项核心指标的两次运行差值绝对值（noise.abs_diff）
    spec2_abs_diff = {
        "item1_discovery_recall_exhaustive": {
            k: _diff_stat(
                s1["item1_discovery_recall"]["exhaustive_layer"]["by_4class"][k]["value"] if k in s1["item1_discovery_recall"]["exhaustive_layer"]["by_4class"] else s1["item1_discovery_recall"]["exhaustive_layer"]["total"]["value"],
                s2["item1_discovery_recall"]["exhaustive_layer"]["by_4class"][k]["value"] if k in s2["item1_discovery_recall"]["exhaustive_layer"]["by_4class"] else s2["item1_discovery_recall"]["exhaustive_layer"]["total"]["value"],
            )
            for k in NON_PERSON_4CLASS + ["total"]
        },
        "item1_discovery_recall_exhaustive_7class": {
            k: _diff_stat(
                s1["item1_discovery_recall"]["exhaustive_layer"]["by_7class"].get(k, {}).get("value"),
                s2["item1_discovery_recall"]["exhaustive_layer"]["by_7class"].get(k, {}).get("value"),
            )
            for k in ("place", "organization", "object", "ability")
        },
        "item2_link_precision_exhaustive": {
            k: _diff_stat(
                s1["item2_link_precision"]["exhaustive_layer"]["by_4class"][k]["value"] if k in s1["item2_link_precision"]["exhaustive_layer"]["by_4class"] else s1["item2_link_precision"]["exhaustive_layer"]["total"]["value"],
                s2["item2_link_precision"]["exhaustive_layer"]["by_4class"][k]["value"] if k in s2["item2_link_precision"]["exhaustive_layer"]["by_4class"] else s2["item2_link_precision"]["exhaustive_layer"]["total"]["value"],
            )
            for k in NON_PERSON_4CLASS + ["total"]
        },
        "item2_link_precision_all_layers": {
            k: _diff_stat(
                s1["item2_link_precision"]["all_layers"]["by_4class"][k]["value"] if k in s1["item2_link_precision"]["all_layers"]["by_4class"] else s1["item2_link_precision"]["all_layers"]["total"]["value"],
                s2["item2_link_precision"]["all_layers"]["by_4class"][k]["value"] if k in s2["item2_link_precision"]["all_layers"]["by_4class"] else s2["item2_link_precision"]["all_layers"]["total"]["value"],
            )
            for k in PROPER_TYPES + ["total"]
        },
        "item3_b3_f1_exhaustive": {
            k: _diff_stat(
                s1["item3_b3_matched_gold_id"]["exhaustive_layer"]["by_4class"][k]["b3_f1"] if k in s1["item3_b3_matched_gold_id"]["exhaustive_layer"]["by_4class"] else s1["item3_b3_matched_gold_id"]["exhaustive_layer"]["total"]["b3_f1"],
                s2["item3_b3_matched_gold_id"]["exhaustive_layer"]["by_4class"][k]["b3_f1"] if k in s2["item3_b3_matched_gold_id"]["exhaustive_layer"]["by_4class"] else s2["item3_b3_matched_gold_id"]["exhaustive_layer"]["total"]["b3_f1"],
            )
            for k in NON_PERSON_4CLASS + ["total"]
        },
        "item3_b3_f1_all_layers": {
            k: _diff_stat(
                s1["item3_b3_matched_gold_id"]["all_layers"]["by_4class"][k]["b3_f1"] if k in s1["item3_b3_matched_gold_id"]["all_layers"]["by_4class"] else s1["item3_b3_matched_gold_id"]["all_layers"]["total"]["b3_f1"],
                s2["item3_b3_matched_gold_id"]["all_layers"]["by_4class"][k]["b3_f1"] if k in s2["item3_b3_matched_gold_id"]["all_layers"]["by_4class"] else s2["item3_b3_matched_gold_id"]["all_layers"]["total"]["b3_f1"],
            )
            for k in PROPER_TYPES + ["total"]
        },
        "item4_generic_cross_chapter_overmerge_exhaustive": _diff_stat(
            s1["item4_generic_cross_chapter_overmerge"]["exhaustive_layer"]["overmerge_rate_among_matched"]["value"],
            s2["item4_generic_cross_chapter_overmerge"]["exhaustive_layer"]["overmerge_rate_among_matched"]["value"],
        ),
        "item5_coarse_type_accuracy_entity_level": _diff_stat(
            s1["item5_coarse_type_accuracy"]["all_layers"]["entity_level"]["total"]["kind_or_alt_kinds"]["value"],
            s2["item5_coarse_type_accuracy"]["all_layers"]["entity_level"]["total"]["kind_or_alt_kinds"]["value"],
        ),
        "item5_coarse_type_accuracy_mention_level_exhaustive": _diff_stat(
            s1["item5_coarse_type_accuracy"]["exhaustive_layer"]["mention_level"]["total"]["kind_or_alt_kinds"]["value"],
            s2["item5_coarse_type_accuracy"]["exhaustive_layer"]["mention_level"]["total"]["kind_or_alt_kinds"]["value"],
        ),
        "item6_hard_cases_by_type": {
            t: _diff_stat(
                s1["item6_hard_cases_by_type"][t]["value"],
                s2["item6_hard_cases_by_type"][t]["value"],
            )
            for t in ("same", "different", "generic_no_merge", "scored_total")
        },
        "item7_unmatched_pred_non_person_in_sampled_chapters": _diff_stat(
            s1["item7_unmatched_pred_non_person_in_sampled_chapters"]["unmatched_against_exhaustive_count"],
            s2["item7_unmatched_pred_non_person_in_sampled_chapters"]["unmatched_against_exhaustive_count"],
        ),
    }

    return {
        "run1_id": m1["run_id"],
        "run2_id": m2["run_id"],
        "exhaustive_co_matched_partition_b3": {
            "strict_all_mentions": co_b3_ex_all_strict,
            "strict_entity_bearing": co_b3_ex_entity_strict,
            "overlap_all_mentions": co_b3_ex_all_overlap,
        },
        "all_layers_co_matched_partition_b3": co_b3_all_layers_strict,
        "spec2_metrics_diff": spec2_abs_diff,
        "entity_count_active": {
            "run1": m1["entity_count_active"],
            "run2": m2["entity_count_active"],
            "delta": m2["entity_count_active"] - m1["entity_count_active"],
            "abs_diff": abs(m2["entity_count_active"] - m1["entity_count_active"]),
        },
        "singleton_count": {
            "run1": m1["singleton_count"],
            "run2": m2["singleton_count"],
            "delta": m2["singleton_count"] - m1["singleton_count"],
            "abs_diff": abs(m2["singleton_count"] - m1["singleton_count"]),
        },
        "entities_by_kind_delta": {
            k: {
                "run1": m1["entities_by_kind"][k],
                "run2": m2["entities_by_kind"][k],
                "delta": m2["entities_by_kind"][k] - m1["entities_by_kind"][k],
                "abs_diff": abs(m2["entities_by_kind"][k] - m1["entities_by_kind"][k]),
            }
            for k in PROPER_TYPES
        },
        "entities_by_kind_7class_delta": {
            k: {
                "run1": m1["entities_by_kind_7class"][k],
                "run2": m2["entities_by_kind_7class"][k],
                "delta": m2["entities_by_kind_7class"][k] - m1["entities_by_kind_7class"][k],
                "abs_diff": abs(m2["entities_by_kind_7class"][k] - m1["entities_by_kind_7class"][k]),
            }
            for k in SEVEN_KINDS
        },
        "mention_count_active": {
            "run1": m1["mention_count_active"],
            "run2": m2["mention_count_active"],
            "delta": m2["mention_count_active"] - m1["mention_count_active"],
            "abs_diff": abs(m2["mention_count_active"] - m1["mention_count_active"]),
        },
        "mentions_by_kind_delta": {
            k: {
                "run1": m1["mentions_by_kind"][k],
                "run2": m2["mentions_by_kind"][k],
                "delta": m2["mentions_by_kind"][k] - m1["mentions_by_kind"][k],
                "abs_diff": abs(m2["mentions_by_kind"][k] - m1["mentions_by_kind"][k]),
            }
            for k in PROPER_TYPES
        },
        "b3_f1_diff_roster_and_hard_cases": b3_diff_rh,
        "b3_f1_diff_all_three_layers": b3_diff_all,
        "mention_surface_jaccard": fraction(surf_inter, surf_union),
        "mention_span_jaccard": fraction(span_inter, span_union),
        "entity_canonical_name_jaccard": fraction(ename_inter, ename_union),
        "status_and_retry_diff": {
            "run1_status": m1["status"],
            "run2_status": m2["status"],
            "run1_invocations": m1["invocation_count"],
            "run2_invocations": m2["invocation_count"],
            "run1_wall_seconds": m1["wall_seconds"],
            "run2_wall_seconds": m2["wall_seconds"],
            "run1_failed_attempts": m1["total_model_metrics"].get("failed_attempts", m1["total_model_metrics"].get("failed", 0)),
            "run2_failed_attempts": m2["total_model_metrics"].get("failed_attempts", m2["total_model_metrics"].get("failed", 0)),
        },
    }


def evaluate_type_gold(result_paths: list[Path], gold_dir: Path) -> dict:
    roster_doc = json.loads((gold_dir / "roster.json").read_text(encoding="utf-8"))
    mentions_doc = json.loads((gold_dir / "mentions.json").read_text(encoding="utf-8"))
    hard_cases_doc = json.loads((gold_dir / "hard_cases.json").read_text(encoding="utf-8"))

    result_docs = [json.loads(p.read_text(encoding="utf-8")) for p in result_paths]
    run_evals = [
        evaluate_single_result(p, doc, roster_doc, mentions_doc, hard_cases_doc)
        for p, doc in zip(result_paths, result_docs)
    ]

    comparison = (
        compare_multiple_runs(run_evals, result_docs, roster_doc, mentions_doc, hard_cases_doc)
        if len(run_evals) >= 2
        else None
    )
    return {
        "schema_version": 1,
        "guideline": roster_doc.get("guideline", "0.14a-T1-T5"),
        "book": roster_doc.get("book"),
        "source_sha256": roster_doc.get("source_sha256"),
        "runs": run_evals,
        "noise": comparison,
        "multi_run_comparison": comparison,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0.14a 多类型金标准分类别评测工具")
    parser.add_argument("--result", required=True, action="append", type=Path, help="result.json 路径（可多次传入对比多次运行）")
    parser.add_argument("--gold", required=True, type=Path, help="金标准目录（含 roster.json, mentions.json, hard_cases.json）")
    parser.add_argument("--output", required=False, type=Path, help="输出评测 JSON 路径")
    args = parser.parse_args(argv)

    report = evaluate_type_gold(args.result, args.gold)
    text_out = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text_out, encoding="utf-8")
    else:
        print(text_out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
