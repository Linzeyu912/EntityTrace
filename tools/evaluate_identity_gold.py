"""基于名册金标准与提及抽样标注评估身份归组准确性（支持 R3.5 严格地址匹配、难例精确率、非实体误入多成员组与 R5 待审分列）。"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entitytrace.roster import normalize_surface
from entitytrace.util import write_json


def fraction(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def match_mentions(gold_mentions: list[dict], pred_mentions: list[dict], strict_only: bool = True) -> dict[str, dict]:
    """将金标准样本（含实体样本与 not_entity 样本）与预测提及做一对一匹配；一个预测提及最多匹配一个金标准位置。"""
    candidates: list[tuple[int, int, int, int, str, dict]] = []
    for g in gold_mentions:
        if g["gold_id"] == "uncertain" or g["gold_id"] is None:
            continue
        g_start, g_end = int(g["start"]), int(g["end"])
        g_len = g_end - g_start
        if g_len <= 0:
            continue
        for p in pred_mentions:
            p_start, p_end = int(p["start"]), int(p["end"])
            overlap = min(g_end, p_end) - max(g_start, p_start)
            if overlap <= 0:
                continue
            exact = int(g_start == p_start and g_end == p_end)
            if strict_only and not exact:
                continue
            if not exact and overlap * 2 < g_len:
                continue
            len_diff = abs((p_end - p_start) - g_len)
            candidates.append((exact, overlap, -len_diff, -p_start, g["sample_id"], p))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    matched_gold: dict[str, dict] = {}
    used_pred: set[str] = set()
    for _, _, _, _, sample_id, pred in candidates:
        if sample_id in matched_gold or pred["id"] in used_pred:
            continue
        matched_gold[sample_id] = pred
        used_pred.add(pred["id"])
    return matched_gold


def infer_predicted_entity_gold_id(
    pred_eid: str,
    by_pred_all_gids: dict[str, list[str]],
    pred_entity_obj: dict | None,
    alias_norm_to_unique_gid: dict[str, str],
) -> str | None:
    """推断某预测实体对应的主导金标准实体 ID（优先依据已匹配实体层/全样本多数投票，其次依据无歧义名册别名）。"""
    gids = [g for g in by_pred_all_gids.get(pred_eid, []) if not g.startswith("__not_entity__")]
    if gids:
        counts = collections.Counter(gids)
        top_gid, _ = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return top_gid
    if pred_entity_obj:
        cand_names = [pred_entity_obj.get("name", "")] + list(pred_entity_obj.get("aliases", [])) + list(pred_entity_obj.get("names", []))
        matched_gids = [
            alias_norm_to_unique_gid[normalize_surface(s)]
            for s in cand_names
            if s and normalize_surface(s) in alias_norm_to_unique_gid
        ]
        if matched_gids:
            return collections.Counter(matched_gids).most_common(1)[0][0]
    return None


def compute_subset_metrics(
    gold_subset: list[dict],
    matched_map: dict[str, dict],
    pred_owner: dict[str, str],
    pred_entity_by_id: dict[str, dict],
    roster_entities: dict[str, dict],
    out_of_roster_entities: dict[str, dict],
    declared_major_ids: set[str],
    alias_norm_to_unique_gid: dict[str, str],
    is_half_split: bool = False,
) -> dict:
    all_entity_defs = {**roster_entities, **out_of_roster_entities}
    evaluatable = [g for g in gold_subset if g["gold_id"] not in {"not_entity", "uncertain", None}]
    not_entity_items = [g for g in gold_subset if g["gold_id"] == "not_entity"]
    uncertain_items = [g for g in gold_subset if g["gold_id"] == "uncertain"]

    matched_items = []
    for g in evaluatable:
        pred = matched_map.get(g["sample_id"])
        if pred is not None and pred["id"] in pred_owner:
            matched_items.append((g, pred, pred_owner[pred["id"]]))

    # R3.5：not_entity 样本若被并入多成员实体，单列为 not_entity_linked 并计作精确率错误
    not_entity_linked = []
    for g in not_entity_items:
        pred = matched_map.get(g["sample_id"])
        if pred is not None and pred["id"] in pred_owner:
            peid = pred_owner[pred["id"]]
            pent = pred_entity_by_id.get(peid, {})
            member_count = len(pent.get("mention_ids", []))
            if member_count >= 2:
                not_entity_linked.append({
                    "sample_id": g["sample_id"],
                    "surface": g["surface"],
                    "start": g["start"],
                    "end": g["end"],
                    "predicted_entity_id": peid,
                    "predicted_entity_name": pent.get("name", ""),
                    "predicted_entity_size": member_count,
                })

    discovery_recall = fraction(len(matched_items), len(evaluatable))
    if not matched_items and not not_entity_linked:
        return {
            "sample_total": len(gold_subset),
            "evaluatable_mentions": len(evaluatable),
            "not_entity_mentions": len(not_entity_items),
            "not_entity_linked_count": 0,
            "not_entity_linked": [],
            "uncertain_mentions": len(uncertain_items),
            "matched_mentions": 0,
            "unmatched_mentions": len(evaluatable),
            "discovery_recall_sample": discovery_recall,
            "b3_precision": None,
            "b3_recall": None,
            "b3_f1": None,
            "hard_stratum_precision": {"status": "not_applicable", "numerator": 0, "denominator": 0, "value": None},
            "major_entity_coverage": {"status": "not_applicable", "value": None, "major_entity_count": 0, "entities": []},
            "splits_per_entity": {},
            "overmerges": [],
            "major_entity_overmerges": [],
            "out_of_roster_in_major_entity_count": 0,
            "out_of_roster_in_major_details": [],
        }

    by_gold: dict[str, list[str]] = collections.defaultdict(list)
    by_pred: dict[str, list[str]] = collections.defaultdict(list)
    for g, _, pred_eid in matched_items:
        gid = g["gold_id"]
        by_gold[gid].append(pred_eid)
        by_pred[pred_eid].append(gid)

    for ne in not_entity_linked:
        by_pred[ne["predicted_entity_id"]].append(f"__not_entity__{ne['sample_id']}")

    prec_sum = 0.0
    rec_sum = 0.0
    for g, _, pred_eid in matched_items:
        gid = g["gold_id"]
        pred_cluster = by_pred[pred_eid]
        gold_cluster = by_gold[gid]
        same_both = sum(1 for x in pred_cluster if x == gid)
        prec_sum += same_both / len(pred_cluster)
        rec_sum += same_both / len(gold_cluster)

    n_matched = len(matched_items)
    n_prec_denom = n_matched + len(not_entity_linked)
    b3_p = (prec_sum / n_prec_denom) if n_prec_denom > 0 else 0.0
    b3_r = (rec_sum / n_matched) if n_matched > 0 else 0.0
    b3_f1 = (2 * b3_p * b3_r / (b3_p + b3_r)) if (b3_p + b3_r) > 0 else 0.0

    # R3.2 / R3.5：主要实体集合与覆盖率
    major_gold_ids = set(declared_major_ids) if declared_major_ids else {gid for gid, peids in by_gold.items() if len(peids) >= 10}
    min_matched_for_major = 3 if is_half_split else (4 if len(gold_subset) < 250 else 6)

    major_details = []
    splits_per_entity = {}
    for gid, pred_eids in sorted(by_gold.items()):
        counts = collections.Counter(pred_eids)
        splits_per_entity[gid] = {
            "canonical": all_entity_defs.get(gid, {}).get("canonical", gid),
            "matched_count": len(pred_eids),
            "predicted_group_count": len(counts),
            "dominant_group_count": counts.most_common(1)[0][1],
            "is_major": gid in major_gold_ids,
        }
        if gid in major_gold_ids and len(pred_eids) >= (1 if is_half_split else 1):
            dom_eid, dom_cnt = counts.most_common(1)[0]
            cov = dom_cnt / len(pred_eids)
            major_details.append({
                "gold_id": gid,
                "canonical": all_entity_defs.get(gid, {}).get("canonical", gid),
                "matched_count": len(pred_eids),
                "dominant_entity_id": dom_eid,
                "dominant_count": dom_cnt,
                "predicted_group_count": len(counts),
                "coverage": round(cov, 6),
            })

    sufficient_major = [m for m in major_details if m["matched_count"] >= min_matched_for_major]
    if not major_gold_ids or not sufficient_major:
        major_cov_block = {
            "status": "not_applicable",
            "value": None,
            "major_entity_count": len(major_details),
            "entities": major_details,
        }
    else:
        avg_major_cov = round(sum(item["coverage"] for item in major_details) / len(major_details), 6)
        major_cov_block = {
            "status": "ok",
            "value": avg_major_cov,
            "major_entity_count": len(major_details),
            "entities": major_details,
        }

    # R3.5：难例层精确率（hard_stratum_precision：难例中被链接进多成员实体的样本，所属实体正确的比例）
    hard_linked_total = 0
    hard_linked_correct = 0
    hard_b3_prec_sum = 0.0
    for g, _, pred_eid in matched_items:
        if g.get("stratum") != "hard":
            continue
        pent = pred_entity_by_id.get(pred_eid, {})
        if len(pent.get("mention_ids", [])) < 2:
            continue
        hard_linked_total += 1
        dom_gid = infer_predicted_entity_gold_id(pred_eid, by_pred, pent, alias_norm_to_unique_gid)
        pred_cluster = by_pred[pred_eid]
        same_both = sum(1 for x in pred_cluster if x == g["gold_id"])
        hard_b3_prec_sum += same_both / len(pred_cluster)
        if dom_gid == g["gold_id"]:
            hard_linked_correct += 1

    for ne in not_entity_linked:
        sample_obj = next((g for g in not_entity_items if g["sample_id"] == ne["sample_id"]), None)
        if sample_obj and sample_obj.get("stratum") == "hard":
            hard_linked_total += 1

    if hard_linked_total == 0:
        hard_prec_block = {
            "status": "not_applicable",
            "numerator": 0,
            "denominator": 0,
            "value": None,
            "b3_precision_on_linked": None,
        }
    else:
        hard_prec_block = {
            "status": "ok",
            "numerator": hard_linked_correct,
            "denominator": hard_linked_total,
            "value": round(hard_linked_correct / hard_linked_total, 6),
            "b3_precision_on_linked": round(hard_b3_prec_sum / hard_linked_total, 6),
        }

    # overmerges、major_entity_overmerges 与名册外人物并入主要实体计数
    overmerges = []
    major_overmerges = []
    oor_in_major_count = 0
    oor_in_major_details = []

    for pred_eid, gids in sorted(by_pred.items()):
        real_gids = [g for g in gids if not g.startswith("__not_entity__")]
        distinct = sorted(set(real_gids))
        if len(distinct) > 1:
            counts = collections.Counter(real_gids)
            entry = {
                "predicted_entity_id": pred_eid,
                "predicted_entity_name": pred_entity_by_id.get(pred_eid, {}).get("name", ""),
                "gold_entities": [
                    {
                        "gold_id": gid,
                        "canonical": all_entity_defs.get(gid, {}).get("canonical", gid),
                        "count_in_group": counts[gid],
                        "is_major": gid in major_gold_ids,
                        "is_out_of_roster": gid.startswith("x"),
                    }
                    for gid in distinct
                ],
            }
            overmerges.append(entry)
            if sum(1 for gid in distinct if gid in major_gold_ids) >= 2:
                major_overmerges.append(entry)
            if any(gid in major_gold_ids for gid in distinct):
                for gid in distinct:
                    if gid.startswith("x"):
                        c_x = counts[gid]
                        oor_in_major_count += c_x
                        oor_in_major_details.append({
                            "predicted_entity_id": pred_eid,
                            "predicted_entity_name": pred_entity_by_id.get(pred_eid, {}).get("name", ""),
                            "out_of_roster_gold_id": gid,
                            "out_of_roster_canonical": all_entity_defs.get(gid, {}).get("canonical", gid),
                            "count": c_x,
                        })

    return {
        "sample_total": len(gold_subset),
        "evaluatable_mentions": len(evaluatable),
        "not_entity_mentions": len(not_entity_items),
        "not_entity_linked_count": len(not_entity_linked),
        "not_entity_linked": not_entity_linked,
        "uncertain_mentions": len(uncertain_items),
        "matched_mentions": n_matched,
        "unmatched_mentions": len(evaluatable) - n_matched,
        "discovery_recall_sample": discovery_recall,
        "b3_precision": round(b3_p, 6),
        "b3_recall": round(b3_r, 6),
        "b3_f1": round(b3_f1, 6),
        "hard_stratum_precision": hard_prec_block,
        "major_entity_coverage": major_cov_block,
        "splits_per_entity": splits_per_entity,
        "overmerges": overmerges,
        "major_entity_overmerges": major_overmerges,
        "out_of_roster_in_major_entity_count": oor_in_major_count,
        "out_of_roster_in_major_details": oor_in_major_details,
    }


def extract_identity_stage_metrics(manifest: dict | None, review_doc: dict | None) -> dict:
    review_items = review_doc.get("items", []) if isinstance(review_doc, dict) else []
    by_stage = collections.Counter(item.get("stage", "unknown") for item in review_items)
    grounding_reviews = by_stage.get("grounding", 0) + by_stage.get("discovery", 0)
    identity_items = [item for item in review_items if item.get("stage", "unknown") not in {"grounding", "discovery"}]
    identity_reviews = len(identity_items)

    # R5：同时报告去重疑点数和受影响提及数
    dedup_keys: set[str] = set()
    affected_mentions_sum = 0
    for item in identity_items:
        key = str(
            item.get("id")
            or (item.get("stage"), item.get("reason"), item.get("surface"), item.get("name"), item.get("left_name"), item.get("right_name"))
        )
        dedup_keys.add(key)
        imp = item.get("impact")
        affected_mentions_sum += int(imp) if isinstance(imp, int) and imp > 0 else 1

    if not isinstance(manifest, dict):
        return {
            "total_review_items": len(review_items) if review_doc else None,
            "identity_review_items": identity_reviews if review_doc else None,
            "deduplicated_identity_issues": len(dedup_keys) if review_doc else None,
            "identity_review_affected_mentions": affected_mentions_sum if review_doc else None,
            "grounding_discovery_review_items": grounding_reviews if review_doc else None,
            "review_by_stage": dict(by_stage),
            "identity_seconds": None,
            "identity_network_attempts": None,
            "identity_successful_requests": None,
            "identity_failed_attempts": None,
            "identity_cache_hits": None,
        }

    invocations = manifest.get("invocations", [])
    identity_seconds = 0.0
    has_identity_seconds = False
    for inv in invocations:
        st = inv.get("stages", {}).get("identity")
        if isinstance(st, dict) and not st.get("resumed", False) and isinstance(st.get("seconds"), (int, float)):
            identity_seconds += float(st["seconds"])
            has_identity_seconds = True
    if not has_identity_seconds and isinstance(manifest.get("identity_seconds"), (int, float)):
        identity_seconds = float(manifest["identity_seconds"])
        has_identity_seconds = True

    total_metrics = manifest.get("total_model_metrics", {})
    by_stage_metrics = total_metrics.get("by_stage", {})
    id_attempts = 0
    id_success = 0
    id_failed = 0
    id_cached = 0
    for st_name, st_val in by_stage_metrics.items():
        if st_name in {"chapters", "chapter-verify", "chapter-title", "discovery"}:
            continue
        id_attempts += st_val.get("attempts", 0)
        id_success += st_val.get("success", 0)
        id_failed += st_val.get("failed", 0)
        id_cached += st_val.get("cached", 0)

    total_reviews = len(review_items) if review_doc is not None else manifest.get("counts", {}).get("review_items")
    return {
        "total_review_items": total_reviews,
        "identity_review_items": identity_reviews if review_doc is not None else total_reviews,
        "deduplicated_identity_issues": len(dedup_keys) if review_doc is not None else total_reviews,
        "identity_review_affected_mentions": affected_mentions_sum if review_doc is not None else total_reviews,
        "grounding_discovery_review_items": grounding_reviews if review_doc is not None else 0,
        "review_by_stage": dict(by_stage),
        "identity_seconds": round(identity_seconds, 4) if has_identity_seconds else None,
        "identity_network_attempts": id_attempts,
        "identity_successful_requests": id_success,
        "identity_failed_attempts": id_failed,
        "identity_cache_hits": id_cached,
    }


def evaluate_result(
    result_doc: dict,
    roster_doc: dict,
    mentions_doc: dict,
    manifest_doc: dict | None = None,
    review_doc: dict | None = None,
) -> dict:
    if result_doc.get("source", {}).get("text_sha256") and roster_doc.get("source_sha256"):
        if result_doc["source"]["text_sha256"] != roster_doc["source_sha256"]:
            raise ValueError("result.json 原文指纹与金标准名册不匹配")

    pred_mentions = []
    pred_owner: dict[str, str] = {}
    pred_entity_by_id: dict[str, dict] = {}
    for entity in result_doc.get("entities", []):
        pred_entity_by_id[entity["id"]] = entity
        for mid in entity.get("mention_ids", []):
            pred_owner[mid] = entity["id"]
    for m in result_doc.get("mentions", []):
        if m.get("identity_status") == "not_entity_suggested" or m.get("link_method") == "not_entity_suggested":
            continue
        if m.get("entity_id") and m["id"] not in pred_owner:
            pred_owner[m["id"]] = m["entity_id"]
        pred_mentions.append(m)

    roster_entities = {e["gold_id"]: e for e in roster_doc.get("entities", [])}
    out_of_roster_entities = {
        e["gold_id"]: e
        for e in mentions_doc.get("out_of_roster_entities", roster_doc.get("out_of_roster_entities", []))
        if isinstance(e, dict) and "gold_id" in e
    }
    declared_major_ids = set(mentions_doc.get("major_entity_ids", roster_doc.get("major_entity_ids", [])))

    # 建立无歧义名册别名 -> gold_id 映射（用于单样本预测簇归属核对）
    ambig_norms = {normalize_surface(x["surface"]) for x in roster_doc.get("ambiguous_surfaces", [])}
    alias_owners: dict[str, set[str]] = collections.defaultdict(set)
    for gid, ent in roster_entities.items():
        for al in ent.get("aliases", []):
            ns = normalize_surface(al)
            if ns not in ambig_norms:
                alias_owners[ns].add(gid)
    alias_norm_to_unique_gid = {ns: next(iter(gids)) for ns, gids in alias_owners.items() if len(gids) == 1}

    gold_mentions = mentions_doc.get("mentions", [])

    # R3.5：主指标采用严格地址匹配（起止完全一致），重叠匹配（>=50%）单列另报
    matched_strict = match_mentions(gold_mentions, pred_mentions, strict_only=True)
    matched_overlap = match_mentions(gold_mentions, pred_mentions, strict_only=False)

    strict_metrics = compute_subset_metrics(
        gold_mentions,
        matched_strict,
        pred_owner,
        pred_entity_by_id,
        roster_entities,
        out_of_roster_entities,
        declared_major_ids,
        alias_norm_to_unique_gid,
        is_half_split=False,
    )
    overlap_metrics = compute_subset_metrics(
        gold_mentions,
        matched_overlap,
        pred_owner,
        pred_entity_by_id,
        roster_entities,
        out_of_roster_entities,
        declared_major_ids,
        alias_norm_to_unique_gid,
        is_half_split=False,
    )

    book_id = roster_doc.get("book_id", "")
    halves = {}
    if book_id in {"pride-prejudice", "jiuming-qiyuan"} or len(result_doc.get("chapters", [])) >= 20:
        split_ch = 30 if book_id == "pride-prejudice" or len(result_doc.get("chapters", [])) > 40 else 18
        first_half = [g for g in gold_mentions if g.get("chapter_index", 1) <= split_ch]
        second_half = [g for g in gold_mentions if g.get("chapter_index", 1) > split_ch]
        halves = {
            "split_chapter_inclusive": split_ch,
            "first_half": compute_subset_metrics(
                first_half,
                matched_strict,
                pred_owner,
                pred_entity_by_id,
                roster_entities,
                out_of_roster_entities,
                declared_major_ids,
                alias_norm_to_unique_gid,
                is_half_split=True,
            ),
            "second_half": compute_subset_metrics(
                second_half,
                matched_strict,
                pred_owner,
                pred_entity_by_id,
                roster_entities,
                out_of_roster_entities,
                declared_major_ids,
                alias_norm_to_unique_gid,
                is_half_split=True,
            ),
        }

    entities = result_doc.get("entities", [])
    singletons = sum(1 for e in entities if len(e.get("mention_ids", [])) == 1)
    stage_info = extract_identity_stage_metrics(manifest_doc, review_doc)

    return {
        "book": roster_doc.get("book"),
        "book_id": book_id,
        "run_id": result_doc.get("run_id"),
        "schema_version": result_doc.get("schema_version"),
        "source_sha256": roster_doc.get("source_sha256"),
        "gold_metadata": {
            "labeler": mentions_doc.get("labeler", roster_doc.get("labeler")),
            "human_review": mentions_doc.get("human_review", roster_doc.get("human_review", "pending")),
            "sampling_version": mentions_doc.get("sampling_version", "v0"),
            "roster_entities": len(roster_entities),
            "out_of_roster_entities": len(out_of_roster_entities),
            "major_entities": len(declared_major_ids),
            "sampled_mentions": len(gold_mentions),
            "annotation_seconds": mentions_doc.get("annotation_seconds"),
        },
        "global_counts": {
            "predicted_mentions": len(result_doc.get("mentions", [])),
            "active_mentions": len(pred_mentions),
            "entities": len(entities),
            "singletons": singletons,
            "largest_group": max((len(e.get("mention_ids", [])) for e in entities), default=0),
            **stage_info,
        },
        "primary_metrics": strict_metrics,
        "overlap_match_metrics": {
            "matched_mentions": overlap_metrics["matched_mentions"],
            "discovery_recall_sample": overlap_metrics["discovery_recall_sample"],
            "b3_precision": overlap_metrics["b3_precision"],
            "b3_recall": overlap_metrics["b3_recall"],
            "b3_f1": overlap_metrics["b3_f1"],
            "hard_stratum_precision": overlap_metrics["hard_stratum_precision"]["value"],
            "major_entity_coverage": overlap_metrics["major_entity_coverage"]["value"],
            "not_entity_linked_count": overlap_metrics["not_entity_linked_count"],
        },
        "halves": halves,
    }


def evaluate_paths(result_path: Path, gold_dir: Path, output_path: Path | None = None, mentions_filename: str = "mentions.json") -> dict:
    ref_dir = gold_dir / "reference" if (gold_dir / "reference").exists() else gold_dir
    roster_doc = json.loads((ref_dir / "roster.json").read_text(encoding="utf-8"))
    mentions_doc = json.loads((ref_dir / mentions_filename).read_text(encoding="utf-8"))
    result_doc = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.parent / "manifest.json"
    review_path = result_path.parent / "review.json"
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    review_doc = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None

    report = evaluate_result(result_doc, roster_doc, mentions_doc, manifest_doc, review_doc)
    report["artifact_hashes"] = {
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "roster_sha256": hashlib.sha256((ref_dir / "roster.json").read_bytes()).hexdigest(),
        "mentions_sha256": hashlib.sha256((ref_dir / mentions_filename).read_bytes()).hexdigest(),
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--mentions-file", type=str, default="mentions.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_paths(args.result, args.gold, args.output, mentions_filename=args.mentions_file)
    summary = {
        "book_id": report["book_id"],
        "run_id": report["run_id"],
        "discovery_recall_sample": report["primary_metrics"]["discovery_recall_sample"]["value"],
        "b3_precision": report["primary_metrics"]["b3_precision"],
        "b3_recall": report["primary_metrics"]["b3_recall"],
        "b3_f1": report["primary_metrics"]["b3_f1"],
        "hard_stratum_precision": report["primary_metrics"]["hard_stratum_precision"]["value"],
        "major_entity_coverage": report["primary_metrics"]["major_entity_coverage"]["value"],
        "major_entity_overmerges": len(report["primary_metrics"]["major_entity_overmerges"]),
        "out_of_roster_in_major": report["primary_metrics"]["out_of_roster_in_major_entity_count"],
        "not_entity_linked": report["primary_metrics"]["not_entity_linked_count"],
        "entities": report["global_counts"]["entities"],
        "total_review_items": report["global_counts"]["total_review_items"],
        "identity_review_items": report["global_counts"]["identity_review_items"],
        "identity_seconds": report["global_counts"]["identity_seconds"],
        "identity_network_attempts": report["global_counts"]["identity_network_attempts"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
