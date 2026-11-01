"""阶段 G：叠加式实体名册更正与撤回（零模型调用、可撤回、零更正逐字节不变）。"""
from __future__ import annotations

import collections
from pathlib import Path

from .discovery import KINDS
from .roster import VERSION as ROSTER_VERSION
from .storage import Store
from .util import digest, utc_now

ORDERED_KINDS = sorted(KINDS)
VALID_OPS = {"merge", "split-surface", "assign", "not-entity", "revoke"}
VALID_SOURCES = {"user", "agent"}


class CorrectionConflictError(ValueError):
    """期望的 result_revision 与当前实际版本不一致（错误类别 conflict）。"""

    et_category = "conflict"


def _choose_display_name(surfaces: collections.Counter[str]) -> str:
    if not surfaces:
        return ""
    return sorted(surfaces, key=lambda s: (-surfaces[s], -len(s), s))[0]


def _majority_provisional_kind(mentions: list[dict]) -> str:
    counts: collections.Counter[str] = collections.Counter()
    for m in mentions:
        for k in m.get("provisional_kinds", []):
            if k in KINDS:
                counts[k] += 1
    if not counts:
        return "unknown"
    return min(counts, key=lambda k: (-counts[k], ORDERED_KINDS.index(k)))


def _recompute_entity(
    eid: str,
    mids: list[str],
    mentions_by_id: dict[str, dict],
    chapter_order: dict[str, int],
    aliases: list[str],
    kind: str,
    identity_status: str,
) -> dict:
    ordered = sorted(
        dict.fromkeys(mids),
        key=lambda mid: (mentions_by_id[mid]["start"], mentions_by_id[mid]["end"], mid),
    )
    surf_counts = collections.Counter(mentions_by_id[mid]["text"] for mid in ordered)
    first_m = mentions_by_id[ordered[0]]
    first_ch = min(
        (mentions_by_id[mid]["chapter_id"] for mid in ordered),
        key=lambda cid: chapter_order.get(cid, 999999),
    )
    methods_count = collections.Counter(mentions_by_id[mid]["link_method"] for mid in ordered)
    return {
        "id": eid,
        "name": _choose_display_name(surf_counts),
        "names": sorted(surf_counts),
        "aliases": sorted(set(aliases)),
        "mention_ids": ordered,
        "mention_count": len(ordered),
        "start": first_m["start"],
        "first_chapter_id": first_ch,
        "kind": kind,
        "identity_status": identity_status,
        "link_methods": dict(methods_count),
    }


def partition_active_corrections(records: list[dict]) -> tuple[list[dict], list[dict], set[str], dict[str, str]]:
    revoked_by: dict[str, str] = {}
    for rec in records:
        if rec.get("op") == "revoke":
            for target_id in rec.get("args", {}).get("revoked_ids", []):
                revoked_by.setdefault(target_id, rec["id"])
    revoked_set = set(revoked_by)
    active_user = [
        rec for rec in records
        if rec.get("op") != "revoke" and rec.get("source") == "user" and rec["id"] not in revoked_set
    ]
    active_agent = [
        rec for rec in records
        if rec.get("op") != "revoke" and rec.get("source") == "agent" and rec["id"] not in revoked_set
    ]
    return active_user, active_agent, revoked_set, revoked_by


def _init_replay_state(roster_out: dict, raw_reviews: list[dict]) -> tuple[dict[str, dict], dict[str, dict], list[dict], list[dict]]:
    # 未受影响的实体保留原字典副本，确保未被触碰的实体字典逐字节不变
    entities_by_id = {e["id"]: dict(e) for e in roster_out["entities"]}
    mentions_by_id = {m["id"]: dict(m) for m in roster_out["mentions"]}
    identity_reviews = [dict(r) for r in roster_out["reviews"] if str(r.get("stage", "")).startswith("roster")]
    non_identity_reviews = [dict(r) for r in raw_reviews if not str(r.get("stage", "")).startswith("roster")]
    return entities_by_id, mentions_by_id, identity_reviews, non_identity_reviews


def _apply_single_op_in_place(
    corr: dict,
    entities_by_id: dict[str, dict],
    mentions_by_id: dict[str, dict],
    chapter_order: dict[str, int],
    identity_reviews: list[dict],
    non_identity_reviews: list[dict],
) -> None:
    op = corr["op"]
    args = corr["args"]
    cid = corr["id"]
    resolves = corr.get("resolves")

    target_review = None
    if resolves is not None:
        for r in identity_reviews:
            if r["id"] == resolves:
                target_review = r
                break
        if target_review is None:
            for r in non_identity_reviews:
                if r["id"] == resolves:
                    target_review = r
                    break
        if target_review is None:
            raise ValueError(f"待审项不存在：{resolves}")

    if op == "merge":
        target_id = args["target_entity_id"]
        merged_id = args["merged_entity_id"]
        if target_id == merged_id:
            raise ValueError("合并的目标实体与被并实体不能相同")
        if target_id not in entities_by_id:
            raise ValueError(f"目标实体不存在：{target_id}")
        if merged_id not in entities_by_id:
            raise ValueError(f"被并实体不存在：{merged_id}")
        ent_a = entities_by_id[target_id]
        ent_b = entities_by_id.pop(merged_id)
        for mid in ent_b["mention_ids"]:
            m = dict(mentions_by_id[mid])
            m["entity_id"] = target_id
            m["link_method"] = "user_correction"
            m["link_refs"] = [*m.get("link_refs", []), cid]
            m.pop("identity_status", None)
            mentions_by_id[mid] = m
        merged_mids = [*ent_a["mention_ids"], *ent_b["mention_ids"]]
        merged_aliases = sorted(set(ent_a.get("aliases", [])) | set(ent_b.get("aliases", [])))
        entities_by_id[target_id] = _recompute_entity(
            target_id, merged_mids, mentions_by_id, chapter_order, merged_aliases, ent_a["kind"], "user_corrected"
        )

    elif op == "split-surface":
        entity_id = args["entity_id"]
        surface = args["surface"]
        if entity_id not in entities_by_id:
            raise ValueError(f"待拆分实体不存在：{entity_id}")
        ent = entities_by_id[entity_id]
        moved_mids = [mid for mid in ent["mention_ids"] if mentions_by_id[mid]["text"] == surface]
        kept_mids = [mid for mid in ent["mention_ids"] if mentions_by_id[mid]["text"] != surface]
        if not moved_mids:
            raise ValueError(f"实体 {entity_id} 中不存在称呼：{surface}")
        if not kept_mids:
            raise ValueError(f"实体 {entity_id} 的全部提及均为称呼“{surface}”，拆了等于没拆")
        new_id = digest(["correction-split", cid])
        for mid in moved_mids:
            m = dict(mentions_by_id[mid])
            m["entity_id"] = new_id
            m["link_method"] = "user_correction"
            m["link_refs"] = [*m.get("link_refs", []), cid]
            m.pop("identity_status", None)
            mentions_by_id[mid] = m
        kept_aliases = [a for a in ent.get("aliases", []) if a != surface]
        entities_by_id[entity_id] = _recompute_entity(
            entity_id, kept_mids, mentions_by_id, chapter_order, kept_aliases, ent["kind"], "user_corrected"
        )
        new_kind = _majority_provisional_kind([mentions_by_id[mid] for mid in moved_mids])
        entities_by_id[new_id] = _recompute_entity(
            new_id, moved_mids, mentions_by_id, chapter_order, [surface], new_kind, "user_corrected"
        )
        for mid in kept_mids:
            km = mentions_by_id[mid]
            if km.get("link_method") == "c3_model":
                identity_reviews.append({
                    "id": digest([cid, mid]),
                    "stage": "correction-c3-affected",
                    "reason": f"实体拆分称呼“{surface}”后，原经 C3 模型链接留在该实体的提及建议复核",
                    "correction_id": cid,
                    "entity_id": entity_id,
                    "split_entity_id": new_id,
                    "mention_id": mid,
                    "surface": km["text"],
                    "impact": 1,
                    "evidence": [km["address"]],
                    "options": [entity_id, new_id, "new_entity", "not_entity"],
                    "status": "pending",
                })

    elif op == "assign":
        mid = args["mention_id"]
        target = args["target"]
        if mid not in mentions_by_id:
            raise ValueError(f"提及不存在：{mid}")
        old_m = mentions_by_id[mid]
        old_eid = old_m["entity_id"]
        if target != "new":
            if target not in entities_by_id:
                raise ValueError(f"目标实体不存在：{target}")
            if old_eid == target:
                raise ValueError(f"提及 {mid} 已属于目标实体 {target}")
            dest_eid = target
        else:
            dest_eid = digest(["correction-new", cid])
        old_ent = entities_by_id[old_eid]
        rem_mids = [x for x in old_ent["mention_ids"] if x != mid]
        m = dict(old_m)
        m["entity_id"] = dest_eid
        m["link_method"] = "user_correction"
        m["link_refs"] = [*m.get("link_refs", []), cid]
        m.pop("identity_status", None)
        mentions_by_id[mid] = m
        if not rem_mids:
            entities_by_id.pop(old_eid)
        else:
            rem_names = {mentions_by_id[x]["text"] for x in rem_mids}
            rem_aliases = [a for a in old_ent.get("aliases", []) if a in rem_names]
            entities_by_id[old_eid] = _recompute_entity(
                old_eid, rem_mids, mentions_by_id, chapter_order, rem_aliases, old_ent["kind"], "user_corrected"
            )
        if target != "new":
            dest_ent = entities_by_id[dest_eid]
            dest_mids = [*dest_ent["mention_ids"], mid]
            entities_by_id[dest_eid] = _recompute_entity(
                dest_eid, dest_mids, mentions_by_id, chapter_order, dest_ent.get("aliases", []), dest_ent["kind"], "user_corrected"
            )
        else:
            new_kind = _majority_provisional_kind([m])
            entities_by_id[dest_eid] = _recompute_entity(
                dest_eid, [mid], mentions_by_id, chapter_order, [m["text"]], new_kind, "user_corrected"
            )

    elif op == "not-entity":
        mid = args["mention_id"]
        if mid not in mentions_by_id:
            raise ValueError(f"提及不存在：{mid}")
        old_m = mentions_by_id[mid]
        if old_m.get("identity_status") == "not_entity_confirmed":
            raise ValueError(f"提及 {mid} 已确认为非实体")
        old_eid = old_m["entity_id"]
        ne_id = digest(["correction-not-entity", mid])
        old_ent = entities_by_id[old_eid]
        rem_mids = [x for x in old_ent["mention_ids"] if x != mid]
        m = dict(old_m)
        m["entity_id"] = ne_id
        m["link_method"] = "user_correction"
        m["link_refs"] = [*m.get("link_refs", []), cid]
        m["identity_status"] = "not_entity_confirmed"
        mentions_by_id[mid] = m
        if not rem_mids:
            entities_by_id.pop(old_eid)
        else:
            rem_names = {mentions_by_id[x]["text"] for x in rem_mids}
            rem_aliases = [a for a in old_ent.get("aliases", []) if a in rem_names]
            entities_by_id[old_eid] = _recompute_entity(
                old_eid, rem_mids, mentions_by_id, chapter_order, rem_aliases, old_ent["kind"], "user_corrected"
            )
        entities_by_id[ne_id] = _recompute_entity(
            ne_id, [mid], mentions_by_id, chapter_order, [], "unknown", "not_entity_confirmed"
        )
    else:
        raise ValueError(f"不支持的更正操作：{op}")

    if target_review is not None:
        target_review["status"] = "resolved"
        target_review["resolved_by"] = cid


def replay_user_corrections(
    roster_out: dict,
    raw_reviews: list[dict],
    chapters: list[dict],
    active_user_corrections: list[dict],
    algorithm: str = ROSTER_VERSION,
) -> dict:
    chapter_order = {c["id"]: idx for idx, c in enumerate(chapters)}
    entities_by_id, mentions_by_id, identity_reviews, non_identity_reviews = _init_replay_state(roster_out, raw_reviews)
    for corr in active_user_corrections:
        _apply_single_op_in_place(corr, entities_by_id, mentions_by_id, chapter_order, identity_reviews, non_identity_reviews)

    final_entities = sorted(entities_by_id.values(), key=lambda e: (e["start"], e["id"]))
    updated_mentions = sorted(mentions_by_id.values(), key=lambda m: (m["start"], m["end"], m["id"]))

    surface_to_entity_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for m in updated_mentions:
        if m.get("identity_status") in {"not_entity_suggested", "not_entity_confirmed"} or m.get("link_method") == "not_entity_suggested":
            continue
        surface_to_entity_counts[m["text"]][m["entity_id"]] += 1

    ambiguous_surfaces = []
    for surf, e_counts in sorted(surface_to_entity_counts.items()):
        if len(e_counts) > 1:
            ambiguous_surfaces.append({
                "surface": surf,
                "total_mentions": sum(e_counts.values()),
                "entities": [
                    {"entity_id": eid, "count": cnt}
                    for eid, cnt in e_counts.most_common()
                ],
            })
    ambiguous_surfaces.sort(key=lambda item: (-item["total_mentions"], item["surface"]))

    identity_reviews.sort(key=lambda r: (-r.get("impact", 1), r["id"]))
    all_reviews = identity_reviews + non_identity_reviews

    method_totals = collections.Counter(m["link_method"] for m in updated_mentions)
    pending_identity_reviews = sum(1 for r in identity_reviews if r.get("status", "pending") == "pending")
    roster_summary = {
        "algorithm": algorithm,
        "entity_count": len(final_entities),
        "active_entity_count": sum(
            1 for e in final_entities
            if e["identity_status"] not in {"not_entity_suggested", "not_entity_confirmed"}
        ),
        "singleton_count": sum(1 for e in final_entities if e["mention_count"] == 1),
        "link_methods": dict(method_totals),
        "c3_rejected_name_mismatch": method_totals.get("c3_rejected_name_mismatch", 0),
        "identity_review_count": pending_identity_reviews,
        "ambiguous_surface_count": len(ambiguous_surfaces),
    }

    return {
        "entities": final_entities,
        "mentions": updated_mentions,
        "identity_reviews": identity_reviews,
        "non_identity_reviews": non_identity_reviews,
        "all_reviews": all_reviews,
        "ambiguous_surfaces": ambiguous_surfaces,
        "roster_summary": roster_summary,
    }


def _find_broken_dependents(
    roster_out: dict,
    raw_reviews: list[dict],
    chapters: list[dict],
    remaining_user: list[dict],
) -> list[str]:
    chapter_order = {c["id"]: idx for idx, c in enumerate(chapters)}
    entities_by_id, mentions_by_id, identity_reviews, non_identity_reviews = _init_replay_state(roster_out, raw_reviews)
    broken_ids: list[str] = []
    for corr in remaining_user:
        snap_entities = dict(entities_by_id)
        snap_mentions = dict(mentions_by_id)
        snap_id_rev = [dict(r) for r in identity_reviews]
        snap_non_id_rev = [dict(r) for r in non_identity_reviews]
        try:
            _apply_single_op_in_place(
                corr, snap_entities, snap_mentions, chapter_order, snap_id_rev, snap_non_id_rev
            )
        except ValueError:
            broken_ids.append(corr["id"])
        else:
            entities_by_id = snap_entities
            mentions_by_id = snap_mentions
            identity_reviews = snap_id_rev
            non_identity_reviews = snap_non_id_rev
    return broken_ids


def _find_duplicate_record(
    op: str,
    op_args: dict,
    source: str,
    active_user: list[dict],
    active_agent: list[dict],
    current_entities: list[dict],
    current_mentions: list[dict],
) -> dict | None:
    if source == "agent":
        for rec in reversed(active_agent):
            if rec["op"] == op and rec["args"] == op_args:
                return rec
        return None

    entities_by_id = {e["id"]: e for e in current_entities}
    mentions_by_id = {m["id"]: m for m in current_mentions}
    for rec in reversed(active_user):
        if rec["op"] != op or rec["args"] != op_args:
            continue
        if op == "merge":
            if op_args["merged_entity_id"] not in entities_by_id:
                return rec
        elif op == "split-surface":
            ent = entities_by_id.get(op_args["entity_id"])
            if ent is not None and not any(mentions_by_id[mid]["text"] == op_args["surface"] for mid in ent["mention_ids"]):
                return rec
        elif op == "assign":
            m = mentions_by_id.get(op_args["mention_id"])
            expected_eid = digest(["correction-new", rec["id"]]) if op_args["target"] == "new" else op_args["target"]
            if m is not None and m.get("entity_id") == expected_eid:
                return rec
        elif op == "not-entity":
            m = mentions_by_id.get(op_args["mention_id"])
            if m is not None and m.get("identity_status") == "not_entity_confirmed":
                return rec
    return None


def apply_correction(
    store: Store,
    run_id: str,
    op: str,
    op_args: dict,
    source: str,
    expect_revision: str,
    output_root: Path,
    resolves: str | None = None,
    note: str | None = None,
    cascade: bool = False,
) -> dict:
    from .pipeline import build_export_bundle, write_export_bundle

    if op not in VALID_OPS:
        raise ValueError(f"不支持的更正操作：{op}")
    if source not in VALID_SOURCES:
        raise ValueError("必须指定 --source user|agent")
    if not expect_revision:
        raise ValueError("必须指定 --expect-revision")

    run = store.run(run_id)
    algo = run["config"].get("identity_algorithm")
    if algo != ROSTER_VERSION:
        raise ValueError(f"更正仅支持 entity-roster-v1 运行，不支持 {algo}")
    state = run["state"]
    complete = all(state.get(name, {}).get("complete", False) for name in ("chapters", "discovery", "identity"))
    saved_roster = store.records(run_id, "roster_output")
    if not complete or not saved_roster:
        raise ValueError("更正仅支持已完成的 entity-roster-v1 运行，当前运行尚未完成")

    before_bundle = build_export_bundle(store, run_id)
    before_rev = before_bundle["result"]["result_revision"]
    if expect_revision != before_rev:
        raise CorrectionConflictError("结果版本已变化，请重新读取")

    before_reviews = before_bundle["review"]["items"]
    if resolves is not None:
        if op == "revoke":
            raise ValueError("revoke 操作不支持 --resolves")
        if not any(r["id"] == resolves for r in before_reviews):
            raise ValueError(f"待审项不存在：{resolves}")

    all_records = store.corrections(run_id)
    active_user, active_agent, revoked_set, revoked_by = partition_active_corrections(all_records)
    roster_out = saved_roster[0]
    raw_reviews = store.records(run_id, "review")
    chapters = store.records(run_id, "chapter")

    if op == "revoke":
        target_id = op_args["correction_id"]
        target_rec = next((r for r in all_records if r["id"] == target_id and r.get("op") != "revoke"), None)
        if target_rec is None:
            raise ValueError(f"更正记录不存在：{target_id}")
        if source == "agent" and target_rec.get("source") != "agent":
            raise ValueError("agent 只能撤回 agent 建议")
        if target_id in revoked_set:
            output = write_export_bundle(store, run_id, before_bundle, output_root)
            return {
                "correction_id": revoked_by[target_id],
                "status": "duplicate",
                "result_revision_before": before_rev,
                "result_revision": before_rev,
                "affected_entity_ids": [],
                "output_dir": str(output.resolve()),
            }
        if target_rec.get("source") == "user":
            remaining_user = [r for r in active_user if r["id"] != target_id]
            broken_ids = _find_broken_dependents(roster_out, raw_reviews, chapters, remaining_user)
            if broken_ids and not cascade:
                raise ValueError(
                    f"撤回更正 {target_id} 会使后续更正失效：{', '.join(broken_ids)}；如需一并撤回请加 --cascade"
                )
            to_revoke = [target_id, *broken_ids]
        else:
            to_revoke = [target_id]

        revoke_args = {"correction_id": target_id, "cascade": bool(cascade), "revoked_ids": to_revoke}
        created_at = utc_now()

        def build_revoke(seq: int) -> dict:
            cid = digest([run_id, seq, "revoke", revoke_args, source])
            return {
                "id": cid,
                "seq": seq,
                "op": "revoke",
                "args": revoke_args,
                "source": source,
                "note": note,
                "resolves": None,
                "base_revision": before_rev,
                "created": created_at,
            }

        saved = store.append_correction(run_id, build_revoke)
        after_bundle = build_export_bundle(store, run_id)
        output = write_export_bundle(store, run_id, after_bundle, output_root)
        before_map = {e["id"]: e for e in before_bundle["result"]["entities"]}
        after_map = {e["id"]: e for e in after_bundle["result"]["entities"]}
        affected = sorted(eid for eid in (set(before_map) | set(after_map)) if before_map.get(eid) != after_map.get(eid))
        return {
            "correction_id": saved["id"],
            "status": "revoked",
            "result_revision_before": before_rev,
            "result_revision": after_bundle["result"]["result_revision"],
            "affected_entity_ids": affected,
            "output_dir": str(output.resolve()),
        }

    # 非 revoke 操作：先检查幂等重复提交
    dup = _find_duplicate_record(
        op, op_args, source, active_user, active_agent,
        before_bundle["result"]["entities"], before_bundle["result"]["mentions"],
    )
    if dup is not None:
        output = write_export_bundle(store, run_id, before_bundle, output_root)
        return {
            "correction_id": dup["id"],
            "status": "duplicate",
            "result_revision_before": before_rev,
            "result_revision": before_rev,
            "affected_entity_ids": [],
            "output_dir": str(output.resolve()),
        }

    # 在落盘前演练校验本次更正是否合法（不写任何记录；失败直接抛异常）
    dry_probe = {
        "id": "__dry_run_probe__",
        "seq": 0,
        "op": op,
        "args": op_args,
        "source": source,
        "note": note,
        "resolves": resolves if source == "user" else None,
        "base_revision": before_rev,
        "created": "",
    }
    replay_user_corrections(roster_out, raw_reviews, chapters, [*active_user, dry_probe], algorithm=algo)

    created_at = utc_now()

    def build_op_record(seq: int) -> dict:
        cid = digest([run_id, seq, op, op_args, source])
        return {
            "id": cid,
            "seq": seq,
            "op": op,
            "args": op_args,
            "source": source,
            "note": note,
            "resolves": resolves,
            "base_revision": before_rev,
            "created": created_at,
        }

    saved = store.append_correction(run_id, build_op_record)
    after_bundle = build_export_bundle(store, run_id)
    output = write_export_bundle(store, run_id, after_bundle, output_root)
    before_map = {e["id"]: e for e in before_bundle["result"]["entities"]}
    after_map = {e["id"]: e for e in after_bundle["result"]["entities"]}
    affected = sorted(eid for eid in (set(before_map) | set(after_map)) if before_map.get(eid) != after_map.get(eid))
    return {
        "correction_id": saved["id"],
        "status": "applied" if source == "user" else "suggested",
        "result_revision_before": before_rev,
        "result_revision": after_bundle["result"]["result_revision"],
        "affected_entity_ids": affected,
        "output_dir": str(output.resolve()),
    }
