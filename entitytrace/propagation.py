"""规格三：同任务专名确定性补全（Rule T）与冗余定位待审合并消解（零模型调用）。"""
from __future__ import annotations

import collections

from .discovery import _word_boundary
from .roster import _unit_for_mention, normalize_surface
from .text import address, exact_matches, is_dirty_surface, validate_span
from .util import digest


def propagate_same_task_proper(
    source: dict,
    chapters: list[dict],
    discovery_tasks: list[dict],
    c1_results: dict[str, dict],
    entities: list[dict],
    mentions: list[dict],
    ambiguous_surfaces: list[dict],
    roster_summary: dict,
    downgraded_norms: set[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict, list[dict]]:
    """在身份归组（C1/C2/C3/C4）完成后，对同任务内无歧义 C1 专名做确定性字面补全。

    若无任何新增补全提及，原样返回输入对象引用，确保无可补时导出逐字节不变。
    """
    text = source["text"]
    source_id = source["id"]
    chapter_order = {c["id"]: idx for idx, c in enumerate(chapters)}
    ent_by_id = {e["id"]: e for e in entities}

    mentions_by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for m in mentions:
        mentions_by_task[m["task_id"]].append(m)

    downgraded = downgraded_norms or set()
    new_mentions: list[dict] = []
    for dtask in discovery_tasks:
        tid = dtask["id"]
        c1 = c1_results.get(tid)
        prop_norms = (set(c1.get("declared_proper_norms", [])) - downgraded) if c1 else set()
        weak_norms = set(c1.get("weak_norms", [])) if c1 else set()
        c1_req_id = c1.get("request_id") if c1 else None
        t_mentions = mentions_by_task.get(tid, [])
        by_s: dict[str, list[dict]] = collections.defaultdict(list)
        for m in t_mentions:
            by_s[m["text"]].append(m)

        qualifying_t: dict[str, str] = {}
        for s, ms in by_s.items():
            if len(s) < 2 or is_dirty_surface(s):
                continue
            ns = normalize_surface(s)
            if ns not in prop_norms or ns in weak_norms:
                continue
            eids = {m["entity_id"] for m in ms}
            if len(eids) != 1:
                continue
            eid = next(iter(eids))
            ent = ent_by_id[eid]
            if ent.get("identity_status") in {"unresolved_singleton", "not_entity_suggested"}:
                continue
            if any(
                m.get("link_method") in {"unresolved", "not_entity_suggested", "c3_rejected_name_mismatch"}
                for m in ms
            ):
                continue
            qualifying_t[s] = eid

        all_q: dict[str, tuple[str, str]] = {s: (eid, "same_task_proper") for s, eid in qualifying_t.items()}
        if not all_q:
            continue

        existing_spans = [(m["start"], m["end"]) for m in t_mentions]
        masked_spans = dtask.get("masked_spans") or []
        cands = []
        for s, (eid, method) in all_q.items():
            for a, b in exact_matches(text, s, dtask["start"], dtask["end"]):
                if not _word_boundary(text, a, b, s):
                    continue
                if masked_spans and any(max(a, sa) < min(b, sb) for sa, sb in masked_spans):
                    continue
                if any(max(a, ea) < min(b, eb) for ea, eb in existing_spans):
                    continue
                cands.append((a, b, s, eid, method))

        if not cands:
            continue

        cands.sort(key=lambda x: (-(x[1] - x[0]), x[0], x[2]))
        accepted_spans: list[tuple[int, int]] = []
        unit_by_id = {u["id"]: u for u in dtask["units"]}
        for a, b, s, eid, method in cands:
            if any(max(a, aa) < min(b, ab) for aa, ab in accepted_spans):
                continue
            accepted_spans.append((a, b))
            span = address(source_id, a, b)
            validate_span(text, span, source_id, s)
            mid = digest([source_id, a, b, s])
            u_id = _unit_for_mention({"start": a, "end": b}, dtask["units"])
            u = unit_by_id.get(u_id, dtask["units"][0])
            ev_span = address(source_id, u["start"], u["end"])
            ent = ent_by_id[eid]
            new_m = {
                "id": mid,
                "text": s,
                "start": a,
                "end": b,
                "address": span,
                "chapter_id": dtask["chapter_id"],
                "task_id": tid,
                "provisional_kinds": [ent["kind"]] if ent.get("kind") else [],
                "observation_ids": [],
                "evidence": [ev_span],
                "entity_id": eid,
                "link_method": method,
                "link_refs": [c1_req_id] if (method == "same_task_proper" and c1_req_id) else [],
            }
            new_mentions.append(new_m)

    if not new_mentions:
        return entities, mentions, ambiguous_surfaces, roster_summary, []

    updated_mentions = sorted([*mentions, *new_mentions], key=lambda m: (m["start"], m["end"], m["id"]))
    by_mid = {m["id"]: m for m in updated_mentions}
    new_mids_by_eid: dict[str, list[dict]] = collections.defaultdict(list)
    for nm in new_mentions:
        new_mids_by_eid[nm["entity_id"]].append(nm)

    updated_entities: list[dict] = []
    for ent in entities:
        added = new_mids_by_eid.get(ent["id"])
        if not added:
            updated_entities.append(ent)
            continue
        ent_copy = dict(ent)
        mids = sorted(
            dict.fromkeys([*ent["mention_ids"], *(nm["id"] for nm in added)]),
            key=lambda mid: (by_mid[mid]["start"], by_mid[mid]["end"], mid),
        )
        lmethods = dict(ent.get("link_methods", {}))
        for nm in added:
            m_name = nm["link_method"]
            lmethods[m_name] = lmethods.get(m_name, 0) + 1
        ent_copy["mention_ids"] = mids
        ent_copy["mention_count"] = len(mids)
        ent_copy["start"] = by_mid[mids[0]]["start"]
        ent_copy["first_chapter_id"] = min(
            (by_mid[mid]["chapter_id"] for mid in mids),
            key=lambda cid: chapter_order.get(cid, 999999),
        )
        ent_copy["names"] = sorted({by_mid[mid]["text"] for mid in mids})
        ent_copy["link_methods"] = lmethods
        updated_entities.append(ent_copy)

    surface_to_entity_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for m in updated_mentions:
        if m.get("link_method") == "not_entity_suggested":
            continue
        surface_to_entity_counts[m["text"]][m["entity_id"]] += 1

    updated_ambiguous = []
    for surf, e_counts in sorted(surface_to_entity_counts.items()):
        if len(e_counts) > 1:
            updated_ambiguous.append({
                "surface": surf,
                "total_mentions": sum(e_counts.values()),
                "entities": [{"entity_id": eid, "count": cnt} for eid, cnt in e_counts.most_common()],
            })
    updated_ambiguous.sort(key=lambda item: (-item["total_mentions"], item["surface"]))

    method_totals = collections.Counter(m["link_method"] for m in updated_mentions)
    updated_summary = dict(roster_summary)
    updated_summary["entity_count"] = len(updated_entities)
    updated_summary["active_entity_count"] = sum(
        1 for e in updated_entities if e.get("identity_status") != "not_entity_suggested"
    )
    updated_summary["singleton_count"] = sum(1 for e in updated_entities if e["mention_count"] == 1)
    updated_summary["link_methods"] = dict(method_totals)
    updated_summary["ambiguous_surface_count"] = len(updated_ambiguous)

    return updated_entities, updated_mentions, updated_ambiguous, updated_summary, new_mentions


def deduplicate_grounding_reviews(
    source: dict,
    discovery_tasks: list[dict],
    observations: list[dict],
    mentions: list[dict],
    reviews: list[dict],
) -> tuple[list[dict], list[dict]]:
    """对原因为“原文引用内找不到完整称呼”的定位待审，若该称呼在同一任务内每处出现均已被提及覆盖，则标记为 resolved。"""
    text = source["text"]
    obs_by_id = {o["id"]: o for o in observations}
    dtask_by_id = {t["id"]: t for t in discovery_tasks}
    task_spans: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for m in mentions:
        task_spans[m["task_id"]].append((m["start"], m["end"]))

    updated_reviews: list[dict] = []
    resolved_items: list[dict] = []
    for item in reviews:
        if (
            item.get("stage") == "grounding"
            and item.get("reason") == "原文引用内找不到完整称呼"
            and item.get("status", "pending") == "pending"
        ):
            obs = obs_by_id.get(item.get("observation_id", ""))
            if obs is not None:
                tid = obs["task_id"]
                surf = obs["text"]
                dtask = dtask_by_id.get(tid)
                if dtask is not None:
                    hits = [
                        h
                        for h in exact_matches(text, surf, dtask["start"], dtask["end"])
                        if _word_boundary(text, *h, surf)
                    ]
                    tspans = task_spans.get(tid, [])
                    if hits and all(any(max(a, ta) < min(b, tb) for ta, tb in tspans) for a, b in hits):
                        new_item = {
                            **item,
                            "status": "resolved",
                            "resolution": "auto_same_task_covered",
                        }
                        updated_reviews.append(new_item)
                        resolved_items.append(new_item)
                        continue
        updated_reviews.append(item)

    return updated_reviews, resolved_items


def resolve_redundant_grounding_reviews(
    source: dict,
    discovery_tasks: list[dict],
    observations: list[dict],
    mentions: list[dict],
    reviews: list[dict],
) -> tuple[list[dict], list[dict]]:
    """导出期消解两类冗余定位待审（纯函数，零模型调用，不修改输入）。

    - B1：同一条发现记录的其它引用已成功定位，且该称呼就在本任务正文内
      （reported_refs 错指已覆盖场景）→ ``auto_ref_mismatch_covered``。
    - A1：该称呼不在本任务正文内，但落在前后 250 码点邻文，且相邻任务已在
      该邻文范围登记同一称呼（跨任务切片误报已覆盖场景）
      → ``auto_context_margin_covered``。
    """
    text = source["text"]
    margin = 250
    obs_by_id = {o["id"]: o for o in observations}
    dtask_by_id = {
        t["id"]: t
        for t in discovery_tasks
        if isinstance(t.get("start"), int) and isinstance(t.get("end"), int)
    }
    mentions_by_surface: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for m in mentions:
        mentions_by_surface[m.get("text", "")].append((m["start"], m.get("task_id", "")))

    updated_reviews: list[dict] = []
    resolved_items: list[dict] = []
    for item in reviews:
        new_item = None
        if item.get("stage") == "grounding" and item.get("status", "pending") == "pending":
            obs = obs_by_id.get(item.get("observation_id", ""))
            dtask = dtask_by_id.get(obs["task_id"]) if obs else None
            if dtask is not None:
                s, e = dtask["start"], dtask["end"]
                surf = obs.get("text", "")
                covered = obs.get("status") == "grounded" or bool(obs.get("mention_ids"))
                if covered and surf in text[s:e]:
                    new_item = {
                        **item,
                        "status": "resolved",
                        "resolved_by": "auto_ref_mismatch_covered",
                    }
                elif surf and surf not in text[s:e] and (
                    surf in text[max(0, s - margin):s] or surf in text[e:e + margin]
                ):
                    lo, hi = max(0, s - margin), e + margin
                    if any(
                        m_task_id != obs["task_id"]
                        and (lo <= m_start < s or e <= m_start < hi)
                        for m_start, m_task_id in mentions_by_surface.get(surf, [])
                    ):
                        new_item = {
                            **item,
                            "status": "resolved",
                            "resolved_by": "auto_context_margin_covered",
                        }
        if new_item is not None:
            updated_reviews.append(new_item)
            resolved_items.append(new_item)
        else:
            updated_reviews.append(item)

    return updated_reviews, resolved_items
