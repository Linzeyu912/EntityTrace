"""阶段 F：只读状态、名册、实体详情、待审、原文摘录与实体优先级排名查询（零模型调用，按需构建并复用只读查询快照）。"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .discovery import KINDS
from .pipeline import (
    PROTOCOL,
    build_export_bundle,
    infer_incomplete_stage_error,
    make_error_dict,
)
from .roster import VERSION as ROSTER_VERSION
from .storage import Store
from .text import address, validate_span

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
SNAPSHOT_VERSION = 2
DEFAULT_CONTEXT = 80
MAX_ENTITY_CONTEXT = 400
MAX_QUOTE_CONTEXT = 4000


def _validate_page(limit: int, offset: int) -> None:
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"分页参数 --limit 须在 1—{MAX_LIMIT} 之间")
    if offset < 0:
        raise ValueError("分页参数 --offset 不能为负数")


def _validate_context(context: int, max_context: int) -> None:
    if not 0 <= context <= max_context:
        raise ValueError(f"上下文参数 --context 须在 0—{max_context} 之间")


def _paginate(items: list, limit: int, offset: int) -> tuple[list, int, int | None]:
    total = len(items)
    page = items[offset:offset + limit]
    next_offset = (offset + limit) if (offset + limit) < total else None
    return page, total, next_offset


def _load_roster_snapshot(
    store: Store,
    run_id: str,
    need_entities: bool = False,
    need_reviews: bool = False,
) -> tuple[dict, dict]:
    run = store.run(run_id)
    algo = run["config"].get("identity_algorithm")
    if algo != ROSTER_VERSION:
        raise ValueError(f"只读查询仅支持 entity-roster-v1 运行，不支持 {algo}")
    fp = store.query_fingerprint(run_id)
    cols = ["fingerprint", "meta_json"]
    if need_entities:
        cols.append("entities_json")
    if need_reviews:
        cols.append("reviews_json")
    sql = f"SELECT {', '.join(cols)} FROM query_snapshots WHERE run_id=?"
    rows = store.rows(sql, (run_id,))
    if not rows or rows[0]["fingerprint"] != fp:
        build_export_bundle(store, run_id)
        rows = store.rows(sql, (run_id,))
    row = rows[0]
    snap: dict = {"meta": json.loads(row["meta_json"])}
    if need_entities:
        snap["entities"] = json.loads(row["entities_json"])
    if need_reviews:
        snap["reviews"] = json.loads(row["reviews_json"])
    # 旧版快照缺少新字段（snapshot_version / chapter_stats）时自动重建，不报错
    stale = snap["meta"].get("snapshot_version") != SNAPSHOT_VERSION or (
        need_entities and snap["entities"] and "chapter_stats" not in snap["entities"][0]
    )
    if stale:
        build_export_bundle(store, run_id)
        rows = store.rows(sql, (run_id,))
        row = rows[0]
        snap = {"meta": json.loads(row["meta_json"])}
        if need_entities:
            snap["entities"] = json.loads(row["entities_json"])
        if need_reviews:
            snap["reviews"] = json.loads(row["reviews_json"])
    return run, snap


def _load_source_text(store: Store, source_id: str) -> str:
    rows = store.rows("SELECT text FROM sources WHERE id=?", (source_id,))
    if not rows:
        raise ValueError("原文版本不存在")
    return rows[0]["text"]


def query_status(store: Store, run_id: str, output_root: Path) -> dict:
    run, snap = _load_roster_snapshot(store, run_id)
    meta = snap["meta"]
    state = run["state"]
    invocations = store.records(run_id, "invocation")
    last_inv_id = state.get("last_invocation")
    latest_inv = next(
        (inv for inv in reversed(invocations) if inv.get("id") == last_inv_id),
        invocations[-1] if invocations else {},
    )
    inv_stages = latest_inv.get("stages") or {}
    stages_info = {
        name: {
            "complete": bool((state.get(name) or {}).get("complete", False)),
            "seconds": float((inv_stages.get(name) or {}).get("seconds", 0.0)),
            "resumed": bool((inv_stages.get(name) or {}).get("resumed", False)),
        }
        for name in ("chapters", "discovery", "identity")
    }
    raw_err = state.get("last_error")
    if isinstance(raw_err, dict) and "category" in raw_err:
        last_error: dict | None = make_error_dict(raw_err["category"], str(raw_err.get("message", "")))
    elif isinstance(raw_err, str) and raw_err:
        last_error = make_error_dict("local", raw_err)
    elif meta["status"] == "incomplete":
        last_error = infer_incomplete_stage_error(state)
    else:
        last_error = None
    resolved_out = (output_root / run_id[:16]).resolve()
    return {
        "protocol": PROTOCOL,
        "command": "status",
        "run_id": run_id,
        "result_revision": meta["result_revision"],
        "error": None,
        "status": meta["status"],
        "stages": stages_info,
        "last_error": last_error,
        "counts": meta["counts"],
        "output_dir": str(resolved_out),
        "files": {
            "result": str(resolved_out / "result.json"),
            "review": str(resolved_out / "review.json"),
            "manifest": str(resolved_out / "manifest.json"),
        },
    }


def query_entities(
    store: Store,
    run_id: str,
    kind: str | None = None,
    min_mentions: int | None = None,
    surface: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict:
    _validate_page(limit, offset)
    if kind is not None and kind not in KINDS:
        raise ValueError(f"不支持的实体粗类：{kind}（可选：{', '.join(sorted(KINDS))}）")
    if min_mentions is not None and min_mentions < 0:
        raise ValueError("--min-mentions 不能为负数")
    if surface is not None and surface == "":
        raise ValueError("--surface 不能为空字符串")
    _, snap = _load_roster_snapshot(store, run_id, need_entities=True)
    meta = snap["meta"]
    matched: list[dict] = []
    for ent in snap["entities"]:
        if kind is not None and ent["kind"] != kind:
            continue
        if min_mentions is not None and ent["mention_count"] < min_mentions:
            continue
        surf_count = 0
        if surface is not None:
            surf_count = int((ent.get("surface_counts") or {}).get(surface, 0))
            if surf_count == 0:
                continue
        item = {
            "id": ent["id"],
            "name": ent["name"],
            "kind": ent["kind"],
            "mention_count": ent["mention_count"],
            "first_chapter_id": ent["first_chapter_id"],
            "identity_status": ent["identity_status"],
        }
        if surface is not None:
            item["surface_count"] = surf_count
        matched.append(item)
    matched.sort(key=lambda e: (-e["mention_count"], e["id"]))
    page, total, next_offset = _paginate(matched, limit, offset)
    return {
        "protocol": PROTOCOL,
        "command": "entities",
        "run_id": run_id,
        "result_revision": meta["result_revision"],
        "error": None,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "items": page,
    }


def _is_review_related_to_entity(review_item: dict, entity_id: str, ent_mids: set[str], ent_spans: set[tuple[int, int]]) -> bool:
    if review_item.get("entity_id") == entity_id:
        return True
    if review_item.get("split_entity_id") == entity_id:
        return True
    if review_item.get("rejected_entity_id") == entity_id:
        return True
    if review_item.get("mention_id") in ent_mids:
        return True
    options = review_item.get("options")
    if isinstance(options, list) and entity_id in options:
        return True
    stage = str(review_item.get("stage", ""))
    if stage.startswith("roster") or stage.startswith("correction"):
        for ev in review_item.get("evidence") or []:
            if isinstance(ev, dict) and (ev.get("start"), ev.get("end")) in ent_spans:
                return True
    return False


def query_entity(
    store: Store,
    run_id: str,
    entity_id: str,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    context: int = DEFAULT_CONTEXT,
) -> dict:
    _validate_page(limit, offset)
    _validate_context(context, MAX_ENTITY_CONTEXT)
    _, snap = _load_roster_snapshot(store, run_id, need_reviews=True)
    meta = snap["meta"]
    ent_rows = store.rows(
        "SELECT entity_json, mentions_json FROM query_entity_mentions WHERE run_id=? AND entity_id=?",
        (run_id, entity_id),
    )
    if not ent_rows:
        raise ValueError(f"实体不存在：{entity_id}")
    ent = json.loads(ent_rows[0]["entity_json"])
    ent_mentions = json.loads(ent_rows[0]["mentions_json"])
    text = _load_source_text(store, meta["source_id"])
    chapters_by_id = {c["id"]: c for c in meta["chapters"]}
    ent_mids = {m["id"] for m in ent_mentions}
    ent_spans = {(m["start"], m["end"]) for m in ent_mentions}

    related_reviews = [
        r for r in snap["reviews"]
        if r.get("status", "pending") == "pending"
        and _is_review_related_to_entity(r, entity_id, ent_mids, ent_spans)
    ]
    related_reviews.sort(
        key=lambda r: (-(r["impact"] if isinstance(r.get("impact"), int) else 1), r["id"])
    )
    review_ids = [r["id"] for r in related_reviews]

    page_mentions, total, next_offset = _paginate(ent_mentions, limit, offset)
    mention_items = []
    for m in page_mentions:
        ch = chapters_by_id[m["chapter_id"]]
        lo = max(ch["start"], m["start"] - context)
        hi = min(ch["end"], m["end"] + context)
        mention_items.append({
            "id": m["id"],
            "chapter_id": m["chapter_id"],
            "start": m["start"],
            "end": m["end"],
            "text": m["text"],
            "link_method": m["link_method"],
            "snippet": text[lo:hi],
        })
    return {
        "protocol": PROTOCOL,
        "command": "entity",
        "run_id": run_id,
        "result_revision": meta["result_revision"],
        "error": None,
        "entity_id": ent["id"],
        "name": ent["name"],
        "names": ent.get("names", []),
        "aliases": ent.get("aliases", []),
        "kind": ent["kind"],
        "identity_status": ent["identity_status"],
        "mention_count": ent["mention_count"],
        "first_chapter_id": ent["first_chapter_id"],
        "link_methods": ent.get("link_methods", {}),
        "review_ids": review_ids,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "mentions": mention_items,
    }


def query_review(
    store: Store,
    run_id: str,
    status: str = "pending",
    stage: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict:
    _validate_page(limit, offset)
    if status not in {"pending", "resolved", "all"}:
        raise ValueError("--status 仅支持 pending、resolved 或 all")
    _, snap = _load_roster_snapshot(store, run_id, need_reviews=True)
    meta = snap["meta"]
    matched: list[dict] = []
    for raw_item in snap["reviews"]:
        item_status = raw_item.get("status", "pending")
        if status != "all" and item_status != status:
            continue
        if stage is not None and raw_item.get("stage") != stage:
            continue
        item = dict(raw_item)
        if not isinstance(item.get("impact"), int):
            item["impact"] = 1
        item["status"] = item_status
        matched.append(item)
    matched.sort(key=lambda r: (-r["impact"], r["id"]))
    page, total, next_offset = _paginate(matched, limit, offset)
    return {
        "protocol": PROTOCOL,
        "command": "review",
        "run_id": run_id,
        "result_revision": meta["result_revision"],
        "error": None,
        "status_filter": status,
        "stage_filter": stage,
        "counts": meta["review_counts"],
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "items": page,
    }


def query_quote_json(
    store: Store,
    run_id: str,
    start: int,
    end: int,
    context: int = DEFAULT_CONTEXT,
) -> dict:
    _validate_context(context, MAX_QUOTE_CONTEXT)
    _, snap = _load_roster_snapshot(store, run_id)
    meta = snap["meta"]
    source_id = meta["source_id"]
    full_text = _load_source_text(store, source_id)
    span_text = validate_span(full_text, address(source_id, start, end), source_id)
    chapters = meta["chapters"]
    chapter_id = None
    for ch in chapters:
        if ch["start"] <= start < end <= ch["end"]:
            chapter_id = ch["id"]
            break
    if chapter_id is None:
        for ch in chapters:
            if ch["start"] <= start < ch["end"]:
                chapter_id = ch["id"]
                break
    return {
        "protocol": PROTOCOL,
        "command": "quote",
        "run_id": run_id,
        "result_revision": meta["result_revision"],
        "error": None,
        "start": start,
        "end": end,
        "text": span_text,
        "context_before": full_text[max(0, start - context):start],
        "context_after": full_text[end:min(len(full_text), end + context)],
        "chapter_id": chapter_id,
    }


def query_rank(
    store: Store,
    run_id: str,
    kind: str | None = None,
    from_chapter: int | None = None,
    to_chapter: int | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict:
    """实体优先级排名：按 sqrt(范围内提及数 × 范围内章节数) 排序，零模型调用、只读。"""
    _validate_page(limit, offset)
    if kind is not None and kind not in KINDS:
        raise ValueError(f"不支持的实体粗类：{kind}（可选：{', '.join(sorted(KINDS))}）")
    _, snap = _load_roster_snapshot(store, run_id, need_entities=True)
    meta = snap["meta"]
    chapters = meta["chapters"]
    body_seq: dict[str, int] = {}
    seq = 0
    for ch in chapters:
        if ch.get("kind") == "chapter":
            seq += 1
            body_seq[ch["id"]] = seq
    doc_order = {ch["id"]: idx for idx, ch in enumerate(chapters)}
    total_body = len(body_seq)

    if from_chapter is None and to_chapter is None:
        # 缺省全书：统计全部提及（包括卷题/附章区间内的提及）
        selected_ids = {ch["id"] for ch in chapters}
        range_info = {
            "from_chapter": 1 if total_body else None,
            "to_chapter": total_body if total_body else None,
            "chapter_count": total_body,
        }
    else:
        if total_body == 0:
            raise ValueError("当前运行无正文章节，不支持按章序范围筛选")
        n = 1 if from_chapter is None else from_chapter
        m_ = total_body if to_chapter is None else to_chapter
        if not (1 <= n <= total_body):
            raise ValueError(f"--from-chapter 须在 1—{total_body} 之间")
        if not (1 <= m_ <= total_body):
            raise ValueError(f"--to-chapter 须在 1—{total_body} 之间")
        if n > m_:
            raise ValueError("--from-chapter 不能大于 --to-chapter")
        selected_ids = {cid for cid, s in body_seq.items() if n <= s <= m_}
        range_info = {"from_chapter": n, "to_chapter": m_, "chapter_count": m_ - n + 1}

    scored: list[dict] = []
    for ent in snap["entities"]:
        ch_stats = ent.get("chapter_stats") or {}
        m = 0
        touched_ids: list[str] = []
        min_start: int | None = None
        for cid, stat in ch_stats.items():
            if cid not in selected_ids:
                continue
            cnt = int(stat.get("count", 0))
            if cnt <= 0:
                continue
            m += cnt
            touched_ids.append(cid)
            s_min = stat.get("min_start")
            if isinstance(s_min, int) and (min_start is None or s_min < min_start):
                min_start = s_min
        if m == 0:
            continue
        c = len(touched_ids)
        touched_ids.sort(key=lambda cid: doc_order[cid])
        body_seqs = [body_seq[cid] for cid in touched_ids if cid in body_seq]
        raw_score = math.sqrt(m * c)
        scored.append({
            "id": ent["id"],
            "name": ent["name"],
            "kind": ent["kind"],
            "identity_status": ent["identity_status"],
            "_m": m,
            "_c": c,
            "_raw": raw_score,
            "_min_start": min_start if min_start is not None else 0,
            "first_chapter_id": touched_ids[0],
            "last_chapter_id": touched_ids[-1],
            "span_chapters": (max(body_seqs) - min(body_seqs) + 1) if body_seqs else 0,
        })

    def sort_key(e: dict) -> tuple:
        # 第一级 (m*c, raw_score) 降序与未舍入 sqrt(m*c) 完全等价且无浮点抖动
        return (-(e["_m"] * e["_c"]), -e["_raw"], -e["_m"], -e["_c"], e["_min_start"], e["id"])

    by_kind: dict[str, list[dict]] = {}
    for e in scored:
        by_kind.setdefault(e["kind"], []).append(e)
    rank_map: dict[str, int] = {}
    for k, members in by_kind.items():
        members.sort(key=sort_key)
        for rank, e in enumerate(members, start=1):
            rank_map[e["id"]] = rank
    for e in scored:
        e["kind_rank"] = rank_map[e["id"]]

    if kind is not None:
        ordered = sorted(by_kind.get(kind, []), key=lambda e: e["kind_rank"])
    else:
        ordered = sorted(scored, key=sort_key)
    page, total, next_offset = _paginate(ordered, limit, offset)
    items = [
        {
            "id": e["id"],
            "name": e["name"],
            "kind": e["kind"],
            "identity_status": e["identity_status"],
            "kind_rank": e["kind_rank"],
            "score": round(e["_raw"], 6),
            "mention_count_in_range": e["_m"],
            "chapter_count_in_range": e["_c"],
            "first_chapter_id": e["first_chapter_id"],
            "last_chapter_id": e["last_chapter_id"],
            "span_chapters": e["span_chapters"],
        }
        for e in page
    ]
    return {
        "protocol": PROTOCOL,
        "command": "rank",
        "run_id": run_id,
        "result_revision": meta["result_revision"],
        "error": None,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "range": range_info,
        "items": items,
    }
