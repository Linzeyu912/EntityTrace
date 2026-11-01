"""阶段 F：只读状态、名册、实体详情、待审与原文摘录查询（零模型调用，统一由 build_export_bundle 内存构建）。"""
from __future__ import annotations

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


def _require_roster_bundle(store: Store, run_id: str) -> tuple[dict, dict]:
    run = store.run(run_id)
    algo = run["config"].get("identity_algorithm")
    if algo != ROSTER_VERSION:
        raise ValueError(f"只读查询仅支持 entity-roster-v1 运行，不支持 {algo}")
    bundle = build_export_bundle(store, run_id)
    return run, bundle


def query_status(store: Store, run_id: str, output_root: Path) -> dict:
    _, bundle = _require_roster_bundle(store, run_id)
    result = bundle["result"]
    manifest = bundle["manifest"]
    state = manifest["state"]
    invocations = manifest.get("invocations") or []
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
    elif result["status"] == "incomplete":
        last_error = infer_incomplete_stage_error(state)
    else:
        last_error = None
    resolved_out = (output_root / run_id[:16]).resolve()
    return {
        "protocol": PROTOCOL,
        "command": "status",
        "run_id": run_id,
        "result_revision": result["result_revision"],
        "error": None,
        "status": result["status"],
        "stages": stages_info,
        "last_error": last_error,
        "counts": manifest["counts"],
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
    _, bundle = _require_roster_bundle(store, run_id)
    result = bundle["result"]
    mentions_by_id = {m["id"]: m for m in result["mentions"]}
    matched: list[dict] = []
    for ent in result["entities"]:
        if kind is not None and ent["kind"] != kind:
            continue
        if min_mentions is not None and ent["mention_count"] < min_mentions:
            continue
        surf_count = 0
        if surface is not None:
            surf_count = sum(1 for mid in ent["mention_ids"] if mentions_by_id[mid]["text"] == surface)
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
        "result_revision": result["result_revision"],
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
    _, bundle = _require_roster_bundle(store, run_id)
    result = bundle["result"]
    ent = next((e for e in result["entities"] if e["id"] == entity_id), None)
    if ent is None:
        raise ValueError(f"实体不存在：{entity_id}")
    text = bundle["text"]
    chapters_by_id = {c["id"]: c for c in result["chapters"]}
    mentions_by_id = {m["id"]: m for m in result["mentions"]}
    ent_mentions = [mentions_by_id[mid] for mid in ent["mention_ids"]]
    ent_mentions.sort(key=lambda m: (m["start"], m["end"], m["id"]))
    ent_mids = {m["id"] for m in ent_mentions}
    ent_spans = {(m["start"], m["end"]) for m in ent_mentions}

    related_reviews = [
        r for r in bundle["review"]["items"]
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
        "result_revision": result["result_revision"],
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
    _, bundle = _require_roster_bundle(store, run_id)
    result = bundle["result"]
    review_doc = bundle["review"]
    matched: list[dict] = []
    for raw_item in review_doc["items"]:
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
        "result_revision": result["result_revision"],
        "error": None,
        "status_filter": status,
        "stage_filter": stage,
        "counts": review_doc["counts"],
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
    _, bundle = _require_roster_bundle(store, run_id)
    result = bundle["result"]
    full_text = bundle["text"]
    source_id = result["source"]["id"]
    span_text = validate_span(full_text, address(source_id, start, end), source_id)
    chapters = result["chapters"]
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
        "result_revision": result["result_revision"],
        "error": None,
        "start": start,
        "end": end,
        "text": span_text,
        "context_before": full_text[max(0, start - context):start],
        "context_after": full_text[end:min(len(full_text), end + context)],
        "chapter_id": chapter_id,
    }
