from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from .chapters import detect_chapters
from .config import load_services
from .corrections import CorrectionConflictError, partition_active_corrections, replay_user_corrections
from .discovery import discover
from .identity import group_entities
from .resolution import VERSION as RESOLUTION_VERSION, annotate_groups, resolve_groups
from .roster import (
    DIRECT_VERSION as ROSTER_DIRECT_VERSION,
    VERSION as ROSTER_VERSION,
    build_incomplete_roster_output,
    resolve_roster,
)
from .models import BudgetError, ModelError, Models, RequestBudget
from .storage import Store
from .text import import_text, validate_span
from .util import digest, dumps, implementation_hash, utc_now, write_bytes, write_json

PROTOCOL = "entitytrace.v1"
ALLOWED_ERROR_CATEGORIES = frozenset({
    "input", "config", "quota", "network", "model_output", "budget", "conflict", "local",
})
RETRYABLE_CATEGORIES = frozenset({"network", "budget", "model_output"})
EXIT_CODE_BY_CATEGORY = {
    "network": 2,
    "budget": 2,
    "model_output": 2,
    "quota": 2,
    "input": 1,
    "config": 1,
    "conflict": 1,
    "local": 1,
}
_CONFIG_VALUE_HINTS = (
    "模型密钥",
    "TypeSafe/Jev 凭证",
    "服务地址不是已核对的官方地址",
    "Decision 模型需要明确版本",
)
_LOCAL_INVARIANT_HINTS = (
    "保存的原文指纹不匹配",
    "原文版本不存在",
    "提及地址不属于对应章节",
    "实体分组未恰好覆盖所有提及",
    "身份判断引用不存在的提及",
    "章题原文核验失败",
    "同一位置出现冲突章题",
    "章题范围相互重叠",
    "章节范围无法完整还原全文",
)


def make_error_dict(category: str, message: str) -> dict:
    cat = category if category in ALLOWED_ERROR_CATEGORIES else "local"
    return {
        "category": cat,
        "message": message,
        "retryable": cat in RETRYABLE_CATEGORIES,
    }


def classify_exception(exc: BaseException) -> dict:
    if isinstance(exc, CorrectionConflictError):
        return make_error_dict("conflict", str(exc))
    tagged = getattr(exc, "et_category", None)
    if tagged in ALLOWED_ERROR_CATEGORIES:
        msg = str(exc) if str(exc) else f"{tagged} 错误：{type(exc).__name__}"
        return make_error_dict(tagged, msg)
    if isinstance(exc, BudgetError):
        cat = "network" if (exc.category == "network" or exc.stop_reason == "network_outage") else "budget"
        return make_error_dict(cat, str(exc))
    if isinstance(exc, ModelError):
        status = getattr(exc, "status", None)
        code = getattr(exc, "code", None)
        msg = str(exc)
        if code == "network_error" or status in {500, 502, 503, 504, 529}:
            return make_error_dict("network", msg)
        if (
            status in {402, 429}
            or "余额或硬额度不足" in msg
            or code in {"billing_error", "quota_exceeded", "insufficient_quota", "rate_limit_exceeded"}
        ):
            return make_error_dict("quota", msg)
        if status in {401, 403}:
            return make_error_dict("config", msg)
        return make_error_dict("model_output", msg)
    if isinstance(exc, RuntimeError) and getattr(exc, "code", None) == "network_error":
        return make_error_dict("network", str(exc))
    if isinstance(exc, (UnicodeDecodeError, LookupError)):
        return make_error_dict("input", f"输入文本解码失败：{exc}")
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return make_error_dict("input", f"输入路径不可访问：{exc}")
    if isinstance(exc, ValueError):
        msg = str(exc)
        if any(h in msg for h in _CONFIG_VALUE_HINTS):
            return make_error_dict("config", msg)
        if any(h in msg for h in _LOCAL_INVARIANT_HINTS):
            return make_error_dict("local", msg)
        return make_error_dict("input", msg)
    msg = str(exc) if isinstance(exc, RuntimeError) and str(exc) else f"本地运行错误：{type(exc).__name__}"
    return make_error_dict("local", msg)


def infer_incomplete_stage_error(state: dict, stop_after: str = "identity") -> dict | None:
    for name in ("chapters", "discovery", "identity"):
        st = state.get(name)
        if st is None:
            continue
        if not st.get("complete", False):
            failures = st.get("failures") or []
            if failures:
                msgs = [str(f.get("error", "")) for f in failures]
                if any("余额或硬额度不足" in m or "HTTP 402" in m or "HTTP 429" in m for m in msgs):
                    return make_error_dict("quota", f"阶段 {name} 未完成（{len(failures)} 个任务失败）：{msgs[0]}")
                if any("HTTP 401" in m or "HTTP 403" in m for m in msgs):
                    return make_error_dict("config", f"阶段 {name} 未完成（{len(failures)} 个任务失败）：{msgs[0]}")
                if any(
                    "网络失败或超时" in m
                    or any(f"HTTP {code}" in m for code in (500, 502, 503, 504, 529))
                    for m in msgs
                ):
                    return make_error_dict("network", f"阶段 {name} 未完成（{len(failures)} 个任务失败）：{msgs[0]}")
                return make_error_dict(
                    "model_output",
                    f"阶段 {name} 未完成：{len(failures)} 个任务因模型输出或格式错误失败，保留断点可续跑",
                )
            issues = int(st.get("issues", 0) or 0)
            if issues > 0:
                return make_error_dict("input", "章节结构无法自动确认，重跑结果相同，见 `review --stage chapters`")
            return make_error_dict("model_output", f"阶段 {name} 未完成，保留断点可续跑")
        if name == stop_after:
            return None
    return None

IDENTITY_ALGORITHMS = {
    "roster": ROSTER_VERSION,
    ROSTER_VERSION: ROSTER_VERSION,
    "groups-v2": RESOLUTION_VERSION,
    RESOLUTION_VERSION: RESOLUTION_VERSION,
}


def metrics(store: Store, run_id: str, invocation: str | None = None) -> dict:
    sql, args = "SELECT * FROM calls WHERE run_id=?", (run_id,)
    if invocation:
        sql += " AND invocation=?"
        args += (invocation,)
    rows = store.rows(sql, args)
    result = {"network_attempts": 0, "successful_requests": 0, "failed_attempts": 0, "cache_hits": 0,
              "input_tokens_reported": 0, "output_tokens_reported": 0, "request_seconds_sum": 0.0, "by_stage": {}}
    for row in rows:
        stage = result["by_stage"].setdefault(row["stage"], {"attempts": 0, "success": 0, "failed": 0, "cached": 0, "seconds_sum": 0.0})
        if row["status"] == "cached":
            result["cache_hits"] += 1
            stage["cached"] += 1
            continue
        result["network_attempts"] += 1
        result["request_seconds_sum"] += row["seconds"]
        stage["attempts"] += 1
        stage["seconds_sum"] += row["seconds"]
        usage = json.loads(row["detail"]).get("usage", {})
        if isinstance(usage, dict) and usage:
            result["input_tokens_reported"] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            result["output_tokens_reported"] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        if row["status"] == "success":
            result["successful_requests"] += 1
            stage["success"] += 1
        else:
            result["failed_attempts"] += 1
            stage["failed"] += 1
    return result


def build_export_bundle(store: Store, run_id: str) -> dict:
    run = store.run(run_id)
    source = store.source(run["source_id"])
    text, metadata = source["text"], source["metadata"]
    if hashlib.sha256(source["raw"]).hexdigest() != metadata["raw_sha256"] or hashlib.sha256(text.encode("utf-8")).hexdigest() != metadata["text_sha256"]:
        raise ValueError("保存的原文指纹不匹配，停止导出")
    chapters = store.records(run_id, "chapter")
    mentions = store.records(run_id, "mention")
    observations = store.records(run_id, "observation")
    decisions = store.records(run_id, "decision")
    reviews = store.records(run_id, "review")
    algo = run["config"].get("identity_algorithm")
    roster_mode = algo in {ROSTER_VERSION, ROSTER_DIRECT_VERSION}
    by_chapter = {c["id"]: c for c in chapters}
    state = run["state"]
    complete = all(state.get(name, {}).get("complete", False) for name in ("chapters", "discovery", "identity"))

    if roster_mode:
        raw_ids = {m["id"] for m in mentions}
        saved_roster = store.records(run_id, "roster_output")
        roster_out = saved_roster[0] if saved_roster else build_incomplete_roster_output(source, mentions, algorithm=algo)
        active_user, active_agent, _, _ = partition_active_corrections(store.corrections(run_id))
        if active_user:
            replayed = replay_user_corrections(roster_out, reviews, chapters, active_user, algorithm=algo)
            entities = replayed["entities"]
            mentions = replayed["mentions"]
            identity_reviews = replayed["identity_reviews"]
            non_identity_reviews = replayed["non_identity_reviews"]
            all_reviews = replayed["all_reviews"]
            roster_summary = replayed["roster_summary"]
            ambiguous_surfaces = replayed["ambiguous_surfaces"]
        else:
            entities = [dict(e) for e in roster_out["entities"]]
            mentions = [dict(m) for m in roster_out["mentions"]]
            non_identity_reviews = [r for r in reviews if not str(r.get("stage", "")).startswith("roster")]
            identity_reviews = [r for r in roster_out["reviews"] if str(r.get("stage", "")).startswith("roster")]
            all_reviews = identity_reviews + non_identity_reviews
            roster_summary = roster_out["roster_summary"]
            ambiguous_surfaces = roster_out["ambiguous_surfaces"]
        for mention in mentions:
            validate_span(text, mention["address"], source["id"], mention["text"])
            chapter = by_chapter[mention["chapter_id"]]
            if not chapter["start"] <= mention["start"] < mention["end"] <= chapter["end"]:
                raise ValueError("提及地址不属于对应章节")
            mention["chapter_start"] = mention["start"] - chapter["start"]
            mention["chapter_end"] = mention["end"] - chapter["start"]
            for span in mention["evidence"]:
                validate_span(text, span, source["id"])
        ids = {m["id"] for m in mentions}
        if (
            not (raw_ids <= ids)
            or any(m.get("link_method") != "same_task_proper" for m in mentions if m["id"] not in raw_ids)
            or sorted(mid for e in entities for mid in e["mention_ids"]) != sorted(ids)
        ):
            raise ValueError("实体分组未恰好覆盖所有提及")
        pending_identity = [r for r in identity_reviews if r.get("status", "pending") == "pending"]
        pending_non_identity = [r for r in non_identity_reviews if r.get("status", "pending") == "pending"]
        pending_all = pending_identity + pending_non_identity
        has_resolved = bool(active_user) or (len(pending_all) != len(all_reviews))
        status = "incomplete" if not complete else "completed_with_review" if pending_all else "completed"
        limitations = [
            "名册条目 ID 由成员首处地址与强称呼集合摘要生成，不是跨版本永久编号。",
            "发现阶段未覆盖的提及不进入名册归组；原文定位通过不代表实体和身份语义正确。",
            "无比较或未决的单独提及（unresolved_singleton）仅表示证据不足，不能据此断言与所有其他实体不同。",
        ]
        if active_user:
            limitations.append("含用户更正，被更正实体的 ID 规则见 corrections")
        result = {
            "schema_version": 3,
            "run_id": run_id,
            "status": status,
            "source": metadata,
            "chapters": chapters,
            "mentions": mentions,
            "observations": observations,
            "entities": entities,
            "roster_summary": roster_summary,
            "ambiguous_surfaces": ambiguous_surfaces,
            "scope": "实体提及与名册链接身份归组；不含描述、服饰或经历提取",
            "limitations": limitations,
        }
        if active_user:
            result["corrections"] = active_user
        if run["config"].get("source_run_id"):
            result["source_run_id"] = run["config"]["source_run_id"]
        result["result_revision"] = digest(result)
        review_counts = {
            "total": len(all_reviews),
            "identity_reviews": len(identity_reviews),
            "grounding_discovery_reviews": len(non_identity_reviews),
        }
        if has_resolved:
            review_counts["pending"] = len(pending_all)
            review_counts["resolved"] = len(all_reviews) - len(pending_all)
        review_doc = {
            "schema_version": 1,
            "run_id": run_id,
            "result_revision": result["result_revision"],
            "items": all_reviews,
            "counts": review_counts,
            "instruction": "按 impact 降序排列；自动模型判断不冒充人工答案。",
        }
        if active_agent:
            review_doc["agent_suggestions"] = active_agent
        invocations = store.records(run_id, "invocation")
        identity_seconds = round(
            sum(
                inv.get("stages", {}).get("identity", {}).get("seconds", 0.0)
                for inv in invocations
                if not inv.get("stages", {}).get("identity", {}).get("resumed", False)
            ),
            4,
        )
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "result_revision": result["result_revision"],
            "status": status,
            "source": metadata,
            "config": run["config"],
            "state": state,
            "invocations": invocations,
            "identity_seconds": identity_seconds,
            "total_model_metrics": metrics(store, run_id),
            "counts": {
                "chapters": sum(c["kind"] == "chapter" for c in chapters),
                "sections": len(chapters),
                "chapter_path_counts": state.get("chapters", {}).get("path_counts", {}),
                "mentions": len(mentions),
                "observations": len(observations),
                "entities": len(entities),
                "singletons": sum(1 for e in entities if len(e["mention_ids"]) == 1),
                "unlocated_observations": sum(o["status"] == "unlocated" for o in observations),
                "decisions": len(decisions),
                "review_items": len(pending_all) if has_resolved else len(all_reviews),
                "identity_review_items": len(pending_identity) if has_resolved else len(identity_reviews),
                "grounding_discovery_review_items": len(pending_non_identity) if has_resolved else len(non_identity_reviews),
            },
            "validation": {
                "source_hashes": True,
                "mention_addresses": True,
                "entity_membership": True,
                "semantic_accuracy_measured": False,
            },
            "files": ["result.json", "review.json", "decisions.jsonl", "requests.jsonl", "source.txt", "source.original"],
        }
        if run["config"].get("source_run_id"):
            manifest["source_run_id"] = run["config"]["source_run_id"]
        if run["config"].get("arm"):
            manifest["arm"] = run["config"]["arm"]
        return {"result": result, "review": review_doc, "manifest": manifest, "decisions": decisions, "text": text, "raw": source["raw"]}

    identities = store.records(run_id, "identity")
    entities, conflicts = group_entities(mentions, identities)
    group_resolution = algo == RESOLUTION_VERSION
    if group_resolution:
        entities = annotate_groups(entities, store.records(run_id, "entity_annotation"))
    reviews.extend(conflicts)
    for mention in mentions:
        validate_span(text, mention["address"], source["id"], mention["text"])
        chapter = by_chapter[mention["chapter_id"]]
        if not chapter["start"] <= mention["start"] < mention["end"] <= chapter["end"]:
            raise ValueError("提及地址不属于对应章节")
        mention["chapter_start"] = mention["start"] - chapter["start"]
        mention["chapter_end"] = mention["end"] - chapter["start"]
        for span in mention["evidence"]:
            validate_span(text, span, source["id"])
    ids = {m["id"] for m in mentions}
    if sorted(mid for e in entities for mid in e["mention_ids"]) != sorted(ids):
        raise ValueError("实体分组未恰好覆盖所有提及")
    for pair in identities:
        if pair["left"] not in ids or pair["right"] not in ids:
            raise ValueError("身份判断引用不存在的提及")
    status = "incomplete" if not complete else "completed_with_review" if reviews else "completed"
    result = {"schema_version": 2 if group_resolution else 1, "run_id": run_id, "status": status, "source": metadata,
              "chapters": chapters, "mentions": mentions, "observations": observations, "entities": entities,
              "identity_relations": identities, "scope": "实体提及与身份归组；不含描述、服饰或经历提取",
              "limitations": ["候选召回未采用语义向量检索，可能漏掉没有字面联系的化名。",
                              "模型分数门槛尚未标注校准；原文定位通过不代表实体和身份语义正确。",
                              "无比较关系的单独提及为暂定实体，不能据此断言与所有其他实体不同。"]}
    result["result_revision"] = digest(result)
    review_doc = {"schema_version": 1, "run_id": run_id, "result_revision": result["result_revision"],
                  "items": reviews, "instruction": "按问题编号回答；自动模型判断不冒充人工答案。"}
    invocations = store.records(run_id, "invocation")
    manifest = {"schema_version": 1, "run_id": run_id, "result_revision": result["result_revision"], "status": status,
                "source": metadata, "config": run["config"], "state": state, "invocations": invocations,
                "total_model_metrics": metrics(store, run_id), "counts": {"chapters": sum(c["kind"] == "chapter" for c in chapters),
                "sections": len(chapters), "mentions": len(mentions), "observations": len(observations), "entities": len(entities),
                "unlocated_observations": sum(o["status"] == "unlocated" for o in observations), "decisions": len(decisions),
                "identity_pairs": len(identities), "review_items": len(reviews), "grouping_conflicts": len(conflicts)},
                "validation": {"source_hashes": True, "mention_addresses": True, "entity_membership": True,
                               "semantic_accuracy_measured": False},
                "files": ["result.json", "review.json", "decisions.jsonl", "requests.jsonl", "source.txt", "source.original"]}
    if group_resolution:
        manifest["resolution_rounds"] = store.records(run_id, "resolution_round")
    return {"result": result, "review": review_doc, "manifest": manifest, "decisions": decisions, "text": text, "raw": source["raw"]}


def write_export_bundle(store: Store, run_id: str, bundle: dict, output_root: Path, output_dir: Path | None = None) -> Path:
    output = output_dir if output_dir is not None else (output_root / run_id[:16])
    write_json(output / "result.json", bundle["result"])
    write_json(output / "review.json", bundle["review"])
    write_bytes(output / "decisions.jsonl", "".join(dumps(d) + "\n" for d in bundle["decisions"]).encode("utf-8"))
    requests = store.rows("SELECT DISTINCT r.* FROM request_cache r JOIN calls c ON r.id=c.request_id WHERE c.run_id=? ORDER BY r.id", (run_id,))
    write_bytes(output / "requests.jsonl", "".join(dumps({"id": r["id"], "service": r["service"], "model": r["model"],
        "request": json.loads(r["request"]), "response": json.loads(r["response"])}) + "\n" for r in requests).encode("utf-8"))
    write_bytes(output / "source.txt", bundle["text"].encode("utf-8"))
    write_bytes(output / "source.original", bundle["raw"])
    write_json(output / "manifest.json", bundle["manifest"])
    return output


def export_run(store: Store, run_id: str, output_root: Path, output_dir: Path | None = None) -> Path:
    bundle = build_export_bundle(store, run_id)
    return write_export_bundle(store, run_id, bundle, output_root, output_dir)


def run_pipeline(input_path: Path, db_path: Path, output_root: Path, config_path: Path,
                 discovery_concurrency: int = 6, decision_concurrency: int = 3, task_chars: int = 9000,
                 encoding: str = "utf-8", stop_after: str = "identity", log=print,
                 identity_rounds: int = 12, group_candidates: int = 3,
                 max_requests: int | None = None, max_seconds: float | None = None,
                 max_network_failures: int = 9,
                 identity_algorithm: str = "roster",
                 receipt_out: dict | None = None) -> tuple[str, Path, bool]:
    wall_begin = time.monotonic()
    if identity_algorithm not in IDENTITY_ALGORITHMS:
        raise ValueError(f"不支持的身份归组算法：{identity_algorithm}")
    algo_version = IDENTITY_ALGORITHMS[identity_algorithm]
    if not 1 <= discovery_concurrency <= 32 or not 1 <= decision_concurrency <= 16 or not 500 <= task_chars <= 20000:
        raise ValueError("并发数或任务长度超出本工具支持范围")
    if algo_version == RESOLUTION_VERSION and (not 1 <= identity_rounds <= 24 or not 1 <= group_candidates <= 3):
        raise ValueError("身份轮次须为 1—24，候选组上限须为 1—3")
    budget = RequestBudget(max_requests, max_seconds, max_network_failures=max_network_failures)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    store = Store(db_path)
    begin, invocation = time.monotonic(), uuid.uuid4().hex
    started, stages = utc_now(), {}
    run_id = None
    caught_exc: Exception | None = None
    try:
        try:
            source = import_text(store, input_path, encoding)
        except Exception as exc:
            exc.et_category = "input"  # type: ignore[attr-defined]
            raise
        try:
            services = load_services(config_path, require_decision=(algo_version == RESOLUTION_VERSION))
        except Exception as exc:
            exc.et_category = "config"  # type: ignore[attr-defined]
            raise
        # 预算参数（max_requests / max_seconds / max_network_failures）不写入 config 摘要，保证扩大预算可续跑同一 run_id
        config = {"implementation": implementation_hash(), "models": {k: v.public() for k, v in services.items()},
                  "discovery_concurrency": discovery_concurrency, "decision_concurrency": decision_concurrency,
                  "task_chars": task_chars, "offset_unit": "unicode_code_point", "scope": "entity-identity-v2",
                  "identity_algorithm": algo_version}
        if algo_version == RESOLUTION_VERSION:
            config["identity_rounds"] = identity_rounds
            config["group_candidates"] = group_candidates
        run_id = digest([source["id"], config])
        store.execute("INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], dumps(config), "{}", started, started))
        models = Models(store, run_id, invocation, services, discovery_concurrency, decision_concurrency, budget=budget)
        log(f"运行编号：{run_id}\n原文：{source['metadata']['code_points']} 个码点；开始处理 {input_path.name}")
        state = store.run(run_id)["state"]
        for name in ("chapters", "discovery", "identity"):
            tick = time.monotonic()
            reused = state.get(name, {}).get("complete", False)
            if reused:
                log(f"复用已完成阶段：{name}")
                outcome = state[name]
            elif name == "chapters":
                outcome = detect_chapters(store, models, source, run_id, log, concurrency=discovery_concurrency)
                outcome.pop("chapters", None)
            elif name == "discovery":
                outcome = discover(store, models, source, run_id, store.records(run_id, "chapter"), task_chars, discovery_concurrency, log)
            elif algo_version == ROSTER_VERSION:
                outcome = resolve_roster(store, models, source, run_id,
                                         concurrency=discovery_concurrency, log=log, task_chars=task_chars)
            else:
                outcome = resolve_groups(store, models, source, run_id, decision_concurrency, log,
                                         identity_rounds, group_candidates)
            stages[name] = {"seconds": round(time.monotonic() - tick, 4), "resumed": reused}
            state[name] = outcome
            store.state(run_id, **{name: outcome})
            if not outcome["complete"] or name == stop_after:
                break
    except Exception as exc:
        caught_exc = exc
        if not run_id:
            store.close()
            raise
    if run_id is None:
        store.close()
        raise ValueError("尚未建立运行")
    if budget.stop_reason == "network_outage" or (isinstance(caught_exc, BudgetError) and caught_exc.category == "network"):
        error: dict | None = {
            "category": "network",
            "stop_reason": "network_outage",
            "message": f"连续网络失败停止（network_outage）：连续 {budget.consecutive_network_failures} 次传输失败；保留未完成断点以便恢复",
            "retryable": True,
        }
    elif isinstance(caught_exc, RuntimeError) and getattr(caught_exc, "code", None) == "network_error":
        error = {
            "category": "network",
            "stop_reason": "network_error",
            "message": str(caught_exc),
            "retryable": True,
        }
    elif budget.stop_reason in {"request_budget", "time_budget"} or isinstance(caught_exc, BudgetError):
        reason = budget.stop_reason or getattr(caught_exc, "stop_reason", None) or "request_budget"
        error = {
            "category": "budget",
            "stop_reason": reason,
            "message": f"请求或时间预算停止（{reason}）：本次已发起 {budget.count} 次请求；保留未完成断点以便恢复",
            "retryable": True,
        }
    elif caught_exc is not None:
        error = classify_exception(caught_exc)
    else:
        error = None
    invocation_record = {"id": invocation, "started": started, "ended": utc_now(), "seconds": round(time.monotonic() - begin, 4),
                         "stages": stages, "model_metrics": metrics(store, run_id, invocation),
                         "budget": budget.summary(),
                         "peak_active_requests": models.peaks if 'models' in locals() else {}, "error": error}
    store.put(run_id, "invocation", invocation_record)
    store.state(run_id, last_invocation=invocation, last_error=error)
    try:
        export_begin = time.monotonic()
        bundle = build_export_bundle(store, run_id)
        output = write_export_bundle(store, run_id, bundle, output_root)
        invocation_record.update(export_seconds=round(time.monotonic() - export_begin, 4),
                                 wall_seconds=round(time.monotonic() - wall_begin, 4),
                                 timing_scope="从管线函数入口、数据库初始化至主输出导出完成；不含最终计时回执写入与进程启动退出",
                                 finished_output_at=utc_now())
        store.put(run_id, "invocation", invocation_record)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["invocations"] = store.records(run_id, "invocation")
        write_json(output / "manifest.json", manifest)
        state = store.run(run_id)["state"]
        complete = budget.stop_reason is None and all(state.get(name, {}).get("complete", False) for name in ("chapters", "discovery", "identity"))
        if receipt_out is not None:
            resolved_out = output.resolve()
            stages_receipt = {
                name: {
                    "complete": bool(state.get(name, {}).get("complete", False)),
                    "seconds": float(stages.get(name, {}).get("seconds", 0.0)),
                    "resumed": bool(stages.get(name, {}).get("resumed", False)),
                }
                for name in ("chapters", "discovery", "identity")
            }
            if isinstance(error, dict):
                receipt_error: dict | None = make_error_dict(error["category"], error["message"])
            elif not complete:
                halted = getattr(models, "halted", None) if 'models' in locals() else None
                if halted:
                    halt_msg, _, _ = next(iter(halted.values()))
                    receipt_error = make_error_dict("quota", halt_msg)
                else:
                    receipt_error = infer_incomplete_stage_error(state, stop_after=stop_after)
            else:
                receipt_error = None
            pending_reviews = [
                r for r in bundle["review"]["items"]
                if r.get("status", "pending") == "pending"
            ]
            pending_reviews.sort(
                key=lambda r: (-(r["impact"] if isinstance(r.get("impact"), int) else 1), r["id"])
            )
            review_top = [
                {
                    "id": r["id"],
                    "stage": r.get("stage", ""),
                    "reason": r.get("reason", ""),
                    "impact": r["impact"] if isinstance(r.get("impact"), int) else 1,
                }
                for r in pending_reviews[:10]
            ]
            receipt_out.clear()
            receipt_out.update({
                "protocol": PROTOCOL,
                "command": "run",
                "run_id": run_id,
                "result_revision": bundle["result"]["result_revision"],
                "status": bundle["result"]["status"],
                "stages": stages_receipt,
                "output_dir": str(resolved_out),
                "files": {
                    "result": str(resolved_out / "result.json"),
                    "review": str(resolved_out / "review.json"),
                    "manifest": str(resolved_out / "manifest.json"),
                },
                "counts": {
                    "mentions": manifest["counts"]["mentions"],
                    "entities": manifest["counts"]["entities"],
                    "review_items": manifest["counts"]["review_items"],
                },
                "review_top": review_top,
                "error": receipt_error,
            })
        log(f"输出目录：{output.resolve()}\n本次主输出完成用时：{invocation_record['wall_seconds']:.2f} 秒；状态：{'流程完成，查看待审清单' if complete else '未完成，可恢复'}")
        if error:
            msg = f"[{error['category']}] {error['message']}" if isinstance(error, dict) else str(error)
            log("错误：" + msg)
        return run_id, output, complete
    except Exception as exc:
        exc.et_run_id = run_id  # type: ignore[attr-defined]
        raise
    finally:
        store.close()
