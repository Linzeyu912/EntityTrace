"""名册链接（C 组）与称呼直连（B 组）对照实验工具：只读复用源数据库发现结果，在独立冷缓存实验库执行。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entitytrace.config import load_services
from entitytrace.discovery import plan_tasks
from entitytrace.models import Models
from entitytrace.pipeline import metrics
from entitytrace.roster import DIRECT_VERSION, VERSION, resolve_roster, resolve_surface_direct
from entitytrace.storage import Store
from entitytrace.text import validate_span
from entitytrace.util import digest, dumps, implementation_hash, utc_now, write_bytes, write_json
from tools.bounded_identity_run import RequestBudget
from tools.identity_calibration import ExperimentBudget


def read_source_run(source_db: Path, source_run_id: str) -> dict:
    uri = f"file:{source_db.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        run_rows = conn.execute("SELECT * FROM runs WHERE id=?", (source_run_id,)).fetchall()
        if not run_rows:
            raise ValueError(f"源数据库中不存在运行编号：{source_run_id}")
        run_row = dict(run_rows[0])
        run_config = json.loads(run_row["config"])
        run_state = json.loads(run_row["state"])

        src_rows = conn.execute("SELECT * FROM sources WHERE id=?", (run_row["source_id"],)).fetchall()
        if not src_rows:
            raise ValueError("源数据库中不存在对应原文")
        src_row = dict(src_rows[0])
        source = {
            "id": src_row["id"],
            "metadata": json.loads(src_row["metadata"]),
            "text": src_row["text"],
            "raw": src_row["raw"],
        }

        def get_records(kind: str) -> list[dict]:
            rows = conn.execute(
                "SELECT data FROM records WHERE run_id=? AND kind=? ORDER BY start,id",
                (source_run_id, kind),
            ).fetchall()
            return [json.loads(r["data"]) for r in rows]

        chapters = get_records("chapter")
        mentions = get_records("mention")
        observations = get_records("observation")
        links = get_records("link")
        reviews = get_records("review")

        saved_discovery_tasks = conn.execute(
            "SELECT id, status FROM tasks WHERE run_id=? AND stage='discovery'",
            (source_run_id,),
        ).fetchall()
        saved_task_ids = {r["id"] for r in saved_discovery_tasks if r["status"] == "completed"}
    finally:
        conn.close()

    task_chars = int(run_config.get("task_chars", 9000))
    rebuilt_tasks = plan_tasks(source, chapters, task_chars)
    rebuilt_ids = {t["id"] for t in rebuilt_tasks}
    if rebuilt_ids != saved_task_ids:
        raise ValueError(
            f"重建发现任务 ID 与源运行不一致：rebuilt={len(rebuilt_ids)}, saved={len(saved_task_ids)}"
        )

    return {
        "source": source,
        "source_run_id": source_run_id,
        "source_config": run_config,
        "source_state": run_state,
        "task_chars": task_chars,
        "chapters": chapters,
        "mentions": mentions,
        "observations": observations,
        "links": links,
        "reviews": reviews,
    }


def seed_experiment_store(store: Store, src_bundle: dict, exp_run_id: str, exp_config: dict) -> None:
    source = src_bundle["source"]
    started = utc_now()
    store.execute(
        "INSERT OR IGNORE INTO sources VALUES(?,?,?,?)",
        (source["id"], dumps(source["metadata"]), source["text"], source["raw"]),
    )
    initial_state = {
        "chapters": {"complete": True, "reused_from": src_bundle["source_run_id"]},
        "discovery": {"complete": True, "reused_from": src_bundle["source_run_id"]},
    }
    store.execute(
        "INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?)",
        (exp_run_id, source["id"], dumps(exp_config), dumps(initial_state), started, started),
    )
    # 批量写入复用的章节、提及、观察、关系线索和定位待审
    existing_mentions = store.rows("SELECT COUNT(*) AS n FROM records WHERE run_id=? AND kind='mention'", (exp_run_id,))[0]["n"]
    if existing_mentions == 0:
        grounding_reviews = [r for r in src_bundle["reviews"] if r.get("stage") in {"grounding", "discovery"}]
        with store.lock, store.db:
            for kind, items in [
                ("chapter", src_bundle["chapters"]),
                ("mention", src_bundle["mentions"]),
                ("observation", src_bundle["observations"]),
                ("link", src_bundle["links"]),
                ("review", grounding_reviews),
            ]:
                for rec in items:
                    store.db.execute(
                        "INSERT OR REPLACE INTO records VALUES(?,?,?,?,?)",
                        (exp_run_id, kind, rec["id"], rec.get("start"), dumps(rec)),
                    )


def export_experiment(
    store: Store,
    src_bundle: dict,
    exp_run_id: str,
    exp_config: dict,
    roster_out: dict,
    output_dir: Path,
    complete: bool,
    identity_seconds: float,
    wall_seconds: float,
    budget: RequestBudget,
    invocation_id: str,
    error_msg: str | None,
) -> dict:
    source = src_bundle["source"]
    text, metadata = source["text"], source["metadata"]
    chapters = src_bundle["chapters"]
    observations = src_bundle["observations"]
    grounding_reviews = [r for r in src_bundle["reviews"] if r.get("stage") in {"grounding", "discovery"}]

    entities = roster_out["entities"]
    mentions = roster_out["mentions"]
    identity_reviews = [r for r in roster_out["reviews"] if r.get("stage") not in {"grounding", "discovery"}]
    all_reviews = identity_reviews + grounding_reviews

    by_chapter = {c["id"]: c for c in chapters}
    for m in mentions:
        validate_span(text, m["address"], source["id"], m["text"])
        ch = by_chapter[m["chapter_id"]]
        m["chapter_start"] = m["start"] - ch["start"]
        m["chapter_end"] = m["end"] - ch["start"]
        for span in m["evidence"]:
            validate_span(text, span, source["id"])

    ids = {m["id"] for m in mentions}
    if sorted(mid for e in entities for mid in e["mention_ids"]) != sorted(ids):
        raise ValueError("实验导出校验失败：实体分组未恰好覆盖所有提及")

    status = "incomplete" if not complete else ("completed_with_review" if all_reviews else "completed")
    result_doc = {
        "schema_version": 3,
        "run_id": exp_run_id,
        "source_run_id": src_bundle["source_run_id"],
        "status": status,
        "source": metadata,
        "chapters": chapters,
        "mentions": mentions,
        "observations": observations,
        "entities": entities,
        "roster_summary": roster_out["roster_summary"],
        "ambiguous_surfaces": roster_out["ambiguous_surfaces"],
        "scope": "实体提及与名册链接身份归组对照实验；复用已完成发现结果",
    }
    result_doc["result_revision"] = digest(result_doc)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result_doc)
    write_json(
        output_dir / "review.json",
        {
            "schema_version": 1,
            "run_id": exp_run_id,
            "result_revision": result_doc["result_revision"],
            "items": all_reviews,
            "counts": {
                "total": len(all_reviews),
                "identity_reviews": len(identity_reviews),
                "grounding_discovery_reviews": len(grounding_reviews),
            },
            "instruction": "按 impact 降序排列；自动模型判断不冒充人工答案。",
        },
    )

    requests = store.rows(
        "SELECT DISTINCT r.* FROM request_cache r JOIN calls c ON r.id=c.request_id WHERE c.run_id=? ORDER BY r.id",
        (exp_run_id,),
    )
    write_bytes(
        output_dir / "requests.jsonl",
        "".join(
            dumps({
                "id": r["id"],
                "service": r["service"],
                "model": r["model"],
                "request": json.loads(r["request"]),
                "response": json.loads(r["response"]),
            })
            + "\n"
            for r in requests
        ).encode("utf-8"),
    )

    total_model_metrics = metrics(store, exp_run_id)
    inv_record = {
        "id": invocation_id,
        "started": utc_now(),
        "seconds": round(identity_seconds, 4),
        "wall_seconds": round(wall_seconds, 4),
        "stages": {
            "chapters": {"seconds": 0.0, "resumed": True},
            "discovery": {"seconds": 0.0, "resumed": True},
            "identity": {"seconds": round(identity_seconds, 4), "resumed": False, "complete": complete},
        },
        "model_metrics": metrics(store, exp_run_id, invocation_id),
        "budget": {
            "max_requests": budget.max_requests,
            "max_seconds": budget.max_seconds,
            "network_attempts": budget.count,
            "stop_reason": budget.stop_reason,
            "network_errors": dict(budget.network_errors),
        },
        "error": error_msg,
    }
    store.put(exp_run_id, "invocation", inv_record)
    invocations = store.records(exp_run_id, "invocation")
    total_identity_seconds = round(
        sum(
            inv.get("stages", {}).get("identity", {}).get("seconds", 0.0)
            for inv in invocations
            if not inv.get("stages", {}).get("identity", {}).get("resumed", False)
        ),
        4,
    )

    manifest_doc = {
        "schema_version": 2,
        "run_id": exp_run_id,
        "source_run_id": src_bundle["source_run_id"],
        "arm": exp_config["arm"],
        "result_revision": result_doc["result_revision"],
        "status": status,
        "source": metadata,
        "config": exp_config,
        "invocations": invocations,
        "identity_seconds": total_identity_seconds,
        "total_model_metrics": total_model_metrics,
        "counts": {
            "chapters": sum(c["kind"] == "chapter" for c in chapters),
            "sections": len(chapters),
            "mentions": len(mentions),
            "observations": len(observations),
            "entities": len(entities),
            "singletons": sum(1 for e in entities if len(e["mention_ids"]) == 1),
            "review_items": len(all_reviews),
            "identity_review_items": len(identity_reviews),
            "grounding_discovery_review_items": len(grounding_reviews),
        },
        "budget": inv_record["budget"],
        "files": ["result.json", "review.json", "requests.jsonl", "manifest.json"],
    }
    write_json(output_dir / "manifest.json", manifest_doc)
    return manifest_doc


def run_experiment(args: argparse.Namespace) -> int:
    wall_begin = time.perf_counter()
    src_bundle = read_source_run(args.source_db, args.source_run)
    services = load_services(args.config)

    exp_config = {
        "implementation": implementation_hash(),
        "arm": args.arm,
        "identity_algorithm": DIRECT_VERSION if args.arm == "B" else VERSION,
        "source_run_id": args.source_run,
        "task_chars": src_bundle["task_chars"],
        "discovery_concurrency": args.concurrency,
        "models": {k: v.public() for k, v in services.items()} if args.arm == "C" else {},
        "budget": {"max_requests": args.max_requests, "max_seconds": args.max_seconds},
    }
    exp_run_id = digest(["roster-exp-v1", src_bundle["source"]["id"], args.source_run, args.arm])

    store = Store(args.db)
    budget = RequestBudget(args.max_requests, args.max_seconds)
    invocation_id = uuid.uuid4().hex
    error_msg = None
    complete = False
    roster_out = None
    id_begin = time.perf_counter()

    try:
        seed_experiment_store(store, src_bundle, exp_run_id, exp_config)
        if args.arm == "B":
            roster_out = resolve_surface_direct(
                src_bundle["source"],
                src_bundle["chapters"],
                src_bundle["mentions"],
                src_bundle["links"],
                src_bundle["reviews"],
            )
            complete = True
        else:
            models = Models(
                store,
                exp_run_id,
                invocation_id,
                services,
                discovery_concurrency=args.concurrency,
                decision_concurrency=3,
                opener=budget,
            )
            outcome = resolve_roster(
                store,
                models,
                src_bundle["source"],
                exp_run_id,
                concurrency=args.concurrency,
                log=lambda msg: print(msg, flush=True),
                task_chars=src_bundle["task_chars"],
            )
            complete = bool(outcome.get("complete"))
            if complete:
                saved = store.records(exp_run_id, "roster_output")
                roster_out = saved[0]
            else:
                error_msg = f"阶段 {outcome.get('stage')} 未完成，失败数：{len(outcome.get('failures', []))}"
    except (ExperimentBudget, Exception) as exc:
        error_msg = str(exc)
        complete = False
    finally:
        id_seconds = time.perf_counter() - id_begin

    try:
        if roster_out is None:
            # 未完成时按已完成部分或暂定单项导出 incomplete 快照，保留断点
            fallback = resolve_surface_direct(
                src_bundle["source"],
                src_bundle["chapters"],
                src_bundle["mentions"],
                [],
                src_bundle["reviews"],
            )
            roster_out = fallback
        wall_seconds = time.perf_counter() - wall_begin
        manifest = export_experiment(
            store,
            src_bundle,
            exp_run_id,
            exp_config,
            roster_out,
            args.output,
            complete,
            id_seconds,
            wall_seconds,
            budget,
            invocation_id,
            error_msg,
        )
    finally:
        store.close()

    print(
        json.dumps(
            {
                "arm": args.arm,
                "run_id": exp_run_id,
                "status": manifest["status"],
                "entities": manifest["counts"]["entities"],
                "identity_review_items": manifest["counts"]["identity_review_items"],
                "total_review_items": manifest["counts"]["review_items"],
                "identity_seconds": manifest["identity_seconds"],
                "network_attempts": manifest["total_model_metrics"]["network_attempts"],
                "cache_hits": manifest["total_model_metrics"]["cache_hits"],
                "budget_stop_reason": budget.stop_reason,
                "error": error_msg,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if complete else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=ROOT / "data/entitytrace.sqlite3")
    parser.add_argument("--source-run", type=str, required=True)
    parser.add_argument("--arm", choices=["B", "C"], required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / ".env.models.local")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=2400)
    args = parser.parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
