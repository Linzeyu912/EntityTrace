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

from entitytrace.chapters import detect_chapters
from entitytrace.config import load_services
from entitytrace.discovery import discover, plan_tasks
from entitytrace.models import ExperimentBudget, Models, RequestBudget
from entitytrace.pipeline import export_run, metrics
from entitytrace.roster import DIRECT_VERSION, VERSION, resolve_roster, resolve_surface_direct
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import digest, dumps, implementation_hash, utc_now, write_json


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
    ch_complete = bool(run_state.get("chapters", {}).get("complete", False))
    disc_complete = bool(run_state.get("discovery", {}).get("complete", False))
    if ch_complete and disc_complete:
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
    src_state = src_bundle.get("source_state", {})
    ch_state = dict(src_state.get("chapters", {}))
    disc_state = dict(src_state.get("discovery", {}))
    ch_state["complete"] = bool(ch_state.get("complete", False))
    ch_state["reused_from"] = src_bundle["source_run_id"]
    disc_state["complete"] = bool(disc_state.get("complete", False))
    disc_state["reused_from"] = src_bundle["source_run_id"]
    initial_state = {
        "chapters": ch_state,
        "discovery": disc_state,
    }
    store.execute(
        "INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?)",
        (exp_run_id, source["id"], dumps(exp_config), dumps(initial_state), started, started),
    )
    store.state(exp_run_id, **initial_state)
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
    roster_out: dict | None,
    output_dir: Path,
    complete: bool,
    identity_seconds: float,
    wall_seconds: float,
    budget: RequestBudget,
    invocation_id: str,
    error_msg: str | None,
    chapters_seconds: float = 0.0,
    discovery_seconds: float = 0.0,
    full_cold_pipeline: bool = False,
) -> dict:
    """委托生产 export_run 统一导出 schema_version=3 产物，不保留第二份导出实现。"""
    if roster_out is not None and not store.records(exp_run_id, "roster_output"):
        store.put(exp_run_id, "roster_output", {"id": "roster_output", "start": 0, **roster_out})
    store.state(exp_run_id, identity={"complete": complete, "algorithm": exp_config["identity_algorithm"]})
    inv_record = {
        "id": invocation_id,
        "started": utc_now(),
        "seconds": round(chapters_seconds + discovery_seconds + identity_seconds, 4),
        "wall_seconds": round(wall_seconds, 4),
        "stages": {
            "chapters": {"seconds": round(chapters_seconds, 4), "resumed": not full_cold_pipeline},
            "discovery": {"seconds": round(discovery_seconds, 4), "resumed": not full_cold_pipeline},
            "identity": {"seconds": round(identity_seconds, 4), "resumed": False, "complete": complete},
        },
        "model_metrics": metrics(store, exp_run_id, invocation_id),
        "budget": budget.summary(),
        "error": error_msg,
    }
    store.put(exp_run_id, "invocation", inv_record)
    store.state(exp_run_id, last_invocation=invocation_id, last_error=error_msg)
    out_path = export_run(store, exp_run_id, output_dir, output_dir=output_dir)
    manifest_doc = json.loads((out_path / "manifest.json").read_text(encoding="utf-8"))
    manifest_doc["budget"] = inv_record["budget"]
    write_json(out_path / "manifest.json", manifest_doc)
    return manifest_doc


def run_experiment(args: argparse.Namespace) -> int:
    wall_begin = time.perf_counter()
    services = load_services(args.config)
    store = Store(args.db)
    budget = RequestBudget(args.max_requests, args.max_seconds)
    invocation_id = uuid.uuid4().hex
    error_msg = None
    complete = False
    roster_out = None
    src_bundle = None
    chapters_seconds = 0.0
    discovery_seconds = 0.0
    id_seconds = 0.0
    full_cold_pipeline = args.source_file is not None

    try:
        if args.source_file is not None:
            source = import_text(store, args.source_file)
            exp_config = {
                "implementation": implementation_hash(),
                "arm": args.arm,
                "identity_algorithm": DIRECT_VERSION if args.arm == "B" else VERSION,
                "source_run_id": None,
                "task_chars": args.task_chars,
                "discovery_concurrency": args.concurrency,
                "models": {k: v.public() for k, v in services.items()},
                "budget": {"max_requests": args.max_requests, "max_seconds": args.max_seconds},
                "mode": "cold_full_pipeline",
            }
            exp_run_id = digest(["roster-full-v1", source["id"], args.arm])
            started = utc_now()
            store.execute(
                "INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?)",
                (exp_run_id, source["id"], dumps(exp_config), "{}", started, started),
            )
            models = Models(
                store,
                exp_run_id,
                invocation_id,
                services,
                discovery_concurrency=args.concurrency,
                decision_concurrency=3,
                budget=budget,
            )
            log_fn = lambda msg: print(msg, flush=True)
            state = store.run(exp_run_id)["state"]
            if not state.get("chapters", {}).get("complete", False):
                ch_t0 = time.perf_counter()
                ch_out = detect_chapters(store, models, source, exp_run_id, log_fn)
                ch_out.pop("chapters", None)
                chapters_seconds = time.perf_counter() - ch_t0
                store.state(exp_run_id, chapters=ch_out)
                if not ch_out.get("complete", False) or not store.records(exp_run_id, "chapter") or ch_out.get("scanned_code_points") != len(source["text"]):
                    raise RuntimeError("章节阶段未完成")
            if not state.get("discovery", {}).get("complete", False):
                disc_t0 = time.perf_counter()
                disc_out = discover(
                    store,
                    models,
                    source,
                    exp_run_id,
                    store.records(exp_run_id, "chapter"),
                    args.task_chars,
                    args.concurrency,
                    log_fn,
                )
                discovery_seconds = time.perf_counter() - disc_t0
                store.state(exp_run_id, discovery=disc_out)
                if not disc_out.get("complete"):
                    raise RuntimeError("发现阶段未完成")
            src_bundle = {
                "source": source,
                "source_run_id": exp_run_id,
                "source_config": exp_config,
                "source_state": store.run(exp_run_id)["state"],
                "task_chars": args.task_chars,
                "chapters": store.records(exp_run_id, "chapter"),
                "mentions": store.records(exp_run_id, "mention"),
                "observations": store.records(exp_run_id, "observation"),
                "links": store.records(exp_run_id, "link"),
                "reviews": store.records(exp_run_id, "review"),
            }
        else:
            src_bundle = read_source_run(args.source_db, args.source_run)
            exp_config = {
                "implementation": implementation_hash(),
                "arm": args.arm,
                "identity_algorithm": DIRECT_VERSION if args.arm == "B" else VERSION,
                "source_run_id": args.source_run,
                "task_chars": src_bundle["task_chars"],
                "discovery_concurrency": args.concurrency,
                "models": {k: v.public() for k, v in services.items()} if args.arm == "C" else {},
                "budget": {"max_requests": args.max_requests, "max_seconds": args.max_seconds},
                "mode": "recompute_from_cache" if args.recompute_from_cache else "experiment",
            }
            exp_run_id = digest(["roster-exp-v1", src_bundle["source"]["id"], args.source_run, args.arm])
            seed_experiment_store(store, src_bundle, exp_run_id, exp_config)
            exp_state = store.run(exp_run_id)["state"]
            if not exp_state.get("chapters", {}).get("complete", False):
                raise RuntimeError("章节阶段未完成")
            if not exp_state.get("discovery", {}).get("complete", False):
                raise RuntimeError("发现阶段未完成")
            if args.recompute_from_cache:
                with store.lock, store.db:
                    store.db.execute("DELETE FROM tasks WHERE run_id=? AND stage LIKE 'roster%'", (exp_run_id,))
                    store.db.execute("DELETE FROM calls WHERE run_id=?", (exp_run_id,))
                    store.db.execute(
                        "DELETE FROM records WHERE run_id=? AND kind IN ('roster_output', 'invocation')",
                        (exp_run_id,),
                    )
            models = Models(
                store,
                exp_run_id,
                invocation_id,
                services,
                discovery_concurrency=args.concurrency,
                decision_concurrency=3,
                budget=budget,
            )

        id_begin = time.perf_counter()
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
        id_seconds = time.perf_counter() - id_begin
    except (ExperimentBudget, Exception) as exc:
        error_msg = str(exc)
        complete = False
        if "id_begin" in locals() and id_seconds == 0.0:
            id_seconds = time.perf_counter() - id_begin

    try:
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
            chapters_seconds=chapters_seconds,
            discovery_seconds=discovery_seconds,
            full_cold_pipeline=full_cold_pipeline,
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
                "chapters_seconds": round(chapters_seconds, 4),
                "discovery_seconds": round(discovery_seconds, 4),
                "identity_seconds": manifest["identity_seconds"],
                "wall_seconds": round(wall_seconds, 4),
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
    parser.add_argument("--source-run", type=str, default=None)
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--task-chars", type=int, default=9000)
    parser.add_argument("--recompute-from-cache", action="store_true")
    parser.add_argument("--arm", choices=["B", "C"], required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / ".env.models.local")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=2400)
    args = parser.parse_args()
    if not args.source_run and not args.source_file:
        parser.error("必须提供 --source-run 或 --source-file")
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
