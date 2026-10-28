from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import export_run, run_pipeline
from .storage import Store
from .text import address, validate_span


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="EntityTrace：发现实体提及、按证据判断身份并导出 JSON")
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run", help="处理全文；相同输入及配置自动复用断点")
    run.add_argument("input", type=Path)
    run.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    run.add_argument("--output", type=Path, default=Path("outputs"))
    run.add_argument("--config", type=Path, default=Path(".env.models.local"))
    run.add_argument("--encoding", default="utf-8")
    run.add_argument("--discovery-concurrency", type=int, default=6)
    run.add_argument("--decision-concurrency", type=int, default=3)
    run.add_argument("--task-chars", type=int, default=9000)
    run.add_argument("--stop-after", choices=["chapters", "discovery", "identity"], default="identity")
    export = subs.add_parser("export", help="从已保存数据重新导出，不调用模型")
    export.add_argument("run_id")
    export.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    export.add_argument("--output", type=Path, default=Path("outputs"))
    quote = subs.add_parser("quote", help="按全文码点地址取回原文，不调用模型")
    quote.add_argument("run_id")
    quote.add_argument("start", type=int)
    quote.add_argument("end", type=int)
    quote.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _, _, complete = run_pipeline(args.input, args.db, args.output, args.config, args.discovery_concurrency,
                                         args.decision_concurrency, args.task_chars, args.encoding, args.stop_after,
                                         log=lambda message: print(message, flush=True))
            return 0 if complete else 2
        store = Store(args.db)
        try:
            if args.command == "export":
                print(export_run(store, args.run_id, args.output).resolve())
            else:
                source = store.source(store.run(args.run_id)["source_id"])
                print(validate_span(source["text"], address(source["id"], args.start, args.end), source["id"]))
        finally:
            store.close()
        return 0
    except Exception as exc:
        print("执行失败：" + (str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__), file=sys.stderr)
        return 2
