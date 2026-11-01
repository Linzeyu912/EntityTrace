from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corrections import apply_correction
from .pipeline import export_run, run_pipeline
from .storage import Store
from .text import address, validate_span


def _add_correct_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["user", "agent"], default=argparse.SUPPRESS,
                        help="更正来源：user（立即生效）或 agent（仅写入建议）（必填）")
    parser.add_argument("--expect-revision", default=argparse.SUPPRESS,
                        help="期望的当前 result_revision，不一致时拒绝写入（必填）")
    parser.add_argument("--resolves", default=argparse.SUPPRESS, help="本次更正对应解决的待审项 review_id")
    parser.add_argument("--note", default=argparse.SUPPRESS, help="更正说明或依据备注")
    parser.add_argument("--db", type=Path, default=argparse.SUPPRESS, help="数据库路径，默认 data/entitytrace.sqlite3")
    parser.add_argument("--output", type=Path, default=argparse.SUPPRESS, help="输出根目录，默认 outputs")


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
    run.add_argument("--identity-algorithm", choices=["roster", "groups-v2"], default="roster",
                     help="身份归组算法：roster（默认，entity-roster-v1 实体名册链接）或 groups-v2（旧版实体组消歧）")
    run.add_argument("--identity-rounds", type=int, default=12,
                     help="实体组消歧最大轮次，默认 12（仅对 --identity-algorithm groups-v2 生效）")
    run.add_argument("--group-candidates", type=int, default=3,
                     help="每次最多比较的候选组数，默认 3（仅对 --identity-algorithm groups-v2 生效）")
    run.add_argument("--max-requests", type=int, default=None, help="本次调用最大新网络请求预算，默认不限；超限保留断点并以退出码 2 停止")
    run.add_argument("--max-seconds", type=float, default=None, help="本次调用最大运行秒数预算，默认不限；超限保留断点并以退出码 2 停止")
    export = subs.add_parser("export", help="从已保存数据重新导出，不调用模型")
    export.add_argument("run_id")
    export.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    export.add_argument("--output", type=Path, default=Path("outputs"))
    quote = subs.add_parser("quote", help="按全文码点地址取回原文，不调用模型")
    quote.add_argument("run_id")
    quote.add_argument("start", type=int)
    quote.add_argument("end", type=int)
    quote.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))

    correct = subs.add_parser("correct", help="对已完成的名册运行叠加确定性更正或撤回，不调用模型")
    correct.add_argument("run_id")
    _add_correct_common_args(correct)
    op_common = argparse.ArgumentParser(add_help=False)
    _add_correct_common_args(op_common)
    correct_ops = correct.add_subparsers(dest="op", required=True)

    c_merge = correct_ops.add_parser("merge", parents=[op_common], help="将被并实体的全部提及并入目标实体（保留目标实体 ID）")
    c_merge.add_argument("target_entity_id")
    c_merge.add_argument("merged_entity_id")

    c_split = correct_ops.add_parser("split-surface", parents=[op_common], help="将实体中指定称呼的提及拆为新实体")
    c_split.add_argument("entity_id")
    c_split.add_argument("surface")

    c_assign = correct_ops.add_parser("assign", parents=[op_common], help="将单个提及指派到指定实体或新建实体（new）")
    c_assign.add_argument("mention_id")
    c_assign.add_argument("target")

    c_not_ent = correct_ops.add_parser("not-entity", parents=[op_common], help="将单个提及确认为非实体")
    c_not_ent.add_argument("mention_id")

    c_revoke = correct_ops.add_parser("revoke", parents=[op_common], help="撤回指定更正记录")
    c_revoke.add_argument("correction_id")
    c_revoke.add_argument("--cascade", action="store_true", help="一并撤回依赖该更正的后续更正")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _, _, complete = run_pipeline(args.input, args.db, args.output, args.config, args.discovery_concurrency,
                                         args.decision_concurrency, args.task_chars, args.encoding, args.stop_after,
                                         log=lambda message: print(message, flush=True),
                                         identity_rounds=args.identity_rounds, group_candidates=args.group_candidates,
                                         max_requests=args.max_requests, max_seconds=args.max_seconds,
                                         identity_algorithm=args.identity_algorithm)
            return 0 if complete else 2
        db_path = getattr(args, "db", Path("data/entitytrace.sqlite3"))
        store = Store(db_path)
        try:
            if args.command == "export":
                print(export_run(store, args.run_id, args.output).resolve())
            elif args.command == "correct":
                if args.op == "merge":
                    op_args = {"target_entity_id": args.target_entity_id, "merged_entity_id": args.merged_entity_id}
                elif args.op == "split-surface":
                    op_args = {"entity_id": args.entity_id, "surface": args.surface}
                elif args.op == "assign":
                    op_args = {"mention_id": args.mention_id, "target": args.target}
                elif args.op == "not-entity":
                    op_args = {"mention_id": args.mention_id}
                else:
                    op_args = {"correction_id": args.correction_id}
                receipt = apply_correction(
                    store,
                    args.run_id,
                    args.op,
                    op_args,
                    source=getattr(args, "source", ""),
                    expect_revision=getattr(args, "expect_revision", ""),
                    output_root=getattr(args, "output", Path("outputs")),
                    resolves=getattr(args, "resolves", None),
                    note=getattr(args, "note", None),
                    cascade=getattr(args, "cascade", False),
                )
                print(json.dumps(receipt, ensure_ascii=False))
            else:
                source = store.source(store.run(args.run_id)["source_id"])
                print(validate_span(source["text"], address(source["id"], args.start, args.end), source["id"]))
        finally:
            store.close()
        return 0
    except Exception as exc:
        print("执行失败：" + (str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__), file=sys.stderr)
        return 2
