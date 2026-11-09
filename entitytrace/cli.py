from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import OPENAI_DEFAULT_CONCURRENCY, load_services, read_env
from .corrections import apply_correction
from .describe import run_describe
from .pipeline import (
    EXIT_CODE_BY_CATEGORY,
    PROTOCOL,
    classify_exception,
    export_run,
    run_pipeline,
)
from .query import (
    DEFAULT_CONTEXT,
    DEFAULT_LIMIT,
    query_entities,
    query_entity,
    query_quote_json,
    query_rank,
    query_review,
    query_status,
)
from .storage import Store
from .text import address, validate_span

KNOWN_COMMANDS = frozenset({"run", "status", "entities", "entity", "review", "quote", "rank", "export", "correct", "describe"})
DEFAULT_JSON_COMMANDS = frozenset({"status", "entities", "entity", "review", "rank", "correct", "describe"})


class _CliHelpExit(Exception):
    """捕获 --help 正常退出，便于进程内调用返回 0。"""


class _CliUsageError(ValueError):
    """命令行参数错误，映射为 input 类别与退出码 1。"""

    et_category = "input"


class _CliParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CliUsageError(f"命令行参数错误：{message}")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            print(message, end="", file=sys.stderr)
        if status == 0:
            raise _CliHelpExit()
        raise _CliUsageError((message or "命令行参数错误").strip())


def _add_correct_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["user", "agent"], default=argparse.SUPPRESS,
                        help="更正来源：user（立即生效）或 agent（仅写入建议）（必填）")
    parser.add_argument("--expect-revision", default=argparse.SUPPRESS,
                        help="期望的当前 result_revision，不一致时拒绝写入（必填）")
    parser.add_argument("--resolves", default=argparse.SUPPRESS, help="本次更正对应解决的待审项 review_id")
    parser.add_argument("--note", default=argparse.SUPPRESS, help="更正说明或依据备注")
    parser.add_argument("--db", type=Path, default=argparse.SUPPRESS, help="数据库路径，默认 data/entitytrace.sqlite3")
    parser.add_argument("--output", type=Path, default=argparse.SUPPRESS, help="输出根目录，默认 outputs")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="输出单行 JSON 回执（默认已启用）")


def _detect_invocation_context(raw_argv: list[str]) -> tuple[str | None, str | None, bool]:
    cmd: str | None = None
    cmd_idx = -1
    for idx, tok in enumerate(raw_argv):
        if not tok.startswith("-"):
            if tok in KNOWN_COMMANDS:
                cmd = tok
                cmd_idx = idx
            break
    run_id: str | None = None
    if cmd and cmd != "run" and cmd_idx >= 0:
        for tok in raw_argv[cmd_idx + 1:]:
            if not tok.startswith("-"):
                run_id = tok
                break
    json_mode = ("--json" in raw_argv) or (cmd in DEFAULT_JSON_COMMANDS)
    return cmd, run_id, json_mode


def _emit_json_line(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _build_failed_run_receipt(err: dict) -> dict:
    return {
        "protocol": PROTOCOL,
        "command": "run",
        "run_id": None,
        "result_revision": None,
        "status": "failed",
        "stages": {
            name: {"complete": False, "seconds": 0.0, "resumed": False}
            for name in ("chapters", "discovery", "identity")
        },
        "output_dir": None,
        "files": None,
        "counts": None,
        "review_top": [],
        "error": err,
        "generator_provider": None,
        "generator_model": None,
        "generator_timeout_seconds": None,
        "discovery_concurrency": None,
    }


def _build_parser() -> _CliParser:
    parser = _CliParser(description="EntityTrace：发现实体提及、按证据判断身份并导出 JSON")
    subs = parser.add_subparsers(dest="command", required=True, parser_class=_CliParser)

    run = subs.add_parser("run", help="处理全文；相同输入及配置自动复用断点")
    run.add_argument("input", type=Path)
    run.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    run.add_argument("--output", type=Path, default=Path("outputs"))
    run.add_argument("--config", type=Path, default=Path(".env.models.local"))
    run.add_argument("--encoding", default="utf-8")
    run.add_argument("--generator", choices=["deepseek", "openai"], default=None,
                     help="生成服务商，默认读取 GENERATOR_PROVIDER 配置（缺省 deepseek）")
    run.add_argument("--discovery-concurrency", type=int, default=None,
                     help="发现阶段并发；未显式给出时 deepseek 为 6、openai 为 8")
    run.add_argument("--decision-concurrency", type=int, default=3)
    run.add_argument("--task-chars", type=int, default=9000)
    run.add_argument("--stop-after", choices=["chapters", "discovery", "identity"], default="identity")
    run.add_argument("--identity-algorithm", choices=["roster", "groups-v2"], default="roster",
                     help="身份归组算法：roster（默认，entity-roster-v1 实体名册链接）或 groups-v2（旧版实体组消歧）")
    run.add_argument("--identity-rounds", type=int, default=12,
                     help="实体组消歧最大轮次，默认 12（仅对 --identity-algorithm groups-v2 生效）")
    run.add_argument("--group-candidates", type=int, default=3,
                     help="每次最多比较的候选组数，默认 3（仅对 --identity-algorithm groups-v2 生效）")
    run.add_argument("--max-requests", type=int, default=None,
                     help="本次调用最大新网络请求预算，默认不限；超限保留断点并以退出码 2 停止")
    run.add_argument("--max-seconds", type=float, default=None,
                     help="本次调用最大运行秒数预算，默认不限；超限保留断点并以退出码 2 停止")
    run.add_argument("--json", action="store_true", help="将进度写往 stderr，并在 stdout 输出单行机器可读 JSON 回执")

    status = subs.add_parser("status", help="只读查询运行阶段状态、错误、计数与产物路径（默认 JSON，不调用模型）")
    status.add_argument("run_id")
    status.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    status.add_argument("--output", type=Path, default=Path("outputs"))
    status.add_argument("--json", action="store_true", help="输出单行 JSON（默认已启用）")

    entities = subs.add_parser("entities", help="只读分页查询实体名册，支持按粗类、最少提及数或精确称呼筛选（默认 JSON）")
    entities.add_argument("run_id")
    entities.add_argument("--kind", default=None, help="按实体粗类筛选")
    entities.add_argument("--min-mentions", type=int, default=None, help="最少提及次数下限")
    entities.add_argument("--surface", default=None, help="仅返回含该精确称呼提及的实体，并附 surface_count")
    entities.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="单页条数上限（1—200，默认 50）")
    entities.add_argument("--offset", type=int, default=0, help="分页起始偏移（默认 0）")
    entities.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    entities.add_argument("--json", action="store_true", help="输出单行 JSON（默认已启用）")

    entity = subs.add_parser("entity", help="只读查询单个实体的别名、链接方法分布、相关待审与提及分页片段（默认 JSON）")
    entity.add_argument("run_id")
    entity.add_argument("entity_id")
    entity.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="提及单页条数上限（1—200，默认 50）")
    entity.add_argument("--offset", type=int, default=0, help="提及分页起始偏移（默认 0）")
    entity.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="前后各取多少码点片段（0—400，默认 80，不跨章节边界）")
    entity.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    entity.add_argument("--json", action="store_true", help="输出单行 JSON（默认已启用）")

    review = subs.add_parser("review", help="只读分页查询按 impact 排序的待审项（默认 pending、JSON）")
    review.add_argument("run_id")
    review.add_argument("--status", default="pending", help="筛选状态：pending（默认）、resolved 或 all")
    review.add_argument("--stage", default=None, help="按产生阶段筛选（如 roster-c3-link、discovery 等）")
    review.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="单页条数上限（1—200，默认 50）")
    review.add_argument("--offset", type=int, default=0, help="分页起始偏移（默认 0）")
    review.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    review.add_argument("--json", action="store_true", help="输出单行 JSON（默认已启用）")

    rank = subs.add_parser("rank", help="只读实体优先级排名：按范围内提及数与章节数计算 score 排序（默认 JSON）")
    rank.add_argument("run_id")
    rank.add_argument("--kind", default=None, help="按实体粗类筛选（person/place/object/ability/concept/organization）")
    rank.add_argument("--from-chapter", type=int, default=None, help="正文章序起点（含，缺省与 --to-chapter 均未给时为全书）")
    rank.add_argument("--to-chapter", type=int, default=None, help="正文章序终点（含，只给 --from-chapter 时默认到最后一章）")
    rank.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="单页条数上限（1—200，默认 50）")
    rank.add_argument("--offset", type=int, default=0, help="分页起始偏移（默认 0）")
    rank.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    rank.add_argument("--json", action="store_true", help="输出单行 JSON（默认已启用）")

    describe = subs.add_parser("describe", help="抽取 top-K 实体的描述事实与状态变化事件（默认 JSON，调用模型）")
    describe.add_argument("run_id")
    describe.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    describe.add_argument("--method", choices=["d1", "d2"], required=True,
                          help="d1：按发现任务块抽取；d2：按 top-K 实体的提及窗口抽取")
    describe.add_argument("--top-k", type=int, default=20, help="每个实际类别取前多少实体（1—200，默认 20）")
    describe.add_argument("--kinds", default="person,place,object,ability",
                          help="逗号分隔的实体类别，默认 person,place,object,ability")
    describe.add_argument("--topk-file", type=Path, default=None,
                          help="预登记 top-K 名单 JSON；给出时不重新排名")
    describe.add_argument("--concurrency", type=int, default=8, help="同时进行的生成请求上限（1—32，默认 8）")
    describe.add_argument("--generator", choices=["deepseek", "openai"], default=None,
                          help="生成服务商，默认读取 GENERATOR_PROVIDER 配置（缺省 deepseek）")
    describe.add_argument("--config", type=Path, default=Path(".env.models.local"))
    describe.add_argument("--max-requests", type=int, default=None,
                          help="本次调用最大新网络请求预算，默认不限；超限保留断点并以退出码 2 停止")
    describe.add_argument("--max-seconds", type=float, default=None,
                          help="本次调用最大运行秒数预算，默认不限；超限保留断点并以退出码 2 停止")
    describe.add_argument("--out", type=Path, default=None, help="将描述事实导出 JSON 写入该路径")

    export = subs.add_parser("export", help="从已保存数据重新导出，不调用模型")
    export.add_argument("run_id")
    export.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))
    export.add_argument("--output", type=Path, default=Path("outputs"))

    quote = subs.add_parser("quote", help="按全文码点地址取回原文，不调用模型")
    quote.add_argument("run_id")
    quote.add_argument("start", type=int)
    quote.add_argument("end", type=int)
    quote.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="开启 --json 时前后各附带多少码点上下文（默认 80）")
    quote.add_argument("--json", action="store_true", help="输出带前后上下文、chapter_id 与 result_revision 的单行 JSON")
    quote.add_argument("--db", type=Path, default=Path("data/entitytrace.sqlite3"))

    correct = subs.add_parser("correct", help="对已完成的名册运行叠加确定性更正或撤回，不调用模型")
    correct.add_argument("run_id")
    _add_correct_common_args(correct)
    op_common = _CliParser(add_help=False)
    _add_correct_common_args(op_common)
    correct_ops = correct.add_subparsers(dest="op", required=True, parser_class=_CliParser)

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

    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    cmd_name, known_run_id, json_mode = _detect_invocation_context(raw_argv)
    parser = _build_parser()
    try:
        args = parser.parse_args(raw_argv)
        cmd_name = args.command
        known_run_id = getattr(args, "run_id", None)
        json_mode = bool(getattr(args, "json", False)) or (cmd_name in DEFAULT_JSON_COMMANDS)

        if args.command == "run":
            log_stream = sys.stderr if args.json else sys.stdout
            receipt: dict = {}
            discovery_concurrency = args.discovery_concurrency
            if discovery_concurrency is None:
                local_env = read_env(args.config)
                provider = (
                    args.generator
                    or os.environ.get("GENERATOR_PROVIDER", local_env.get("GENERATOR_PROVIDER", "deepseek"))
                ).strip()
                discovery_concurrency = OPENAI_DEFAULT_CONCURRENCY if provider == "openai" else 6
            _, _, complete = run_pipeline(
                args.input,
                args.db,
                args.output,
                args.config,
                discovery_concurrency,
                args.decision_concurrency,
                args.task_chars,
                args.encoding,
                args.stop_after,
                log=lambda message: print(message, file=log_stream, flush=True),
                identity_rounds=args.identity_rounds,
                group_candidates=args.group_candidates,
                max_requests=args.max_requests,
                max_seconds=args.max_seconds,
                identity_algorithm=args.identity_algorithm,
                generator=args.generator,
                receipt_out=receipt,
            )
            if args.json:
                _emit_json_line(receipt)
            if complete:
                return 0
            err_obj = receipt.get("error")
            if isinstance(err_obj, dict):
                return EXIT_CODE_BY_CATEGORY.get(err_obj.get("category", "model_output"), 2)
            return 2

        db_path = getattr(args, "db", Path("data/entitytrace.sqlite3"))
        if not db_path.exists() or not db_path.is_file():
            raise FileNotFoundError(f"数据库文件不存在：{db_path}")
        store = Store(db_path)
        try:
            if args.command == "status":
                _emit_json_line(query_status(store, args.run_id, args.output))
            elif args.command == "entities":
                _emit_json_line(
                    query_entities(
                        store,
                        args.run_id,
                        kind=args.kind,
                        min_mentions=args.min_mentions,
                        surface=args.surface,
                        limit=args.limit,
                        offset=args.offset,
                    )
                )
            elif args.command == "entity":
                _emit_json_line(
                    query_entity(
                        store,
                        args.run_id,
                        args.entity_id,
                        limit=args.limit,
                        offset=args.offset,
                        context=args.context,
                    )
                )
            elif args.command == "review":
                _emit_json_line(
                    query_review(
                        store,
                        args.run_id,
                        status=args.status,
                        stage=args.stage,
                        limit=args.limit,
                        offset=args.offset,
                    )
                )
            elif args.command == "rank":
                _emit_json_line(
                    query_rank(
                        store,
                        args.run_id,
                        kind=args.kind,
                        from_chapter=args.from_chapter,
                        to_chapter=args.to_chapter,
                        limit=args.limit,
                        offset=args.offset,
                    )
                )
            elif args.command == "describe":
                services = load_services(args.config, generator=args.generator)
                result = run_describe(
                    store,
                    args.run_id,
                    args.method,
                    services=services,
                    top_k=args.top_k,
                    kinds=args.kinds,
                    topk_file=args.topk_file,
                    concurrency=args.concurrency,
                    out=args.out,
                    max_requests=args.max_requests,
                    max_seconds=args.max_seconds,
                )
                _emit_json_line({
                    "ok": True,
                    "describe_id": result["describe_id"],
                    "run_id": result["run_id"],
                    "method": result["method"],
                    "counts": result["counts"],
                    "out": str(args.out.resolve()) if args.out is not None else None,
                })
            elif args.command == "export":
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
                envelope = {
                    "protocol": PROTOCOL,
                    "command": "correct",
                    "run_id": args.run_id,
                    "error": None,
                    **receipt,
                }
                _emit_json_line(envelope)
            else:
                if args.json:
                    _emit_json_line(query_quote_json(store, args.run_id, args.start, args.end, context=args.context))
                else:
                    source = store.source(store.run(args.run_id)["source_id"])
                    print(validate_span(source["text"], address(source["id"], args.start, args.end), source["id"]))
        finally:
            store.close()
        return 0
    except _CliHelpExit:
        return 0
    except Exception as exc:
        err = classify_exception(exc)
        et_run_id = getattr(exc, "et_run_id", None) or known_run_id
        if cmd_name == "run" and getattr(exc, "et_run_id", None) is None:
            exit_code = 1
        else:
            exit_code = EXIT_CODE_BY_CATEGORY.get(err["category"], 1)
        if json_mode:
            if cmd_name == "run" and getattr(exc, "et_run_id", None) is None:
                _emit_json_line(_build_failed_run_receipt(err))
            else:
                _emit_json_line({
                    "protocol": PROTOCOL,
                    "command": cmd_name or "unknown",
                    "run_id": et_run_id if cmd_name != "run" else getattr(exc, "et_run_id", None),
                    "error": err,
                })
        print("执行失败：" + err["message"], file=sys.stderr)
        return exit_code
