"""带独立请求／时长预算的全文验证入口，复用本地任务和响应缓存。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entitytrace import pipeline
from entitytrace.models import ExperimentBudget, Models, RequestBudget
from entitytrace.util import implementation_hash, utc_now, write_json


def run(args):
    if args.receipt.exists():
        raise ValueError('实验回执已存在，请使用新的文件名')
    begin = time.perf_counter()
    report = {'started_at': utc_now(), 'input': str(args.input),
              'production_implementation_hash': implementation_hash(),
              'identity_rounds': args.identity_rounds, 'group_candidates': 3,
              'discovery_concurrency': 6, 'decision_concurrency': 3,
              'budget': {'max_requests': args.max_requests, 'max_seconds': args.max_seconds,
                         'max_consecutive_network_failures': 9},
              'scope': 'warm_task_and_response_cache_full_pipeline', 'status': 'running'}
    write_json(args.receipt, report)
    budget_info = {}
    try:
        rid, output, complete = pipeline.run_pipeline(
            args.input, args.db, args.output_root, args.config,
            identity_rounds=args.identity_rounds, group_candidates=3,
            max_requests=args.max_requests, max_seconds=args.max_seconds,
            log=lambda message: print(message, flush=True),
        )
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        invocation = next(v for v in manifest['invocations'] if v['id'] == manifest['state']['last_invocation'])
        budget_info = invocation.get('budget', {})
        report.update(run_id=rid, output=str(output), complete=complete,
                      status='completed' if complete else 'incomplete', invocation=invocation,
                      result_revision=json.loads((output / 'result.json').read_text(encoding='utf-8'))['result_revision'])
    except Exception as exc:
        report.update(status='local_error', error='本地验证错误：' + type(exc).__name__)
        raise
    finally:
        report.update(network_attempts=budget_info.get('network_attempts', 0),
                      budget_stop_reason=budget_info.get('stop_reason'),
                      network_error_types=budget_info.get('network_errors', {}),
                      wall_seconds_before_receipt=time.perf_counter() - begin, finished_at=utc_now())
        write_json(args.receipt, report)
    print(json.dumps({k: report[k] for k in ('run_id', 'status', 'network_attempts', 'budget_stop_reason',
                                            'wall_seconds_before_receipt')}, ensure_ascii=False), flush=True)
    return 0 if report['complete'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--db', type=Path, default=ROOT / 'data/entitytrace.sqlite3')
    parser.add_argument('--config', type=Path, default=ROOT / '.env.models.local')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--identity-rounds', type=int, default=3)
    parser.add_argument('--max-requests', type=int, default=6500)
    parser.add_argument('--max-seconds', type=float, default=2700)
    args = parser.parse_args()
    if not 1 <= args.identity_rounds <= 12 or not 0 <= args.max_requests <= 6500 or not 0 < args.max_seconds <= 2700:
        parser.error('验证预算须为 1—12 轮、0—6500 次新请求、0—2700 秒')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
