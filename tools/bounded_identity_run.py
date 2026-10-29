"""带独立请求／时长预算的全文验证入口，复用本地任务和响应缓存。"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entitytrace import pipeline
from entitytrace.models import Models
from entitytrace.util import implementation_hash, utc_now, write_json
from tools.identity_calibration import ExperimentBudget


class RequestBudget:
    def __init__(self, max_requests, max_seconds, opener=None, max_network_failures=9):
        self.max_requests, self.max_seconds = max_requests, max_seconds
        self.started = time.perf_counter()
        self.count, self.stop_reason = 0, None
        self.max_network_failures = max_network_failures
        self.consecutive_network_failures = 0
        self.network_errors = collections.Counter()
        self.lock = threading.Lock()
        self.opener = opener or urllib.request.urlopen

    def __call__(self, request, timeout):
        with self.lock:
            remaining = self.max_seconds - (time.perf_counter() - self.started)
            if self.stop_reason == 'network_outage':
                raise ExperimentBudget('连续网络失败，实验停止新请求并保留断点')
            if self.count >= self.max_requests or remaining <= 0:
                self.stop_reason = 'request_budget' if self.count >= self.max_requests else 'time_budget'
                raise ExperimentBudget('全文实验预算耗尽；保留未完成任务以便恢复')
            self.count += 1
        try:
            # 在预算包装层读完响应，连接阶段和读取阶段失败都能计入停止条件。
            with self.opener(request, timeout=min(timeout, remaining)) as response:
                body = response.read()
        except urllib.error.HTTPError:
            # HTTP 错误由生产客户端解析、重试或触发硬额度保护。
            with self.lock:
                self.consecutive_network_failures = 0
            raise
        except OSError as exc:
            with self.lock:
                self.consecutive_network_failures += 1
                self.network_errors[type(exc).__name__] += 1
                if self.consecutive_network_failures >= self.max_network_failures:
                    self.stop_reason = 'network_outage'
            raise
        with self.lock:
            self.consecutive_network_failures = 0
        return io.BytesIO(body)


def run(args):
    if args.receipt.exists():
        raise ValueError('实验回执已存在，请使用新的文件名')
    begin = time.perf_counter()
    budget = RequestBudget(args.max_requests, args.max_seconds)
    report = {'started_at': utc_now(), 'input': str(args.input),
              'production_implementation_hash': implementation_hash(),
              'identity_rounds': args.identity_rounds, 'group_candidates': 3,
              'discovery_concurrency': 6, 'decision_concurrency': 3,
              'budget': {'max_requests': args.max_requests, 'max_seconds': args.max_seconds,
                         'max_consecutive_network_failures': budget.max_network_failures},
              'scope': 'warm_task_and_response_cache_full_pipeline', 'status': 'running'}
    write_json(args.receipt, report)
    original = pipeline.Models

    class BudgetModels(Models):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs, opener=budget)

    pipeline.Models = BudgetModels
    try:
        rid, output, complete = pipeline.run_pipeline(args.input, args.db, args.output_root, args.config,
            identity_rounds=args.identity_rounds, group_candidates=3, log=lambda message: print(message, flush=True))
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        invocation = next(v for v in manifest['invocations'] if v['id'] == manifest['state']['last_invocation'])
        report.update(run_id=rid, output=str(output), complete=complete,
                      status='completed' if complete else 'incomplete', invocation=invocation,
                      result_revision=json.loads((output / 'result.json').read_text(encoding='utf-8'))['result_revision'])
    except Exception as exc:
        report.update(status='local_error', error='本地验证错误：' + type(exc).__name__)
        raise
    finally:
        pipeline.Models = original
        report.update(network_attempts=budget.count, budget_stop_reason=budget.stop_reason,
                      network_error_types=dict(budget.network_errors),
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
