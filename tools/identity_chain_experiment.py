"""冻结提及对的完整单候选复核对照；在线入口不读取标签，不修改生产默认值。"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entitytrace import resolution
from entitytrace.config import load_services
from entitytrace.identity import group_entities
from entitytrace.models import ModelError, Models
from entitytrace.storage import Store
from entitytrace.util import digest, implementation_hash, utc_now, write_json
from tools.identity_calibration import ExperimentBudget, fraction, load_inputs, load_labels

VERSION = 'identity-chain-experiment-v1'
POLICIES = {'p95': .95, 'p90': .90}


def make_task(case):
    targets = [{k: t[k] for k in ('id', 'text', 'start', 'end', 'address', 'chapter_id', 'provisional_kinds')}
               for t in case['targets']]
    mentions = {t['id']: t for t in targets}
    groups = group_entities(targets, [])[0]
    owner = {mid: g for g in groups for mid in g['mention_ids']}
    left, right = [t['id'] for t in targets]
    a, b = owner[left], owner[right]
    pair = {'id': digest([left, right]), 'left': left, 'right': right,
            'start': min(t['start'] for t in targets), 'reasons': ['固定实验提及对'],
            'extra_evidence': case.get('extra_evidence', [])}
    edge = {'id': resolution.comparison_id(a, b), 'groups': [a['id'], b['id']], 'pair': pair}
    return mentions, {'id': digest([VERSION, edge]), 'round': 1, 'stage': 'identity-groups',
        'target': a['id'], 'groups': [a, b], 'edges': [edge], 'cannot_link': [], 'start': pair['start']}


def run_case(models, source, case, probability):
    mentions, task = make_task(case)
    original = resolution.POLICY
    # 仅供单进程串行实验。重新绑定而非原地改字典，保留每条决策当时的门槛。
    resolution.POLICY = {**original, 'min_probability': probability, 'min_confidence': .8}
    try:
        records = resolution.resolve_task(models, source, mentions, task)
    finally:
        resolution.POLICY = original
    relation = next(r for kind, r in records if kind == 'identity')
    decisions = [{k: d[k] for k in ('id', 'stage', 'request_id', 'comparison_id', 'answer', 'policy')}
                 for kind, d in records if kind == 'decision']
    return {'choice': relation['choice'], 'accepted': relation['accepted'], 'reason': relation['reason'],
            'relation_id': relation['id'], 'decision_ids': relation['decision_ids'], 'decisions': decisions}


def online(args):
    begin = time.perf_counter()
    paths = {key: args.out.with_name(args.out.stem + '.' + key + '.sqlite3') for key in POLICIES}
    if args.out.exists() or any(p.exists() for p in paths.values()):
        raise ValueError('实验回执或独立缓存已存在，请使用新的输出路径')
    inputs, sources = load_inputs(args.inputs)
    services = load_services(ROOT / '.env.models.local')
    count = 0

    def opener(request, timeout):
        nonlocal count
        remaining = args.max_seconds - (time.perf_counter() - begin)
        if count >= args.max_requests or remaining <= 0:
            raise ExperimentBudget('实验请求数或时长预算耗尽')
        count += 1
        return urllib.request.urlopen(request, timeout=min(timeout, remaining))

    report = {'tool_version': VERSION, 'started_at': utc_now(), 'inputs_sha256': digest(inputs),
        'production_implementation_hash': implementation_hash(),
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model': services['decision'].model, 'concurrency': 1, 'attempts_per_request': 1,
        'selected_case_ids': [c['id'] for c in inputs['cases']],
        'policies': {k: {'min_probability': p, 'min_confidence': .8} for k, p in POLICIES.items()},
        'budget': {'max_requests': args.max_requests, 'max_seconds': args.max_seconds},
        'status': 'running', 'results': [], 'calls': [], 'errors': [],
        'scope': 'single_candidate_production_resolve_task_not_full_clustering'}
    stores = {}
    current = {}
    try:
        for key, path in paths.items():
            stores[key] = Store(path)
        models = {key: Models(store, VERSION, utc_now(), services, decision_concurrency=1,
                             timeout=45, attempts=1, opener=opener) for key, store in stores.items()}
        for index, case in enumerate(inputs['cases']):
            order = list(POLICIES) if index % 2 == 0 else list(reversed(POLICIES))
            for key in order:
                current = {'id': case['id'], 'policy': key}
                if time.perf_counter() - begin >= args.max_seconds:
                    raise ExperimentBudget('实验时长预算耗尽')
                tick = time.perf_counter()
                result = run_case(models[key], sources[case['source']], case, POLICIES[key])
                report['results'].append({**current, **result, 'wall_seconds': time.perf_counter() - tick})
                write_json(args.out, report)
        report['status'] = 'completed'
    except (ModelError, ExperimentBudget) as exc:
        report['status'] = 'budget_exhausted' if isinstance(exc, ExperimentBudget) else 'blocked'
        report['errors'].append({**current, 'message': str(exc), 'http_status': getattr(exc, 'status', None),
                                 'code': getattr(exc, 'code', 'experiment_budget')})
    except Exception as exc:
        report['status'] = 'local_error'
        report['errors'].append({**current, 'message': '实验本地错误：' + type(exc).__name__})
        raise
    finally:
        for key, store in stores.items():
            report['calls'].extend({'policy': key, **c} for c in store.rows(
                'SELECT stage,request_id,status,seconds,detail,created FROM calls'))
            store.close()
        report['network_attempts'] = count
        report['expected_results'] = len(inputs['cases']) * len(POLICIES)
        report['missing_results'] = report['expected_results'] - len(report['results'])
        report['wall_seconds_before_receipt'] = time.perf_counter() - begin
        write_json(args.out, report)
    print(json.dumps({k: report[k] for k in ('status', 'network_attempts', 'missing_results')}, ensure_ascii=False))
    return 0 if report['status'] == 'completed' else 2


def outcome_metrics(rows):
    matrix = {truth: {key: 0 for key in ('same', 'different', 'review', 'unmeasured')}
              for truth in ('same', 'different')}
    for row in rows:
        choice = 'review' if row['choice'] == 'uncertain' else row['choice']
        matrix[row['label']][choice] += 1
    tp, fp = matrix['same']['same'], matrix['different']['same']
    tn, fn = matrix['different']['different'], matrix['same']['different']
    n, accepted = len(rows), tp + fp + tn + fn
    return {'n': n, 'matrix': matrix, 'correct': tp + tn, 'wrong': fp + fn,
        'review': sum(v['review'] for v in matrix.values()),
        'unmeasured': sum(v['unmeasured'] for v in matrix.values()),
        'auto_accuracy': fraction(tp + tn, accepted), 'auto_coverage': fraction(accepted, n),
        'merge_precision': fraction(tp, tp + fp), 'merge_recall': fraction(tp, sum(matrix['same'].values())),
        'false_merge': fp, 'false_different': fn}


def score(args):
    inputs, sources = load_inputs(args.inputs)
    document, labels = load_labels(args.labels, inputs, sources)
    trial = json.loads(args.answers.read_text(encoding='utf-8'))
    known = {c['id']: c for c in inputs['cases']}
    if trial['tool_version'] != VERSION or trial['inputs_sha256'] != digest(inputs):
        raise ValueError('实验工具或冻结输入版本不一致')
    if trial['selected_case_ids'] != list(known) or trial['policies'] != {
            k: {'min_probability': p, 'min_confidence': .8} for k, p in POLICIES.items()}:
        raise ValueError('在线样本或门槛不符合冻结实验范围')
    results = {}
    for row in trial['results']:
        key = row['id'], row['policy']
        if key in results or row['id'] not in known or row['policy'] not in POLICIES:
            raise ValueError('结果重复或没有对应实验任务')
        if row['choice'] not in {'same', 'different', 'uncertain'} or row['accepted'] != (row['choice'] != 'uncertain'):
            raise ValueError('身份结果与接受状态不一致')
        results[key] = row
    expected = len(known) * len(POLICIES)
    if trial['expected_results'] != expected or trial['missing_results'] != expected - len(results):
        raise ValueError('缺失结果计数不一致')
    if trial['status'] == 'completed' and len(results) != expected:
        raise ValueError('完成状态仍缺少实验结果')
    table, timing = [], {}
    for policy in POLICIES:
        rows = [{'id': cid, 'label': labels[cid]['label'],
                 'choice': results.get((cid, policy), {}).get('choice', 'unmeasured')} for cid in known]
        for language in ('all', 'zh', 'en'):
            subset = [r for r in rows if language == 'all' or known[r['id']]['source'] == language]
            table.append({'policy': policy, 'language': language, **outcome_metrics(subset)})
        calls = [c for c in trial['calls'] if c['policy'] == policy]
        timing[policy] = {'status_counts': dict(collections.Counter(c['status'] for c in calls)),
            'stage_counts': dict(collections.Counter(c['stage'] for c in calls if c['status'] != 'cached')),
            'request_seconds_sum': sum(c['seconds'] for c in calls),
            'completed_case_seconds_sum': sum(r['wall_seconds'] for r in trial['results'] if r['policy'] == policy)}
    changes = collections.defaultdict(list)
    for cid in known:
        def status(policy):
            choice = results.get((cid, policy), {}).get('choice', 'unmeasured')
            return ('review' if choice == 'uncertain' else 'unmeasured' if choice == 'unmeasured'
                    else 'correct' if choice == labels[cid]['label'] else 'wrong')
        changes[status('p95') + '_to_' + status('p90')].append(cid)
    report = {'tool_version': VERSION, 'inputs_sha256': digest(inputs), 'labels_sha256': digest(document),
        'annotation_status': 'agent_reviewed_pending_independent_human_review',
        'scope': trial['scope'], 'trial_status': trial['status'],
        'trial_file_sha256': hashlib.sha256(args.answers.read_bytes()).hexdigest(),
        'network_attempts': trial['network_attempts'], 'missing_results': trial['missing_results'],
        'metrics': table, 'timing': timing, 'transitions': dict(changes), 'errors': trial['errors']}
    write_json(args.out, report)
    print(json.dumps({'status': 'scored', 'trial_status': trial['status']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['online', 'score'])
    parser.add_argument('--inputs', type=Path, default=ROOT / 'test/identity-calibration/inputs.json')
    parser.add_argument('--labels', type=Path, default=ROOT / 'test/identity-calibration/reference/labels.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--answers', type=Path)
    parser.add_argument('--max-requests', type=int, default=256)
    parser.add_argument('--max-seconds', type=float, default=900)
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 256 or not 0 < args.max_seconds <= 900:
        parser.error('实验预算须为 1—256 次请求、0—900 秒')
    if args.mode == 'score' and (not args.answers or args.answers.resolve() == args.out.resolve()):
        parser.error('score 需要 --answers，且不能覆盖在线回执')
    return {'online': online, 'score': score}[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
