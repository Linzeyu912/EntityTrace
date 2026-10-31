"""小规模身份诊断；标签仅供评分，生产门槛和历史数据库保持不变。"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sqlite3
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entitytrace.config import load_services
from entitytrace.identity import group_entities, plan_pairs
from entitytrace.models import ExperimentBudget, ModelError, Models, check_choices
from entitytrace.resolution import IDENTITY, ask_cases, identity_case
from entitytrace.storage import Store
from entitytrace.text import validate_span
from entitytrace.util import digest, implementation_hash, utc_now, write_json

VERSION = 'identity-calibration-v1'
MODES = {'narrow': ('identity-group', 180), 'wide': ('identity-group-expanded', 400)}
POLICIES = [(p, c) for p in (.95, .90, .85, .80, 0.) for c in (.8, .7, 0.)]


def load_inputs(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    sources = {}
    for key, metadata in data['sources'].items():
        raw = (ROOT / metadata['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata['raw_sha256']:
            raise ValueError('原文文件指纹不匹配：' + key)
        text = raw.removeprefix(b'\xef\xbb\xbf').decode('utf-8')
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != metadata['text_sha256']:
            raise ValueError('定位文本指纹不匹配：' + key)
        sid = digest(['text-v1', metadata['raw_sha256'], 'utf-8', raw.startswith(b'\xef\xbb\xbf'), metadata['text_sha256']])
        if sid != metadata['id']:
            raise ValueError('原文版本编号不匹配：' + key)
        sources[key] = {'id': sid, 'text': text}
    ids = [c['id'] for c in data['cases']]
    if len(set(ids)) != len(ids):
        raise ValueError('样本编号重复')
    for case in data['cases']:
        source = sources[case['source']]
        if len(case['targets']) != 2 or case['targets'][0]['id'] == case['targets'][1]['id']:
            raise ValueError('每个样本必须有两个不同位置的目标提及')
        for target in case['targets']:
            validate_span(source['text'], target['address'], source['id'], target['text'])
            if (target['start'], target['end']) != (target['address']['start'], target['address']['end']):
                raise ValueError('提及地址不一致')
        for span in case.get('extra_evidence', []):
            validate_span(source['text'], span, source['id'])
    return data, sources


def load_labels(path, inputs, sources):
    raw = json.loads(path.read_text(encoding='utf-8'))
    labels = {row['id']: row for row in raw['cases']}
    if len(labels) != len(raw['cases']) or set(labels) != {c['id'] for c in inputs['cases']}:
        raise ValueError('标注和样本编号必须一一对应')
    if raw['version'] != inputs['version']:
        raise ValueError('标注版本与样本不匹配')
    for case in inputs['cases']:
        label, source = labels[case['id']], sources[case['source']]
        if label['label'] not in {'same', 'different'} or not label['evidence']:
            raise ValueError('标签无效或缺少依据')
        for evidence in label['evidence']:
            span = {'text_version_id': source['id'], 'offset_unit': 'unicode_code_point',
                    'start': evidence['start'], 'end': evidence['end']}
            validate_span(source['text'], span, source['id'], evidence['quote'])
    return raw, labels


def make_case(case, source, mode):
    # 显式白名单：即使输入意外带入 label/rationale，也不进入模型状态。
    targets = [{k: t[k] for k in ('id', 'text', 'start', 'end', 'address', 'chapter_id', 'provisional_kinds')}
               for t in case['targets']]
    mentions = {m['id']: m for m in targets}
    groups = group_entities(targets, [])[0]
    owner = {mid: group for group in groups for mid in group['mention_ids']}
    left, right = [t['id'] for t in targets]
    return identity_case(source, owner[left], owner[right], mentions,
        {'left': left, 'right': right, 'extra_evidence': case.get('extra_evidence', [])},
        'initial' if mode == 'narrow' else 'expanded')


def routed(answer, probability, confidence):
    if answer is None:
        return 'unmeasured'
    choice = answer['choice']
    return choice if (choice in {'same', 'different'} and answer['probabilities'][choice] >= probability
                      and answer['confidence'] >= confidence) else 'review'


def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}


def metrics(rows, mode, p, c):
    matrix = {truth: {v: 0 for v in ('same', 'different', 'review', 'unmeasured')} for truth in ('same', 'different')}
    for row in rows:
        matrix[row['label']][routed(row['answers'].get(mode), p, c)] += 1
    tp, fp = matrix['same']['same'], matrix['different']['same']
    tn, fn = matrix['different']['different'], matrix['same']['different']
    n = len(rows)
    accepted = tp + fp + tn + fn
    positive = sum(row['label'] == 'same' for row in rows)
    unmeasured = sum(v['unmeasured'] for v in matrix.values())
    return {'n': n, 'matrix': matrix, 'correct': tp + tn, 'wrong': fp + fn,
            'review': sum(v['review'] for v in matrix.values()), 'unmeasured': unmeasured,
            'auto_accuracy': fraction(tp + tn, accepted), 'auto_coverage': fraction(accepted, n),
            'correct_auto_coverage': fraction(tp + tn, n), 'merge_precision': fraction(tp, tp + fp),
            'merge_recall': fraction(tp, positive), 'false_merge': fp, 'false_different': fn,
            'missed_same_including_unmeasured': positive - tp}


def transitions(rows, before, after):
    categories = collections.defaultdict(list)
    for row in rows:
        a = routed(row['answers'].get(before[0]), *before[1:])
        b = routed(row['answers'].get(after[0]), *after[1:])
        def status(value):
            if value in {'review', 'unmeasured'}:
                return value
            return 'correct' if value == row['label'] else 'wrong'
        categories[status(a) + '_to_' + status(b)].append(row['id'])
    return dict(sorted(categories.items()))


def read_record(db, run, kind, rid):
    row = db.execute('SELECT data FROM records WHERE run_id=? AND kind=? AND id=?', (run, kind, rid)).fetchone()
    if row is None:
        raise ValueError('缺少历史记录：' + rid)
    return json.loads(row[0])


def historical_timing(db, rid):
    rows = db.execute("SELECT id,status,seconds,detail,created FROM calls WHERE request_id=? AND status!='cached' ORDER BY id", (rid,)).fetchall()
    used = []
    for row in rows:
        used.append(dict(row))
        if row['status'] == 'success':
            break
    if not used or used[-1]['status'] != 'success':
        raise ValueError('成功缓存缺少原始调用计时')
    return {'attempts': len(used), 'failed_attempts': len(used)-1,
            'request_seconds_sum': sum(x['seconds'] for x in used),
            'failed_seconds_sum': sum(x['seconds'] for x in used[:-1]),
            'success_seconds': used[-1]['seconds'], 'created': used[-1]['created'],
            'active_requests': json.loads(used[-1]['detail']).get('active_requests')}


def check_history(db, decision, source, mode, targets):
    payload = decision['input']
    radius = MODES[mode][1]
    representatives = []
    for side in ('A', 'B'):
        group = payload[side]
        if group['member_count'] != 1 or len(group['representatives']) != 1:
            raise ValueError('配对样本必须是单提及组，避免代表数变化混入对照')
        rep = group['representatives'][0]
        span = rep['address']
        validate_span(source['text'], span, source['id'], rep['target_mention'])
        lo, hi = max(0, span['start']-radius), min(len(source['text']), span['end']+radius)
        if rep['context_address']['start'] != lo or rep['context_address']['end'] != hi:
            raise ValueError('历史上下文范围不是指定对照半径')
        validate_span(source['text'], rep['context_address'], source['id'], rep['before']+rep['target_mention']+rep['after'])
        representatives.append(rep)
    if {r['mention_id'] for r in representatives} != {t['id'] for t in targets}:
        raise ValueError('历史目标与标注目标不一致')
    for extra in payload['extra_evidence']:
        validate_span(source['text'], extra['address'], source['id'], extra['text'])
    cached = db.execute('SELECT request,response FROM request_cache WHERE id=?', (decision['request_id'],)).fetchone()
    if cached is None:
        raise ValueError('缺少用于核对问题及响应的 HTTP 缓存')
    request, response = [json.loads(value) for value in cached]
    check_choices(response, request['questions'], request['model'])
    qid = decision['question_id']
    if response['answers'][qid] != decision['answer']:
        raise ValueError('决策与缓存响应不一致')
    expected = {'extra_evidence': [e['text'] for e in payload['extra_evidence']]}
    for side in ('A', 'B'):
        g = payload[side]
        expected[side] = {'member_count': 1, 'sampled': False, 'representatives': [
            {k: r[k] for k in ('before', 'target_mention', 'after', 'provisional_kinds')} for r in g['representatives']]}
    if request['state']['cases'][qid] != expected:
        raise ValueError('保存证据不等于实际模型输入')
    question = request['questions'][qid]
    if question['criteria'] != IDENTITY:
        raise ValueError('历史选项改变，不能放入同提示对照')
    fixed = {side: [{k: r[k] for k in ('address', 'mention_id', 'target_mention', 'provisional_kinds')}
                   for r in payload[side]['representatives']] for side in ('A', 'B')}
    fixed['extra_evidence'] = payload['extra_evidence']
    fixed['instructions'] = re.sub(r'^针对 cases\.c\d+：', '', question['instructions'])
    fixed['notice'] = request['state']['notice']
    fixed['model'] = request['model']
    return fixed, {'request_id': decision['request_id'], 'question_count': len(request['questions']),
                   'payload_sha256': digest(request), 'input_characters': len(json.dumps(expected, ensure_ascii=False)),
                   'timing': historical_timing(db, decision['request_id'])}


def replay_rows(db, inputs, sources, labels):
    rows = []
    for case in inputs['cases']:
        h = case['historical']
        row = {'id': case['id'], 'language': case['source'], 'cohort': case['cohort'],
               'label': labels[case['id']]['label'], 'category': labels[case['id']]['category'],
               'answers': {}, 'requests': {}}
        if h:
            comparison = read_record(db, h['run_id'], 'group_comparison', h['comparison_id'])
            decisions = [read_record(db, h['run_id'], 'decision', did) for did in comparison['decision_ids']]
            fixed = []
            for mode, (stage, _) in MODES.items():
                chosen = [d for d in decisions if d['stage'] == stage]
                if len(chosen) != 1:
                    raise ValueError('配对阶段缺失或重复')
                decision = chosen[0]
                basis, info = check_history(db, decision, sources[case['source']], mode, case['targets'])
                fixed.append(basis)
                row['answers'][mode] = decision['answer']
                row['requests'][mode] = info
            if fixed[0] != fixed[1]:
                raise ValueError('除了邻文宽度之外，目标／提示／类别／关系证据也发生改变：' + case['id'])
            row['same_fixed_case_fields'] = True
            row['both_requests_single_case'] = all(x['question_count'] == 1 for x in row['requests'].values())
            row['saved_relation'] = comparison['choice']
            row['saved_reason'] = comparison['reason']
            row['guard_present'] = any(d['stage'] == 'identity-group-guard' for d in decisions)
            row['saved_guard_answers'] = [d['answer'] for d in decisions if d['stage'] == 'identity-group-guard']
        rows.append(row)
    return rows


def candidate_diagnostics(db, inputs):
    result = []
    for source_key, source in inputs['sources'].items():
        def records(kind):
            return [json.loads(x[0]) for x in db.execute('SELECT data FROM records WHERE run_id=? AND kind=?', (source['baseline_run_id'], kind))]
        mentions, links = records('mention'), records('link')
        pairs = plan_pairs(mentions, links)
        edges = {frozenset((p['left'], p['right'])) for p in pairs}
        parent = {m['id']: m['id'] for m in mentions}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for pair in pairs:
            parent[find(pair['left'])] = find(pair['right'])
        for case in inputs['cases']:
            if case['source'] != source_key:
                continue
            a, b = [t['id'] for t in case['targets']]
            result.append({'id': case['id'], 'direct_candidate': frozenset((a,b)) in edges,
                           'candidate_graph_connected': find(a) == find(b)})
    return result


def summarize(rows):
    cohorts = {'paired': [r for r in rows if r['cohort'] == 'cached_paired']}
    cohorts.update({lang: [r for r in cohorts['paired'] if r['language'] == lang] for lang in ('zh', 'en')})
    cohorts['single_case_requests'] = [r for r in cohorts['paired'] if r['both_requests_single_case']]
    cohorts['without_overlap'] = [r for r in cohorts['paired'] if r['category'] != 'overlap']
    table = []
    for name, subset in cohorts.items():
        for mode in MODES:
            for p,c in POLICIES:
                table.append({'cohort': name, 'mode': mode, 'min_probability': p, 'min_confidence': c,
                              **metrics(subset, mode, p, c)})
    comparisons = []
    for p,c in POLICIES:
        comparisons.append({'before': ['narrow',p,c], 'after': ['wide',p,c],
                            'transitions': transitions(cohorts['paired'], ('narrow',p,c), ('wide',p,c))})
    timings = {}
    for mode in MODES:
        requests = {r['requests'][mode]['request_id']: r['requests'][mode] for r in cohorts['paired']}
        timings[mode] = {'unique_requests': len(requests),
            'original_network_attempts': sum(v['timing']['attempts'] for v in requests.values()),
            'original_failed_attempts': sum(v['timing']['failed_attempts'] for v in requests.values()),
            'original_request_seconds_sum': sum(v['timing']['request_seconds_sum'] for v in requests.values()),
            'original_success_seconds_sum': sum(v['timing']['success_seconds'] for v in requests.values()),
            'original_failed_seconds_sum': sum(v['timing']['failed_seconds_sum'] for v in requests.values()),
            'original_success_latency_median': statistics.median(v['timing']['success_seconds'] for v in requests.values()),
            'question_counts': dict(collections.Counter(v['question_count'] for v in requests.values()))}
    return table, comparisons, timings


def compact_report(report):
    """只导出诊断计数与审计引用；不提交整批模型输入或响应缓存。"""
    summary = {k:v for k,v in report.items() if k not in {'cases','metrics'}}
    summary['metrics'] = []
    for row in report['metrics']:
        if row['cohort'] != 'paired' and (row['min_probability'],row['min_confidence']) not in {(.95,.8),(.9,.8),(.85,.8),(0,0)}:
            continue
        summary['metrics'].append({k:row[k] for k in ('cohort','mode','min_probability','min_confidence',
            'n','correct','wrong','review','unmeasured','matrix')})
    summary['cases'] = []
    for row in report['cases']:
        item = {k:v for k,v in row.items() if k not in {'answers','requests','saved_guard_answers'}}
        item['decisions'] = {mode:{'choice':a['choice'],'selected_probability':a['probabilities'][a['choice']],
                                  'confidence':a['confidence'],**row['requests'][mode]}
                             for mode,a in row['answers'].items()}
        item['saved_guards'] = [{'choice':a['choice'],'selected_probability':a['probabilities'][a['choice']],
                                'confidence':a['confidence']} for a in row.get('saved_guard_answers',[])]
        summary['cases'].append(item)
    return summary


def replay(args):
    start = time.perf_counter()
    inputs, sources = load_inputs(args.inputs)
    label_document, labels = load_labels(args.labels, inputs, sources)
    db = sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = replay_rows(db, inputs, sources, labels)
        candidates = candidate_diagnostics(db, inputs)
    finally:
        db.close()
    table, changes, timings = summarize(rows)
    report = {'schema_version': 1, 'tool_version': VERSION, 'created_at': utc_now(),
              'inputs_sha256': digest(inputs), 'labels_sha256': digest(label_document),
              'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'production_implementation_hash': implementation_hash(),
              'annotation_status': 'agent_reviewed_pending_independent_human_review',
              'independent_human_gold': False, 'production_policy_changed': False,
              'network_attempts_this_replay': 0, 'scope': 'single_identity_decisions_not_end_to_end_clustering',
              'paired_cohort_n': sum(r['cohort']=='cached_paired' for r in rows),
              'unmeasured_challenge_n': sum(r['cohort']=='challenge_unmeasured' for r in rows),
              'cases': rows, 'metrics': table, 'evidence_transitions': changes,
              'historical_timing': timings, 'candidate_diagnostics': candidates,
              'timing_limit': '历史请求延迟及并发耗时之和，含同批其他问题；不能作为本样本独立墙钟或新证据的在线耗时。',
              'selection_limit': '历史配对队列条件化于触发扩证据，非随机样本、非独立留出；有些请求同批其他问题不同。'}
    if args.probe:
        probe = json.loads(args.probe.read_text(encoding='utf-8'))
        report['online_probe'] = {k: probe[k] for k in ('started', 'status', 'error', 'http_status', 'error_code', 'wall_seconds') if k in probe}
        report['online_probe']['case_id'] = 'p25'
        report['online_probe']['network_attempts'] = len(probe['calls'])
        report['online_probe']['request_id'] = probe['calls'][0]['request_id']
    report['analysis_seconds_before_export'] = time.perf_counter()-start
    write_json(args.out, report)
    if args.summary_out:
        write_json(args.summary_out, compact_report(report))
    print(json.dumps({'status':'completed','paired_cases':report['paired_cohort_n'],
                      'unmeasured_challenges':report['unmeasured_challenge_n'],'output':str(args.out)}, ensure_ascii=False))


def online(args):
    # 本入口不加载 reference/labels.json，也不把样本 ID 或评测类别送入模型。
    begin = time.perf_counter()
    if args.out.exists():
        raise ValueError('实验回执已存在，请使用新的输出路径：'+str(args.out))
    db_paths = [args.out.with_name(args.out.stem+f'.repeat-{r+1}.sqlite3') for r in range(args.repeats)]
    if any(p.exists() for p in db_paths):
        raise ValueError('独立重复要求新数据库，请使用新的输出路径')
    inputs, sources = load_inputs(args.inputs)
    selected = [c for c in inputs['cases'] if args.cohort == 'all' or c['cohort'] == args.cohort]
    services = load_services(ROOT/'.env.models.local')
    count = 0
    def opener(request, timeout):
        nonlocal count
        remaining = args.max_seconds-(time.perf_counter()-begin)
        if count >= args.max_requests or remaining <= 0:
            raise ExperimentBudget('实验请求数或时长预算耗尽')
        count += 1
        return urllib.request.urlopen(request, timeout=min(timeout, remaining))
    report = {'tool_version':VERSION,'started_at':utc_now(),'inputs_sha256':digest(inputs),
              'production_implementation_hash':implementation_hash(),
              'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model':services['decision'].model,'concurrency':1,'attempts_per_request':1,
              'selected_case_ids':[c['id'] for c in selected],
              'budget':{'max_requests':args.max_requests,'max_seconds':args.max_seconds,'repeats':args.repeats},
              'status':'running','answers':[],'calls':[],'errors':[]}
    try:
        for repeat in range(args.repeats):
            db_path = db_paths[repeat]
            store = Store(db_path)
            models = Models(store, VERSION, utc_now(), services, decision_concurrency=1,
                            timeout=45, attempts=1, opener=opener)
            try:
                for index, case in enumerate(selected):
                    # AB/BA 交替，并在第二次重复倒转，减小固定先后顺序影响。
                    order = ['narrow','wide'] if (index+repeat)%2==0 else ['wide','narrow']
                    for mode in order:
                        if time.perf_counter()-begin >= args.max_seconds:
                            raise ExperimentBudget('实验时长预算耗尽')
                        task_start = time.perf_counter()
                        built = make_case(case, sources[case['source']], mode)
                        result = ask_cases(models, [built], 'identity-calibration')[built['id']]
                        report['answers'].append({'id':case['id'],'mode':mode,'repeat':repeat+1,
                            'answer':result['answer'],'request_id':result['request_id'],
                            'wall_seconds':time.perf_counter()-task_start})
                        write_json(args.out, report)
            finally:
                report['calls'].extend(store.rows('SELECT stage,request_id,status,seconds,detail,created FROM calls'))
                store.close()
        report['status'] = 'completed'
    except (ModelError, ExperimentBudget) as exc:
        report['status'] = 'budget_exhausted' if isinstance(exc, ExperimentBudget) else 'blocked'
        report['errors'].append({'message':str(exc),'http_status':getattr(exc,'status',None),
                                 'code':getattr(exc,'code','experiment_budget')})
    except Exception as exc:
        report['status'] = 'local_error'
        report['errors'].append({'message':'实验本地错误：'+type(exc).__name__})
        raise
    finally:
        report['network_attempts'] = count
        report['expected_answers'] = len(selected)*2*args.repeats
        report['missing_answers'] = report['expected_answers']-len(report['answers'])
        report['wall_seconds_before_receipt'] = time.perf_counter()-begin
        write_json(args.out, report)
    print(json.dumps({'status':report['status'],'network_attempts':count,'missing_answers':report['missing_answers']}, ensure_ascii=False))
    return 0 if report['status']=='completed' else 2


def score_online(args):
    """在线结束后独立读标签评分；未返回的问题保留为未测。"""
    inputs, sources = load_inputs(args.inputs)
    label_document, labels = load_labels(args.labels, inputs, sources)
    trial = json.loads(args.answers.read_text(encoding='utf-8'))
    if trial['inputs_sha256'] != digest(inputs):
        raise ValueError('在线结果与冻结输入版本不同')
    selected = trial['selected_case_ids']
    known = {c['id']:c for c in inputs['cases']}
    if len(set(selected)) != len(selected) or not set(selected) <= known.keys():
        raise ValueError('在线样本编号重复或不属于冻结输入')
    repeats = trial['budget']['repeats']
    if repeats not in (1,2):
        raise ValueError('重复次数不符合实验协议')
    answers = {}
    for row in trial['answers']:
        key = row['repeat'], row['id'], row['mode']
        if key in answers or row['id'] not in selected or row['mode'] not in MODES or not 1 <= row['repeat'] <= repeats:
            raise ValueError('在线响应重复或没有对应实验问题')
        check_choices({'model':trial['model'],'answers':{'c0':row['answer']}},
                      {'c0':{'criteria':IDENTITY}}, trial['model'])
        answers[key] = row['answer']
    expected = len(selected)*len(MODES)*repeats
    if trial['expected_answers'] != expected or trial['missing_answers'] != expected-len(answers):
        raise ValueError('在线回执缺失计数不一致')
    if trial['status'] == 'completed' and len(answers) != expected:
        raise ValueError('在线实验标为完成但仍缺响应')
    table, changes = [], []
    for repeat in range(1,repeats+1):
        rows = [{'id':cid,'label':labels[cid]['label'],
                 'answers':{mode:answers.get((repeat,cid,mode)) for mode in MODES}} for cid in selected]
        for language in ('all','zh','en'):
            subset = [row for row in rows if language == 'all' or known[row['id']]['source']==language]
            for mode in MODES:
                for p,c in POLICIES:
                    table.append({'repeat':repeat,'language':language,'mode':mode,
                        'min_probability':p,'min_confidence':c,**metrics(subset,mode,p,c)})
        changes.append({'repeat':repeat,'raw_choice_transitions':transitions(rows,('narrow',0,0),('wide',0,0))})
    result = {'tool_version':VERSION,'inputs_sha256':digest(inputs),'labels_sha256':digest(label_document),
              'annotation_status':'agent_reviewed_pending_independent_human_review',
              'scope':'single_identity_decisions_not_end_to_end_clustering','trial_status':trial['status'],
              'trial_file_sha256':hashlib.sha256(args.answers.read_bytes()).hexdigest(),
              'network_attempts':trial['network_attempts'],'missing_answers':trial['missing_answers'],
              'trial_wall_seconds_before_receipt':trial['wall_seconds_before_receipt'],
              'metrics':table,'transitions':changes,'errors':trial['errors']}
    write_json(args.out,result)
    print(json.dumps({'status':'scored','trial_status':trial['status'],'missing_answers':trial['missing_answers']},ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['replay','online','score'])
    parser.add_argument('--inputs', type=Path, default=ROOT/'test/identity-calibration/inputs.json')
    parser.add_argument('--labels', type=Path, default=ROOT/'test/identity-calibration/reference/labels.json')
    parser.add_argument('--db', type=Path, default=ROOT/'data/entitytrace.sqlite3')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--summary-out', type=Path, help='精简诊断数据；省略整批模型输入与原始响应')
    parser.add_argument('--answers', type=Path, help='score 模式读取的在线回执')
    parser.add_argument('--cohort', choices=['cached_paired','challenge_unmeasured','all'], default='cached_paired')
    parser.add_argument('--repeats', type=int, choices=[1,2], default=2)
    parser.add_argument('--max-requests', type=int, default=96)
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 96 or not 0 < args.max_seconds <= 600:
        parser.error('实验预算须为 1—96 次请求、0—600 秒')
    if args.mode == 'score' and (not args.answers or args.answers.resolve() == args.out.resolve()):
        parser.error('score 需要 --answers，且输出不能覆盖在线回执')
    return {'replay':replay,'online':online,'score':score_online}[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
