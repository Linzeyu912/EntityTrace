from __future__ import annotations

import argparse
import copy
import io
import json
import sqlite3
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from entitytrace.config import Service
from entitytrace.resolution import ask_cases
from entitytrace.util import digest, dumps
from tools.identity_calibration import (ROOT, check_history, load_inputs, load_labels,
    make_case, metrics, online, routed, score_online, transitions)


def answer(choice='same', probability=.9, confidence=.85):
    return {'type':'choice','choice':choice,'confidence':confidence,
            'probabilities':{k:probability if k==choice else (1-probability)/2
                             for k in ('same','different','uncertain')}}


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.sources = load_inputs(ROOT/'test/identity-calibration/inputs.json')

    def test_frozen_labels_and_source_addresses_validate(self):
        document, labels = load_labels(ROOT/'test/identity-calibration/reference/labels.json', self.inputs, self.sources)
        self.assertEqual(len(labels),32)
        self.assertFalse(document['independent_human_gold'])
        paired = [labels[c['id']]['label'] for c in self.inputs['cases'] if c['cohort']=='cached_paired']
        self.assertEqual(paired.count('same'),12)
        self.assertEqual(paired.count('different'),12)

    def test_threshold_keeps_wrong_and_unmeasured_in_denominators(self):
        rows = [
            {'id':'1','label':'same','answers':{'narrow':answer()}},
            {'id':'2','label':'different','answers':{'narrow':answer()}},
            {'id':'3','label':'same','answers':{}},
        ]
        result = metrics(rows,'narrow',.9,.8)
        self.assertEqual((result['correct'],result['wrong'],result['unmeasured']),(1,1,1))
        self.assertEqual(result['auto_accuracy']['value'],.5)
        self.assertEqual(result['auto_coverage']['value'],2/3)
        self.assertEqual(result['merge_recall']['value'],.5)
        self.assertEqual(result['false_merge'],1)
        self.assertIsNone(metrics(rows,'narrow',.95,.8)['auto_accuracy']['value'])

    def test_uncertain_is_not_accepted_even_without_thresholds(self):
        self.assertEqual(routed(answer('uncertain',.99,.99),0,0),'review')

    def test_error_to_review_is_separate_from_correction(self):
        rows = [
            {'id':'1','label':'same','answers':{'narrow':answer('different'),'wide':answer('same',.5,.2)}},
            {'id':'2','label':'same','answers':{'narrow':answer('different'),'wide':answer()}},
            {'id':'3','label':'different','answers':{'narrow':answer('different'),'wide':answer()}},
        ]
        value = transitions(rows,('narrow',.9,.8),('wide',.9,.8))
        self.assertEqual(value,{'wrong_to_review':['1'],'wrong_to_correct':['2'],'correct_to_wrong':['3']})

    def test_evidence_modes_do_not_leak_labels_or_change_targets(self):
        case = copy.deepcopy(self.inputs['cases'][0])
        case.update(label='DO_NOT_SEND_LABEL',rationale='DO_NOT_SEND_RATIONALE')
        a = make_case(case,self.sources[case['source']],'narrow')
        b = make_case(case,self.sources[case['source']],'wide')
        self.assertEqual(a['instructions'],b['instructions'])
        self.assertEqual(a['options'],b['options'])
        for side in ('A','B'):
            ar,br = [v['state'][side]['representatives'][0] for v in (a,b)]
            self.assertEqual(ar['target_mention'],br['target_mention'])
            self.assertEqual(ar['provisional_kinds'],br['provisional_kinds'])
            self.assertEqual(len(ar['before']),180)
            self.assertEqual(len(br['before']),400)
        self.assertNotIn('DO_NOT_SEND',dumps(a))

    def test_history_detects_payload_and_address_corruption(self):
        source = self.sources['zh']
        case = self.inputs['cases'][0]
        built = make_case(case,source,'narrow')
        class Fake:
            def choices(self,state,questions,stage):
                self.payload = {'model':'jev-1.13.0','state':state,'questions':questions}
                self.response = {'model':'jev-1.13.0','answers':{qid:answer() for qid in questions}}
                return self.response['answers'],'request'
        model = Fake()
        decision = ask_cases(model,[built],'identity-group')[built['id']]
        db = sqlite3.connect(':memory:')
        db.row_factory = sqlite3.Row
        db.executescript('CREATE TABLE request_cache(id,request,response); CREATE TABLE calls(id,request_id,status,seconds,detail,created);')
        db.execute('INSERT INTO request_cache VALUES(?,?,?)',('request',dumps(model.payload),dumps(model.response)))
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?)',(1,'request','success',1.,'{"active_requests":1}','now'))
        check_history(db,decision,source,'narrow',case['targets'])
        bad = copy.deepcopy(decision)
        bad['input']['A']['representatives'][0]['before'] += '!'
        with self.assertRaises(ValueError):
            check_history(db,bad,source,'narrow',case['targets'])
        payload = copy.deepcopy(model.payload)
        payload['state']['cases']['c0']['A']['representatives'][0]['target_mention'] = 'wrong'
        db.execute('UPDATE request_cache SET request=?',(dumps(payload),))
        with self.assertRaises(ValueError):
            check_history(db,decision,source,'narrow',case['targets'])
        db.close()

    def online_args(self,path):
        return argparse.Namespace(inputs=ROOT/'test/identity-calibration/inputs.json',out=path,
            cohort='cached_paired',repeats=1,max_requests=1,max_seconds=600)

    def test_online_hard_quota_stops_at_one_request_and_keeps_missing(self):
        def quota(request,timeout):
            raise urllib.error.HTTPError(request.full_url,402,'billing',{},
                io.BytesIO(b'{"detail":{"error_type":"billing_error"}}'))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json'
            with patch('tools.identity_calibration.load_services',return_value={'decision':Service('decision','https://example.invalid','jev-1.13.0','test')}), patch('urllib.request.urlopen',side_effect=quota) as http:
                self.assertEqual(online(self.online_args(path)),2)
            report=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(http.call_count,1)
            self.assertEqual(report['status'],'blocked')
            self.assertEqual(report['missing_answers'],48)
            self.assertEqual(len(report['answers']),0)
            self.assertEqual(report['errors'][0]['http_status'],402)

    def test_online_request_budget_does_not_masquerade_as_model_error(self):
        def success(request,timeout):
            payload=json.loads(request.data)
            return io.BytesIO(dumps({'model':'jev-1.13.0','answers':{qid:answer() for qid in payload['questions']}}).encode())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json'
            with patch('tools.identity_calibration.load_services',return_value={'decision':Service('decision','https://example.invalid','jev-1.13.0','test')}), patch('urllib.request.urlopen',side_effect=success) as http:
                self.assertEqual(online(self.online_args(path)),2)
            report=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(http.call_count,1)
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertEqual(report['missing_answers'],47)
            self.assertEqual(len(report['calls']),1)
            before=path.read_bytes()
            with self.assertRaises(ValueError):
                online(self.online_args(path))
            self.assertEqual(path.read_bytes(),before)

    def score_fixture(self,folder):
        path=Path(folder)/'online.json'
        trial={'inputs_sha256':digest(self.inputs),'selected_case_ids':['p01','p02'],
               'budget':{'repeats':1},'model':'jev-1.13.0','status':'blocked',
               'answers':[{'repeat':1,'id':'p01','mode':'narrow','answer':answer()}],
               'expected_answers':4,'missing_answers':3,'network_attempts':2,
               'wall_seconds_before_receipt':1.5,'errors':[{'http_status':402}]}
        path.write_text(dumps(trial),encoding='utf-8')
        args=argparse.Namespace(inputs=ROOT/'test/identity-calibration/inputs.json',
            labels=ROOT/'test/identity-calibration/reference/labels.json',answers=path,out=Path(folder)/'score.json')
        return args,trial

    def test_scoring_partial_online_trial_preserves_missing_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            args,_=self.score_fixture(folder)
            score_online(args)
            result=json.loads(args.out.read_text(encoding='utf-8'))
            row=next(x for x in result['metrics'] if x['language']=='all' and x['mode']=='wide')
            self.assertEqual((row['n'],row['unmeasured'],row['review']),(2,2,0))
            self.assertEqual(result['trial_status'],'blocked')

    def test_scoring_rejects_duplicate_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            args,trial=self.score_fixture(folder)
            trial['answers']*=2
            args.answers.write_text(dumps(trial),encoding='utf-8')
            with self.assertRaises(ValueError):
                score_online(args)

    def test_scoring_rejects_changed_input_version(self):
        with tempfile.TemporaryDirectory() as folder:
            args,trial=self.score_fixture(folder)
            trial['inputs_sha256']='wrong'
            args.answers.write_text(dumps(trial),encoding='utf-8')
            with self.assertRaises(ValueError):
                score_online(args)


if __name__ == '__main__':
    unittest.main()
