from __future__ import annotations

import argparse
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace import resolution
from entitytrace.config import Service
from entitytrace.models import ModelError
from entitytrace.util import dumps
from tools.identity_calibration import load_inputs
from tools.identity_chain_experiment import ROOT, online, outcome_metrics, run_case, score


def answer(choice='same', probability=.92):
    return {'type': 'choice', 'choice': choice, 'confidence': .9,
            'probabilities': {v: probability if v == choice else (1 - probability) / 2
                              for v in ('same', 'different', 'uncertain')}}


class Model:
    def __init__(self, guard='same'):
        self.guard, self.calls = guard, []

    def choices(self, state, questions, stage):
        self.calls.append((state, questions, stage))
        if stage == 'identity-group-guard' and self.guard == 'fail':
            raise ModelError('模拟复核请求失败')
        choice = self.guard if stage == 'identity-group-guard' else 'same'
        return {key: answer(choice) for key in questions}, str(len(self.calls))


class ChainExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.sources = load_inputs(ROOT / 'test/identity-calibration/inputs.json')

    def test_relaxed_policy_keeps_guard_and_restores_default_on_failure(self):
        original = resolution.POLICY
        model = Model('fail')
        with self.assertRaises(ModelError):
            run_case(model, self.sources['zh'], self.inputs['cases'][0], .9)
        self.assertIs(resolution.POLICY, original)
        self.assertEqual([c[2] for c in model.calls], ['identity-group', 'identity-group-guard'])

    def test_disagreement_remains_review_and_recorded_policy_is_immutable(self):
        case = self.inputs['cases'][0]
        result = run_case(Model('different'), self.sources['zh'], case, .9)
        self.assertEqual((result['choice'], result['accepted']), ('uncertain', False))
        strict = run_case(Model(), self.sources['zh'], case, .95)
        self.assertEqual(strict['choice'], 'uncertain')
        self.assertTrue(all(d['policy']['min_probability'] == .9 for d in result['decisions']))
        self.assertEqual(resolution.POLICY['min_probability'], .95)

    def test_labels_and_case_metadata_never_enter_model_questions(self):
        case = copy.deepcopy(self.inputs['cases'][0])
        case.update(label='SECRET_LABEL', rationale='SECRET_RATIONALE', cohort='SECRET_COHORT')
        case['targets'][0]['label'] = 'SECRET_TARGET'
        model = Model()
        run_case(model, self.sources['zh'], case, .9)
        sent = dumps(model.calls)
        self.assertNotIn('SECRET_', sent)
        self.assertNotIn(case['id'], sent)

    def test_metrics_preserve_review_missing_and_false_merges(self):
        result = outcome_metrics([
            {'label': 'same', 'choice': 'uncertain'}, {'label': 'same', 'choice': 'unmeasured'},
            {'label': 'different', 'choice': 'same'}, {'label': 'different', 'choice': 'different'}])
        self.assertEqual((result['n'], result['review'], result['unmeasured'], result['wrong']), (4, 1, 1, 1))
        self.assertEqual(result['merge_recall'], {'numerator': 0, 'denominator': 2, 'value': 0.0})
        self.assertEqual(outcome_metrics([])['auto_accuracy']['value'], None)

    def test_budget_interruption_retains_incomplete_tasks_and_calls(self):
        def success(request, timeout):
            payload = json.loads(request.data)
            return io.BytesIO(dumps({'model': 'jev-1.13.0',
                'answers': {qid: answer() for qid in payload['questions']}}).encode())
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(inputs=ROOT / 'test/identity-calibration/inputs.json',
                labels=Path('must-not-read-online.json'), out=Path(folder) / 'online.json',
                max_requests=1, max_seconds=60)
            with patch('tools.identity_chain_experiment.load_services', return_value={
                    'decision': Service('decision', 'https://example.invalid', 'jev-1.13.0', 'test')}), \
                    patch('urllib.request.urlopen', side_effect=success) as http, \
                    patch('tools.identity_chain_experiment.load_labels', side_effect=AssertionError('标签泄漏')):
                self.assertEqual(online(args), 2)
            trial = json.loads(args.out.read_text(encoding='utf-8'))
            self.assertEqual(http.call_count, 1)
            self.assertEqual((trial['status'], trial['missing_results']), ('budget_exhausted', 64))
            self.assertEqual(len(trial['calls']), 1)
            self.assertEqual(resolution.POLICY['min_probability'], .95)
            args.answers, args.out = args.out, Path(folder) / 'score.json'
            args.labels = ROOT / 'test/identity-calibration/reference/labels.json'
            score(args)
            result = json.loads(args.out.read_text(encoding='utf-8'))
            first = result['metrics'][0]
            self.assertEqual((first['n'], first['unmeasured']), (32, 32))
            self.assertIsNone(first['auto_accuracy']['value'])


if __name__ == '__main__':
    unittest.main()
