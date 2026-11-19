import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import offline_eval as e
import predictor as p
from test_predictor import candles, market


def sample():
    m = market()
    m['status'], m['result'] = 'finalized', 'yes'
    opened = p.timestamp(m['open_time'])
    m['close_time'] = e.iso(opened + 900)
    return e.make_case(m, candles())


class EvalTests(unittest.TestCase):
    def test_cutoff_and_label_separation(self):
        case = sample()
        cutoff = p.timestamp(case['context']['observed_at'])
        self.assertEqual(case['label'], 1)
        self.assertNotIn('label', case['context'])
        self.assertNotIn('result', case['context'])
        self.assertTrue(all(r[0] + 60 <= cutoff - 60
                            for r in case['context']['underlying']['completed_candles']))
        bad = copy.deepcopy(case)
        bad['context']['result'] = 'yes'
        with self.assertRaises(ValueError):
            e.validate_case(bad)
        bad = copy.deepcopy(case)
        bad['context']['underlying']['completed_candles'][-1][0] = cutoff
        with self.assertRaises(ValueError):
            e.validate_case(bad)
        bad = copy.deepcopy(case)
        bad['context']['underlying']['last_close'] += 100
        with self.assertRaises(ValueError):
            e.validate_case(bad)

    def test_metrics_and_hashes(self):
        brier, acc, loss = e.metrics([.7, .2], [1, 0])
        self.assertAlmostEqual(brier, .065)
        self.assertEqual(acc, 1)
        self.assertEqual(e.metrics([.5, .5], [1, 0])[:2], (.25, .5))
        self.assertGreater(loss, 0)
        case = sample()
        key = e.cache_key(case, 'model', 'engine')
        self.assertNotEqual(key, e.cache_key(case, 'other', 'engine'))
        self.assertNotEqual(key, e.cache_key(case, 'model', 'changed-engine'))
        changed = copy.deepcopy(case)
        changed['label'] = 0
        self.assertEqual(key, e.cache_key(changed, 'model', 'engine'))
        changed['context']['floor_strike'] += 1
        self.assertNotEqual(key, e.cache_key(changed, 'model', 'engine'))

    def test_paid_calls_cached_and_report_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = e.connect_cache(Path(tmp) / 'cache.sqlite3')
            case = sample()
            try:
                with patch.dict(e.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
                     patch.object(p, 'forecast', return_value=({'p_up': .7, 'reason': 'x'}, {'usage': {}})) as call, \
                     contextlib.redirect_stdout(io.StringIO()):
                    e.run(db, [case], ['a', 'b'], 2)
                    e.run(db, [case], ['a', 'b'], 2)
                self.assertEqual(call.call_count, 2)
                for invocation in call.call_args_list:
                    self.assertEqual(invocation.args[0], case['context'])
                    self.assertNotIn('label', invocation.args[0])
                    self.assertEqual(invocation.kwargs, e.EVAL_OPTIONS)
                output = io.StringIO()
                with patch.object(p, 'request', side_effect=AssertionError('network forbidden')), \
                     contextlib.redirect_stdout(output):
                    e.report(db, [case], ['a', 'b'])
                self.assertIn('1/1 cases completed', output.getvalue())
                self.assertIn('0.0900', output.getvalue())
                self.assertIn('Always 50%', output.getvalue())
            finally:
                db.close()

    def test_failure_coverage_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = e.connect_cache(Path(tmp) / 'cache.sqlite3')
            case = sample()
            try:
                def forecast(context, model, **options):
                    if model == 'bad':
                        raise ValueError('bad response')
                    return {'p_up': .7}, {}
                with patch.dict(e.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
                     patch.object(p, 'forecast', side_effect=forecast) as call, \
                     contextlib.redirect_stdout(io.StringIO()):
                    e.run(db, [case], ['good', 'bad'], 2)
                    e.run(db, [case], ['good', 'bad'], 2)
                self.assertEqual(call.call_count, 2)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    e.report(db, [case], ['good', 'bad'])
                self.assertIn('bad: coverage 0/1', output.getvalue())
                self.assertIn('No common cases', output.getvalue())
            finally:
                db.close()

    def test_dataset_roundtrip_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cases.jsonl'
            case = sample()
            path.write_text(json.dumps(case) + '\n')
            self.assertEqual(e.load_cases(path), [case])
            path.write_text((json.dumps(case) + '\n') * 2)
            with self.assertRaises(ValueError):
                e.load_cases(path)

    def test_reasoning_priority_and_budget(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"p_up":0.55,"reason":"noise"}'}}]}
        for model in ('moonshotai/Kimi-K3.1-Pro', *p.REASONING_MODELS, p.MODEL):
            with patch.dict(e.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
                 patch.object(p, 'request', return_value=response) as call:
                p.forecast({}, model, **e.EVAL_OPTIONS)
            payload = call.call_args.kwargs['payload']
            self.assertEqual(payload['service_tier'], 'priority')
            self.assertEqual(payload['max_tokens'], 8192)
            self.assertEqual(call.call_args.kwargs['timeout'], 180)
            if model == p.MODEL:
                self.assertNotIn('reasoning_effort', payload)
            else:
                self.assertEqual(payload['reasoning_effort'], 'high')
        self.assertNotEqual(e.engine_hash(), e.engine_hash({**e.EVAL_OPTIONS, 'reasoning_effort': 'none'}))
        self.assertNotEqual(e.engine_hash(), e.engine_hash({**e.EVAL_OPTIONS, 'service_tier': 'default'}))

    def test_decision_models_ignore_chat_flags(self):
        response = {'answers': {'up': {'type': 'noul', 'noul': .6}}}
        for model in p.DECISION_MODELS:
            with patch.dict(e.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
                 patch.object(p, 'request', return_value=response) as call:
                p.forecast({}, model, **e.EVAL_OPTIONS)
            payload = call.call_args.kwargs['payload']
            for field in ('reasoning_effort', 'service_tier', 'max_tokens'):
                self.assertNotIn(field, payload)

    def test_truncated_reasoning_is_failure(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': ''}}]}
        with patch.dict(e.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
             patch.object(p, 'request', return_value=response):
            with self.assertRaisesRegex(ValueError, 'includes reasoning'):
                p.forecast({}, 'moonshotai/Kimi-K3.1-Pro', **e.EVAL_OPTIONS)

    def test_kimi_live_fast_mode(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"p_up":0.55,"reason":"noise"}'}}]}
        with patch.dict(e.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
             patch.object(p, 'request', return_value=response) as call:
            p.forecast({}, 'moonshotai/Kimi-K3.1-Pro')
        self.assertEqual(call.call_args.kwargs['payload']['reasoning_effort'], 'none')


if __name__ == '__main__':
    unittest.main()
