import argparse
import contextlib
import io
import json
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import predictor as p
import compare


def market():
    now = time.time()
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
    return dict(ticker='BTC-TEST', title='BTC up?', status='active', result='',
                open_time=iso(now - 120), close_time=iso(now + 600),
                floor_strike=63000, rules_primary='Average at close >= average at open.',
                yes_bid_dollars='.49', yes_ask_dollars='.51', no_ask_dollars='.51',
                yes_bid_size_fp='10', yes_ask_size_fp='10')


def candles():
    start = int(time.time() // 60) * 60
    return [[start - 60 * i, 62990, 63010, 63000, 63000 + i, 2] for i in range(100)]


class Tests(unittest.TestCase):
    def test_forecast_validation(self):
        self.assertEqual(p.parse_forecast('{"p_up":0.6,"reason":"x"}')['p_up'], .6)
        for value in ('true', 'NaN', '1.1', '-0.1', '"0.6"', 'null'):
            with self.assertRaises(ValueError):
                p.parse_forecast('{"p_up":' + value + ',"reason":"x"}')

    def test_decision_model_api(self):
        context = {'ticker': 'BTC-TEST'}
        for model in p.DECISION_MODELS:
            response = {'model': model, 'answers': {'up': {'type': 'noul', 'noul': .62}},
                        'usage': {'input_tokens': 100, 'output_tokens': 1}}
            with patch.dict(p.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
                 patch.object(p, 'request', return_value=response) as call:
                prediction, raw = p.forecast(context, model)
            self.assertEqual(prediction['p_up'], .62)
            self.assertIs(raw, response)
            self.assertEqual(call.call_args.args[0], 'https://api.deepinfra.com/v1/decisions')
            payload = call.call_args.kwargs['payload']
            self.assertEqual(payload['state'], context)
            self.assertEqual(payload['questions']['up']['type'], 'noul')
            self.assertNotIn('messages', payload)
            self.assertNotIn('max_tokens', payload)
        for invalid in (True, float('nan'), 1.2, '0.5'):
            with patch.dict(p.os.environ, {'DEEPINFRA_API_KEY': 'test-key'}), \
                 patch.object(p, 'request', return_value={'answers': {'up': {'type': 'noul', 'noul': invalid}}}):
                with self.assertRaises(ValueError):
                    p.forecast(context, p.DECISION_MODELS[0])

    def test_paired_inputs_quotes_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            databases = {model: p.connect(Path(tmp) / f'{i}.sqlite3') for i, model in enumerate(p.DECISION_MODELS)}
            m = market()
            args = argparse.Namespace(min_edge=.05, fee_rate=.07)
            try:
                with patch.object(p, 'open_markets', return_value=[m]), \
                     patch.object(p, 'request', return_value=candles()), \
                     patch.object(p, 'forecast', return_value=({'p_up': .6, 'reason': 'x'}, {})) as call, \
                     patch.object(p, 'kalshi', return_value={'market': m}) as refresh, \
                     contextlib.redirect_stdout(io.StringIO()):
                    compare.compare_once(databases, args)
                    compare.compare_once(databases, args)
                self.assertEqual(call.call_count, 2)
                self.assertEqual(refresh.call_count, 1)
                self.assertIs(call.call_args_list[0].args[0], call.call_args_list[1].args[0])
                rows = [db.execute('SELECT * FROM predictions').fetchone() for db in databases.values()]
                self.assertEqual(rows[0]['context_json'], rows[1]['context_json'])
                self.assertEqual(rows[0]['decision_market_json'], rows[1]['decision_market_json'])
                self.assertEqual({r['model'] for r in rows}, set(p.DECISION_MODELS))
            finally:
                for db in databases.values():
                    db.close()

    def test_eligibility(self):
        m = market()
        self.assertTrue(p.eligible(m, time.time(), 90))
        self.assertFalse(p.eligible(m, time.time() + 590, 90))
        m['result'] = 'yes'
        self.assertFalse(p.eligible(m, time.time(), 90))

    def test_completed_candles_only(self):
        now = time.time()
        context = p.candle_context(candles(), now)
        self.assertTrue(all(r[0] + 60 <= now for r in context['completed_candles']))
        with self.assertRaises(ValueError):
            p.candle_context(candles(), now + 400)
        with self.assertRaises(ValueError):
            p.candle_context(candles()[::2], now)

    def test_decisions(self):
        m = market()
        self.assertEqual(p.decision(m, .5, .05, .07)[1], None)
        self.assertEqual(p.decision(m, .7, .05, .07)[1], 'yes')
        self.assertEqual(p.decision(m, .3, .05, .07)[1], 'no')
        self.assertEqual(p.estimated_fee(.5, .07), .02)
        m['yes_bid_size_fp'] = '0'
        self.assertIsNone(p.decision(m, .3, .05, .07)[1])
        m['yes_bid_dollars'] = '.8'
        with self.assertRaises(ValueError):
            p.decision(m, .7, .05, .07)

    def test_forward_lifecycle_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = p.connect(Path(tmp) / 'db.sqlite3')
            m = market()
            args = argparse.Namespace(series='KXBTC15M', min_seconds=90, model='test', min_edge=.05, fee_rate=.07)
            response = {'usage': {'total_tokens': 10}}
            with patch.object(p, 'open_markets', return_value=[m]), \
                 patch.object(p, 'request', return_value=candles()), \
                 patch.object(p, 'forecast', return_value=({'p_up': .7, 'reason': 'test'}, response)) as call, \
                 patch.object(p, 'kalshi', return_value={'market': m}), \
                 contextlib.redirect_stdout(io.StringIO()):
                p.predict(db, args)
                p.predict(db, args)
                self.assertEqual(call.call_count, 1)
            row = db.execute('SELECT * FROM predictions').fetchone()
            self.assertLess(p.timestamp(row['recorded_at']), p.timestamp(row['close_time']))
            self.assertEqual(row['side'], 'yes')
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('INSERT INTO predictions SELECT * FROM predictions')
            db.rollback()
            # Closing alone is not settlement; only finalized yes/no outcomes count.
            with db:
                db.execute("UPDATE predictions SET close_time='2020-01-01T00:00:00Z'")
            m['status'], m['result'] = 'closed', ''
            with patch.object(p, 'kalshi', return_value={'market': m}):
                p.settle(db)
            self.assertEqual(db.execute('SELECT count(*) FROM outcomes').fetchone()[0], 0)
            m['status'], m['result'] = 'finalized', 'yes'
            with patch.object(p, 'kalshi', return_value={'market': m}), contextlib.redirect_stdout(io.StringIO()):
                p.settle(db)
                p.settle(db)
            self.assertEqual(db.execute('SELECT count(*) FROM outcomes').fetchone()[0], 1)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                p.report(db)
            self.assertIn('Brier 0.0900', output.getvalue())
            self.assertIn('net P&L $+0.47', output.getvalue())
            db.close()

    def test_late_response_discarded_and_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = p.connect(Path(tmp) / 'db.sqlite3')
            m, closed = market(), market()
            closed['status'] = 'closed'
            args = argparse.Namespace(series='KXBTC15M', min_seconds=90, model='test', min_edge=.05, fee_rate=.07)
            with patch.object(p, 'open_markets', return_value=[m]), \
                 patch.object(p, 'request', return_value=candles()), \
                 patch.object(p, 'forecast', return_value=({'p_up': .7, 'reason': 'x'}, {})) as call, \
                 patch.object(p, 'kalshi', return_value={'market': closed}), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    p.predict(db, args)
                p.predict(db, args)
                self.assertEqual(call.call_count, 1)
            self.assertEqual(db.execute('SELECT count(*) FROM predictions').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT count(*) FROM attempts').fetchone()[0], 1)
            db.close()


if __name__ == '__main__':
    unittest.main()
