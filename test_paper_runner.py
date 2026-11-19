import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import paper_runner as r
import predictor as p
from test_predictor import market, candles


def study_market():
    m = market()
    now = time.time()
    m['open_time'] = p.datetime.fromtimestamp(now - 310, p.timezone.utc).isoformat()
    m['close_time'] = p.datetime.fromtimestamp(now + 590, p.timezone.utc).isoformat()
    m['event_ticker'] = 'BTC-EVENT'
    return m


def context(m):
    return dict(observed_at=p.utcnow(), ticker=m['ticker'], close_time=m['close_time'],
                floor_strike=m['floor_strike'], underlying=p.candle_context(candles(), time.time()))


def snapshot(m):
    return dict(market=m, received_at=p.utcnow(), book_requested_at=p.utcnow(),
                book={'orderbook_fp': {'yes_dollars': [['.49', '20']], 'no_dollars': [['.49', '20']]}},
                series={'fee_type': 'quadratic', 'fee_multiplier': 1}, event={})


class PaperTests(unittest.TestCase):
    def test_fixed_timing(self):
        m = study_market()
        self.assertTrue(r.eligible_window(m, time.time()))
        self.assertFalse(r.eligible_window(m, time.time() - 100))
        self.assertFalse(r.eligible_window(m, time.time() + 100))

    def test_depth_liquidity_and_slippage(self):
        book = {'orderbook_fp': {'yes_dollars': [['.49', '10']],
                                'no_dollars': [['.48', '.75'], ['.49', '.25']]}}
        midpoint, quotes = r.book_quotes(book)
        self.assertAlmostEqual(midpoint, .5)
        self.assertAlmostEqual(quotes['yes'], .5175)
        side, entry, fee, edge = r.paper_entry(.7, quotes, .07)
        self.assertEqual(side, 'yes')
        self.assertAlmostEqual(entry, .5275)
        self.assertEqual(fee, .02)
        self.assertAlmostEqual(edge, .7 - .5275 - .02)
        self.assertIsNone(r.paper_entry(.5, quotes, .07)[0])
        book['orderbook_fp']['no_dollars'] = [['.49', '.5']]
        self.assertIsNone(r.book_quotes(book)[1]['yes'])
        book['orderbook_fp']['no_dollars'] = [['.8', '1']]
        with self.assertRaises(ValueError):
            r.book_quotes(book)

    def test_fees_overrides_and_unknown(self):
        self.assertAlmostEqual(r.fee_rate({'fee_type': 'quadratic', 'fee_multiplier': 2}, {}), .14)
        self.assertAlmostEqual(r.fee_rate({'fee_type': 'quadratic', 'fee_multiplier': 2},
                                        {'fee_multiplier_override': 0}), 0)
        with self.assertRaises(ValueError):
            r.fee_rate({'fee_type': 'unknown', 'fee_multiplier': 1}, {})

    def test_manifest_pinned_and_single_process_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'study.sqlite3'
            db = r.connect(path)
            try:
                r.pin_study(db, 1)
                r.pin_study(db, 1)
                with self.assertRaises(ValueError):
                    r.pin_study(db, 2)
                with r.collector_lock(path):
                    with self.assertRaises(RuntimeError):
                        with r.collector_lock(path):
                            pass
            finally:
                db.close()

    def test_forward_lifecycle_cap_and_paired_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = r.connect(Path(tmp) / 'study.sqlite3')
            r.pin_study(db, 1)
            m = study_market()
            captured = context(m)
            try:
                with patch.object(p, 'open_markets', return_value=[m]), \
                     patch.object(p, 'prepare_context', return_value=captured), \
                     patch.object(p, 'forecast', return_value=({'p_up': .7}, {'usage': {'estimated_cost': .01}})) as call, \
                     patch.object(r, 'fresh_snapshot', return_value=snapshot(m)) as snap, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(r.collect_once(db))
                    self.assertFalse(r.collect_once(db))
                self.assertEqual(call.call_count, 5)
                self.assertEqual(snap.call_count, 1)
                for invocation in call.call_args_list:
                    self.assertIs(invocation.args[0], captured)
                    self.assertEqual(invocation.kwargs, r.OPTIONS)
                self.assertEqual(db.execute("SELECT count(*) FROM calls WHERE side='yes'").fetchone()[0], 5)
                c = db.execute('SELECT * FROM cohorts').fetchone()
                self.assertLess(p.timestamp(c['decision_at']), p.timestamp(c['close_time']))
                with db:
                    db.execute("UPDATE cohorts SET close_time='2020-01-01T00:00:00Z'")
                closed = {**m, 'status': 'closed', 'result': ''}
                with patch.object(p, 'kalshi', return_value={'market': closed}):
                    r.settle(db)
                self.assertIsNone(db.execute('SELECT result FROM cohorts').fetchone()[0])
                final = {**m, 'status': 'finalized', 'result': 'yes'}
                with patch.object(p, 'kalshi', return_value={'market': final}), contextlib.redirect_stdout(io.StringIO()):
                    r.settle(db)
                output = io.StringIO()
                with patch.object(p, 'request', side_effect=AssertionError('network forbidden')), contextlib.redirect_stdout(output):
                    r.report(db)
                self.assertIn('complete paired settled cases 1/1', output.getvalue())
                self.assertIn('0.0900', output.getvalue())
            finally:
                db.close()

    def test_error_and_late_quote_are_not_fabricated_fills(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = r.connect(Path(tmp) / 'study.sqlite3')
            r.pin_study(db, 1)
            m = study_market()
            try:
                def forecast(ctx, model, **kwargs):
                    if model == r.MODELS[0]:
                        raise RuntimeError('provider failed')
                    return {'p_up': .7}, {}
                with patch.object(p, 'open_markets', return_value=[m]), \
                     patch.object(p, 'prepare_context', return_value=context(m)), \
                     patch.object(p, 'forecast', side_effect=forecast) as call, \
                     patch.object(r, 'fresh_snapshot', side_effect=ValueError('too late')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    r.collect_once(db)
                    r.collect_once(db)
                self.assertEqual(call.call_count, 5)
                self.assertEqual(db.execute("SELECT count(*) FROM calls WHERE status='error'").fetchone()[0], 1)
                self.assertEqual(db.execute('SELECT count(*) FROM calls WHERE side IS NOT NULL').fetchone()[0], 0)
                self.assertEqual(db.execute('SELECT state FROM cohorts').fetchone()[0], 'discarded')
            finally:
                db.close()

    def test_interrupted_paid_attempts_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = r.connect(Path(tmp) / 'study.sqlite3')
            r.pin_study(db, 1)
            with db:
                db.execute('INSERT INTO cohorts (ticker,state,context_json,observed_at,close_time) VALUES (?,?,?,?,?)',
                           ('BTC', 'collecting', '{}', p.utcnow(), p.utcnow()))
                db.execute('INSERT INTO calls (ticker,model,status,started_at) VALUES (?,?,?,?)',
                           ('BTC', r.MODELS[0], 'pending', p.utcnow()))
            r.recover_interrupted(db)
            self.assertEqual(db.execute('SELECT status FROM calls').fetchone()[0], 'interrupted')
            self.assertEqual(db.execute('SELECT state FROM cohorts').fetchone()[0], 'discarded')
            db.close()

    def test_pnl_and_drawdown(self):
        trades, pnl, spent, dd = r.pnl_stats([('yes', .5, .02), ('yes', .5, .02), (None, None, None)], [1, 0, 1])
        self.assertEqual(trades, 2)
        self.assertAlmostEqual(pnl, -.04)
        self.assertAlmostEqual(spent, 1.04)
        self.assertAlmostEqual(dd, .52)


if __name__ == '__main__':
    unittest.main()
