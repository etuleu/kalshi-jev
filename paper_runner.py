#!/usr/bin/env python3
"""Forward-only, five-model BTC study. Public market reads and paper fills only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time
from urllib.parse import quote

import predictor as p

MODELS = (*p.DECISION_MODELS, 'moonshotai/Kimi-K3.1-Pro', *p.REASONING_MODELS)
OPTIONS = dict(reasoning_effort='high', service_tier='priority', max_tokens=8192, timeout=180)
DEFAULT_DB = p.ROOT / 'data' / 'forward-v1.sqlite3'
STRATEGY = dict(series='KXBTC15M', sample_after_seconds=300, sample_tolerance_seconds=60,
                min_edge=.05, slippage_dollars=.01, contracts=1, base_fee_rate=.07,
                min_seconds_to_close=60, max_context_age_seconds=240,
                fee_rounding='ceil-to-cent; conservative estimate, not account-specific fees')


def dumps(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def manifest(max_windows):
    return dict(models=MODELS, options=OPTIONS, strategy=STRATEGY, max_windows=max_windows,
                code_hash=hashlib.sha256(Path(__file__).read_bytes() + Path(p.__file__).read_bytes()).hexdigest())


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS study (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cohorts (
            ticker TEXT PRIMARY KEY, state TEXT NOT NULL, context_json TEXT NOT NULL,
            observed_at TEXT NOT NULL, close_time TEXT NOT NULL, decision_at TEXT,
            snapshot_json TEXT, market_p REAL, proxy_p REAL, error TEXT,
            result INTEGER CHECK(result IN (0,1)), settled_at TEXT, settlement_json TEXT
        );
        CREATE TABLE IF NOT EXISTS calls (
            ticker TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, seconds REAL, p_up REAL,
            response_json TEXT, error TEXT, side TEXT, entry REAL, fee REAL, edge REAL,
            PRIMARY KEY(ticker,model)
        );
    ''')
    return db


def pin_study(db, max_windows):
    value = dumps(manifest(max_windows))
    with db:
        db.execute('INSERT OR IGNORE INTO study VALUES (1,?,?)', (value, p.utcnow()))
    if db.execute('SELECT manifest FROM study').fetchone()[0] != value:
        raise ValueError('Study settings/code changed. Use a NEW --db; do not mix strategies mid-study.')


@contextmanager
def collector_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another collector/settler is using {path}') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def eligible_window(market, now):
    age = now - p.timestamp(market['open_time'])
    return (p.eligible(market, now, 300)
            and STRATEGY['sample_after_seconds'] <= age <=
            STRATEGY['sample_after_seconds'] + STRATEGY['sample_tolerance_seconds'])


def proxy_probability(context):
    data = context['underlying']
    sigma = data['one_minute_log_return_stdev']
    spot, target = data['last_close'], context['floor_strike']
    minutes = (p.timestamp(context['close_time']) - (data['completed_candles'][-1][0] + 60)) / 60
    if sigma <= 0:
        return 1. if spot > target else 0. if spot < target else .5
    z = math.log(spot / target) / (sigma * math.sqrt(minutes))
    return .5 * (1 + math.erf(z / math.sqrt(2)))


def finite(value, low, high):
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError('Invalid quote, size, or fee multiplier')
    return result


def book_quotes(book):
    """Convert public YES/NO bids to asks. Aggregate duplicate levels, require one contract."""
    sides = {}
    for side in ('yes', 'no'):
        levels = {}
        for price, size in book['orderbook_fp'][side + '_dollars']:
            price = finite(price, 0, 1)
            size = finite(size, 0, 1e18)
            if 0 < price < 1 and size:
                levels[price] = levels.get(price, 0) + size
        if not levels:
            raise ValueError('Empty order book')
        sides[side] = levels
    yes_bid, no_bid = max(sides['yes']), max(sides['no'])
    if yes_bid + no_bid > 1 + 1e-9:
        raise ValueError('Crossed order book')
    quotes = {}
    for side, opposing in (('yes', 'no'), ('no', 'yes')):
        remaining, cost = 1., 0.
        for bid, size in sorted(sides[opposing].items(), reverse=True):
            take = min(remaining, size)
            cost += take * (1 - bid)
            remaining -= take
            if remaining <= 1e-9:
                break
        quotes[side] = None if remaining > 1e-9 else cost
    return (yes_bid + (1 - no_bid)) / 2, quotes


def fee_rate(series, event):
    kind = event.get('fee_type_override') or series.get('fee_type')
    multiplier = event.get('fee_multiplier_override')
    if multiplier is None:
        multiplier = series.get('fee_multiplier')
    if kind not in ('quadratic', 'quadratic_with_maker_fees') or multiplier is None:
        raise ValueError('Unknown fee schedule; no paper entry')
    return STRATEGY['base_fee_rate'] * finite(multiplier, 0, 100)


def paper_entry(probability, quotes, rate):
    candidates = []
    for side, prob in (('yes', probability), ('no', 1 - probability)):
        raw = quotes.get(side)
        if raw is None:
            continue
        # Deliberately worse than the depth-weighted executable ask, not an optimistic fill.
        entry = round(raw + STRATEGY['slippage_dollars'], 8)
        if not 0 < entry < 1:
            continue
        fee = p.estimated_fee(entry, rate)
        edge = prob - entry - fee
        candidates.append((edge, side, entry, fee))
    if not candidates:
        return None, None, None, None
    edge, side, entry, fee = max(candidates)
    return (side, entry, fee, edge) if edge >= STRATEGY['min_edge'] else (None, None, None, edge)


def timed_call(context, model):
    started_at, start = p.utcnow(), time.monotonic()
    try:
        prediction, response = p.forecast(context, model, **OPTIONS)
        return dict(status='ok', started_at=started_at, finished_at=p.utcnow(),
                    seconds=time.monotonic() - start, p_up=prediction['p_up'], response=response)
    except Exception as exc:
        # Record provider/schema failures; no automatic retry of a paid attempt.
        return dict(status='error', started_at=started_at, finished_at=p.utcnow(),
                    seconds=time.monotonic() - start, error=f'{type(exc).__name__}: {exc}')


def fresh_snapshot(context):
    ticker = quote(context['ticker'], safe='')
    series = p.kalshi('/series/KXBTC15M')['series']
    event_ticker = context['ticker'].rsplit('-', 1)[0]
    event = p.kalshi('/events/' + quote(event_ticker, safe=''))['event']
    market = p.kalshi('/markets/' + ticker)['market']
    book_started = p.utcnow()
    book = p.kalshi('/markets/' + ticker + '/orderbook', depth=100)
    received = p.utcnow()
    if time.time() - p.timestamp(book_started) > 10:
        raise ValueError('Slow order-book fetch; no decision recorded')
    if (market['ticker'] != context['ticker'] or market['close_time'] != context['close_time']
            or market.get('floor_strike') != context['floor_strike']
            or not p.eligible(market, time.time(), STRATEGY['min_seconds_to_close'])
            or time.time() - p.timestamp(context['observed_at']) > STRATEGY['max_context_age_seconds']):
        raise ValueError('Market/target changed, forecast stale, or too near close')
    return dict(market=market, book=book, series=series, event=event,
                book_requested_at=book_started, received_at=received)


def collect_once(db):
    config = json.loads(db.execute('SELECT manifest FROM study').fetchone()[0])
    if db.execute('SELECT count(*) FROM cohorts').fetchone()[0] >= config['max_windows']:
        print('Collection cap reached; no more paid calls.', flush=True)
        return False
    market = next((m for m in p.open_markets('KXBTC15M')
                   if eligible_window(m, time.time())
                   and not db.execute('SELECT 1 FROM cohorts WHERE ticker=?', (m['ticker'],)).fetchone()), None)
    if market is None:
        return False
    context = p.prepare_context(market)
    finite(context['floor_strike'], 1, 1e12)
    if not eligible_window(market, time.time()):
        return False
    # Unique durable reservation BEFORE any paid request, in one transaction for all models.
    with db:
        db.execute('INSERT INTO cohorts (ticker,state,context_json,observed_at,close_time,proxy_p) VALUES (?,?,?,?,?,?)',
                   (context['ticker'], 'collecting', dumps(context), context['observed_at'], context['close_time'],
                    proxy_probability(context)))
        for model in MODELS:
            db.execute('INSERT INTO calls (ticker,model,status,started_at) VALUES (?,?,?,?)',
                       (context['ticker'], model, 'pending', p.utcnow()))
    print(f"Collecting {context['ticker']} | {len(MODELS)} paid requests | input {context['observed_at']}", flush=True)
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {pool.submit(timed_call, context, model): model for model in MODELS}
        for future in as_completed(futures):
            model, result = futures[future], future.result()
            with db:
                db.execute('''UPDATE calls SET status=?,started_at=?,finished_at=?,seconds=?,p_up=?,response_json=?,error=?
                              WHERE ticker=? AND model=?''',
                           (result['status'], result['started_at'], result['finished_at'], result['seconds'],
                            result.get('p_up'), dumps(result['response']) if 'response' in result else None,
                            result.get('error'), context['ticker'], model))
            print(f"  {model}: {result['status']} ({result['seconds']:.1f}s)", flush=True)
    # Common decision barrier AFTER all calls. Slower models delay everybody; this is intentional.
    try:
        snapshot = fresh_snapshot(context)
        with db:
            db.execute('UPDATE cohorts SET snapshot_json=? WHERE ticker=?', (dumps(snapshot), context['ticker']))
        midpoint, quotes = book_quotes(snapshot['book'])
        rate = None
        try:
            rate = fee_rate(snapshot['series'], snapshot['event'])
        except ValueError as exc:
            snapshot['fee_warning'] = str(exc)
        if rate is not None and snapshot['market'].get('fee_waiver_expiration_time'):
            # Do not claim exact handling of waivers; retaining the fee is conservative.
            snapshot['fee_warning'] = 'Fee waiver present; estimate conservatively does not apply waiver'
        snapshot['estimated_fee_rate'] = rate
        snapshot['quotes_for_one_contract'] = quotes
        # Recheck local time immediately before the durable decision write.
        now = time.time()
        if (p.timestamp(context['close_time']) - now < STRATEGY['min_seconds_to_close']
                or now - p.timestamp(snapshot['received_at']) > 10):
            raise ValueError('Decision arrived late; no paper entries')
        with db:
            db.execute('UPDATE cohorts SET state=?,decision_at=?,snapshot_json=?,market_p=? WHERE ticker=?',
                       ('recorded', p.utcnow(), dumps(snapshot), midpoint, context['ticker']))
            for call in db.execute("SELECT model,p_up FROM calls WHERE ticker=? AND status='ok'", (context['ticker'],)).fetchall():
                entry = paper_entry(call['p_up'], quotes, rate) if rate is not None else (None, None, None, None)
                db.execute('UPDATE calls SET side=?,entry=?,fee=?,edge=? WHERE ticker=? AND model=?',
                           (*entry, context['ticker'], call['model']))
        print(f"Recorded decision-time book: YES midpoint {midpoint:.1%}", flush=True)
    except (RuntimeError, ValueError, KeyError) as exc:
        with db:
            db.execute('UPDATE cohorts SET state=?,error=? WHERE ticker=?', ('discarded', str(exc), context['ticker']))
        print(f'No paper entries: {exc}', flush=True)
    return True


def recover_interrupted(db):
    with db:
        db.execute("UPDATE calls SET status='interrupted',error='Collector interrupted; no automatic retry' WHERE status='pending'")
        db.execute("UPDATE cohorts SET state='discarded',error='Interrupted before shared decision' WHERE state='collecting'")


def settle(db):
    for cohort in db.execute('SELECT ticker,close_time FROM cohorts WHERE result IS NULL').fetchall():
        if p.timestamp(cohort['close_time']) > time.time():
            continue
        ticker = quote(cohort['ticker'], safe='')
        try:
            market = p.kalshi('/markets/' + ticker)['market']
        except RuntimeError as exc:
            if 'HTTP 404 ' not in str(exc):
                print(f"Settlement deferred {cohort['ticker']}: {exc}", flush=True)
                continue
            market = p.kalshi('/historical/markets/' + ticker)['market']
        if market.get('status') not in ('finalized', 'settled') or market.get('result') not in ('yes', 'no'):
            continue
        with db:
            db.execute('UPDATE cohorts SET result=?,settled_at=?,settlement_json=? WHERE ticker=? AND result IS NULL',
                       (int(market['result'] == 'yes'), p.utcnow(), dumps(market), cohort['ticker']))
        print(f"Settled {cohort['ticker']}: {market['result']}", flush=True)


def metrics(probabilities, labels):
    return (statistics.mean((prob - label) ** 2 for prob, label in zip(probabilities, labels)),
            statistics.mean(.5 if prob == .5 else int((prob > .5) == bool(label))
                            for prob, label in zip(probabilities, labels)))


def pnl_stats(entries, labels):
    pnl, spent, trades, peak, drawdown = 0., 0., 0, 0., 0.
    for (side, entry, fee), label in zip(entries, labels):
        if side:
            pnl += int((side == 'yes') == bool(label)) - entry - fee
            spent += entry + fee
            trades += 1
        peak = max(peak, pnl)
        drawdown = max(drawdown, peak - pnl)
    return trades, pnl, spent, drawdown


def report(db):
    study = db.execute('SELECT * FROM study').fetchone()
    if not study:
        print('No study initialized.')
        return
    config = json.loads(study['manifest'])
    cohorts = db.execute('SELECT * FROM cohorts ORDER BY observed_at').fetchall()
    calls = {(r['ticker'], r['model']): r for r in db.execute('SELECT * FROM calls')}
    recorded = [c for c in cohorts if c['state'] == 'recorded']
    paired = [c for c in recorded if c['result'] is not None and all(
        calls.get((c['ticker'], m)) is not None and calls[c['ticker'], m]['status'] == 'ok'
        for m in config['models'])]
    cap = config['max_windows']
    print(f"\nStudy started {study['created_at']} | fixed window cap {cap} (at most {cap * len(config['models'])} paid attempts)")
    print(f'Windows reserved {len(cohorts)}/{cap}; decisions {len(recorded)}; '
          f'official outcomes {sum(c["result"] is not None for c in cohorts)}; '
          f'complete paired settled cases {len(paired)}/{cap}')
    print(f'Discarded windows {sum(c["state"] == "discarded" for c in cohorts)}; '
          f'pending/interrupted/error calls {sum(c["status"] != "ok" for c in calls.values())}')
    reported_cost, missing_cost = 0., 0
    for model in config['models']:
        group = [r for r in calls.values() if r['model'] == model]
        ok = [r for r in group if r['status'] == 'ok']
        latencies = [r['seconds'] for r in group if r['seconds'] is not None]
        print(f'{model}: {len(ok)}/{len(group)} successful; mean latency '
              + (f'{statistics.mean(latencies):.1f}s' if latencies else 'n/a'))
        for row in group:
            raw = json.loads(row['response_json']) if row['response_json'] else {}
            cost = raw.get('usage', {}).get('estimated_cost')
            if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0:
                reported_cost += cost
            else:
                missing_cost += 1
    print(f'Provider-reported estimated inference cost ${reported_cost:.4f}; '
          f'{missing_cost} calls lack dollar estimates (not total billing).')
    if not paired:
        return
    labels = [c['result'] for c in paired]
    print(f'\nPAIRED scores only ({len(paired)} identical cases); UP rate {statistics.mean(labels):.1%}')
    print(f'{"Model/baseline":47} {"Brier↓":>8} {"Acc↑":>7} {"Trades":>7} {"P&L est.":>10} {"ROI":>8} {"Max DD":>8}')
    scores = []
    for model in config['models']:
        group = [calls[c['ticker'], model] for c in paired]
        brier, accuracy = metrics([r['p_up'] for r in group], labels)
        stats = pnl_stats([(r['side'], r['entry'], r['fee']) for r in group], labels)
        scores.append((model, brier, accuracy, stats))
    for name, probabilities in [('Market decision-time midpoint', [c['market_p'] for c in paired]),
                                ('Driftless price/volatility proxy', [c['proxy_p'] for c in paired]),
                                ('Always 50%', [.5] * len(paired))]:
        entries = []
        for probability, cohort in zip(probabilities, paired):
            snapshot = json.loads(cohort['snapshot_json'])
            rate = snapshot['estimated_fee_rate']
            entry = paper_entry(probability, snapshot['quotes_for_one_contract'], rate) if rate is not None else (None, None, None, None)
            entries.append(entry[:3])
        brier, accuracy = metrics(probabilities, labels)
        scores.append((name, brier, accuracy, pnl_stats(entries, labels)))
    for name, brier, accuracy, (trades, pnl, spent, drawdown) in sorted(scores, key=lambda r: r[1]):
        roi = f'{pnl / spent:+.1%}' if spent else 'n/a'
        print(f'{name:47} {brier:8.4f} {accuracy:6.1%} {trades:7} ${pnl:+8.2f} {roi:>8} ${drawdown:6.2f}')
    print('Daily UTC paired counts:', ', '.join(f'{day}: {sum(c["observed_at"][:10] == day for c in paired)}'
          for day in sorted({c['observed_at'][:10] for c in paired})))
    print('P&L uses displayed depth + $0.01 adverse slippage and cent-rounded estimated fees; NOT guaranteed fills.')
    print('P&L excludes inference costs. No actual orders, account-specific fees, or trading-edge claims.')
    print('Adjacent windows are correlated; keep settings fixed and inspect held-out/day-by-day results.')


def main():
    p.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['predict', 'watch', 'settle', 'report'])
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--max-windows', type=int, default=500, help='Hard cap on reserved windows, not guaranteed successes')
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.max_windows <= 2000 or not 5 <= args.interval <= 60:
        parser.error('Require max-windows 1..2000 and interval 5..60')
    if args.command == 'report':
        if not args.db.exists():
            parser.error('No study yet. Start predict/watch first.')
        db = sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            report(db)
        finally:
            db.close()
        return
    if args.command in ('predict', 'watch') and not os.environ.get('DEEPINFRA_API_KEY'):
        parser.error('Set DEEPINFRA_API_KEY in .env')
    with collector_lock(args.db):
        db = connect(args.db)
        try:
            if args.command == 'settle':
                settle(db)
                report(db)
                return
            pin_study(db, args.max_windows)
            recover_interrupted(db)
            print(f'PAPER ONLY. High reasoning + priority where supported. Cap {args.max_windows * len(MODELS)} paid attempts; '
                  'this is NOT a dollar spending cap. No automatic paid retries.', flush=True)
            if args.command == 'predict':
                collect_once(db)
                report(db)
                return
            while True:
                try:
                    settle(db)
                    collected = collect_once(db)
                    if collected:
                        report(db)
                    total = db.execute('SELECT count(*) FROM cohorts').fetchone()[0]
                    pending = db.execute('SELECT count(*) FROM cohorts WHERE result IS NULL').fetchone()[0]
                    if total >= args.max_windows and not pending:
                        print('Window cap reached and all outcomes resolved; stopped.', flush=True)
                        report(db)
                        break
                except (RuntimeError, ValueError, KeyError, sqlite3.Error) as exc:
                    print(f'Skipped cycle: {exc}', flush=True)
                time.sleep(args.interval)
        finally:
            db.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. In-flight provider requests may still be billed; reservations prevent automatic retries.')
    except (RuntimeError, ValueError, KeyError, OSError, sqlite3.Error) as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
