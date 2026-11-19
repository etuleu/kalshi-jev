#!/usr/bin/env python3
"""Frozen BTC forecasting eval: build once, run hosted models, rescore fully offline."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time
from urllib.parse import urlencode

import predictor as p

DATASET = p.ROOT / 'evals' / 'btc-small.jsonl'
CACHE = p.ROOT / 'data' / 'offline-eval.sqlite3'
MODELS = (*p.DECISION_MODELS, 'moonshotai/Kimi-K3.1-Pro', *p.REASONING_MODELS)
EVAL_OPTIONS = {'reasoning_effort': 'high', 'service_tier': 'priority', 'max_tokens': 8192, 'timeout': 180}
EXPERIMENT = 'Retrospective fixed-cutoff forecast. No market quotes supplied. Coinbase is a proxy.'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def make_case(market, rows):
    opened, close = p.timestamp(market['open_time']), p.timestamp(market['close_time'])
    if (market.get('result') not in ('yes', 'no') or market.get('status') not in ('finalized', 'settled')
            or close - opened != 900):
        raise ValueError('Need an officially resolved 15-minute market')
    cutoff = opened + 300  # Five minutes into the window, ten minutes remaining.
    # Add one minute of publication lag, not just exclusion of the partial candle.
    rows = [r for r in rows if r[0] + 60 <= cutoff - 60]
    context = {
        'observed_at': iso(cutoff), 'ticker': market['ticker'], 'title': market['title'],
        'close_time': market['close_time'], 'seconds_to_close': close - cutoff,
        'floor_strike': market['floor_strike'], 'rules_primary': market['rules_primary'],
        'rules_secondary': market.get('rules_secondary'),
        'underlying': p.candle_context(rows, cutoff), 'experiment': EXPERIMENT,
    }
    case = {
        'id': market['ticker'], 'context': context, 'label': int(market['result'] == 'yes'),
        'provenance': {'source': 'Kalshi official result + Coinbase historical candles',
                       'open_time': market['open_time'], 'built_at': p.utcnow(),
                       'kind': 'retrospective reconstruction, not a recorded live snapshot'},
    }
    validate_case(case)
    return case


def validate_case(case):
    if type(case.get('label')) is not int or case['label'] not in (0, 1):
        raise ValueError('Expected a binary official label')
    context = case['context']
    allowed = {'observed_at', 'ticker', 'title', 'close_time', 'seconds_to_close', 'floor_strike',
               'rules_primary', 'rules_secondary', 'underlying', 'experiment'}
    if set(context) != allowed or case['id'] != context['ticker']:
        raise ValueError('Unexpected context fields or mismatched case ID')
    cutoff, close = p.timestamp(context['observed_at']), p.timestamp(context['close_time'])
    opened = p.timestamp(case['provenance']['open_time'])
    if cutoff != opened + 300 or close - cutoff != 600 or context['seconds_to_close'] != 600:
        raise ValueError('Expected fixed five-minute observation / ten-minute forecast horizon')
    target = context['floor_strike']
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
        raise ValueError('Missing or invalid opening settlement target')
    if context['experiment'] != EXPERIMENT:
        raise ValueError('Unexpected experiment instructions')
    rows = context['underlying']['completed_candles']
    if any(r[0] + 60 > cutoff - 60 for r in rows):
        raise ValueError('Future or insufficiently lagged candle in context')
    if context['underlying'] != p.candle_context(rows, cutoff):
        raise ValueError('Derived features do not match pre-cutoff candles')


def load_cases(path, limit=None):
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for case in cases:
        validate_case(case)
        if case['id'] in ids:
            raise ValueError('Duplicate case ID')
        ids.add(case['id'])
    if not cases:
        raise ValueError('Empty dataset')
    return cases[:limit] if limit else cases


def build(path, count):
    if path.exists():
        raise ValueError(f'{path} already exists; use a new --dataset to build another frozen set')
    markets = p.kalshi('/markets', series_ticker='KXBTC15M', status='settled', limit=100)['markets']
    markets = sorted(markets, key=lambda m: m['close_time'], reverse=True)
    cases = []
    for market in markets:
        cutoff = p.timestamp(market['open_time']) + 300
        params = urlencode({'granularity': 60, 'start': iso(cutoff - 7200), 'end': iso(cutoff)})
        try:
            rows = p.request('https://api.exchange.coinbase.com/products/BTC-USD/candles?' + params)
            case = make_case(market, rows)
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"Skipping {market['ticker']}: {exc}", flush=True)
            continue
        cases.append(case)
        print(f"Frozen {len(cases)}/{count}: {case['id']}", flush=True)
        if len(cases) == count:
            break
        time.sleep(.15)
    if len(cases) != count:
        raise ValueError(f'Only {len(cases)} valid cases available; no incomplete dataset written')
    cases.sort(key=lambda c: c['context']['observed_at'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        for case in cases:
            out.write(canonical(case) + '\n')
    print(f'Saved {len(cases)} cases to {path}; UP labels: {sum(c["label"] for c in cases)}')


def connect_cache(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('''CREATE TABLE IF NOT EXISTS runs (
        key TEXT PRIMARY KEY, model TEXT NOT NULL, case_id TEXT NOT NULL,
        status TEXT NOT NULL, started_at TEXT NOT NULL, seconds REAL,
        p_up REAL, response_json TEXT, error TEXT
    )''')
    db.commit()
    return db


def engine_hash(options=None):
    # Includes prompts, adapter code, eval version AND requested inference settings.
    source = Path(p.__file__).read_bytes() + Path(__file__).read_bytes()
    return hashlib.sha256(source + canonical(EVAL_OPTIONS if options is None else options).encode()).hexdigest()


def cache_key(case, model, engine):
    # Label/provenance intentionally excluded: neither is part of inference.
    return digest({'context': case['context'], 'model': model, 'engine': engine})


def timed_forecast(context, model, options):
    start = time.monotonic()
    prediction, response = p.forecast(context, model, **options)
    response = {**response, '_eval_requested_options': options}
    return prediction['p_up'], response, time.monotonic() - start


def run(db, cases, models, workers, retry_errors=False, options=None):
    options = dict(EVAL_OPTIONS if options is None else options)
    if not os.environ.get('DEEPINFRA_API_KEY'):
        raise ValueError('Set DEEPINFRA_API_KEY in .env')
    engine = engine_hash(options)
    print('Requested chat settings:', canonical(options), flush=True)
    print('Reasoning applies to supported reasoners only; priority may fall back to standard. '
          'Drex decision requests do not accept these chat flags.', flush=True)
    tasks = []
    for model in models:
        for case in cases:
            key = cache_key(case, model, engine)
            prior = db.execute('SELECT status FROM runs WHERE key=?', (key,)).fetchone()
            if prior and not (retry_errors and prior['status'] == 'error'):
                continue  # Includes interrupted pending calls: never silently charge again.
            tasks.append((key, model, case))
    print(f'{len(tasks)} fresh paid requests; {len(cases) * len(models) - len(tasks)} cached/reserved. '
          f'Workers={workers}.', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for key, model, case in tasks:
            with db:
                if retry_errors:
                    db.execute("DELETE FROM runs WHERE key=? AND status='error'", (key,))
                db.execute('INSERT INTO runs (key,model,case_id,status,started_at) VALUES (?,?,?,?,?)',
                           (key, model, case['id'], 'pending', p.utcnow()))
            # Only context is passed to the model worker. Labels never enter its payload.
            futures[pool.submit(timed_forecast, case['context'], model, options)] = (key, model, case['id'])
        for future in as_completed(futures):
            key, model, case_id = futures[future]
            try:
                probability, response, seconds = future.result()
                p.parse_forecast(json.dumps({'p_up': probability, 'reason': ''}))
                with db:
                    db.execute("UPDATE runs SET status='ok',p_up=?,seconds=?,response_json=?,error=NULL WHERE key=?",
                               (probability, seconds, canonical(response), key))
                print(f'{model} | {case_id} | UP {probability:.1%} | {seconds:.2f}s', flush=True)
            except (RuntimeError, ValueError, KeyError) as exc:
                with db:
                    db.execute("UPDATE runs SET status='error',error=? WHERE key=?", (str(exc), key))
                print(f'{model} | {case_id} | ERROR: {exc}', flush=True)


def metrics(probabilities, labels):
    n = len(labels)
    if not n:
        raise ValueError('No cases to score')
    brier = statistics.mean((prob - label) ** 2 for prob, label in zip(probabilities, labels))
    accuracy = statistics.mean(.5 if prob == .5 else int((prob > .5) == bool(label))
                               for prob, label in zip(probabilities, labels))
    log_loss = -statistics.mean(label * math.log(max(1e-9, prob))
                               + (1 - label) * math.log(max(1e-9, 1 - prob))
                               for prob, label in zip(probabilities, labels))
    return brier, accuracy, log_loss


def baseline(case):
    # Driftless log-price diffusion around the known target; simple reference, not calibrated.
    context = case['context']
    price = context['underlying']['last_close']
    sigma = context['underlying']['one_minute_log_return_stdev']
    last_end = context['underlying']['completed_candles'][-1][0] + 60
    minutes = (p.timestamp(context['close_time']) - last_end) / 60
    if sigma <= 0:
        return 1.0 if price > context['floor_strike'] else 0.0 if price < context['floor_strike'] else .5
    z = math.log(price / context['floor_strike']) / (sigma * math.sqrt(minutes))
    return .5 * (1 + math.erf(z / math.sqrt(2)))


def report(db, cases, models, options=None):
    engine = engine_hash(options)
    print('Requested chat settings:', canonical(EVAL_OPTIONS if options is None else options))
    results = {}
    for model in models:
        results[model] = {}
        for case in cases:
            row = db.execute('SELECT * FROM runs WHERE key=?', (cache_key(case, model, engine),)).fetchone()
            if row and row['status'] == 'ok':
                results[model][case['id']] = row
        print(f'{model}: coverage {len(results[model])}/{len(cases)}')
    common = [c for c in cases if all(c['id'] in results[m] for m in models)]
    print(f'\nPAIRED leaderboard: {len(common)}/{len(cases)} cases completed by ALL selected models.')
    if not common:
        print('No common cases. Run models first or select models with cached successes.')
        return
    labels = [c['label'] for c in common]
    print(f'Observed UP rate: {statistics.mean(labels):.1%}; scores below use only this shared subset.')
    print(f'{"Model / baseline":48} {"Brier ↓":>9} {"Acc ↑":>8} {"LogLoss ↓":>10} {"Mean sec":>9}')
    scores = []
    for model in models:
        rows = [results[model][c['id']] for c in common]
        scores.append((model, metrics([r['p_up'] for r in rows], labels), statistics.mean(r['seconds'] for r in rows)))
    scores.extend([
        ('Always 50%', metrics([.5] * len(common), labels), 0),
        ('Always UP (imbalance check)', metrics([1.] * len(common), labels), 0),
        ('Always DOWN (imbalance check)', metrics([0.] * len(common), labels), 0),
        ('Driftless price/volatility proxy', metrics([baseline(c) for c in common], labels), 0),
    ])
    for name, (brier, accuracy, loss), seconds in sorted(scores, key=lambda x: x[1][0]):
        print(f'{name:48} {brier:9.4f} {accuracy:7.1%} {loss:10.4f} {seconds:9.2f}')
    print('\nRetrospective smoke test, NOT evidence of trading edge. No historical quotes => no P&L.')
    print('Adjacent windows are correlated. Use fresh held-out cases before making performance claims.')


def main():
    p.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'run', 'report'])
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--count', type=int, default=8, help='Number of resolved windows to freeze')
    parser.add_argument('--limit', type=int, help='Evaluate only the first N cases')
    parser.add_argument('--models', nargs='+', default=list(MODELS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reasoning-effort', choices=['none', 'low', 'medium', 'high'], default='high')
    parser.add_argument('--service-tier', choices=['default', 'priority', 'flex'], default='priority')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Output budget including reasoning')
    parser.add_argument('--timeout', type=int, default=180, help='Per-request timeout in seconds')
    parser.add_argument('--retry-errors', action='store_true', help='Explicitly pay to retry failed calls')
    args = parser.parse_args()
    if (not 1 <= args.count <= 100 or not 1 <= args.workers <= 8
            or (args.limit is not None and args.limit < 1)
            or not 400 <= args.max_tokens <= 32768 or not 1 <= args.timeout <= 600):
        parser.error('Require count 1..100, workers 1..8, positive limit, tokens 400..32768, timeout 1..600')
    options = {'reasoning_effort': args.reasoning_effort, 'service_tier': args.service_tier,
               'max_tokens': args.max_tokens, 'timeout': args.timeout}
    args.models = list(dict.fromkeys(args.models))
    if args.command == 'build':
        build(args.dataset, args.count)
        return
    cases = load_cases(args.dataset, args.limit)
    db = connect_cache(args.cache)
    try:
        if args.command == 'run':
            run(db, cases, args.models, args.workers, args.retry_errors, options)
        report(db, cases, args.models, options)
    finally:
        db.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped; submitted calls may still be billed. Pending attempts are not automatically retried.')
    except (RuntimeError, ValueError, KeyError, OSError, sqlite3.Error) as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
