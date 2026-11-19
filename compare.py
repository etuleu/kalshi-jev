#!/usr/bin/env python3
"""Paired, paper-only Drex forecasts using identical inputs and decision-time quotes."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote

import predictor as p


def compare_once(databases, args):
    market = next((m for m in p.open_markets('KXBTC15M')
                   if p.eligible(m, time.time(), 90)
                   and all(not p.attempted(db, m['ticker']) for db in databases.values())), None)
    if market is None:
        print('No new jointly eligible BTC window.')
        return
    context = p.prepare_context(market)
    # If either reservation fails, no model call is made. Reservations prevent retries.
    for db in databases.values():
        p.reserve_attempt(db, context)
    print(f"Paired forecast: {market['ticker']} | snapshot {context['observed_at']}", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=len(databases)) as pool:
        futures = {model: pool.submit(p.forecast, context, model) for model in databases}
        for model, future in futures.items():
            try:
                results[model] = future.result()
            except (RuntimeError, ValueError, KeyError) as exc:
                print(f'{model}: inference failed ({exc}); attempt reserved, no automatic retry.')
    # Persist responses even if the subsequent quote refresh or freshness checks fail.
    for model, (_, response) in results.items():
        with databases[model]:
            databases[model].execute('UPDATE attempts SET response_json=? WHERE ticker=?',
                                     (json.dumps(response), context['ticker']))
    if not results:
        return
    fresh = p.kalshi('/markets/' + quote(context['ticker'], safe=''))['market']
    for model, (prediction, response) in results.items():
        settings = argparse.Namespace(model=model, min_edge=args.min_edge, fee_rate=args.fee_rate)
        try:
            p.record_prediction(databases[model], settings, context, prediction, response, fresh=fresh)
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f'{model}: forecast discarded ({exc})')


def main():
    p.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['predict', 'settle', 'report', 'watch'])
    parser.add_argument('--data-dir', type=Path, default=p.ROOT / 'data' / 'drex-comparison')
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--min-edge', type=float, default=.05)
    parser.add_argument('--fee-rate', type=float, default=.07)
    args = parser.parse_args()
    if (args.interval < 15 or not 0 <= args.min_edge <= 1 or not 0 <= args.fee_rate <= 1):
        parser.error('Require interval >=15 and edge/fee-rate in [0,1]')
    if args.command in ('predict', 'watch') and not os.environ.get('DEEPINFRA_API_KEY'):
        parser.error('Set DEEPINFRA_API_KEY in .env')
    with ExitStack() as stack:
        databases = {model: stack.enter_context(closing(p.connect(args.data_dir / (model.split('/')[-1] + '.sqlite3'))))
                     for model in p.DECISION_MODELS}
        if args.command == 'predict':
            compare_once(databases, args)
        elif args.command in ('settle', 'report'):
            for model, db in databases.items():
                print(f'\n{model}')
                (p.settle if args.command == 'settle' else p.report)(db)
        else:
            print('Paired paper watch: up to TWO paid inferences per new BTC window. Ctrl-C stops.', flush=True)
            while True:
                try:
                    for db in databases.values():
                        p.settle(db)
                    compare_once(databases, args)
                except (RuntimeError, ValueError, KeyError, sqlite3.IntegrityError) as exc:
                    print(f'Skipped cycle: {exc}', flush=True)
                time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
    except (RuntimeError, ValueError, KeyError, sqlite3.IntegrityError) as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
