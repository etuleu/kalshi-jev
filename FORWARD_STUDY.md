# Five-model forward paper study

`paper_runner.py` is a separate, forward-only collector for Drex Max, Drex Finance, Kimi K3.1 Pro, DeepSeek V4.1 Flash 1017, and GLM 5.3. **No real orders or Kalshi account access.** The old `compare.py` is still the legacy two-Drex experiment; do not run both unintentionally.

## Commands

```sh
python3 paper_runner.py watch              # foreground; Ctrl-C stops
python3 paper_runner.py report             # fully offline, read-only; safe while collecting
python3 paper_runner.py settle             # official results, when collector isn't running
python3 paper_runner.py predict            # one eligible window only, otherwise returns
```

Default database: `data/forward-v1.sqlite3`. `report` needs an existing initialized study. The collector automatically settles outcomes during watch; a separate `settle` process is unnecessary and rejected by its lock while watching.

Background launch (only if not already started):

```sh
nohup python3 -u paper_runner.py watch > data/forward-watch.log 2>&1 &
echo $! > data/forward-watch.pid
```

```sh
tail -f data/forward-watch.log
python3 paper_runner.py report
kill "$(cat data/forward-watch.pid)"  # verify the PID is your collector first
```

Keep the Mac awake and connected (for example `caffeinate -i -w PID` in another terminal). No launch agent, auto-restart service, or sleep prevention is installed. SQLite WAL permits reporting while collecting; an advisory lock blocks duplicate collectors/settlers on the same DB. Different DB paths are different studies and can still duplicate charges.

## Frozen protocol

- BTC fifteen-minute contracts only; sample **300–360 seconds after market open**. Poll every 15 seconds. Starting mid-window outside that interval skips it; no late catch-up/backfill.
- All five models get exactly the same live input context, target, contract rules, and completed Coinbase candles. Market quotes are withheld from model inputs. Context timing is saved, not backdated to an idealized cutoff.
- Kimi, DeepSeek, and GLM use high reasoning, 8,192 output tokens including reasoning, priority requested, and a 180-second network timeout. Priority may fall back to standard. Drex uses the native decision API without chat flags.
- Every cohort and all five paid-call reservations are committed **before** inference. Unique ticker keys and process locking prevent duplicates within a study. Errors, timeouts, and interrupted calls are recorded and never automatically retried. Submitted calls may still be billed after you stop the process.
- After all model calls finish, fetch a single shared market/fee/order-book snapshot. Fast models intentionally wait for the slowest model: this tests a common decision barrier, not a fastest-response strategy. Model inference duration, completion time, decision time, and the raw depth snapshot are retained.
- Reject changed targets, closed markets, context older than 240 seconds, less than 60 seconds to close, book fetches taking over 10 seconds, and quote receipts over 10 seconds old at the decision write. Public REST data can still be cached; request/receipt timestamps are not exchange-source freshness guarantees.
- Each model independently simulates buying at most **one contract**, holding to settlement. Find the depth-weighted cost for one contract from complementary YES/NO bids; don't invent liquidity when less than one contract is available.
- Add **$0.01 adverse slippage** to that cost, estimate the one-contract taker fee with `ceil_to_cent(0.07 * effective_multiplier * price * (1-price))`, and require a net probability edge of at least 0.05. Otherwise pass. The multiplier/type are read from series/event metadata. Unknown schedules disable entries rather than assuming zero fees. Any fee waiver is conservatively ignored.
- These fees are **estimates**, not verified account-specific exchange charges. Displayed liquidity and slippage allowances do not guarantee fills; queue position, cancellations, execution latency, and the precise settlement/fee rounding process can change realized results.
- The price/volatility baseline and 50% baseline use the same paper-entry rules, shared book, fee schedule, and slippage allowance. Kalshi's decision-time midpoint is also scored. Coinbase remains only a proxy for the CF Benchmarks settlement index.
- Official finalized/settled YES/NO outcomes label the cases. Closed/unresolved markets are never treated as losses. The historical market endpoint is tried if live lookup returns 404. Exceptions/unusual settlements require manual review.

The manifest pins the strategy, model IDs, inference settings, cohort cap, and source hashes. Changing collector or predictor code/settings requires a **new `--db`** rather than mixing a tuned strategy into the existing study. Hosted model weights can still change behind a stable provider model ID; raw responses are retained where available. Report/settlement commands can inspect an existing study even after source changes.

## Scope, costs, and stopping

The default cap is **500 reserved windows**, at most **2,500 paid inference attempts**. Errors count toward the cap. This is a window/attempt cap, **not a dollar spending cap**. At current prices and the original eight-case high-reasoning usage, the approximate total was $20; actual usage, priority premiums, timeout billing, and price changes can make it different. Provider dollar estimates are incomplete (especially for decision calls and failures), so the report explicitly marks missing costs.

Five hundred consecutive fifteen-minute windows take about **5.2 days** under ideal conditions. Sleeping, outages, and missing the sample interval extend collection time. Because errors count toward the fixed cap, 500 reserved windows may yield fewer than 500 complete paired results. The runner does not silently buy more inference to reach 500 successes. After the cap, it makes no new paid calls, waits for outcomes, then exits; indefinitely unresolved contracts can keep the settlement loop alive until manually stopped.

For a separately capped smoke study, use a new path:

```sh
python3 paper_runner.py watch --db data/forward-smoke.sqlite3 --max-windows 2
```

Do not change the cap for an already initialized study; it is pinned. Existing model caches and legacy Drex data are untouched and never reused as fresh forward predictions.

## Read the report correctly

Progress shows reservations, recorded decisions, official outcomes, and **complete paired settled cases**. Coverage and errors are visible separately. The leaderboard uses only cases with a shared valid decision snapshot and successful forecasts from all five models, so comparisons never mix different test sets. Failed windows can bias this subset; always inspect coverage, not only scores.

Scores include Brier, direction accuracy, trade count, simulated net P&L after estimated entry fees/slippage, return on premium-plus-fees, and peak-to-trough paper drawdown. It also shows latency, incomplete provider-reported inference costs, UP frequency, and UTC daily paired counts. Paper P&L excludes inference costs, and each model's trades are separate hypothetical strategies, not a portfolio that receives the same liquidity five times.

This is evidence collection, not proof of an edge. Adjacent windows are correlated; compare daily/held-out performance, class balance, coverage, costs, and the simple baseline. Repeatedly selecting whichever model looks best introduces selection bias. Don't deploy money based solely on a positive paper total.
