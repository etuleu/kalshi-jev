# Kalshi × DeepInfra: BTC up/down experiment

Forward-only, **paper-only** predictions for `KXBTC15M`. Python 3.10+; standard library only. No Kalshi account, order code, or trading key required.

## Forward study: recommended live collector

```sh
python3 paper_runner.py watch    # five models, fixed protocol, paper only
python3 paper_runner.py report   # progress + paired scores, fully offline
```

The new study samples five minutes into each BTC window, records all five models' inference timings, and evaluates against a common post-inference order book. It models one-contract depth, adverse slippage, and estimated fees with series/event multipliers. Settings are pinned; duplicate collectors are blocked. Default cap: 500 reserved windows / 2,500 paid calls (not a dollar cap), about 5.2 days minimum. Failed attempts count toward the cap.

See [`FORWARD_STUDY.md`](FORWARD_STUDY.md) for costs, background lifecycle, data paths, and assumptions. **Do not leave the legacy two-model watcher running alongside this one unintentionally.**

## Legacy single-model runner

Keep `DEEPINFRA_API_KEY` in `.env` (already configured locally). Optional `DEEPINFRA_MODEL`; see `.env.example`. Environment variables override `.env`.

```sh
python3 -m unittest -v
python3 predictor.py discover       # public data; no model charge
python3 predictor.py predict        # at most one paid inference; nearest eligible BTC window
python3 predictor.py settle         # retrieve official outcomes for saved predictions
python3 predictor.py report         # local metrics, no network
python3 predictor.py watch          # foreground loop; Ctrl-C stops
```

Watch polls every 60 seconds, with at most one inference attempt per contract per database (roughly four an hour while markets run). **DeepInfra charges apply.** Failed/discarded attempts are reserved too and are not automatically retried. No background daemon is installed. Do not run multiple separate databases unless intentionally making separate paid experiments.

Default model: `meta-llama/Llama-3.3-70B-Instruct-Turbo`, temperature 0, maximum 400 output tokens. Override via `--model`. The complete provider response, including available token/cost usage, is saved. Never send the key as prompt content.

## Quick frozen/offline eval

Eight resolved BTC windows are frozen in `evals/btc-small.jsonl`. All five models (Drex Max, Drex Finance, Kimi, DeepSeek V4.1 Flash, GLM 5.3) receive the same pre-cutoff data. Fresh inference uses DeepInfra; reporting and cached replay need no market access.

```sh
python3 offline_eval.py run       # cached responses reused; fresh calls are paid
python3 offline_eval.py report    # fully offline leaderboard
```

The eval enables high reasoning for Kimi/DeepSeek/GLM and requests DeepInfra's priority tier for chat models (unsupported models fall back to standard). It allows 8,192 output tokens including reasoning and a 180-second request timeout. These settings are part of the cache identity; prior non-reasoning runs are not mixed in. The legacy watchers' defaults are unchanged; `paper_runner.py` uses reasoning/priority for the new five-model forward study.

Scores include Brier, accuracy, log loss, latency, and simple baselines, with explicit coverage and paired-case scoring. This tiny retrospective set is imbalanced (7/8 UP) and is **only a smoke test**. It cannot establish a trading edge. See [`evals/README.md`](evals/README.md) for cutoff/leakage controls, cache behavior, and creating a new held-out set.

## Compare Drex Max vs Drex Max Finance

```sh
python3 compare.py predict   # one paired experiment (two paid requests)
python3 compare.py watch     # both models each new eligible BTC window; Ctrl-C stops
python3 compare.py settle
python3 compare.py report
```

Both models receive the **identical context snapshot**, are queried concurrently, and use the **same refreshed decision-time quotes**. Raw responses, input context, and predictions are saved separately in `data/drex-comparison/drex-max.sqlite3` and `data/drex-comparison/drex-max-finance.sqlite3`. The original Llama database is untouched. Watch can make roughly eight paid requests an hour. Nothing runs in the background unless you launch it.

These are **decision models**, not text generators. They use DeepInfra's `/v1/decisions` endpoint, with `state` plus a typed `noul` (yes/no) question. `answers.up.noul` becomes `p_up`; no explanation is generated or fabricated. Chat-only parameters such as temperature and max_tokens are not sent. Token usage is saved, but this endpoint may not return a dollar cost. Public model metadata listed both at $0.12 per million input tokens when checked; pricing can change.

A failed/discarded attempt remains reserved. The comparison selects only windows unattempted by **both** models; it never reruns an old snapshot as a new forward forecast. If one model fails, the other's valid forecast can still be recorded, so compare common tickers when analyzing aggregate performance. A crash between separate database commits can also leave an unpaired record.

For a single Drex model you can also use:

```sh
python3 predictor.py predict --model nace-ai/drex-max --db data/drex-single.sqlite3
```

Use a separate database per model: the single-model runner deduplicates by ticker, not by model. Earlier Llama predictions were made at a different time and are **not** a paired comparison with these Drex predictions. This experiment only evaluates paper outcomes, not financial advice or automated live trading.

## Method

- Fetch exact live contract rules, target, and close time.
- Fetch 90 completed one-minute Coinbase BTC-USD candles, excluding the partial candle. Reject stale/gapped histories.
- Ask for `p_up` (and a brief explanation for chat models only). Kalshi quotes are **withheld from the model**, so the market baseline isn't simply copied.
- Refresh Kalshi quotes after inference. Discard responses if the contract closed, the target changed, less than 30 seconds remain, or inference/data refresh took over 90 seconds. Start inference only with at least 90 seconds left.
- Record a timestamped, unique prediction and observed quotes in `data/experiment.sqlite3`. Never overwrite a prediction. Attempts are recorded separately before paid calls, so crashes do not automatically trigger another call. SQLite is an audit log, not tamper-proof storage.
- Simulate buying **one** YES or NO contract at the observed ask only if its probability minus ask minus estimated fee exceeds `--min-edge` (default 0.05). Otherwise pass. Require displayed size of at least one contract; NO ask liquidity comes from YES bids.
- Use official finalized/settled YES/NO outcomes, not Coinbase returns, to score. Run `settle` again after pending contracts resolve. This initial runner expects outcomes in Kalshi's live dataset; archived/exceptional contracts need manual follow-up.

Predictions include passes in accuracy, Brier score, and calibration buckets; P&L includes trades only. Baselines are a constant 50% forecast and the YES bid/ask midpoint at decision time. Exactly 50% forecasts receive half-credit for accuracy. Results are grouped by model; use a fresh `--db` for a changed strategy/prompt to keep experiments separate.

## Limitations

BTC settlement uses CF Benchmarks' 60-second averages at both endpoints, with ties resolving YES. Coinbase is only a proxy; it is **not** the settlement feed. This is a noisy short-horizon experiment, not evidence that an LLM has an edge.

Paper P&L assumes execution at the refreshed ask, no slippage, and holding until settlement. Displayed liquidity doesn't guarantee a fill. Fee estimate: `ceil_to_cent(0.07 * price * (1-price))` for one taker contract; change the coefficient with `--fee-rate`. This is a conservative approximation, **not a verified account-specific fee calculation**; series/event overrides and precision rules can differ. No real money is used. Small samples and correlated adjacent windows make early win rates misleading.

The runner selects whichever window is active when invoked (not a fixed forecast horizon). Context and remaining time are saved so timing can be analyzed later. A network error or invalid model response skips a prediction; it must never become an invented loss/win.

## Sources

- https://docs.kalshi.com/getting_started/quick_start_market_data
- https://docs.kalshi.com/api-reference/market/get-markets
- https://docs.kalshi.com/api-reference/market/get-market
- https://docs.kalshi.com/getting_started/fee_rounding
- Public REST: `https://external-api.kalshi.com/trade-api/v2`
- Coinbase candles: `https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60`
- DeepInfra chat: `https://api.deepinfra.com/v1/openai/chat/completions`
- DeepInfra decisions: `https://api.deepinfra.com/v1/decisions`
- https://deepinfra.com/nace-ai/drex-max/api
- https://deepinfra.com/nace-ai/drex-max-finance/api
