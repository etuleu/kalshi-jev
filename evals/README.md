# Small frozen BTC eval

`btc-small.jsonl` contains eight consecutive recent resolved Kalshi `KXBTC15M` markets. It is a **retrospective smoke test**, not a trading backtest or a held-out performance benchmark.

## Quick comparison

From the repository root:

```sh
python3 offline_eval.py report           # fully offline; uses saved predictions
python3 offline_eval.py run              # only uncached model calls use DeepInfra
python3 offline_eval.py run --limit 2    # quick two-case check
python3 offline_eval.py run --models nace-ai/drex-max moonshotai/Kimi-K3.1-Pro
```

Default models: Drex Max, Drex Max Finance, Kimi K3.1 Pro, `deepseek-ai/DeepSeek-V4.1-Flash-1017`, and `zai-org/GLM-5.3`. Each gets the same context for each case.

Offline eval defaults:
- `reasoning_effort=high` for Kimi, DeepSeek, and GLM; no reasoning flag for Drex.
- `service_tier=priority` on all chat requests. DeepInfra can fall back to standard for unsupported models; actual tier is retained in the response. In the verified run, Kimi and DeepSeek returned `priority`; GLM returned `default`.
- Chat output budget: **8,192 tokens including reasoning**, temperature 0, JSON output, 180-second per-request timeout. Truncated outputs are errors, not predictions.
- Drex still uses its native decision API, which does **not** accept these chat-only parameters.

Priority has a premium on supported models (currently documented as 50%); reasoning also consumes billed output tokens. This configuration is slower and more expensive than the original non-reasoning smoke test. The live `predictor.py` and `compare.py` callers retain their previous short-request defaults.

Settings can be overridden, and must match when reporting:

```sh
python3 offline_eval.py run --reasoning-effort medium --service-tier default --max-tokens 4096 --timeout 120
python3 offline_eval.py report --reasoning-effort medium --service-tier default --max-tokens 4096 --timeout 120
```

Requested settings are saved as `_eval_requested_options` alongside the raw provider response. The response includes token usage and, where available, reasoning-token counts, estimated cost, and actual service tier.

Llama has been removed from the default eval lineup. Its historical results remain stored. The five remaining models' verified high-reasoning results were carried over after this roster-only change, since their inputs and inference settings did not change.

A fresh complete run makes 40 paid requests, at most four concurrently. Existing successful responses are reused, so repeated runs with unchanged code/data make no new calls. The SQLite cache (`data/offline-eval.sqlite3`) retains raw responses, usage, probabilities, timings, and errors. Cache identity includes model ID, exact context, requested inference settings, and hashes of the predictor and eval source files. The earlier no-reasoning results remain stored but are not mixed into the new leaderboard. Changing either source file invalidates the cache—even a nonfunctional edit. Provider model IDs can change behind the scenes: to intentionally rerun, use a new `--cache` path and note the date.

Errors and interrupted pending calls remain reserved to avoid automatic duplicate charges. `--retry-errors` explicitly pays to retry known errors; it does not retry interrupted pending calls. Use a new cache deliberately if that is needed. Do not launch simultaneous runs against the same cache.

## Inputs and leakage controls

- Fixed observation time: five minutes after market open, **ten minutes before close**.
- Up to 90 completed one-minute Coinbase candles, ending at least **one minute before observation** to allow a publication lag.
- Derived returns and volatility are recomputed exclusively from those candles.
- Opening settlement target and exact contract rules are included. Only allowlisted context fields enter the model request. The label and provenance remain outside the inference context.
- Official Kalshi YES/NO results provide labels, not returns inferred from Coinbase.
- Dataset validation rejects future candles, unexpected context fields, inconsistent features, duplicate IDs, and incorrect horizons.

Historical API data can be revised, and the stored opening target/rules were fetched after resolution, not verified against a contemporaneous snapshot. Models may have memorized historical information; timestamps and tickers remain visible. Do not interpret this as leakage-proof. Forward-recorded live snapshots remain the stronger test.

## Scores

The report uses **only the intersection of successfully completed cases across all selected models**, and separately displays each model's coverage. Failed requests are not silently treated as correct predictions or as losses. Limiting the set or selecting different models can change the shared subset.

- Brier score (lower is better)
- Accuracy (exactly 50% gets half-credit)
- Log loss (lower is better; clipped at 1e-9 for log safety)
- Mean provider-call wall time (cached replay time is not reported as inference latency)

References include always-50%, always-UP/DOWN, and a simple driftless log-price/volatility proxy. The proxy uses Coinbase's last available close, the known opening target, and recent volatility; it does not model CF Benchmarks' closing average accurately. There are **no historical market quotes**, so no market-probability comparison or paper P&L is claimed here.

The included eight cases have **seven UP outcomes**. Accuracy is therefore especially misleading: always-UP scores 87.5%. Adjacent windows share price histories and are correlated. Any model lead is only a tiny-sample observation, not statistically established superiority. Repeated prompt tuning on this set will overfit it.

## Fresh held-out batch

```sh
python3 offline_eval.py build --dataset data/eval-heldout.jsonl --count 24
python3 offline_eval.py run --dataset data/eval-heldout.jsonl
```

Build needs public Kalshi/Coinbase access but no model call. Existing datasets are never overwritten. It selects recent consecutive resolved windows without filtering on outcome; invalid/gapped windows are skipped. For a genuinely new held-out batch, wait for new windows and check that tickers do not overlap the original set. Keep prompts fixed before reviewing held-out labels. Both datasets and model responses can be rescored without market access afterward.
