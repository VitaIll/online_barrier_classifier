# H-005: First PnL number for online_barrier_classifier (round 001)

## Claim
Running `simulate_inventory_aware` on the persisted offline+online prediction stream (`artifacts/online_eval/predictions.parquet`, 31486 test decision-bars covering ~14 months of BTCUSDT 2024) with symmetric triple-barriers φ = c_stop = α = 0.0041113 (the calibrated 90%-quantile of train log-excursions) and a realistic per-side cost of 1bp produces finite, sensible PnL/Sharpe/PSR/drawdown metrics — and the offline+online (= `p_online`) stream's risk-adjusted return is competitive with the offline-only (= `p_offline`) stream across a τ_open ∈ {0.10, 0.15, 0.20, 0.30, 0.50} sweep.

## Mechanism
- Load `predictions.parquet`, slice the matching last-31486 rows of `bars_20m_features.parquet`, and slice the matching minute window of `cleansed_data/BTCUSDT/1m.parquet`.
- Map decision-bar index k → minute index using `open_time` searchsort (per CODE-SCOUT). Specifically, `boundaries["k"] = minute_idx_in_test_slice // M`. Drop test-split bars whose lookahead window `[n_k+1, n_k+M]` crosses the single 80-minute gap recorded in `cleansed_data/BTCUSDT/metadata.json`.
- For each τ_open in {0.10, 0.15, 0.20, 0.30, 0.50}, run the harness on `p_offline` and on `p_final` (= `p_online` literal — CODE-SCOUT confirmed `notebooks/online_eval.ipynb` line 211 sets `p_final = p_online`, no blending).
- Build a **shuffled-signal random-entry null** (preserves the per-τ entry rate but scrambles the signal-to-bar alignment) — this is the drift-matched null per Bailey/Borwein/López de Prado 2014 (LITERATURE-SCOUT cite). Always-on `p≡1` is **forbidden as evaluation benchmark** by CONSTITUTION V.b.
- Note from THEORIST: BTC 2024 +120% drift means a naïve "Sharpe ≈ 0" null is wrong; drift-to-barrier ratio ~0.0073 biases P(TP) − P(SL) by ≈60 bps per always-long trade. The shuffled-signal null inherits that bias automatically, so the falsification compares each strategy's Sharpe to its own per-τ shuffled null — not to zero.

## Predicted effect (THEORIST)
- Trades: offline-only > offline+online at every τ_open (offline mean p ≈ 0.20; online concentrates near base rate 0.097). Expect 2–4× ratio at τ ≥ 0.20.
- Per-trade hit-rate: offline+online higher (online is calibrated; offline overpredicts). Expect online hit-rate at τ=0.20 in 0.18–0.25 vs offline 0.13–0.16.
- Sharpe: ambiguous direction; magnitude ~ 0–1.5 per-strategy; the comparison vs shuffled null is the load-bearing test.

## Seed-noise band
N/A for this round — predictions are deterministic (already persisted; no model retrain). The shuffled-signal null uses 200 bootstraps (`bootstrap_no_skill_pvalue` already implemented at `src/backtest.py:453`); we report `null_sharpe_mean ± null_sharpe_std`. THEORIST predicts that null_sharpe_mean will be slightly POSITIVE (drift bleeds in), not zero. That's expected behavior, not a bug.

## Falsification (THEORIST's pick — clean, harness-internal)
H-005 fails if `simulate_inventory_aware` returns NaN or any non-finite metric (Sharpe, PSR, max_drawdown_log, total_log_return, hit_rate) for any τ_open ∈ {0.10, 0.15, 0.20, 0.30, 0.50} on either p_offline or p_final.

Secondary (informational, not a hard fail): if trade count = 0 at τ=0.10 for either signal, that implies a serialization/mapping bug (predictions never cross such a low threshold), not a market fact.

## Falsification (substantive)
A SHUFFLED-SIGNAL null whose Sharpe ≥ the strategy's Sharpe at the chosen τ_open under the same trade count means the strategy's edge is statistically indistinguishable from random-entry under the same drift. We compute the bootstrap p-value at each τ_open. p-value > 0.5 on every τ for both signals would falsify "the existing offline+online stack has any extractable edge under realistic costs", though a single τ_open giving p < 0.1 is enough to refute the strong falsifier.

## What this round is NOT
- Not an accept-gate. CONSTITUTION V.b requires τ_open chosen via validation Sharpe-grid. We do not have a persisted val-split prediction file (artifact only stores test). We therefore present a τ sweep as a transparent post-hoc grid — the headline numbers are not signed off as the project's PnL accept-gate. A follow-up round (queued below) should re-run offline+online with a persisted validation prediction stream.
- Not a backtest-overfit-corrected number. CSCV / PBO is queued as H-113.

## Decision rule for round 001
- `accept` if (a) all metrics finite at every τ in the sweep, (b) at least one τ has bootstrap p-value < 0.20 for at least one signal, (c) plot is legible after Read-back.
- `iterate` if (a) holds but (b) fails — round produces the first PnL number but no statistical edge demonstrated; queue H-005b (val-split τ_open selection).
- `kill` only if (a) fails — implies the harness or data wiring is broken (no PR-worthy output).

## Spawned follow-up (regardless of decision)
- **H-005b**: re-run `notebooks/offline_train.ipynb` + `notebooks/online_eval.ipynb` with a persisted **validation-split** prediction stream. Use the val Sharpe grid to choose τ_open, then evaluate at fixed τ on test. Backtest-tagged round.
- **H-005c** (deferred to after H-113 lands): apply CSCV PBO to the H-005 sweep to deflate the post-hoc τ grid.

## References
- Local: López de Prado (2018) *AFML*, `C:\Users\vitil\Downloads\Advances in Financial Machine Learning ... Anna's Archive.pdf`, Ch. 13 §13.6.1 pp. 181–183 (symmetric-barrier null heat-maps), Ch. 14 §14.5–14.7 pp. 199–206 (PSR/DSR, HHI, TuW).
- Local: `RESEARCH/literature/INDEX.md` entries [BACKTEST] and [PSR-DSR] (assumed; LITERATURE-SCOUT cited the underlying PDF directly).
- Web fallback: Bailey, Borwein, López de Prado, Zhu (2014) *Pseudo-Mathematics & Financial Charlatanism*, SSRN 2308659 (random-entry null bootstrap).

## Sub-agent IDs
- LITERATURE-SCOUT: a334d36ad948f2999
- CODE-SCOUT: a0b67c6ea9515d7c9
- THEORIST: a88d23c70c01df57a
