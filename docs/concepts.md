# Concepts

One paragraph each. Read top-to-bottom; the order matches the streaming Pipeline.

## Label construction (triple-barrier, upper-only)

The label `y_k = 1[ ln(H_{k+1} / C_k) >= alpha ]` asks: in the **next** decision bar, did the price reach an excursion of at least `alpha` log-return above the current close? `alpha` is calibrated as the 90th-percentile of training-window log-excursions (a fixed number, currently 0.0041113 ≈ 41 bps for the BTCUSDT 2023–2025 slice). The label is one-sided (upper only — no symmetric down-barrier in the label), positive-skew, and uses **only** `H_{k+1}` from the next bar. The boundary close, the next-bar high, and the calibrated `alpha` itself must never appear as feature columns. Per-segment burn-in (`burn_in_bars = max(window) = 96`) is enforced before any label is produced. Source: `src/wagie/offline/label.py`.

## Adaptive Random Forest (ARF) — the streaming online layer

`river.ensemble.ARFClassifier` wrapped by `wagie.pipeline.online_arf.OnlineARFCorrector`. The ARF is a bagged ensemble of online Hoeffding trees with per-leaf NBA prediction, ADWIN drift detection at the per-tree level, and Poisson(λ=6) bagging. It consumes either raw features or `(features, p_offline)` (depending on whether a CatBoost model is wired in) and emits `p_online` per decision bar via `predict_proba_one`. The ARF's value comes from **calibration**, not ranking: `p_online` should be close to the conditional base rate within each volatility regime, and the ECE/Brier of `p_online` are the headline metrics. Drift surfaced by ADWIN is logged so the autonomous loop can correlate calibration shifts with regime breaks. Sibling STREAMING fixes the cold-start NaN window, segment-awareness, and surfaces ADWIN events to the metrics dict.

## Regime cuts

A **fixed regime signal** (default `parkinson_var_rolling_mean_24` — long-window Parkinson volatility) is bucketed into terciles using train-only `qcut` quantiles, then those edges are **frozen** for val + test + live. Each bar carries a `regime_id ∈ {low, med, high}` (or whatever `features.regime_labels` names them). Regime cuts let every metric be reported per-regime: `brier_per_regime`, `ece_per_regime`, `coverage_gap_per_regime`. The high-vol regime is empirically where calibration is hardest (legacy offline ECE ≈ 0.17 vs 0.05 in the low-vol tercile); regime-stratified reporting prevents a global ECE from hiding tail miscalibration. Source: `wagie.features.RegimeCuts` (`src/wagie/features/__init__.py`).

## Threshold gating (`ThresholdGate`)

The post-conformal-removal default decision rule. **Open a long when `p_online >= tau`**, where `tau` is chosen on validation Sharpe (or PR-AUC, or ECE — strategy-realism rule applies — see `RESEARCH/CONSTITUTION.md` §V.b). Single open at a time, sized 1.0, with `take_profit / stop_loss / expiry` from `BrokerConfig`. This replaces `PureConformalGate(in_set_alpha == 1)` because the conformal layer is gone — the calibration-derived gate now collapses to a plain probability threshold on the calibrated `p_online`. Strategy lives in `src/wagie/strategy/__init__.py`; sibling TRADING ships the `ThresholdGate` and `RegimeGated` kinds.

## Accept gates (post-conformal-removal)

A research round "accepts" only when its hypothesis-id, predicted-effect minimum, and minimum-trade count are all satisfied at the bootstrap-CI level — not at point-estimate. Concretely, each spec carries:

- `hypothesis_id` — the H-ID this round tests (e.g. `R-031-replay`).
- `predicted_effect_min` — minimum effect size to accept (e.g. `+0.5` ann-Sharpe lift over baseline).
- `min_n_trades` — minimum trade count to consider the result statistically reliable (e.g. `200`).

The accept gate is satisfied iff: bootstrap-CI lower bound on the headline metric exceeds `predicted_effect_min` AND `n_trades >= min_n_trades` AND ECE/Brier did not regress more than the seed-noise band. Sibling RIGOR ships the bootstrap-CI library and the accept-gate enforcement. Until the gate fires, a round is `iterate` or `kill`, never `accept`.
