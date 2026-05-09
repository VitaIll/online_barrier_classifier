# Phase A backtest report (round 015 — corrected two-layer architecture)

Honest, deflated comparison of the **five legitimate** trading strategies on top of the `online_barrier_classifier` two-layer stack.

**Architecture** (the correction round 015 enforces): the offline CatBoost emits `p_offline`, which is consumed (with `selected_features`) by the streaming online ARFClassifier, which emits `p_online` — the system's actual output. The streaming Mondrian-ACI conformal layer sits on top of `p_online` and produces the per-regime `q_t` and `in_set_α` indicators. Strategies that average or stack `(p_offline, p_online)` are architecturally invalid and have been deleted (this round supersedes rounds 011–013).

**Generated**: round 015.  
**Test slice**: n=31,486 decision boundaries.  
**Costs**: 1 bp/side (round-trip 2 bp).  
**Barriers**: φ = c_stop = α = 0.0041113.  
**τ-grid**: 26 points {0.10..0.60 step 0.02}; **k-grid** (mondrian_aci_size): [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]; **target_rate** (null_random_at_rate): single fixed point at the offline baseline's empirical test rate.

## 0. Headline

**No strategy survives the deflated bar (DSR > 0.95) AND the within-strategy PBO < 0.5 simultaneously.** The strongest test Sharpe is `baseline_offline_tau` at -0.089 (multi-strategy DSR=0.000). The two-layer system as currently fit does not produce tradable alpha at this label / cost / barrier-strategy combination. Phase B (production refactor) and Phase D (feature research, especially the rolling-retrain harness H-310) are the next moves.

Best test Sharpe: `baseline_offline_tau` at **-0.089** (multi-strategy DSR=0.000).  
Cross-strategy CSCV PBO = **0.000**.  
Modal IS-best across 12,870 CSCV combinations: **`baseline_offline_tau`**.

## 1. Strategy comparison

| strategy | knob | knob_star | val_sharpe_at_star | test_sharpe | test_psr | test_n_trades | test_total_log_return | test_max_drawdown_log | test_cdar_5pct_log | test_hit_rate | test_avg_bars_held | p_boot | deflated_sharpe_multi_strategy | walk_forward_sharpe_mean | walk_forward_sharpe_std | within_strategy_pbo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_offline_tau | tau | τ=0.440 | +0.347 | -0.089 | +0.458 | 3427 | -0.024 | +0.361 | +0.276 | +0.525 | +10.773 | +0.000 | +0.000 | +3.534 | +6.404 | +0.000 |
| baseline_online_tau | tau | τ=0.520 | -2.285 | -11.591 | +0.002 | 289 | -0.199 | +0.203 | +0.199 | +0.443 | +5.609 | +1.000 | +0.000 | -8.967 | +19.234 | +0.001 |
| conformal_gate_tau | tau | τ=0.520 | -2.268 | -11.874 | +0.002 | 288 | -0.203 | +0.207 | +0.203 | +0.441 | +5.601 | +0.735 | +0.000 | -13.311 | +12.884 | +0.003 |
| mondrian_aci_size | k | k=20.0 | -5.237 | -5.666 | +0.000 | 2372 | -0.904 | +0.943 | +0.926 | +0.459 | +10.481 | +0.505 | +0.000 | -2.481 | +4.900 | +0.000 |
| null_random_at_rate | target_rate | r=0.123 | -3.752 | -3.963 | +0.000 | 3448 | -0.791 | +0.840 | +0.825 | +0.448 | +18.350 | +1.000 | +0.000 | -4.084 | +1.684 | — |

## 2. CSCV PBO

Per-strategy PBO (16 chunks, C(16,8) = 12,870 combinations):

| strategy | n_knobs | PBO | median logit | modal IS-best knob |
|---|---|---|---|---|
| baseline_offline_tau | 26 | 0.000 | +2.793 | 0.300 |
| baseline_online_tau | 26 | 0.001 | +3.932 | 0.600 |
| conformal_gate_tau | 26 | 0.003 | +3.932 | 0.600 |
| mondrian_aci_size | 6 | 0.000 | +2.398 | 20.000 |
| null_random_at_rate | 1 | nan | +nan | nan |

Cross-strategy CSCV: **PBO = 0.000** (modal IS-best: `baseline_offline_tau`).

## 3. Figures

![equity overlay](RESEARCH/diagrams/phase_A/round_015/equity_overlay.png)

![PBO panel](RESEARCH/diagrams/phase_A/round_015/pbo_panel.png)

![val knob sweep](RESEARCH/diagrams/phase_A/round_015/val_knob_sweep.png)

## 4. Reproduction

```
python scripts/phase_A_round_015.py
```

Idempotent given the persisted artefacts in `artifacts/phase_A/`. Wall ~5 min (ARF replay over val) on the first run; ~1 min on a warm cache.

## 5. What changed vs the deleted rounds 011–013

1. The two combined-signal strategies (`combined_avg_tau`, `combined_stacked_tau`) have been deleted from `src/strategies.py`. They averaged or stacked `p_offline` and `p_online`, which violates the two-layer architecture.
2. `src/inference.py::predict()` now drives the Mondrian-ACI conformal stream off `p_online`, not `p_offline`. The `q_lo_α` and `in_set_α` columns are now legitimate per the round-008 contract.
3. `conformal_gate_tau` and `mondrian_aci_size` now consume `p_online` (not `p_offline`). `mondrian_aci_size`'s confidence score is the per-regime margin of `p_online` over `(1 − q_lo_α)`.
4. The Phase A table has 5 rows, not 7. The two deleted rows are not part of any future deflation.
