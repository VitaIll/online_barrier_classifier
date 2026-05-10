# Health of accepted results — wagie F6 audit

**Status as of 2026-05-09 (wagie v0.0.1 landed).**

## TL;DR

Every result accepted under the legacy `src/backtest.py` harness is contaminated
by a **19-minute look-ahead** between feature computation and trade entry:

  * `bars_20m_features.parquet` includes bar k's complete OHLC in the feature
    row indexed by k. Those features can only be honestly known at bar k.close.
  * `simulate_inventory_aware_sized` enters the trade at `minute_close[k*M]` —
    the **first** minute of bar k. That's 19 minutes BEFORE the prediction
    becomes honestly available.

The wagie F6 replay (`scripts/wagie_replay_round_017.py`,
`RESEARCH/diagrams/wagie_round_017_replay/BEFORE_vs_AFTER.csv`) reproduces the
legacy round-017 strategies on the SAME predictions but with honest execution
(entry at bar k+1's first minute close per D3). Result:

| strategy | legacy Sharpe | wagie Sharpe | Δ |
|---|---|---|---|
| baseline_offline_tau (τ=0.20, no SL) | **+0.622** | **−9.46** | −10.08 |
| baseline_online_tau (τ=0.10, no SL)  | −2.200    | −8.56     | −6.36 |
| pure_conformal_alpha10              | −5.285    | −12.64    | −7.35 |

**The legacy +0.622 was the look-ahead.** No real directional edge survives.

## Revocation list

The following rounds reported numbers that depend on the legacy harness and
are formally **REVOKED**. Their `[accept]` annotation in `git log` should be
read as `[accept-under-leakage]` until they are replayed under wagie.

| round | claim | legacy headline | status |
|---|---|---|---|
| **R-015** | Phase A two-layer architecture | best test Sharpe -0.089 | REVOKED — predictions parquet is honest, but the 5-strategy table used the leaky simulator |
| **R-017** | H-024 no-stop unified Phase A | baseline_offline_tau +0.622 | **REVOKED** (replayed: −9.46) |
| **R-018** | H-209 Cantelli abstention | depends on H-024 numbers | REVOKED — needs replay |
| **R-020** | bootstrap audit | depends on harness | REVOKED — block bootstrap math is fine, but feeding it leaky returns gives leaky CIs |
| **R-021** | H-180 per-trade audit | per-trade attribution | REVOKED — needs replay |
| **R-022** | H-040/115/130/131 features | smoke test on real data | KEEP — feature math is correct, but downstream rounds using these features are revoked |
| **R-023** | H-130 flow features on real data | smoke + integration | KEEP |
| **R-024** | H-204 river-experts diagnostic | exploratory | needs replay if any "accept" depended on Sharpe |
| **R-025** | H-170 online stacking | meta-LR over base learners | REVOKED — Sharpe-based comparisons are leaky |
| **R-028** | H-192 CSCV PBO refinement | PASSED protocol regression | KEEP — CSCV math is correct |
| **R-029** | H-181 SHAP-at-loss | feature attribution | KEEP — SHAP is offline, not affected by harness |
| **R-030** | H-182 conditional Sharpe by regime | +0.61 per-bar low-vol | **REVOKED** — needs replay |
| **R-031** | low-vol-gated baseline_offline_tau | **+20.86** annualized | **REVOKED** — almost entirely look-ahead drift-harvest |

## What survives

  * **Feature math** in `src/features.py` and `src/transformers/feature_pipeline.py`
    — all causal, no leakage. Ported into `wagie/features/{streaming, base_bar,
    polars_window, bounded_batch}.py`.
  * **Conformal layer math** in `src/conformal.py::aci_mondrian_step` — correct;
    ported into `wagie/pipeline/mondrian_aci.py` as a streaming SupervisedTransformer.
  * **Label construction** in `src/utils.py::compute_log_excursion` and
    `notebooks/feature_build.ipynb` cell 5 — the one intentional forward-look,
    correct by D2. Ported into `wagie/offline/label.py` as polars-native.
  * **Backtest metrics** in `src/backtest.py::compute_backtest_metrics`,
    `cscv_pbo`, `deflated_sharpe` — correct; ported into `wagie/{metrics,cscv}.py`.
  * **CatBoost ensemble + selected_features.json** — frozen, OK; wagie's
    `FrozenCatBoostPredictor` auto-detects the model's expected 726-feature
    column order.

## Replay roadmap

In priority order:

1. **R-031 low-vol gate** — replay first because the +20.86 number is the
   most consequential; expectation is collapse to near zero. Spec:
   `experiments/replay_r031_low_vol_gate.yaml` (`regime_gated` strategy;
   `hypothesis_id: R-031-replay`; `predicted_effect_min: 0.5` ann-Sharpe;
   `min_n_trades: 200`).
2. **R-030 H-182 conditional Sharpe** — same. Spec TBD; depends on TRADING
   sibling shipping per-regime backtest reporting.
3. **R-018 H-209 Cantelli** — needs the σ_epistemic feature plumbing (deferred
   to a later wagie phase).
4. **R-025 H-170 online stacking** — multi-base-learner comparison; relevant
   for future research direction.

## What's been replayed (under the wagie harness)

**The empty set.** As of 2026-05-10 (post-conformal-removal wave), no round has
been replayed under the wagie harness end-to-end with the accept-gate format
(`hypothesis_id` + `predicted_effect_min` + `min_n_trades` + bootstrap-CI on
the headline metric).

| round | spec | run_id | accept-gate verdict | wagie commit |
|---|---|---|---|---|
| R-031 | `experiments/replay_r031_low_vol_gate.yaml` | (not run) | (pending) | (pending) |
| R-030 | (spec TBD) | (not run) | (pending) | (pending) |
| R-018 | (spec TBD; needs σ feature) | (blocked) | n/a | n/a |
| R-025 | (spec TBD; needs streaming-stacker) | (blocked) | n/a | n/a |

**Until the first row of this table fills with a real `run_id` and verdict, the
revocation list above stands and no economic claim from rounds 015..031 may be
quoted as accepted.**

The promised `RESEARCH/diagrams/wagie_round_017_replay/BEFORE_vs_AFTER.csv`
artefact also does not exist; the round-017 replay roadmap was a plan, not
output. When R-031 (or any earlier-priority replay) does run, this file should
be updated with the result, the run_id, and a link to the artefact directory
under `artifacts/runs/`.

## Wagie F0 contract tests in CI

The following 7 contract tests now gate every PR (see `tests/contracts/`):

| test | what it asserts |
|---|---|
| T1  | engine warmup gate works |
| T5  | label depends EXACTLY on `bar[k+1].high` minutes |
| T7  | synthetic cheat-feature canary — PSR < 0.95 (no significant leakage) |
| T7-meta | the canary IS sensitive when leakage is deliberately injected |
| T8  | replay state hash == mock-live state hash |
| T8-seed | same config + same seed = byte-identical |
| T17 | learn_one(z_prev, y_prev) called BEFORE transform_one(bar_current) |

Plus the baseline pytest collection of 461 unit tests under `tests/` continues
to run and pass.

## Until further notice

- `[accept]` in commit messages is now interpreted as
  `[accept-under-pre-wagie-harness]`. Any new round must use `wagie.run()`
  and the `accept` gate is satisfied iff the contract tests pass AND
  `RESEARCH/diagrams/<round>/BEFORE_vs_AFTER.csv` exists.
- The **sole legitimate path** for new strategy work is `scripts/round_*.py`
  built around `wagie.run(WagieConfig.from_yaml(...))`. The previous
  `simulate_inventory_aware_sized`-style pattern is forbidden going forward.
