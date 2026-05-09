# Hypothesis Backlog

The autonomous loop targets `online_barrier_classifier`. The architecture is:

```
offline CatBoost  →  p_offline
                     +
              streaming conformal calibration layer  →  P(y=1 | x_k)
                                                          (conditional coverage)
```

The "online stage" (currently River `ARFClassifier`, fed `selected_features + p_offline`) is a **streaming conformal calibration / coverage layer**, not a separate classifier. The differentiator vs the sibling `barrier_classifier` (offline-only) is exactly this layer, and most "online improvements" are framed as conformal-coverage improvements (ACI, Mondrian-ACI, locally-weighted conformal, etc.).

User asks 2 (online) and 6 (UQ) collapse onto one axis: **improve the streaming conformal coverage layer.**

Status legend: `queued` / `in_progress` / `iterate` / `blocked`. Priority P1–P5 = expected α-on-user-goal / cost.

Top of file = highest priority.

---

## Tier 0 — Bootstrap to operational state (after these, the loop is fully self-driving)

### H-101 [P5] Build label + split utilities matching this project's actual flow
- **Owner**: IMPLEMENTER
- **Asks**: ask 5 (engineering hygiene)
- **Mechanism**: extract the label construction (`y_k = 1[ ln(H_{k+1}/C_k) ≥ α ]` with α = 90%-train-quantile) and the chronological split (`train_fraction=0.6`, `val_fraction (of train)=0.2`) from `notebooks/feature_build.ipynb` and `notebooks/offline_train.ipynb` into `src/utils.py` as reusable callables. **Important**: this project has NO embargo, NO walk-forward CV in production — don't blindly port the sibling's utilities, only the parts that match this project's contract. Re-enable `tests/_pending/test_causality.py` and `test_splits.py` after adapting the tests to the actual semantics (no embargo expectations).
- **Falsification**: round-trip — the new utilities reproduce the existing `dataset.parquet` labels and the existing `train/val/test` splits exactly when called with the current config.
- **Status**: queued
- **Cost**: medium (~45 min, requires careful adaptation)

### H-102 [P4] Add sample-weighting utility *adapted* from sibling, with this project's primary metric in mind
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 3 (risk-aware weighting — this project currently has none)
- **Mechanism**: import `compute_barrier_distance_weight` and `compute_time_discount_weight` from sibling (already known correct), but design the integration carefully: the offline CatBoost currently uses `auto_class_weights="SqrtBalanced"` which is what causes the over-prediction problem the online layer is correcting. Adding sample weights ON TOP of SqrtBalanced may double-count. Test paths: (a) keep SqrtBalanced + add only time-discount; (b) drop SqrtBalanced, use barrier-distance weighting only; (c) drop SqrtBalanced, use both. Compare offline calibration outcomes.
- **Falsification**: at least one path produces an offline model whose mean predicted p is closer to the base rate (0.097) than the current 0.205.
- **Status**: queued
- **Cost**: medium

### H-103 [P4] Adopt undef-flag pattern (Section I.4)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5
- **Mechanism**: this project's current feature pipeline silently fills NaNs inside `safe_divide`-style helpers; the sibling's `undef__*` flag-as-input pattern is a strict improvement (preserves the missingness signal). Adapt `feature_pipeline.py` to emit per-undefined-condition flags alongside the value. Re-enable `tests/_pending/test_features.py` once the pattern is in place.
- **Status**: queued
- **Cost**: medium

### H-104 [P3] Investigate adding an explicit embargo between val and test (sibling has it; this project doesn't)
- **Owner**: THEORIST + IMPLEMENTER
- **Asks**: ask 4 (rigorous evaluation)
- **Mechanism**: with M=20 and label horizon = 1 decision bar, val→test boundary leakage is bounded (the last val bar's label uses the first test bar's high — that's a single-bar contamination, not a horizon-spanning one). Quantify the actual leakage by comparing test metrics with embargo ∈ {0, 1, 5, 30, 60} bars. If embargo ≥ 1 changes test ROC by < 0.005, document and accept "embargo=0 is fine for this project". Otherwise add to CONSTITUTION I.2.
- **Status**: queued
- **Cost**: medium

### H-005 [P5] Run inventory-aware backtest on the existing offline+online stack
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 4 — **first PnL number**
- **Mechanism**: load `artifacts/offline_model/model.cbm` and the persisted online predictions in `artifacts/online_eval/predictions.parquet` (already on disk from a previous run). Feed `p_offline` and `p_final` columns into `src/backtest.py::simulate_inventory_aware` with M=20, φ=0.00411 (matched to label α), c_stop ≈ φ initially (symmetric — see strategy-realism CONSTITUTION V.b). Compare offline-only-decisions vs offline+online-decisions vs always-on null under realistic costs. Tag round `backtest` (30-min budget).
- **Falsification**: harness emits finite metrics on real predictions; null Sharpe is approximately 0 (with symmetric barriers) or negative (with c_stop < φ); offline+online Sharpe matches or exceeds offline-only.
- **Status**: queued — does NOT depend on H-101..H-103 (the predictions.parquet file already exists from `notebooks/online_eval.ipynb`'s last run; no need to re-run training).
- **Cost**: 30 min (backtest-tagged budget)

---

## Tier 1 — Streaming conformal coverage layer (asks 2 + 6, the project's differentiator)

The online stage is a streaming conformal layer providing conditional coverage. These rounds explicitly improve coverage validity and tightness, NOT raw ranking.

### H-201 [P5] Baseline coverage diagnostic on the existing online ARF
- **Owner**: IMPLEMENTER + THEORIST
- **Mechanism**: treat the existing River `ARFClassifier` output as if it were a conformal predictor at varying confidence levels (set construction via `predict_proba_one`). Compute marginal empirical coverage AND per-regime coverage on the test split. Quantify *coverage gap* = 1 - α - empirical_coverage by regime. This becomes the baseline every online-stage hypothesis must beat.
- **Falsification**: report finite numbers; a > 5% absolute coverage gap in any regime is the gap to close.
- **Status**: queued — first round attacking ask 2.
- **Cost**: low (no model changes; just measurement)

### H-202 [P5] Adaptive Conformal Inference (Gibbs & Candès 2021)
- **Owner**: LITERATURE-SCOUT + IMPLEMENTER + CRITIC
- **Mechanism**: implement online ACI on top of the offline CatBoost. The threshold `q_t` evolves: `q_{t+1} = q_t + γ(α - 1{y_t ∈ C_t(x_t)})`. Compare marginal + per-regime coverage and set tightness vs the current ARF baseline (H-201). The hypothesis: ACI matches or beats the ARF's de-facto coverage with simpler, theoretically grounded calibration.
- **Falsification**: marginal empirical coverage must converge to 1-α (within 2σ over the stream); per-regime gap must not be worse than ARF baseline.
- **Status**: queued (depends on H-201 baseline)
- **Cost**: medium

### H-203 [P5] Mondrian-ACI hybrid for regime-conditional coverage
- **Mechanism**: extend H-202 to maintain a separate `q_t` per volatility tercile (regime). Closes the per-regime coverage gap that plain ACI may leave open under regime drift.
- **Status**: blocked on H-202.
- **Cost**: medium

### H-204 [P4] River ARF vs SRP vs HAT under coverage-as-metric
- **Mechanism**: paired prequential test of `ARFClassifier`, `SRPClassifier`, `HoeffdingAdaptiveTreeClassifier`. Primary metric: marginal + per-regime coverage at α=0.1, plus set tightness. Secondary: Brier.
- **Status**: queued
- **Cost**: medium

### H-205 [P4] Locally-weighted conformal (kernel-local validity)
- **Mechanism**: at test point x, weight calibration scores by kernel similarity to x. Provides conditional-coverage-by-feature-similarity. Reference: Manokhin Ch. 9.
- **Status**: queued
- **Cost**: medium

### H-206 [P4] CatBoost virtual-ensemble σ_epistemic as a conformal feature
- **Mechanism**: add `sigma_epistemic` from `src/uncertainty.py` to the conformal layer's score function. Tests whether epistemic uncertainty improves coverage tightness.
- **Status**: queued; subsumes the prior H-010.
- **Cost**: medium

### H-207 [P3] ADWIN drift detector triggers ACI threshold reset
- **Mechanism**: feed prediction errors into ADWIN; on detected drift, soft-reset `q_t` toward the prior. Hardens ACI against regime breaks.
- **Status**: queued
- **Cost**: low

---

## Tier 2 — Conformal coverage applied to backtest decisions (ask 4)

### H-011 [P5] Conformal LAC + Mondrian for trade abstention (offline)
- **Mechanism**: use the *batch* split-conformal in `src/conformal.py` (already ported) to gate trade entries — abstain when prediction set is full {0,1}. Combined with H-005's backtest harness for an abstention-aware Sharpe.
- **Status**: queued (depends on H-005)
- **Cost**: medium

### H-208 [P4] Streaming conformal trade gate (online ACI variant)
- **Mechanism**: same as H-011 but using the online ACI threshold from H-202; coverage adapts as the stream evolves.
- **Status**: blocked on H-011 + H-202.
- **Cost**: medium

---

## Tier 3 — Risk-aware weighting beyond what's in the sibling (ask 3)

### H-020 [P4] CVaR-conditional tail upweighting beyond `w_max`
- **Mechanism**: extend `compute_barrier_distance_weight` with a CVaR-α tail multiplier. Concentrates capacity on the worst losses.
- **Status**: blocked on H-102.

### H-021 [P3] Asymmetric Huber-like custom CatBoost loss
- **Status**: queued (medium-high cost; risky with `boosting_type="Ordered"`).

### H-022 [P3] López de Prado meta-labeling layer
- **Mechanism**: second classifier on offline-positive subset, predicts whether the trade actually realizes net-positive payoff.
- **Status**: queued.

### H-023 [P3] First-touch (true triple-barrier) labels
- **Status**: queued; high impact, may invalidate accumulated positives.

---

## Tier 4 — Features (research-driven, ask 1)

### H-040 [P4] Hurst / DFA / sample entropy (time-invariant memory descriptors)
- **Mechanism**: rolling R/S Hurst, DFA exponent, sample entropy on returns. Scale-free under Brownian rescaling. Reference: `Multiscale Stochastic Volatility` (local Downloads).
- **Status**: queued
- **Cost**: medium

### H-041 [P3] Wavelet decomposition energy ratios
- **Status**: queued

### H-042 [P3] HAR-RV residuals as features
- **Mechanism**: fit Heterogeneous Autoregressive Realized Volatility (Corsi 2009) on rolling windows; residual is forward-looking surprise.
- **Status**: queued

### H-044 [P3] Realized higher moments (bipower-corrected skew/kurtosis)
- **Status**: queued

---

## Tier 5 — Production engineering (ask 5)

### H-050 [P4] Feature pipeline as hash-keyed cache
- **Mechanism**: every `compute_*` keyed by `(name, version, config_hash, data_hash)`; cache to `experiments/feature_cache/`.
- **Status**: queued

### H-051 [P4] `compute_*` registry — replace imperative chain in `feature_build` notebook
- **Status**: queued

### H-054 [P3] Pre-commit hook running fast tests
- **Status**: queued

---

## Notes
- Items at `[P5]` are unconditionally scheduled before `[P3]`.
- "BLOCKED" items wait until the unblock-er is `accept`ed.
- New ideas append to the appropriate tier section, NOT the top — priority discipline is enforced.
- Items spawned mid-round are marked with `(spawned by round NNN)` for traceability.
