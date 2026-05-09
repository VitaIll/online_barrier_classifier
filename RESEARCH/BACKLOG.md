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

### H-101 [P5] Port labels + chronological-split + walk-forward-cv utilities
- **Owner**: IMPLEMENTER
- **Asks**: ask 5 (engineering)
- **Mechanism**: bring `construct_labels`, `chronological_split_with_embargo`, `walk_forward_cv`, and the embargo/warmup constants from sibling `barrier_classifier` into this project's `src/utils.py`. Re-enable `tests/_pending/test_causality.py` and `test_splits.py`.
- **Falsification**: tests pass; embargo enforced for any (n, train_frac, val_frac, embargo_k).
- **Status**: queued
- **Cost**: low (~30 min)

### H-102 [P5] Port sample-weighting utilities + tests
- **Owner**: IMPLEMENTER
- **Asks**: ask 3
- **Mechanism**: port `compute_barrier_distance_weight`, `compute_time_discount_weight`, `compute_training_weights` from sibling. Re-enable `tests/_pending/test_weights.py`.
- **Status**: queued
- **Cost**: low

### H-103 [P5] Port feature pipeline + imputation + base-series utilities
- **Owner**: IMPLEMENTER
- **Asks**: ask 5
- **Mechanism**: bring `compute_base_series`, `get_imputation_value`, `create_undef_flags_and_impute`. Re-enable `tests/_pending/test_features.py` and `test_properties.py`.
- **Status**: queued
- **Cost**: medium

### H-005 [P5] Run inventory-aware backtest on the existing offline+online stack
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 4 — **first PnL number**
- **Mechanism**: load existing `artifacts/offline_model/model.cbm` + replay `notebooks/online_eval.ipynb` to produce streaming `p_offline` and `p_final` predictions on the test split; feed both into `src/backtest.py::simulate_inventory_aware`; compare offline-only vs offline+online policy under realistic costs (CONSTITUTION V.b). Tag round `backtest` (30-min budget).
- **Falsification**: harness emits finite metrics on real predictions. The accept criterion is "first economic data point landed", not "Sharpe > X".
- **Status**: queued (after H-101..H-103 to ensure split utilities work)
- **Cost**: 30 min

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
