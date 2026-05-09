# Hypothesis Backlog

The autonomous loop targets `online_barrier_classifier` — the **combined offline + online** system. The differentiator vs the sibling `barrier_classifier` (offline-only research) is the online River ARF correction layer that consumes offline probabilities and improves calibration in a streaming, drift-aware way.

Status legend: `queued` / `in_progress` / `iterate` / `blocked`. Priority P1–P5 reflects expected α-on-user-goal / cost-to-evaluate.

Top of file = highest priority. The loop picks from the top.

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

## Tier 1 — Online model improvements (this project's differentiator; ask 2)

### H-030 [P5] River SRPClassifier vs ARF baseline
- **Owner**: CODE-SCOUT + IMPLEMENTER
- **Mechanism**: replace `ARFClassifier` with `river.ensemble.SRPClassifier` in the online pipeline. Same warmup+test stream. Compare Brier / log-loss / drift detection.
- **Falsification**: paired prequential test on identical streams; SRP must beat ARF on Brier on ≥2/3 seeds.
- **Status**: queued
- **Cost**: low

### H-031 [P4] Hoeffding Adaptive Tree + ALMA stack
- **Mechanism**: `HoeffdingAdaptiveTreeClassifier` + `ALMAClassifier` averaged with ARF. `compose.Stacker` or weighted-mean tracked online.
- **Status**: queued
- **Cost**: medium

### H-032 [P4] Streaming isotonic / Platt calibration in River pipeline
- **Mechanism**: wrap online classifier in `river.calibration.IsotonicCalibrator`. Maintains calibrated probs as drift accumulates.
- **Status**: queued
- **Cost**: low

### H-033 [P3] Online feature standardization
- **Mechanism**: `compose.Pipeline(preprocessing.StandardScaler() | model)`.
- **Status**: queued
- **Cost**: low

### H-034 [P3] ADWIN drift detector on prediction error
- **Mechanism**: feed `(p_k - y_k)^2` into ADWIN; on detected drift, log a marker and optionally reset selected base learners.
- **Status**: queued
- **Cost**: low

---

## Tier 2 — UQ + conformal wired into the offline+online flow (asks 6, 4)

### H-010 [P5] Wire CatBoost virtual-ensemble UQ into 03_offline_train + online_eval
- **Mechanism**: use `src/uncertainty.py::predict_with_decomposed_uq` on the offline CatBoost; expose `sigma_epistemic` to the online layer's input vector (alongside `p_offline`). Test whether the online ARF benefits from epistemic-uncertainty signal.
- **Falsification**: paired comparison: online layer with vs without `sigma_epistemic` feature. Brier delta.
- **Status**: queued (depends on H-103 + a model retrain)
- **Cost**: medium

### H-011 [P5] Conformal LAC + Mondrian for trade abstention
- **Mechanism**: `src/conformal.py` is ported. Use it to produce prediction sets at α = 0.1, 0.2; `conformal_trade_signal` gates the trader's entries. Combined with H-005's backtest harness for an actual abstention-aware Sharpe.
- **Status**: queued (depends on H-005)
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
