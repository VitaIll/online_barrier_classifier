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
- **Status**: ACCEPTED round-004 — `src/utils.py` adds `compute_log_excursion`, `calibrate_alpha`, `construct_labels`, `chronological_split`, `chronological_split_indices`. 21 tests in `tests/test_label_split_utils.py` pass; round-trip on the persisted `bars_20m_features.parquet` (n=78,714) matches notebook cell-3 boundaries exactly (train_end=37,783; val_end=47,228). Parked sibling tests `test_causality.py`/`test_splits.py` retired — semantics didn't match (sibling had embargo + walk-forward; this project has neither). Diagnostic plot `RESEARCH/diagrams/round_004/split_window_visualization.png` shows train/val/test windows on the date axis with per-window positive rates (train p+=0.097, val p+=0.113, test p+=0.097). Unblocks H-105 (NSGA-II HPO needs these splits).
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
- **Status**: ACCEPTED round-001 — finite metrics confirmed on real stream; p_offline τ=0.30 Sharpe=+1.50 PSR=0.995 with bootstrap p<0.001 vs shuffled-signal null. Online layer (p_final=p_online) does NOT add Sharpe at any τ tested — consistent with CONSTITUTION I (online optimizes coverage not ranking). Spawned H-005b + H-005c.
- **Cost**: 30 min (actual: ~7 min wall-clock)

### H-005b [P5] (spawned by round 001) Validation-split τ_open selection for the H-005 sweep
- **Owner**: IMPLEMENTER
- **Asks**: ask 4 (rigorous evaluation per CONSTITUTION V.b)
- **Mechanism**: round 001 picked τ_open as a post-hoc grid because the existing `artifacts/online_eval/` only persists test predictions. Re-run `notebooks/offline_train.ipynb` + `notebooks/online_eval.ipynb` to additionally persist a `predictions_val.parquet` (the chronological val window that was already used for early-stopping). Sweep τ_open on val Sharpe + PR-AUC, pick winner per CONSTITUTION V.b, then evaluate at fixed τ on test. Tag `backtest` (30-min budget). Compare round-001's post-hoc τ=0.30 to the val-chosen τ — if they coincide, the round-001 result holds; if they diverge, the round-001 numbers are demoted to "exploratory only" in REPORT_LATEST.
- **Falsification**: val-chosen τ for p_offline lands inside the round-001 post-hoc grid {0.10..0.50}; if the val-Sharpe-grid optimum is outside this range, the round-001 sweep was insufficiently wide.
- **Status**: queued (depends on a notebook re-run, not on a model retrain)
- **Cost**: medium (notebook surgery + 1 retrain)

### H-005c [P3] (spawned by round 001) CSCV PBO deflation of the H-005 τ sweep
- **Owner**: IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 4 — overfitting check on the post-hoc τ grid
- **Mechanism**: H-113 lands the CSCV / PBO machinery (combinatorially symmetric CV PBO per Bailey et al. 2014). Once H-113 ships, apply it to the round-001 τ_sweep_metrics.csv to compute the PBO of the best post-hoc τ. PBO < 0.5 means the τ=0.30 winner generalizes; PBO ≥ 0.5 means the post-hoc grid was overfit to test. This is the deflation step that justifies promoting round-001 numbers to a "headline" status.
- **Status**: blocked on H-113.
- **Cost**: low (post-H-113)

**Round ordering (post-bootstrap)**:
1. ~~**H-005**~~ — ACCEPTED round-001.
2. ~~**H-201**~~ — ACCEPTED round-002. Coverage baseline established; α=0.20 low-vol gap = -7.9pp is what H-202..H-208 must close.
3. ~~**H-108**~~ — ACCEPTED round-003. `src/ensemble.py` lands; unblocks H-206.
4. ~~**H-101**~~ — ACCEPTED round-004. Label + split utilities in `src/utils.py`; round-trip validated on persisted parquet; unblocks H-105.
5. ~~**H-106**~~ — ACCEPTED round-005. Calibration metrics in `src/utils.py`. Per-regime visual proves online's regime-flat ECE thesis empirically.
6. ~~**H-107**~~ — ACCEPTED round-006. Plot helpers in `src/plotting.py`; weight plots deferred to H-102.
5. **H-106** — regime-stratified calibration helpers (primary metric per CONSTITUTION IV).
6. **H-107** — plot helpers + threshold-analysis CSV (visual-first reporting parity with sibling).
7. **H-202** — Adaptive Conformal Inference on offline output.
8. **H-105** — NSGA-II HPO setup (after H-101 establishes split utilities).
9. **H-102** — sample weighting (carefully, given SqrtBalanced double-count concern).
10. **H-103** — undef-flag pattern.
11. **H-111 / H-112 / H-114 / H-115** — sibling-borne feature groups.
12. **H-203 / H-204 / H-205 / H-206 / H-207 / H-208** — online-stage refinements.

(H-201b, H-201c are P4/P3 follow-ups inserted after current ordering; pick when natural.)

---

## Tier 1 — Streaming conformal coverage layer (asks 2 + 6, the project's differentiator)

The online stage is a streaming conformal layer providing conditional coverage. These rounds explicitly improve coverage validity and tightness, NOT raw ranking.

### H-201 [P5] Baseline coverage diagnostic on the existing online ARF
- **Owner**: IMPLEMENTER + THEORIST
- **Mechanism**: treat the existing River `ARFClassifier` output as if it were a conformal predictor at varying confidence levels (set construction via `predict_proba_one`). Compute marginal empirical coverage AND per-regime coverage on the test split. Quantify *coverage gap* = 1 - α - empirical_coverage by regime. This becomes the baseline every online-stage hypothesis must beat.
- **Falsification**: report finite numbers; a > 5% absolute coverage gap in any regime is the gap to close.
- **Status**: ACCEPTED round-002 — chronological 30/70 cal/eval split on predictions.parquet with parkinson_var_rolling_mean_24 terciles. Finite numbers everywhere; constant-predictor sanity passes (α=0.05 cov=1.0000, α=0.20 cov=0.9181=1-eval_base_rate). Naive `p≥α` over-covers by 8-10pp at α=0.10 → ARF probs not directly usable as conformal thresholds. Mondrian LAC compresses α=0.10 gap to ±3.4pp; α=0.20 still has -7.9pp on low-vol/p_online and -5.8pp on low-vol/p_offline → THE gap H-202..H-208 must close. Spawned H-201b (finer-bin regime), H-201c (cal-fraction sensitivity).
- **Cost**: low (no model changes; just measurement)

### H-201b [P4] (spawned by round 002) Finer regime binning to expose what Mondrian-LAC misses at α=0.20
- **Owner**: IMPLEMENTER
- **Asks**: ask 2 (online coverage)
- **Mechanism**: round 002 used 3-bucket parkinson_var terciles. The persistent low-vol gap at α=0.20 (-7.9pp over-coverage) suggests the low-vol bucket contains a sub-mode that Mondrian-LAC's single q_hat for the whole tercile cannot adapt to. Re-run `scripts/round_002_coverage_baseline.py` with `N_TERCILES=5` (quintiles) and check whether the low-quintile gap at α=0.20 shrinks below ±5pp. If it does, the ARF coverage gap is fundamentally a *resolution* problem (more buckets fix it); if it doesn't, the gap is a probability-quality problem (Mondrian can't fix it; needs ACI / locally-weighted conformal — H-202..H-205).
- **Falsification**: low-quintile gap at α=0.20 stays > 5pp under quintile binning → resolution alone won't help; H-202 work justified.
- **Status**: queued
- **Cost**: low (script tweak + rerun)

### H-201c [P3] (spawned by round 002) Sensitivity of coverage gap to calibration-fraction
- **Owner**: IMPLEMENTER
- **Asks**: ask 2 + ask 4 (rigorous evaluation)
- **Mechanism**: round 002 used a fixed 30/70 chronological cal/eval split. Sweep `CAL_FRAC ∈ {0.20, 0.30, 0.40, 0.50}` and report how max |gap| (Mondrian, α=0.10) and max |gap| (Mondrian, α=0.20) vary. If the gap is monotone-decreasing in cal_frac, the round-002 gap was an n_cal-power problem; if invariant, it's a real conditional-miscalibration signal.
- **Falsification**: gap collapses to within ±0.5pp at cal_frac=0.50 → the round-002 finding is an n_cal-power artifact, not a real gap.
- **Status**: queued
- **Cost**: low (script-level loop + 4× rerun)

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

### H-111 [P4] Group N — barrier-aware features (ported from sibling)
- **Owner**: IMPLEMENTER
- **Asks**: ask 1
- **Mechanism**: this project's barrier construction `α = 0.00411` is a fixed economic threshold, exactly the regime where barrier-aware features (`barrier__z_tight = α / (σ·√M)`, `barrier__emax_ratio = σ·√(2 ln M) / α`, `vol__ratio_short_long`) carry information. Sibling implements them in `barrier_classifier/src/utils.py::compute_barrier_aware_features` (16 windows + 12 vol pairs). Port adapted to this project's `M=20` and `windows=[1,2,4,8,12,24,48,96]`.
- **Status**: queued (high information density; adds ~50 features)
- **Cost**: medium

### H-112 [P4] Group O — excursion features (max drawup/drawdown, max-N-bar returns)
- **Owner**: IMPLEMENTER
- **Asks**: ask 1 — **label-aligned** features
- **Mechanism**: rolling max drawup, max drawdown, max 1-bar / 2-bar returns. Label-aligned because `y_k` itself is a max-future excursion. Sibling: `compute_excursion_features` with chunked stride-trick implementation (avoid 9-window naive O(n·W)).
- **Status**: queued
- **Cost**: medium

### H-114 [P4] Group C+ — volatility decomposition (bipower variation ratio, semivariance up/down/ratio, vol-of-vol)
- **Owner**: IMPLEMENTER
- **Asks**: ask 1
- **Mechanism**: bipower variation = jumpiness proxy (`RV / BPV`); semivariance asymmetry (`SV_down / SV_up` semivolatility ratio); vol-of-vol (rolling std of rolling vol). Sibling: `compute_volatility_decomposition`. The semivariance ratio in particular is well-suited to this project's positive-skew label (only upper barrier).
- **Status**: queued
- **Cost**: medium

### H-115 [P3] Permutation entropy (m=3, τ=1) — complexity / predictability
- **Owner**: IMPLEMENTER
- **Asks**: ask 1
- **Mechanism**: Bandt-Pompe normalized permutation entropy on returns. Sibling: `compute_permutation_entropy` with stable mergesort tie-breaking.
- **Status**: queued
- **Cost**: low

---

## Tier 5 — Production engineering (ask 5)

### H-050 [P4] Feature pipeline as hash-keyed cache
- **Mechanism**: every `compute_*` keyed by `(name, version, config_hash, data_hash)`; cache to `experiments/feature_cache/`.
- **Status**: queued

### H-051 [P4] `compute_*` registry — replace imperative chain in `feature_build` notebook
- **Status**: queued

### H-054 [P3] Pre-commit hook running fast tests
- **Status**: queued

### H-105 [P5] NSGA-II multi-objective Optuna HPO with walk-forward CV (sibling import)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5 — this project currently has NO hyperparameter optimization (`ENABLE_HPO=False` everywhere implicitly).
- **Mechanism**: port the sibling's HPO pattern from `barrier_classifier/notebooks/03_model_training.ipynb`: NSGA-II sampler (objectives = minimize logloss, maximize PR-AUC), walk-forward CV across N folds (each fold's train precedes its val chronologically + embargo), per-trial seed variation (CB_SEED + trial.number), Pareto-frontier visualization, "best trial" selection rule (within 5% of min-logloss, take best PR-AUC), `HPO_DROP_OLDEST_FRAC` speed knob, ordered Pool with timestamp.
- **Falsification**: a fresh HPO run + retrain produces metrics within ±5% of the legacy ROC=0.813 / Brier=0.090 (or beats them).
- **Status**: queued
- **Cost**: medium-high (45+ min wall-clock per HPO run; plan as `backtest`-budget round)

### H-106 [P5] Regime-stratified calibration reporting (sibling import + adaptation)
- **Owner**: IMPLEMENTER
- **Asks**: ask 4 — primary metric per CONSTITUTION IV
- **Mechanism**: port `compute_all_metrics`, `expected_calibration_error`, `calibration_by_regime`, `threshold_analysis` from sibling `src/utils.py`. Wire into `notebooks/offline_train.ipynb` and `notebooks/online_eval.ipynb`. Pick a fixed regime signal (proposal: `parkinson_var_rolling_mean_24` from this project's existing features) and run `pd.qcut(_, 3)` to get terciles. Output `RESEARCH/diagrams/round_NNN/calibration_by_regime.png` per round.
- **Status**: ACCEPTED round-005 — four helpers in `src/utils.py`. 13 tests pass. First per-regime calibration plot on real test stream produces a striking diagnostic: offline ECE scales 0.05/0.10/0.17 (low/med/high vol — overpredicts everywhere, severity rises with vol); online ARF ECE ≈ 0.015 uniformly across regimes (mean_p ≈ base_rate per regime). This is direct empirical proof of the project's design thesis (online sacrifices ranking for regime-conditional calibration) and identifies high-vol as where offline calibration breaks worst — pointing H-202..H-208 at vol-conditional improvement.
- **Cost**: medium

### H-107 [P4] Plot helpers + threshold-analysis CSV (sibling import for visual-first reporting)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5 + visual-first CONSTITUTION rule
- **Mechanism**: port `plot_calibration_curve`, `plot_calibration_by_regime`, `plot_feature_importance`, `plot_threshold_curves`, `plot_weight_profiles`, `plot_weight_distributions` from sibling. The `threshold_analysis.csv` output (precision, recall, trade-rate at sweep of thresholds) is missing from this project's eval artifacts.
- **Status**: ACCEPTED round-006 — `src/plotting.py` lands four helpers (calibration_curve, calibration_by_regime, feature_importance, threshold_curves). 13 smoke tests pass; round-006 visual report card on real data reproduces round-005 ECE numbers exactly (offline 0.108, online 0.015). `plot_weight_*` explicitly deferred to H-102 round. `feature_importance` API improved over sibling: takes `importances` array directly (works with `CatBoostEnsemble` / sklearn / anything), not a model object.
- **Cost**: low

### H-108 [P4] CatBoostEnsemble wrapper class (sibling import)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5
- **Mechanism**: port the `CatBoostEnsemble` class (averages predictions / feature importances / best iterations across N seed-varied CatBoost models). Drop-in replacement for the ad-hoc model handling currently in `notebooks/offline_train.ipynb`. Saves cleanly + reloads via base+`.{i}.cbm` pattern.
- **Status**: ACCEPTED round-003 — `src/ensemble.py` with `CatBoostEnsemble` (predict_proba/feature_importance/best_iteration averaging + save_model + new `load_ensemble` classmethod). 10 tests in `tests/test_ensemble.py` pass. Diagnostic plot `RESEARCH/diagrams/round_003/ensemble_dispersion.png` shows non-trivial seed disagreement (max single-model deviation from ensemble = 0.070). Unblocks H-206.
- **Cost**: low (~20 min)

### H-113 [P3] CSCV — Probability of Backtest Overfit (López de Prado Ch. 11)
- **Owner**: LITERATURE-SCOUT + IMPLEMENTER
- **Asks**: ask 4 — overfitting check for any backtest-tagged round
- **Mechanism**: combinatorially symmetric CV: split test into S equal slices, take all `(S choose S/2)` partitions, rank in-sample SR vs out-of-sample SR per partition; PBO = Pr[best IS ranks below median OOS]. Loop must report PBO whenever a round changes the strategy or hyperparameters. Reference: Bailey et al. (2014) `pseudo-mathematics-and-financial-charlatanism`.
- **Status**: queued (depends on H-005 for at-least-one backtest data point)
- **Cost**: medium

---

## Tier 6 — Sibling reference index

The sibling `C:\Users\vitil\OneDrive\Desktop\barrier_classifier\` is the **read-only reference** for offline-stage practices. Specific practices already queued above as hypotheses:

| Sibling artifact | Imported as | Status |
|---|---|---|
| `compute_barrier_distance_weight` (deep-loss exp upweight, soft cap) | H-102 | queued |
| `compute_time_discount_weight` (geometric δ-decay with floor + cutoff) | H-102 | queued |
| `compute_training_weights` (combined w_dist · w_time, effective-N) | H-102 | queued |
| `chronological_split_with_embargo`, `walk_forward_cv` | H-101 + H-104 | queued |
| `get_imputation_value`, `create_undef_flags_and_impute` (regex lookup) | H-103 | queued |
| `CatBoostEnsemble` class | H-108 | queued |
| NSGA-II Optuna HPO with walk-forward CV (notebook 03) | H-105 | queued |
| `compute_all_metrics`, `expected_calibration_error`, `calibration_by_regime`, `threshold_analysis` | H-106 | queued |
| `plot_calibration_curve`, `plot_feature_importance`, `plot_threshold_curves`, weight plots | H-107 | queued |
| `compute_barrier_aware_features` (Group N) | H-111 | queued |
| `compute_excursion_features` (Group O drawup/drawdown + maxret) | H-112 | queued |
| `compute_volatility_decomposition` (bipower, semivar, vov) | H-114 | queued |
| `compute_permutation_entropy` (Bandt-Pompe) | H-115 | queued |
| `bootstrap_no_skill_pvalue`, `deflated_sharpe` (already in `src/backtest.py`) | done at bootstrap | accept |
| `predict_with_decomposed_uq`, `predictive_intervals` (already in `src/uncertainty.py`) | done at bootstrap | accept |
| `fit_conformal`, `predict_set`, `coverage_by_regime` (already in `src/conformal.py`) | done at bootstrap | accept |

**Rule for sibling-import rounds**: don't blindly copy. Adapt to this project's:
- `M=20` (sibling uses `M=10`); window sets differ
- No EMBARGO (sibling has 60); per-segment burn-in uses `burn_in_bars=96` (sibling uses `K_WARMUP=144`)
- Online stage = streaming conformal coverage (sibling has none of this)
- `auto_class_weights=SqrtBalanced` interaction with sample weighting (H-102)
- Existing notebook flow (sibling has `01_data_download / 02_feature_building / 03_model_training`, this project has `data_download / feature_build / offline_train / online_eval`)

---

## Notes
- Items at `[P5]` are unconditionally scheduled before `[P3]`.
- "BLOCKED" items wait until the unblock-er is `accept`ed.
- New ideas append to the appropriate tier section, NOT the top — priority discipline is enforced.
- Items spawned mid-round are marked with `(spawned by round NNN)` for traceability.
