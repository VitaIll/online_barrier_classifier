# Latest Sprint Report

Regenerated at end of each round.

---

**As of**: 2026-05-09 (Round 0 — bootstrap)

## Status
- **Branches**: `agent/round-000-bootstrap` (loop infrastructure port)
- **Open PRs**: none yet
- **Cron**: registered as scheduled task `barrier-classifier-research-round` (will be renamed to `online-barrier-classifier-research-round`); fireAt-based self-trigger
- **MLflow runs**: none yet
- **Tests**: 24 passed (architecture-agnostic: backtest, conformal, training_helpers); 5 test files parked in `tests/_pending/` awaiting H-101..H-103 utility ports
- **Diagrams**: `RESEARCH/diagrams/` (empty until first round produces one)

## Last 5 LEDGER entries
- 2026-05-09 | 000 | bootstrap | infrastructure ported from sibling

## Round ordering (post-bootstrap, codified in BACKLOG)

1. **H-005** — first PnL via `artifacts/online_eval/predictions.parquet` (already on disk; no port dependency)
2. **H-201** — coverage baseline on existing ARF (measure-only)
3. **H-108** — port CatBoostEnsemble class (cheap; unblocks ensemble work)
4. **H-101** — label + split utilities (round-trip validation)
5. **H-106** — regime-stratified calibration helpers (CONSTITUTION IV primary metric)

Then in priority order: H-107 (plot helpers), H-202 (ACI), H-105 (NSGA-II HPO), H-102 (sample weighting), H-103 (undef flags), H-111..H-115 (sibling-borne features), H-203..H-208 (online-stage refinements).

## Architecture context (verified against `config/*.yaml` + `artifacts/offline_model/config_snapshot.json`)

### Two-stage barrier classifier
- **Decision interval**: M = 20 minutes (every 20 minute bars → one decision bar `X_k`).
- **Label**: `y_k = 1[ ln(H_{k+1} / C_k) ≥ α ]` where α is calibrated to the 90th percentile of training-only log-excursions; current α ≈ **0.00411** (~41 bps).
- **Splits**: chronological. `train_fraction=0.6`, `val_fraction (of train)=0.2`. **No embargo** (open hypothesis H-104).
- **Per-segment warmup**: `burn_in_bars=96` (= max(rolling-windows)). `segment_id` resets at any 1-min gap.

### Offline stage
- CatBoost: 6000 iters, lr=0.007, depth=6, l2=1.0, bootstrap=MVS+mvs_reg=10, grow=SymmetricTree, leaf=Newton×10, **`auto_class_weights=SqrtBalanced`** (this is the source of over-prediction), `boosting_type=Ordered`, `langevin=True`+diffusion_temp=15000, rsm=0.9, subsample=0.65, early_stopping=100. Output → `p_offline`.

### Online stage = streaming conformal coverage layer
- River **ARFClassifier**(`n_models=100`, `max_features=log2`, `lambda_value=6`, `metric=CrossEntropy`, `split_criterion=hellinger`, `max_depth=15`, `leaf_prediction=nba`, `nb_threshold=50`, ADWIN warning δ=0.005 / drift δ=0.0005 / clock=64, `grace_period=200`, `delta=0.001`, `tau=0.01`, `max_size=20.0` MB, `remove_poor_attrs`, `merit_preprune`).
- Input vector `z_k = (top_120_features_k, p_offline_k)`. Output → `p_final` ∈ [0,1].
- **NOT a separate classifier** — its job is input-conditional coverage of `P(y=1|x_k)` in a streaming, drift-aware way.
- Prequential discipline: `predict_proba_one(z_k)` → record → buffer `(z_k, ref_close=C_k)` → when bar `k+1` arrives, compute `y_k = 1[ ln(high_{k+1}/C_k) ≥ α ]` and `learn_one(z_k, y_k)`. Implemented via deque in `notebooks/online_eval.ipynb`.

### Legacy reported results (the floor every modeling round must beat)
- Test n = 31,486; positive rate = **0.0971**.
- **Offline**: ROC=0.813, PR=0.356, Brier=0.090, mean predicted p = 0.205 (over-predicts by ~2×).
- **Online (final)**: ROC=0.799 (slight ranking loss), PR=0.331, **Brier ≈ 0.076 (~16% better)**, mean predicted p = 0.108 (≈ base rate).
- Decile calibration: offline systematically under-empirical across all bins; online sits within ~2pp of the diagonal everywhere.
- **The online stage already does conformal-style coverage in practice** — it just hasn't been formalised. Round H-201 measures and names this baseline; H-202 (ACI) replaces ARF with the formally derived streaming conformal threshold.

### Primary metrics for online-stage rounds
Marginal empirical coverage at α ∈ {0.05, 0.10, 0.20} + per-regime coverage gap (volatility-tercile Mondrian) + prediction-set tightness. NOT raw Brier or ROC alone.

### Local literature anchors
- Gibbs & Candès (2021) **Adaptive Conformal Inference** — streaming threshold update.
- Vovk (2003) Mondrian for class-/feature-conditional coverage.
- Manokhin (2024) *Practical Conformal Prediction* (`Downloads/Valeri Manokhin - ...`) — implementation.
- Lekeufack et al. (2024) *Conformal Decision Theory* (`Downloads/conformal_decision_theory.pdf`) — confidence-band-based decision rule for the trade gate.

## Open questions for the human
1. Cron cadence: every 6h via PowerShell daemon? Or fireAt-based self-trigger only?
2. Anything off-limits in the existing offline pipeline (`src/utils.py`, `notebooks/`)?
3. Backtest budget: 30 min OK for backtest-tagged rounds?
