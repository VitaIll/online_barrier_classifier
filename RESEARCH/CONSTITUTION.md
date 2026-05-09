# Research-Engineering Loop — Constitution

The autonomous loop must obey these invariants. Any round that violates them is rolled back, regardless of metric improvement. The CRITIC sub-agent has veto power on every PR.

## 0. Project architecture snapshot

`online_barrier_classifier` is a **two-stage barrier classifier**:

```
Binance 1m kline → 20m decision bars (DecisionBarAggregator, gap-aware)
   → BaseFeatureExtractor + LagFeatureExtractor + RollingFeatureExtractor + ExpandingFeatureExtractor
   → label y_k = 1[ ln(H_{k+1} / C_k) ≥ α ]   (α = 90%-quantile of train log-excursions)
   → split: chronological, train_fraction=0.6, val_fraction (of train)=0.2
   → OFFLINE: CatBoost (langevin=True, Ordered, MVS, SqrtBalanced) → p_offline
   → top_k_features=120 selected by CatBoost importance
   → ONLINE (streaming conformal coverage layer):
        River ARFClassifier(n_models=100, max_features=log2,
                            split_criterion=hellinger, max_depth=15,
                            leaf_prediction=nba, lambda=6,
                            ADWIN warning δ=0.005, drift δ=0.0005, clock=64)
      consumes (selected_features + p_offline) and outputs p_online (calibrated probability,
      effectively the conformal-coverage estimate).
   → predict-then-learn-with-delayed-label prequential evaluation (notebooks/online_eval.ipynb).
```

**Current accepted constants** (mirror `config/*.yaml` and `artifacts/offline_model/config_snapshot.json`):

| Constant | Value | Source |
|---|---|---|
| `decision_interval` (M, minutes) | 20 | `pipeline.yaml` |
| `windows` (rolling, in decision bars) | `[1, 2, 4, 8, 12, 24, 48, 96]` | `pipeline.yaml` |
| `rolling_stats` | `[mean, var, min, max, iqr, ptp]` | `pipeline.yaml` |
| `lags` | `[1, 2, 3, 5, 10]` | `pipeline.yaml` |
| `expanding_stats` | `[mean, var, skew]` | `pipeline.yaml` |
| `barrier.method` | `quantile` | `pipeline.yaml` |
| `barrier.quantile` | 0.9 | `pipeline.yaml` |
| `burn_in_bars` | 96 (= max windows) | `pipeline.yaml` |
| `train_fraction` | 0.6 | `model.yaml` |
| `val_fraction` (of train) | 0.2 | `model.yaml` |
| `top_k_features` | 120 | `model.yaml` |
| Label α (calibrated) | 0.00411 (90%ile of train log-excursions) | `dataset_metadata.json` / `config_snapshot.json` |
| Symbol / period | BTCUSDT / 2023-01 → 2025-12 | `download.yaml` |
| Cleansed minute rows | 1,578,160 | `data/cleansed_data/BTCUSDT/metadata.json` |
| Test sample count | 31,486 | `artifacts/online_eval/metrics.json` |
| Test positive rate | 0.0971 | `artifacts/online_eval/metrics.json` |

**Legacy results** (pre-loop baseline; this is what every modeling round must improve on):

- Offline test: ROC=0.813, PR=0.356, Brier=0.090
- Online (final) test: ROC=0.799, PR=0.331, Brier ≈ 0.076 (verified by recomputation in earlier session: -16% vs offline)
- Calibration: offline systematically over-predicts (mean p ≈ 0.20 vs base rate 0.097), online is essentially diagonal across deciles

**The economic axis is unmeasured to date.** No PnL number exists yet. H-005 produces the first one.

## I. Causality invariants (HARD; tested)

These reflect `online_barrier_classifier`'s actual implementation, NOT the sibling `barrier_classifier`'s. Constants in [Section 0](#0-project-architecture-snapshot).

1. **No future data in features.** Feature `x_k` is built from minute bars `n ≤ n_k` where `n_k = k × M`, `M = 20`. Past-target features may use only matured labels `y_{<k}`.
2. **Chronological split, no explicit embargo.** Train/val/test are pure chronological cuts at `train_fraction × N` and `(train_fraction + val_fraction × train_size) × N` (val is last 20% of training window). This project does NOT have an `EMBARGO_K` separation — adding one is an open hypothesis (see BACKLOG H-104), not an existing invariant. Until then, every round must verify there is no horizon-overlap leakage between val and test (the label horizon is M=20, so a 1-bar train→test gap is enough but documented).
3. **Per-segment warmup.** `burn_in_bars = 96` (= `max(windows)`). Each segment's first 96 boundaries are dropped before training/evaluation (`bar_in_segment >= burn_in_bars`). Per-segment is enforced because `segment_id` resets at any 1-minute gap (cleansed-data invariant).
4. **No NaN dropping for engineered features.** This project's current implementation does NOT yet use the `undef__*` flag-as-input pattern; NaNs are filled inside the streaming feature extractors (e.g., `safe_divide` returns a neutral value with a flag). When a round adds a new feature that can be undefined, it must follow the same flag-and-impute discipline. Adding the formal `undef__*` flag pattern as a sibling-imported convention is BACKLOG H-103.
5. **Label diagnostics never used as features.** The label is `y_k = 1[ ln(H_{k+1} / C_k) ≥ α ]`, with α calibrated on training-only excursions (90th percentile, currently α ≈ 0.00411). The boundary `close`, `high`, and the calibrated α must never be in any feature column the model sees.
6. **Boundary observation rule** ([spec §5.2](docs/online_barrier_classifier_spec.md)): the decision bar `X_k` is the aggregation of minute bars `[k·M, (k+1)·M)`; features at `k` use only `X_0..X_k`; the label depends only on `H_{k+1}` (next bar's high), which is realized after `k`'s features are fixed.
7. **Prequential discipline (online stage).** The streaming conformal coverage layer must follow predict-then-learn-with-delayed-label. Specifically: at decision bar `k`, predict `p_online_k` using `(features_k + p_offline_k)` first; only when bar `k+1` arrives, compute `y_k` from `H_{k+1}`, then call `learn_one(z_{k}, y_k)` on the streaming model. The current `notebooks/online_eval.ipynb` implements this via a `label_buffer` deque — that idiom is the contract.

## II. Process invariants
1. **Branch-per-round, no automerge.** Each round commits to `agent/round-NNN-<slug>`. Never push to `master`. PRs are opened only after CRITIC approval; the human merges.
2. **One commit per round.** Squash all working commits into a single commit at end-of-round with a structured message (template in `ROUND_TEMPLATE.md`).
3. **Pytest must pass before commit.** `make test` is a precondition; a red test halts the round.
4. **Reproducibility.** All RNGs seeded (numpy, catboost, optuna, river). MLflow run ID logged in `LEDGER.md`. The exact data slice (start/end timestamp, fast-mode flag) recorded in MLflow tags.
5. **FAST_MODE for experimentation.** Hypothesis evaluation uses `FAST_MODE=1` (subsampled data + features) unless the LEDGER explicitly requests a full run for an accept-gate.
6. **Spec consistency.** Any change to feature/label/split logic must update `docs/MINIMAL_PROJECT_SPEC_v2.md` *or* be entered into `docs/ISSUES.md` with rationale. Drift between spec and code is itself a round.

## III. Decision gates
A round resolves to one of three outcomes:
- **`accept`**: hypothesis improves the primary metric on the accept-gate dataset (full data, time-aware, regime-stratified) by an effect size that survives the CRITIC's robustness checks. PR opened.
- **`iterate`**: partial signal; the hypothesis is re-queued to BACKLOG with refinements. No commit unless infrastructural.
- **`kill`**: hypothesis falsified or not worth the cost. Appended to KILL_LIST with reason. Branch deleted.

## IV. Primary success metrics (priority order)
1. **For online-stage rounds (the conformal coverage layer)**: empirical coverage (marginal AND per-regime) at α ∈ {0.05, 0.10, 0.20}; prediction-set tightness (avg interval width); coverage gap = 1 - α - empirical_coverage stratified by volatility tercile. **NOT** raw Brier/ROC — the online layer trades ranking for coverage.
2. **For offline-stage rounds**: out-of-sample probability quality — Brier-Skill-Score (= 1 - Brier_model / Brier_base_rate) + log-loss + Expected Calibration Error (ECE).
3. **Regime-stratified calibration** — ECE/Brier in each volatility tercile (low/med/high). Apply `pd.qcut(vol_proxy, 3)` on a fixed feature-derived signal (e.g., a long-window `parkinson_var_rolling_mean_*` chosen once and kept fixed across rounds).
4. **Risk-adjusted economic metric** — once `src/backtest.py` lands a real-data run (H-005), deflated Sharpe of an inventory-aware policy with realistic transaction costs becomes the headline accept gate. Until then, BSS + coverage are the gates.
4. **Discrimination** (ROC-AUC, PR-AUC) — secondary; cannot be the primary justification for an accept.
5. **Uncertainty-conditioned trade quality** — predictive interval width vs. PnL conditional on UQ thresholds.

ROC-AUC alone never accepts a hypothesis. (See sibling project `online_barrier_classifier` for an example where higher ROC came at calibration's expense.)

## V. Resource and budget rules
- **Wall clock per round**: 20 minutes default. Rounds tagged `backtest` (touching `src/backtest.py` or running an inventory-aware simulation) get 30 minutes. Rounds that exceed budget fall back to `iterate` with a partial result rather than burning compute.
- **MLflow tracking URI**: local SQLite at `experiments/mlruns/`.
- **Compute footprint**: keep ensemble size ≤ 3 in FAST_MODE; ≤ 7 for accept-gate runs.
- **Data footprint**: FAST_MODE = year 2024 only, top-200 features by importance; full mode = full configured range, all features.
- **Visual-first reporting**: every round producing metrics also produces at least one diagnostic plot saved as an MLflow artifact and referenced in the LEDGER entry. Backtest rounds produce equity curve, drawdown profile, and trade-distribution histogram at minimum.
- **Local literature first**: before LITERATURE-SCOUT spawns a web search, it must check `RESEARCH/literature/INDEX.md` for a relevant local PDF/MD. Web search is the fallback when the local corpus is silent.

## V.b. Strategy-realism rule (HARD)
A round's **falsification** tests may use trivial strategies (random `p`, perfect oracle) — those exist only to validate the harness's own behaviour. A round's **evaluation** of model skill must use a **realistic** strategy with all of:
- inventory cap and explicit position-sizing rule (unit-position by default; bet-sized via meta-label-confidence when meta-labeling lands),
- transaction costs deducted round-trip at the configured `c_stop` (or higher for stress runs),
- risk management: a stop-loss barrier, a maximum-open-trades constraint, a max-drawdown circuit-breaker (round halts if equity DD > 30% during the simulation),
- a sensible `τ_open` chosen via validation Sharpe-grid (or PR-AUC-grid), not picked post-hoc on test.

Specifically forbidden as "evaluation" benchmarks:
- always-on (`p ≡ 1`) — has no inventory awareness.
- raw `p > 0.5` without cost deduction — ignores friction.
- buy-and-hold with no exit logic.
- any strategy that opens a new long while a long is already open.

These can appear as **null-skill comparators** in falsification or in bootstrap p-value computation, but never as the headline backtest figure of merit.

## VI. Code health
- **Dead code rule**: anything not referenced by tests or notebooks within 2 rounds → `KILL_LIST` and removed.
- **No new features without a test.** Every `compute_*` function added must have a test asserting at least: shape, dtype, finiteness post-imputation, monotonicity if applicable.
- **The HOUSEKEEPER sub-agent** runs once per week to prune redundant artifacts and surface drift.

## VII. Authorization scope (what the loop may do without asking)
- Create/modify files under `barrier_classifier/`
- Run pytest, train models, log to MLflow
- Commit to `agent/round-NNN-*` branches
- Open draft PRs to `main`
- Append to LEDGER, KILL_LIST, REPORT_LATEST

The loop must NOT (without explicit human authorization):
- Push to `main` or merge any PR
- Force-push to any branch
- Delete data files
- Modify global git config or CI secrets
- Bypass any gate (no `--no-verify`, no skipping CRITIC)
- Change anything in `docs/MINIMAL_PROJECT_SPEC_v2.md` Sections 4–10 without an ISSUES.md entry first
