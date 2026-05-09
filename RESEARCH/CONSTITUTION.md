# Research-Engineering Loop — Constitution

The autonomous loop must obey these invariants. Any round that violates them is rolled back, regardless of metric improvement. The CRITIC sub-agent has veto power on every PR.

## I. Causality invariants (HARD; tested)
1. **No future data in features.** Feature `x_k` may use only bars `n ≤ n_k` where `n_k = k×M`. Past-target features may use only matured labels `y_{<k}`.
2. **Embargo respected.** `EMBARGO_K=60` decision steps between any two adjacent splits in train/val/test. Walk-forward CV inside HPO uses the same embargo.
3. **Per-segment warmup.** Drop the first `K_WARMUP=144` boundaries of each segment before any evaluation. `min(k_test) ≥ K_WARMUP` is asserted.
4. **No NaN dropping for engineered features.** Use the `undef__{feature}` flag-as-input pattern and impute deterministically. Dropping rows because of feature NaNs is forbidden — only label NaNs at series-end may be dropped.
5. **Label diagnostics are not features.** `m_k`, `tau_k`, `phi`, and weight columns (`w_dist`, `w_time`, `weight`) must never appear in `feature_list.json`.
6. **Boundary observation rule** ([spec §5.2](docs/MINIMAL_PROJECT_SPEC_v2.md)): bar `n_k` fully observed; `>n_k` strictly forbidden.

## II. Process invariants
1. **Branch-per-round, no automerge.** Each round commits to `agent/round-NNN-<slug>`. Never push to `main`. PRs are opened only after CRITIC approval; the human merges.
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
1. **Out-of-sample probability quality on the test split** — Brier score, log-loss, Expected Calibration Error (ECE).
2. **Regime-stratified calibration** — ECE/Brier in each volatility tercile (low/med/high).
3. **Risk-adjusted economic metric** — once `src/backtest.py` lands, deflated Sharpe of an inventory-aware policy with realistic transaction costs.
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
