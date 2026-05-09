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

## Top 5 backlog
1. **H-101 [P5]** Port labels + chronological-split + walk-forward-cv utilities (re-enables 2 test files)
2. **H-102 [P5]** Port sample-weighting utilities + tests (re-enables 1 test file)
3. **H-103 [P5]** Port feature pipeline + imputation + base-series utilities (re-enables 2 test files)
4. **H-005 [P5]** Run inventory-aware backtest on existing offline+online stack — first PnL number
5. **H-030 [P5]** River SRPClassifier vs ARF baseline (online ask 2 differentiator)

## Architecture context (online_barrier_classifier specifically)
- Decision interval: M=20 minutes
- Offline: CatBoost (langevin=True) — `artifacts/offline_model/`
- Online: River ARFClassifier correction layer — `notebooks/online_eval.ipynb`
- Combined-system metric: online_corrected ROC=0.799 vs offline ROC=0.813 (legacy run)
  but Brier 0.076 vs 0.090 → **online layer trades ranking for calibration** (~16% Brier reduction)
- The autonomous loop's mission: improve BOTH stages, with primary focus on the online correction layer (ask 2), all under deflated-Sharpe + regime-stratified ECE accept gates.

## Open questions for the human
1. Cron cadence: every 6h via PowerShell daemon? Or fireAt-based self-trigger only?
2. Anything off-limits in the existing offline pipeline (`src/utils.py`, `notebooks/`)?
3. Backtest budget: 30 min OK for backtest-tagged rounds?
