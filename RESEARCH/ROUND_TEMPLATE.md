# Round Playbook (v2 — round-007)

Each scheduled fire executes one round. **Re-arm happens first** (Phase A), so a mid-round crash never kills the loop. Hard wall-clock timeout is enforced by the daemon (`scripts/loop_daemon.ps1`), not by the agent. Compaction runs at the end (Phase 7).

## Phase A — Re-arm IMMEDIATELY (idempotent)

Even if everything else fails, the loop survives:

```python
import datetime as dt
next_fire = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60)).isoformat()
mcp__scheduled-tasks__update_scheduled_task(
    taskId="barrier-classifier-research-round",
    fireAt=next_fire,
)
```

If running under `loop_daemon.ps1`, the daemon handles re-arm — Phase A is a safety net no-op.

## Phase 0 — Init

1. `python scripts/run_round.py --preflight` — halt round on red.
2. Determine `next_id = $(python scripts/run_round.py --next-round-id)`.
3. Read `RESEARCH/CONSTITUTION.md`, top of `BACKLOG.md`, last 10 lines of `LEDGER.md`, recent `KILL_LIST.md`.

## Phase 1 — Pick + plan

1. Pick top non-blocked, non-`in_progress` BACKLOG item. If empty, run HOUSEKEEPER (Phase H).
2. `git checkout -b agent/round-NNN-<slug>` from current `main`.
3. Mark backlog item `in_progress`.

## Phase 2 — Research (parallel)

Spawn LITERATURE-SCOUT + CODE-SCOUT + THEORIST in a **single message**, multiple Agent tool uses. LITERATURE-SCOUT must local-first via `RESEARCH/literature/INDEX.md`.

Synthesize HYPOTHESIS.md on the branch:
```
H-NNN: <one-line claim>
Mechanism: <how it works>
Predicted effect: <metric, magnitude, direction>
Seed-noise band: <baseline metric ± σ from N≥2 reruns>
Falsification: <what observation kills it>
References: <local-pdf paths or web URLs>
```

## Phase 3 — Implement (FAST_MODE)

1. Tests-first for any new compute_*. Diff target ≤ 400 LoC excluding tests.
2. NaN handling via `undef__*` flag-as-input pattern; never drop on feature NaN.
3. Don't add label-derived columns to feature_list (m_k, tau_k, phi, w_dist, w_time, weight).

## Gate 1 — Tests

```
make test
python scripts/critic_check.py --hypothesis H-NNN --claim "<one-liner>"
```

Both must return 0. The CRITIC checklist enforces causality, embargo, leakage, plot-present, ledger-entry-added.

## Phase 4 — Train + Eval

1. `with training_helpers.log_training_run(round_id, hypothesis_id, fast_mode, ...) as run:`
2. Tag run: round_id, hypothesis_id, fast_mode, data_start, data_end, git_sha.
3. Log metrics from `compute_all_metrics`, `expected_calibration_error`, `calibration_by_regime`. UQ rounds add `uq_quality_summary`. Backtest rounds add deflated Sharpe + AURC.
4. Save ≥1 diagnostic plot. **Read the saved PNG back via the Read tool to verify legibility before committing.**
5. **Primary metric is Brier-Skill-Score** = 1 - Brier_model / Brier_base_rate_predictor. Raw Brier is misleading when base rates differ across rounds.

## Gate 2 — CRITIC

Spawn CRITIC sub-agent per `RESEARCH/AGENTS.md`. CRITIC:
1. Runs `python scripts/critic_check.py --json` and parses verdict.
2. Verifies effect size > 1.5 × seed-noise band.
3. Applies multiple-comparisons correction if N_trials in this hypothesis area > 5.
4. Returns APPROVE / REQUEST_CHANGES / VETO.

## Phase 5 — Commit

Squash to one commit:

```
agent/round-NNN: <slug> [<accept|iterate|kill>]

Hypothesis (H-NNN): <one-liner>
Mechanism: <one-liner>
FAST_MODE BSS Δ: <signed value> (vs paired baseline)
Accept-gate result: <if accept>
MLflow run: <run_id>
Tests: <count> passed; CRITIC: <APPROVE|...>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Decision routing:
- `accept`: `gh pr create --draft --base main --head agent/round-NNN-<slug>` with the long-form report.
- `iterate`: re-queue with refinements; commit on branch but no PR.
- `kill`: append `[round NNN] H-NNN: <reason>` to KILL_LIST; compaction will delete the branch after `DEAD_BRANCH_AGE_ROUNDS`.

## Phase 6 — Update state

1. Append one line to `LEDGER.md` (pipe-separated, no prose).
2. Update `BACKLOG.md`: status of the picked item; add any new hypotheses spawned.
3. Regenerate `REPORT_LATEST.md`: status, last 5 LEDGER lines, top 5 BACKLOG, open PRs.

## Phase 7 — Compact (always run, idempotent, ~1s)

```
python scripts/compact_loop_state.py --apply
```

Archives stale LEDGER rows (keeps last 30 hot), strips old "ACCEPTED roundNNN" annotations, moves cold round_NNN diagram dirs to `_archive/`, deletes branches of killed rounds older than 7 rounds, prunes MLflow artifacts > 30 days, cleans `.tmp/`.

## Halt conditions (no commit, but **still re-arm** via Phase A safety)

- Wall-clock budget exceeded (daemon kills the job).
- Pytest red after one fix attempt.
- CRITIC veto with REQUEST_CHANGES exceeding scope.
- Merge conflict on main.
- `data/raw_data/klines_1m.parquet` missing → run HOUSEKEEPER round.

## Phase H — HOUSEKEEPER (when BACKLOG empty or weekly)

1. `python scripts/compact_loop_state.py --apply`
2. `pytest --cov` — surface coverage gaps.
3. Identify stale `src/` files (no test/notebook references); send to KILL_LIST.
4. Audit feature importances over last 5 accepts; propose pruning of consistently-near-zero features.
5. Surface drift between `docs/MINIMAL_PROJECT_SPEC_v2.md` and `src/utils.py`.
6. Open one consolidated PR. Halt loop with fireAt=NOW+1h heartbeat.

## Outputs every round must produce

- One line in `LEDGER.md`.
- One MLflow run with metrics + plots tagged round_id.
- `agent/round-NNN-*` branch (kept ≥7 rounds even if killed).
- Regenerated `REPORT_LATEST.md`.
- `RESEARCH/.loop_heartbeat.json` updated by the daemon.
