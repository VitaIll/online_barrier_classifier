# Round Playbook (v3 — in-session, single-master)

Each iteration of the autonomous loop executes one round in this Claude Code
session. No daemon. No branch creation. Progress is visible in the UI as
the agent works. See [`LOOP_DISCIPLINE.md`](LOOP_DISCIPLINE.md) for the
governing rules.

## Phase 0 — Read state

1. `RESEARCH/CONSTITUTION.md` (invariants).
2. `RESEARCH/LOOP_DISCIPLINE.md` (per-iter structure).
3. Top of `RESEARCH/BACKLOG.md` (next non-blocked, non-`in_progress` item).
4. Last 10 lines of `RESEARCH/LEDGER.md`.
5. Recent `RESEARCH/KILL_LIST.md`.
6. `RESEARCH/AGENTS.md` (sub-agent contracts).
7. `python scripts/run_round.py --preflight` — halt round on red.

## Phase 1 — Pick + plan

1. Pick the top non-blocked, non-`in_progress` BACKLOG item. If empty, run HOUSEKEEPER (Phase H).
2. Mark backlog item `in_progress` (in BACKLOG.md).
3. Determine `next_id = $(python scripts/run_round.py --next-round-id)`.

**No `git checkout -b`.** All work happens directly on `master`.

## Phase 2 — Research (parallel)

Spawn LITERATURE-SCOUT + CODE-SCOUT + THEORIST in a **single message**, multiple Agent tool uses. LITERATURE-SCOUT must local-first via `RESEARCH/literature/INDEX.md`.

Synthesize HYPOTHESIS.md:
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

Both must return 0. The CRITIC checklist enforces causality, leakage, plot-present, ledger-entry-added.

## Phase 4 — Train + Eval

1. `with training_helpers.log_training_run(round_id, hypothesis_id, fast_mode, ...) as run:`
2. Tag run: round_id, hypothesis_id, fast_mode, data_start, data_end, git_sha.
3. Log metrics from `compute_all_metrics`, `expected_calibration_error`, `calibration_by_regime`. UQ rounds add `uq_quality_summary`. Backtest rounds add deflated Sharpe + AURC.
4. Save ≥1 diagnostic plot under `RESEARCH/diagrams/round_NNN/`. **Read the saved PNG back via the Read tool to verify legibility before committing.**
5. **Primary metric is Brier-Skill-Score** = 1 - Brier_model / Brier_base_rate_predictor. Raw Brier is misleading when base rates differ across rounds.

## Gate 2 — CRITIC

Spawn CRITIC sub-agent per `RESEARCH/AGENTS.md`. CRITIC:
1. Runs `python scripts/critic_check.py --json` and parses verdict.
2. Verifies effect size > 1.5 × seed-noise band.
3. Applies multiple-comparisons correction if N_trials in this hypothesis area > 5.
4. Returns APPROVE / REQUEST_CHANGES / VETO.

## Phase 5 — Commit (or kill)

**APPROVE** — commit on master and tag:
```
agent/round-NNN: <summary> (H-xxx) [accept]

Hypothesis (H-NNN): <one-liner>
Mechanism: <one-liner>
FAST_MODE BSS Δ: <signed value> (vs paired baseline)
Accept-gate result: <if accept>
MLflow run: <run_id>
Tests: <count> passed; CRITIC: APPROVE

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```
Then `git tag round-NNN-accepted HEAD`.

**REQUEST_CHANGES** (one iteration attempt; if still red) or **VETO** —
do NOT commit. Append `[round NNN] H-NNN: <reason>` to `KILL_LIST.md`.
Reset working tree if needed (`git checkout -- .`) so master stays clean.

**iterate** outcome (re-queue with refinements without commit) — update BACKLOG status, add a LEDGER row with verdict=`iterate`, no commit.

## Phase 6 — Update state

1. Append one line to `LEDGER.md` (pipe-separated, no prose).
2. Update `BACKLOG.md`: status of the picked item; add any new hypotheses spawned.
3. Regenerate `REPORT_LATEST.md`: status, last 5 LEDGER lines, top 5 BACKLOG.

## Phase 7 — Compact (always run, idempotent, ~1s)

```
python scripts/compact_loop_state.py --apply
```

Archives stale LEDGER rows (keeps last 30 hot), strips old "ACCEPTED roundNNN" annotations, moves cold round_NNN diagram dirs to `_archive/`, prunes MLflow artifacts > 30 days, cleans `.tmp/`.

## Phase 8 — Concise status report

One short paragraph to the user: what was done, what was learned, what's next. No headers, no waffle. End with an explicit verdict (accept/iterate/kill).

## Phase 9 — Schedule next wakeup (under /loop dynamic mode only)

```
ScheduleWakeup(
    delaySeconds=270,    # 270s if next iter follows immediately (cache stays warm)
                         # 1200-1800s if waiting on batch / giving context room
    prompt="<<autonomous-loop-dynamic>>",   # or repeat the user's /loop prompt
    reason="continuing autonomous research loop — next: H-NNN",
)
```

Outside `/loop` dynamic mode: skip Phase 9. The user re-invokes the next round.

## Halt conditions (no commit, NO wakeup scheduled)

- BACKLOG empty → run HOUSEKEEPER (Phase H), then halt.
- Three consecutive killed rounds → halt; loop is wandering.
- Pytest red after one fix attempt → halt; needs human triage.
- Wall-clock > 30 min on a single round → halt that round, summarize partial progress.

## Phase H — HOUSEKEEPER (when BACKLOG empty or weekly)

1. `python scripts/compact_loop_state.py --apply`
2. `pytest --cov` — surface coverage gaps.
3. Identify stale `src/` files (no test/notebook references); send to KILL_LIST.
4. Audit feature importances over last 5 accepts; propose pruning of consistently-near-zero features.
5. Surface drift between `docs/MINIMAL_PROJECT_SPEC_v2.md` and `src/utils.py`.
6. Commit consolidated housekeeping under `housekeeper:` scope. Halt loop.

## Outputs every round must produce

- One line in `LEDGER.md`.
- One MLflow run with metrics + plots tagged round_id.
- ≥1 diagnostic plot under `RESEARCH/diagrams/round_NNN/`.
- Regenerated `REPORT_LATEST.md`.
- For accepted rounds: one commit on master + a `round-NNN-accepted` tag.
