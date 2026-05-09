# Loop Discipline

The autonomous research loop runs as **a single Claude Code session** that
continues itself via `ScheduleWakeup` (or under `/loop` dynamic mode). No
PowerShell daemon, no headless `npx claude-code -p` subprocesses. Progress is
visible in the Claude Code UI as the agent works.

This is modeled on the working pattern from `~/projects/trading_research`
([loop_discipline memory](C:\Users\vitil\.claude\projects\C--Users-vitil-projects-trading-research\memory\loop_discipline.md)).

## Per-iteration structure

Each round MUST follow this five-step structure. Skipping a step → drift.

1. **Read state first.**
   - Top of `RESEARCH/BACKLOG.md` (next non-blocked, non-in_progress item).
   - Last 10 lines of `RESEARCH/LEDGER.md`.
   - Recent `RESEARCH/KILL_LIST.md`.
   - `RESEARCH/CONSTITUTION.md` (invariants).
   - Any memory files relevant to the picked hypothesis.

2. **Pick exactly one focused unit.** A single concrete task that ships
   something measurable. If two related changes look ready to land together,
   pick the smaller one — the second goes in the next iteration.

3. **Execute with verification.** Code runs; tests pass; if the round
   produces metrics, ≥1 diagnostic plot is saved AND read back via the Read
   tool to verify legibility. Spawn LITERATURE-SCOUT / CODE-SCOUT / THEORIST
   in parallel via the Agent tool when the hypothesis is non-trivial. CRITIC
   runs `python scripts/critic_check.py --json` before commit.

4. **Update memory.** Append one row to `LEDGER.md`. Update `BACKLOG.md`
   status of the picked item; add new hypotheses spawned. Regenerate
   `REPORT_LATEST.md`. If a hypothesis dies, append to `KILL_LIST.md`.

5. **Report concise status.** One short paragraph to the user: what was
   done, what was learned, what's next. No waffle, no headers, no emoji.

6. **Schedule next wakeup** (under `/loop` dynamic mode):
   - `270s` if the next iteration follows immediately (cache stays warm).
   - `1200–1800s` if waiting for a batch result or to give context room.
   - Use `ScheduleWakeup` with the same `/loop` prompt, or the literal
     `<<autonomous-loop-dynamic>>` sentinel for autonomous loops.
   - Outside `/loop` mode, skip the wakeup; the user re-invokes the next round.

## Git discipline (simplified — one branch only)

The trading_research project ran without git at all. This repo IS in git, so
we use a minimal version:

- **One branch: `master`.** Every accepted round commits directly to `master`.
  No `agent/round-*` branches, no draft PRs, no merge ceremony.
- **Tag every accepted round** with `round-NNN-accepted` for history.
- **CRITIC must pass before commit.** A failing CRITIC means: fix the issue
  (one attempt), then either commit or kill — never commit a known-bad round.
- **Killed rounds**: append `KILL_LIST.md` with `[round NNN] H-NNN: <reason>`
  and DO NOT commit the failing code. Leave master clean.
- **Commit message**: `agent/round-NNN: <summary> (H-xxx) [accept|hotfix]`,
  ends with `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
- **No `--no-verify`, no `--force`, no `--amend` of pushed commits.**

## Repo hygiene (per iteration)

From the trading_research playbook — applies here too:

- **Only SOTA in the repo.** Old experiment scripts get deleted, not renamed
  `_v2`. Superseded code is gone — `LEDGER.md` is the changelog.
- **No commented-out code, no dead branches, no `_old.py` fallbacks.**
- **Artifacts in `RESEARCH/diagrams/round_NNN/`**, archived after 7 rounds
  via `scripts/compact_loop_state.py`.
- **MEMORY index stays tight.** Per-topic memory files; `MEMORY.md` ≤ 200
  lines (truncates after that).

## Standing checks per round

- **Causality (HARD)**: no future data in features. Run `tests/test_causality.py`
  if it exists. Forbidden columns (`m_k`, `tau_k`, `phi`, `w_dist`, `w_time`,
  `weight`) absent from any new feature_list.
- **Effect-size band**: any accept claim must show effect > 1.5 × seed-noise
  band (≥2 baseline reruns). Otherwise REQUEST_CHANGES.
- **Multiple comparisons**: if `LEDGER.md` shows N_trials > 5 in the same
  hypothesis area, deflate (Bailey-Borwein PBO / Bonferroni / BHY-FDR).
- **Visual-first**: plot saved AND read back before commit (not just glanced
  at — `Read` the PNG via the Read tool).
- **Local literature first**: check `RESEARCH/literature/INDEX.md` before web
  search. Cite chapters, not whole books.

## Forbidden in any iteration

- Two unrelated changes in one round (split into two rounds).
- Skipping memory update because "next time."
- Verbose status reports that don't surface key findings.
- Re-implementing what's already in repo (read first).
- Starting next round before this one's memory update lands.
- Mixing infra changes with experiment work (commit infra under `loop-infra:`
  scope as its own commit).

## Halt conditions (the loop pauses, user resumes)

- BACKLOG empty → HOUSEKEEPER round, then halt with no rescheduled wakeup.
- Three consecutive killed rounds → halt; the loop is wandering.
- Full pytest red after one fix attempt → halt; needs human triage.
- User interrupts the session.

When halted, leave the repo in a clean state and emit a clear final status.
