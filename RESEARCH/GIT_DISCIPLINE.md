# Git & Merge Discipline (Loop Rulebook)

The autonomous loop produces work in rounds. This file defines how round work
enters `master`. The rule is simple:

> **Passes gates → fast-forward to master → delete branch.**
> **Fails gates → log to KILL_LIST → delete branch.**
> No PRs, no drafts, no long-lived branches, no divergence.

## 1. Branch lifecycle

| Phase | Action |
|------|--------|
| Open | At round start: `git checkout master && git checkout -b agent/round-NNN-<slug>`. NNN = zero-padded round index from heartbeat. <slug> = lowercase-hyphen of the BACKLOG hypothesis (e.g., `h005-first-pnl`). |
| Work | Round agent commits on the branch only. Never on master directly while the loop is running. |
| Decide | CRITIC runs `python scripts/critic_check.py --json`. Exit 0 + `must_passed=true` → APPROVE. Otherwise REQUEST_CHANGES or (catastrophic) VETO. |
| Merge | APPROVE → `python scripts/merge_round.py merge --hypothesis H-NNN`. Fast-forwards, tags, smoke-tests, deletes branch, pushes (if origin exists). |
| Kill | REQUEST_CHANGES after one iteration attempt, or VETO → `python scripts/merge_round.py kill --hypothesis H-NNN --reason "<one-line>"`. Deletes branch, appends `KILL_LIST.md`. |

**Invariants:**
- Branch lifetime ≤ 1 round (≤ 30 min wall-clock = daemon timeout).
- `master` is always green: every commit on master corresponds to a round
  whose CRITIC `must_passed=true` AND whose post-merge smoke pytest is green.
- Linear history: only fast-forward merges. No `--no-ff`, no merge commits,
  no rebase of master. If FF fails (master moved), kill the round and re-open.
- No PRs, no drafts. The branch either becomes master or disappears.
- One round = one logical change. If two unrelated edits land in one round,
  CRITIC REQUEST_CHANGES — split into two rounds.

## 2. Commit discipline

Commit message format on a round branch:

```
agent/round-NNN: <imperative summary> (H-xxx)

- bullet of what changed
- bullet of why

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Other allowed scopes (manual, only when the loop is halted):
- `loop-infra:` — daemon, prompts, observability, this file
- `housekeeper:` — HOUSEKEEPER round output
- `hotfix:` — emergency master fix; record in `LEDGER.md` with hypothesis=`hotfix`

Forbidden:
- `--no-verify`, `--force`, `--force-with-lease`, `--amend` of pushed commits
- Mixing infra and round work in one commit (the b3cf2ba mistake)
- Commit messages that don't name the round or hypothesis

## 3. Gate → merge contract

Gates are defined in `scripts/critic_check.py`. **APPROVE** requires all `must`
checks green:

- `branch_naming` — `agent/round-NNN-<slug>`
- `not_on_master` — current branch is not master
- `no_force_no_verify` — clean recent log
- `no_label_features_leakage` — forbidden columns absent from feature lists
- `causality_property_tests` — leakage-blocking tests green
- `pytest_full_suite` — full suite green
- `ledger_entry_added` — ≥1 new LEDGER row in diff

`should` checks (warnings only, do not block APPROVE):
- `diff_size_under_cap`
- `test_for_new_compute`
- `diagnostic_plot_added`

On APPROVE → `merge_round.py merge` will:
1. Verify FF possible (`git merge-base --is-ancestor master HEAD`).
2. `git checkout master && git merge --ff-only <branch>`.
3. Run `pytest -q tests/` on master. On failure, roll back via `ORIG_HEAD`
   and exit non-zero (round must be killed and re-opened from new master).
4. Tag: `round-NNN-accepted`.
5. Delete branch (`git branch -d`).
6. If `origin` exists, push `master` and the tag.

On REQUEST_CHANGES (persistent) or VETO → `merge_round.py kill`:
1. Append `KILL_LIST.md` with `[round-NNN] <UTC> sha=<head>: <reason>`.
2. `git checkout master && git branch -D <branch>`.
3. (Agent has already appended a LEDGER row with verdict=`iterate`/`kill`.)

## 4. DevOps loop hygiene

- **Heartbeat** (`RESEARCH/.loop_heartbeat.json`) is the source of truth for
  `round_index`. Daemon writes it; round agent reads it.
- **Master smoke**: every merge runs `pytest -q tests/` on master after the
  FF. Catches the case where the branch was green in isolation but master
  raced (concurrent commit, env drift).
- **Tags**: every accepted round gets `round-NNN-accepted`. Listed via
  `git tag --list 'round-*-accepted'`.
- **KILL_LIST**: every killed round appends one line. Read by `loop_status`.
- **Push**: post-merge push to `origin` is automatic if `origin` exists. Push
  rejection (race) → fatal log, master left as-is locally. Manual recovery.
- **No long-running branches**: `make loop-status` lists `agent/*` branches.
  Anything older than 1 round is broken — investigate or kill.

## 5. Anti-patterns (rejected on sight)

- Draft PRs that sit waiting for human review.
- Branches kept "for later" or "in case we want to come back."
- Mixed commits that bundle round work with infra changes.
- Cherry-picks across round branches (each round is independent).
- `git stash` / `git reset --soft` to "rescue" partial work between rounds —
  finish the round or kill it, never carry state across.
- Manual edits to master while the daemon is running.
- Commits without a hypothesis ID or round number when the loop is active.

## 6. Where this is enforced

| Rule | Enforced by |
|------|------------|
| Branch naming | `critic_check.py::check_branch_naming` |
| No master commits | `critic_check.py::check_no_main_modifications` |
| No force/no-verify | `critic_check.py::check_no_force_or_skip` |
| LEDGER entry present | `critic_check.py::check_ledger_entry` |
| FF-only merge | `merge_round.py merge` (refuses non-FF) |
| Master smoke | `merge_round.py merge` (post-merge pytest) |
| Branch deletion | `merge_round.py merge\|kill` |
| Tag on accept | `merge_round.py merge` |
| KILL_LIST append | `merge_round.py kill` |

If you change anything in this rulebook, update `critic_check.py` and/or
`merge_round.py` in the same commit. Discipline drift is the leading cause
of loop death by complexity.
