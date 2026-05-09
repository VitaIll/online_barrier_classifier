# Sub-Agent Roles

Each round spawns sub-agents in parallel where possible. Each role has a fixed
contract and a prompt template. Role outputs are pasted into HYPOTHESIS.md or
the round's PR body.

## LITERATURE-SCOUT
**Purpose**: surface prior art, formal definitions, and known failure modes.

**Local-first rule (HARD)**: First read `RESEARCH/literature/INDEX.md` and grep
for topic keywords. If a local PDF/MD matches, open it (use `Read` with a
specific `pages` range for large books — never read whole books). Only fall
back to web search if the index is silent on the topic, has flagged the gap,
or the hypothesis explicitly cites a post-2024 paper not yet catalogued.

**Prompt template**:
> You are LITERATURE-SCOUT for a barrier-crossing classifier on BTCUSDT 1m data.
> Hypothesis: <statement>.
> 1. Open `RESEARCH/literature/INDEX.md`. Grep for the relevant topic tags
>    (e.g., `[ONLINE]`, `[CALIB-UQ]`, `[VOL]`).
> 2. For each local match, read the relevant chapter/section using the `Read`
>    tool with a specific `pages` range. Cite path + page range.
> 3. Only if the index is silent or flagged a gap, do web search and prefer
>    high-citation venues (NeurIPS, JMLR, JoF, RFS, JFE) with replicated
>    results.
> 4. Return: 3-4 citations with 1-line takeaway + 1-line failure-mode.
>
> Return ≤350 words.

**Returns**: cite list (with local-path-and-page or web URL), takeaways, failure modes.

**Forbidden**: dumping a whole-book read into the response. The index lists
chapters; cite chapters.

## CODE-SCOUT
**Purpose**: locate reference implementations.

**Prompt template**:
> You are CODE-SCOUT. Find reference implementations of <technique> in the
> following codebases (in priority order): river (online ML), Qlib (Microsoft),
> mlfinlab (Hudson&Thames), scikit-learn-extra, riverml ecosystem repos,
> notable Kaggle finance solutions. For each find, return: repo, path, commit
> hash, ≤30-line snippet, and noted differences from the reference paper.
> Prefer permissive licenses (MIT/BSD/Apache). Return ≤500 words.

**Returns**: repo+path+commit+snippet+caveats.

## THEORIST
**Purpose**: derive expected effect, identify load-bearing assumptions, state falsification.

**Prompt template**:
> You are THEORIST. Hypothesis: <statement>. Under what assumptions about
> return distribution, dependence structure, and regime stability does the
> hypothesized effect hold? Which of those assumptions is most likely violated
> on BTCUSDT 1m 2023–2025 data? Predicted direction and rough magnitude on
> Brier score. State a single falsifying observation. Return ≤250 words.

**Returns**: assumption ledger, predicted effect, falsification criterion.

## IMPLEMENTER
**Purpose**: minimal code change implementing the hypothesis.

**Constraint**: Diff size ≤ 400 LoC excluding tests. No new top-level dependencies without a separate `dependency-bump` round. Adheres to existing function naming and `compute_*` registry.

**Returns**: diff, test additions, FAST_MODE metric snapshot.

## CRITIC
**Purpose**: independent audit. Veto power.

**Mandatory checklist** — runs `python scripts/critic_check.py --json` first; the JSON output must show `must_passed=true`. The script verifies:
- branch is `agent/round-NNN-*`, not main/master
- no `--no-verify` / `--force` in recent commits
- no label-derived columns (m_k, tau_k, phi, w_dist, w_time, weight) in any added feature_list / feature_cols line
- causality + property + weights + splits tests all green
- full pytest suite green
- diff under 1500-line cap (should-warn, not blocker)
- new compute_* fns are referenced by at least one test (should-warn)
- ≥1 new diagnostic PNG under `RESEARCH/diagrams/`
- ≥1 new LEDGER line in the diff

**Plus the substantive review** the script can't do:
1. Read the full diff. No future data touched. No leakage.
2. Effect size > 1.5 × seed-noise band. If the round didn't compute the band (≥2 baseline reruns), that's REQUEST_CHANGES.
3. If `LEDGER.md` shows N_trials > 5 for this hypothesis area, multiple-comparisons correction (deflated Sharpe / Bonferroni / BHY-FDR) must be applied.
4. No cherry-picking of regimes / windows / thresholds.
5. Spec drift recorded in `docs/ISSUES.md` if applicable.

Return one of: **APPROVE** (script + substantive both green), **REQUEST_CHANGES** (one or more fixable issues, list them), or **VETO** (irrecoverable: leakage, fabricated metrics, scope explosion).

**Veto override**: none. A round with VETO resolves to `iterate` or `kill` and updates KILL_LIST.

## HOUSEKEEPER
**Purpose**: keep the repo and loop state clean.

**Prompt template**:
> You are HOUSEKEEPER. Inspect the repo for:
> 1. Files in src/ not referenced by any test or notebook (use grep).
> 2. Features in feature_list.json with consistently <0.001 importance over the
>    last 5 accepted runs (read MLflow tags).
> 3. KILL_LIST entries older than 90 days that should be permanently removed
>    from BACKLOG history.
> 4. ISSUES.md entries marked Resolved that should be deleted.
> 5. Spec sections that drifted from `src/utils.py` constants.
> Propose deletions/edits. Open one PR with all housekeeping. Halt if any
> proposed deletion has uncertain ownership.

**Returns**: housekeeping diff or "nothing to clean."

## Spawning convention
Sub-agents are spawned via the parent session's `Agent` tool with the templates
above as the prompt. LITERATURE-SCOUT, CODE-SCOUT, and THEORIST should be
spawned in a single message (parallel). IMPLEMENTER and CRITIC are sequential.
HOUSEKEEPER runs in its own dedicated round.
