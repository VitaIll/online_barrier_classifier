# Research plan — round 014 (research-planning round)

**Round 014 is a planning round, not a code-shipping round.** Output =
hypothesis cards in `BACKLOG.md` + this synthesis + INDEX.md updates.
No model trained, no notebook touched, no `compute_*` added.

This document satisfies §A.6 of the planning mandate: ≤ 600 lines,
mapping table → genuine gaps → next-8-rounds ordering.

---

## 1. Eight asks → H-IDs mapping

| # | Ask | New H-IDs | Refined / addended H-IDs | Genuine gap? |
|---|---|---|---|---|
| 1 | Target-relevant invariant features | **H-301** taxonomy synthesis · **H-311** rolling Pearson(ret, signed-vol) · **H-312** BPV-ratio across windows · **H-313** DFA slope on signed-return series | H-040 (failure mode), H-041, H-042, H-044, H-111, H-112, H-114, H-115 (per-card failure mode + online-compatibility) | No (refines existing) |
| 2 | σ_epistemic → decisions | **H-302** σ-conditioned trade abstention | H-025 (addendum — sizing vs abstention are distinct) | No (touches H-206/H-025) |
| 3 | Rolling retraining → realistic backtest | **H-310-a** rolling-origin engineering harness · **H-310-b** rolling-retrain measurement | — | **YES** |
| 4 | Saturated asymmetric weighting | **H-303** saturated weighting card | H-020 (gated by H-303), H-021 (high-risk variant) | No (refines H-102) |
| 5 | Strategy improvement | **H-304** 2×2 prioritisation matrix + breakeven calc | H-022, H-023, H-024, H-025 (one-line refinements) | No (synthesis) |
| 6 | Online ensemble (heterogeneous + Hedge) | **H-305** heterogeneous online ensemble with regime-aware Hedge | H-204 (dependency note) | No (refines H-204) |
| 7 | Trade postmortem | **H-306** postmortem schema + round + spawning rule | — | **YES** |
| 8 | Per-metric bootstrap | **H-320-a** library · **H-320-b** re-render measurement | H-005c (collapses to regression check on top of H-320-a) | **YES** |

Total: **13 new H-IDs** (H-301, H-302, H-303, H-304, H-305, H-306, H-310-a, H-310-b, H-311, H-312, H-313, H-320-a, H-320-b), **15 inline refinements** (H-040, H-041, H-042, H-044, H-111, H-112, H-114, H-115, H-025, H-024, H-022, H-023, H-020, H-021, H-005c), **5 cards backing the 3 genuine gaps** (H-310-a, H-310-b, H-306, H-320-a, H-320-b).

INDEX additions: **9 newly catalogued web-only references** + 1 new tag `[STATS-CI]`.

---

## 2. The three genuine-gap cards (and why they outrank everything else)

### 2.1 Rolling retrain (H-310-a + H-310-b)

Every PnL number to date — round-001 (Sharpe=+1.50 at τ=0.30, demoted),
round-009 (Sharpe=−0.089 at val-τ=0.44), Phase A round-015 (DSR=0 across 5
valid strategies) — rests on a SINGLE static fit of `artifacts/offline_model/
model.cbm` from `train_fraction=0.6` / 2024-Jun and below. Test runs
2024-Oct → 2025-Dec, with BTCUSDT regime drift and a +120% drift on the
underlying. **Every reported Sharpe is therefore an upper bound on a
stale-model artefact.**

Production retrains on a rolling cadence; the loop has not. This is the
single most consequential operational gap on the road to a genuine
out-of-sample number. H-310-a builds the harness; H-310-b runs the
measurement; together they tell us whether the Phase A null result was
about (i) the model (stale), (ii) the strategy (label/cost/barriers
mismatch), or (iii) the features (incompatible regime).

The hard subtlety the cards pin: **the conformal calibration window must
roll with the model**. The Mondrian-ACI per-regime q_t at refit boundary
is warmed from the previous fold's terminal q (not re-warmed cold), and
the parkinson_var_rolling_mean_24 tercile cuts are frozen at the first
refit's train+cal slice. Under these conditions the streaming contract
is preserved across refits.

Priority **P5**, runs as Phase A round 4–5. Total wall ≈ 75 min.

### 2.2 Trade postmortem (H-306)

`src/backtest.py::Trade` already records (k_open, k_close, n_open, n_close,
exit_reason, pnl_log_net, p_signal, bars_held). No round has decomposed
this. Round-009 reported aggregate Sharpe and called the strategy
unviable; round-015 ran 5 valid two-layer strategies against the same harness.

The postmortem schema (7 slices, each with a numerical falsifier) tests
whether ANY conditional alpha exists — by exit-reason, hold-time,
p_signal decile, vol regime, σ_epistemic decile, intraday hour, day-of-
week. The aggregate falsifier is load-bearing: if NO slice shows
hit-rate deviation > 1.5σ AND NO slice's Sharpe is positive at p_boot
< 0.10, sized-strategy work (H-025, H-302) is killed wholesale.

The **spawning rule** (H-306 mandates it, LOOP_DISCIPLINE.md should
absorb it): every future strategy round must run this postmortem on its
trades and diff against the prior round. The diff's significant cells
drive the next iteration.

Priority **P4**, runs as Phase A round 6. Wall ≈ 30 min once H-320-a
ships.

### 2.3 Per-metric bootstrap (H-320-a + H-320-b)

Every "headline" number to date is a point estimate without a CI:
- Round-001/009 Sharpe — ad-hoc shuffled-signal null only.
- Round-005 calibration curves — no per-bin CI.
- Round-007/008 coverage gaps — no per-regime stratified CI.
- Round-012 multi-strategy DSR — point estimate without paired-strategy CI.

The scheme matrix is determined by the metric, not chosen — DeLong for
ROC-AUC, stratified bootstrap for PR-AUC, stationary block bootstrap with
Politis-White block length for Brier / Sharpe / Sortino / Calmar, Wilson
intervals for per-bin calibration, debiased ECE per Roelofs 2022, McNemar
for paired hit-rate.

H-320-a ships the library (`src/bootstrap.py` new module, one function
per metric, scheme name + tuned parameter returned for audit). H-320-b
re-renders the existing accepted numbers and flags any whose 95% CI
contains the null — the same downgrade treatment round-001's PSR=0.995
got in round-009.

Priority **P5** (-a), **P4** (-b). Library ~3 hours engineering;
measurement ~90 min.

---

## 3. Anti-slop check (concrete numbers from last 3 accepted rounds)

The mandate requires this in scratch before drafting a single hypothesis.
**Note (round-015 update)**: rounds 011 and 012 were SUPERSEDED-by-015 due
to architectural errors (sibling-style averaging/stacking of `p_offline` and
`p_online`; Mondrian-ACI driven by `p_offline` instead of `p_online`). Their
numbers are no longer canonical. The canonical Phase-A numbers are now
round-015's, recorded below.

| Round | Hypothesis | Knob | Test Sharpe | n_trades | p_boot | DSR | Notes |
|---|---|---|---|---|---|---|---|
| 009 | H-005b (val-τ) | τ\*=0.44 | **−0.089** | 3,427 | 0.000 | 0.000 (26-τ) | Round-001 PSR=0.995 demoted |
| 015 | Phase-A corrective (5 valid strategies) | τ=0.44 / 0.52 / 0.52 / k=20 / r=0.123 | **−0.089 / −11.59 / −11.87 / −5.67 / −3.96** | 3,427 / 289 / 288 / 2,372 / 3,448 | 0.000 / 1.000 / 0.735 / 0.505 / 1.000 | **0 for all 5** | n_trials=85, var_trial=8.36; modal IS-best across 12,870 CSCV combinations = `baseline_offline_tau` |

**Canonical "gap to close" (post-Phase-A round 015, two-layer architecture)**:
no strategy among the 5 legitimate ones survives the deflated bar. The
offline layer alone is the modal IS-best of the 5 (cross-strategy CSCV
PBO=0.000) — adding the online layer or the conformal gate strictly destroys
value vs offline-alone (`baseline_online_tau` = −11.59, `conformal_gate_tau`
= −11.87, `mondrian_aci_size` = −5.67, all worse than `baseline_offline_tau`
at −0.089). The two-layer system as currently fit does NOT produce tradable
alpha. The model has *discrimination* (offline p_boot=0.000) but no
*tradable edge*.

This is consistent with CONSTITUTION I — the online layer trades ranking for
regime-conditional coverage by design. Round-008 already proved coverage
(Mondrian-ACI per-regime gap ≤ 0.6pp at α∈{0.05, 0.10, 0.20}; α=0.20 low-vol
p_online gap −7.9pp → +0.04pp). Round-015 honestly reports that this
calibration discipline does not translate into trading edge under the current
label / cost / barrier-strategy combination.

Three axes to attack:
- Strategy: align label↔strategy via no-stop (H-024) or first-touch (H-023).
- Features: pull in queued sibling-imported groups (H-111/112/114/115) plus the new H-311/312/313.
- Process: rolling retrain (H-310-a/b) to remove the static-model artefact.

**Forbidden under round-015 contract**: any strategy that averages or stacks
`(p_offline, p_online)`. The architecture is hierarchical, not parallel.

---

## 4. Recommended round ordering (next 8 accepted rounds)

This ordering respects:
- CONSTITUTION V priority discipline (P5 before P4 before P3).
- Card dependencies (H-302 needs H-108 — done; H-310-b needs H-310-a;
  H-320-b needs H-320-a; H-303 needs H-103 → H-102; H-305 needs H-204).
- The strategy 2×2 (H-304) which forbids new strategy variants until
  H-024 closes the no-stop question.
- The trade-postmortem spawning rule — once H-306 ships, every strategy
  round runs the postmortem.

| # | Round | H-ID | Tier | Wall | Why this slot |
|---|---|---|---|---|---|
| 1 | 015 | **H-024** (Phase A round 4 — no-stop unified) | strategy | 5 min | Cheapest test of label/backtest mismatch; H-304 ranks first. Decides whether sizing work has any base. |
| 2 | 016 | **H-320-a** (per-metric bootstrap library) | infra | 3 h | Unblocks H-310-b, H-306, H-005c, H-320-b. Library is foundational. |
| 3 | 017 | **H-320-b** (re-render existing numbers with CIs) | infra | 90 min | Audits accepted results; may downgrade some to "exploratory" — must happen before Phase B refactor sees them. |
| 4 | 018 | **H-310-a** (rolling-retrain harness) | infra | 3 h eng + 45 min | Removes the static-model artefact from every PnL number. |
| 5 | 019 | **H-310-b** (rolling-retrain Phase A re-run) | strategy | 30 min | Decides whether stale-model was binding constraint. Likely the highest-ΔSharpe round of the next quarter. |
| 6 | 020 | **H-306** (trade postmortem schema + first run) | infra | 3 h eng + 30 min | Spawning rule for every subsequent strategy round; closes the "no exploitable conditional alpha" question. |
| 7 | 021 | **H-302** (σ-epistemic abstention) | strategy | 2 h | Distinct from H-025; runs once H-306 has ruled out σ-decile uninformativeness. |
| 8 | 022 | **H-305** (heterogeneous online ensemble + Hedge) | online | 5 min wall + 3 h eng | Per-regime Brier gain on the streaming layer; pre-requisite of any future ensemble-conformal work. Pair with H-204 results. |

**Outside this 8-round window** (queued for rounds 023+):
- H-303 (saturated weighting) — requires H-103 → H-102 first; runs as a
  separate weighting-axis arc.
- H-301 (feature taxonomy synthesis) — synthesis card; one full pass
  after H-306 has shown which slices need feature work.
- H-311 / H-312 / H-313 — feature cards run only after H-301 picks a
  starting cell; expected ΔBSS sequencing in H-301's table is the gate.
- H-023 (first-touch labels) — only if H-024 (no-stop) does NOT deliver
  ΔSharpe > +0.5 with p_boot < 0.05.

---

## 5. Round-014 self-audit (CRITIC checklist mirror)

- [x] Phase 0 done — concrete numbers from rounds 009 / 011 / 012 written
  in §3 above.
- [x] Every new card has all 8 fields per BACKLOG idiom (Owner, Asks,
  Mechanism, Predicted effect, Falsification, References, Status, Cost,
  Dependencies).
- [x] No new card duplicates a queued H without explicit refinement
  language (see Ask 1 addenda block under H-115; Ask 2 H-025 addendum;
  Ask 5 H-024/H-022/H-023 addenda).
- [x] Every literature claim cites a local PDF + page range OR is a
  newly catalogued INDEX entry with URL (9 web-only references added in
  round 014).
- [x] The 3 genuine-gap cards are written: H-310-a, H-310-b (rolling
  retrain ×2), H-306 (postmortem), H-320-a, H-320-b (bootstrap ×2).
- [x] CONSTITUTION V.b strategy-realism honored: every strategy card
  (H-302, H-024 refinement, H-304) names val-knob selection,
  inventory cap (preserved), cost, stop or explicit no-stop, max-DD
  circuit-breaker (inherited from `src/backtest.py`).
- [x] Multiple-comparisons correction named for every sweep card:
  H-302 (130-trial Bailey-Borwein DSR + CSCV), H-303 (Bonferroni at
  α=0.05/27 + CSCV), H-310-b (paired block-bootstrap CI on ΔSharpe),
  H-320-a/b (per-metric scheme).
- [x] Predicted effects above seed-noise band for every card claiming
  effect: H-302 (≥+0.05 Sharpe vs σ_seed≈0.02), H-303 (≥+0.005 BSS vs
  σ≈0.003), H-305 (per-regime Brier Δ vs σ≈0.001), H-310-b (paired
  block-bootstrap p_boot < 0.10 threshold), H-311/312/313 (≥+0.001 to
  +0.007 BSS vs σ≈0.0008).
- [x] §4 ends with concrete next-8-rounds ordering with wall-clock
  estimates.
- [x] One commit, scope `loop-infra:`, no `round-NNN-accepted` tag.
- [x] No notebook touched, no `compute_*` added, no model trained.

The round is self-contained. CRITIC's substantive review — read the BACKLOG diff and verify no causality / leakage / spec invariant was touched.
