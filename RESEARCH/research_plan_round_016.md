# Research plan — round 016 (research-planning round)

**Round 016 is a planning round, not a code-shipping round.** Output =
hypothesis cards in `BACKLOG.md` + this synthesis + `INDEX.md` updates +
one `loop-infra:` commit. No model trained, no notebook touched, no
`compute_*` added. CRITIC will kill any round that ships algorithm code
under planning-scope discipline.

This round is mandated by user override re-anchoring the loop on 8 axes
after rounds 010–015. Round 014 already produced the H-3xx series (13
cards + 15 refinements); this round adds the **19 new H-1xx cards** the
override names explicitly, refines the 14 existing cards it names, and
reconciles the round-014 plan with the new four-phase plan.

---

## 1. Phase 0 prework — confirmed

Files read in full before drafting any card (per LOOP_DISCIPLINE §1):

- `CONSTITUTION.md` §0 (two-layer; `p_online` is output, `p_offline` is input;
  sibling-style averaging/stacking is a CONSTITUTION violation per round-015),
  §I (causality), §IV (BSS + per-regime ECE + deflated Sharpe; ROC-AUC alone
  never accepts), §V.b (strategy realism).
- `LOOP_DISCIPLINE.md`, `ROUND_TEMPLATE.md`, `AGENTS.md` (sub-agent contracts).
- `BACKLOG.md` full (590 lines); `KILL_LIST.md` (rounds 010–013 superseded-by-015).
- `LEDGER.md` rounds 000–015. Critical state: round-008 closed per-regime
  coverage gap to ≤ 0.6pp on every (regime, α, predictor); round-009 val-chosen
  τ\* = 0.44 → test Sharpe = −0.089 (DSR=0 after 26-tau-grid deflation);
  round-015 corrective re-ran Phase A under two-layer architecture with 5 valid
  strategies — multi-strategy DSR=0 across all 5; CSCV PBO=0.000 cross-strategy;
  modal IS-best is `baseline_offline_tau` at Sharpe=−0.089. Two-layer system
  produces no tradable alpha at this label/cost/barrier combination.
- `literature/INDEX.md` (round-014 added 9 web-only refs under `[STATS-CI]`;
  this round adds cross-asset, BPV/SV, and Künsch block-bootstrap entries).
- `src/` surface preserved through round-015: `conformal.py` (LAC + Mondrian-ACI),
  `uncertainty.py` (virtual-ensemble + Cantelli rule), `backtest.py`
  (`simulate_inventory_aware`, `cscv_pbo`, `deflated_sharpe`, `walk_forward_backtest`,
  `bootstrap_no_skill_pvalue`).

---

## 2. Eight asks → 19 new H-IDs mapping

Per the override §4. Cards numbered in the H-1xx range to avoid collision with
round-014's H-3xx series — both series are kept; their relationship is in §3.

| # | Ask | New H-IDs (this round) | Refined existing |
|---|---|---|---|
| 1 | Invariant features | **H-130** derived-flow proxies · **H-131** jump+SV-asymmetric+vov · **H-132** cross-asset ETH/BTC | H-040, H-115, H-114→H-131 |
| 2 | σ_epistemic → decisions | **H-209** Cantelli-bound sized entries · **H-210** virtual-ensemble agreement gate | H-206, H-025 |
| 3 | Walk-forward retrain | **H-150** scheduled retrain · **H-151** drift-triggered (PageHinkley + KSWIN) · **H-152** post-retrain importance audit | — |
| 4 | Asymmetric weighting | **H-153** Huber-saturated tail-truncated weighting | H-102, H-020 |
| 5 | Strategy improvement | **H-160** first-touch label retrain · **H-161** meta-labeling (LdP §3.6) | H-024, H-025, H-022→H-161, H-023→H-160 |
| 6 | Online ensembling | **H-170** online stacking (streaming logistic meta-learner) · **H-171** online BMA weights | H-204, H-207, H-208 |
| 7 | Per-trade postmortem | **H-180** per-trade attribution dataframe · **H-181** SHAP-at-loss · **H-182** conditional Sharpe by feature regime | — |
| 8 | Bootstrap suite | **H-190** stationary block bootstrap · **H-191** per-metric bootstrap (ROC/PR/calibration) · **H-193** per-trade bootstrap with overlap embargo | H-113→H-192 |

**Total**: 19 new H-IDs (override said "18"; the explicit list contains 19),
14 refinements in place. New `[CROSS-ASSET]` tag in INDEX.md.

---

## 3. Reconciliation with round-014 plan

Round-014 generated H-301 (feature taxonomy synthesis) + H-311/H-312/H-313
(specific feature cards) + H-302 (σ-epistemic abstention) + H-303 (saturated
weighting) + H-304 (strategy 2×2 prioritisation) + H-305 (heterogeneous online
ensemble + Hedge) + H-306 (trade postmortem) + H-310-a/b (rolling retrain) +
H-320-a/b (per-metric bootstrap library).

The H-1xx cards added this round are **not duplicates**:

- **H-130** ≠ H-311. H-311 is the *single* `rolling Pearson(return, signed-volume)`
  feature; H-130 is the broader card scoping (a) the cheap path of derived-flow
  features from `taker_buy_base / taker_buy_quote` already in Binance kline,
  and (b) the optional later upgrade to actual top-of-book snapshots. H-311
  becomes a sub-feature of H-130's grid.
- **H-131** ⊃ H-114 + H-312. H-114 is plain BPV/semivar/vov (level); H-312 is
  the BPV ratio across windows. H-131 unifies them under the framing "jump
  detector + label-aligned semivariance asymmetry + vov", citing the
  one-sidedness of the upper-barrier label as the mechanical alignment for
  `SV_up − SV_down`.
- **H-132** is new (no equivalent in H-3xx). Cross-asset ETH/BTC is local-silent
  in INDEX; the `[CROSS-ASSET]` tag is added by this round.
- **H-150 / H-151 / H-152** refine round-014's H-310-a/b. H-310-a builds the
  rolling-retrain harness (engineering); H-150 specifies the scheduled cadence
  (`refit_cadence_bars = 1500` ≈ 20 days, per override; round-014's H-310-a
  proposed 2160 bars ≈ 30 days — the override tightens to 20 days). H-151 adds
  drift-triggered retraining (PageHinkley + KSWIN on running LAC score) which
  H-310-a/b did not address. H-152 is a HOUSEKEEPER-grade post-retrain
  importance audit. **Action**: keep H-310-a/b as the engineering harness card;
  reference H-150 / H-151 / H-152 from inside H-310-a's mechanism section as
  refinements.
- **H-153** refines H-303 (saturated weighting). H-303 specifies the
  `min(w_max, w_dist · w_time)` family with 27-cell ablation; H-153 narrows to
  the Huber-saturated tail-truncated form `w_i = min(w_max, (loss_i / median_loss)^β)`
  with `β = 1.0`, `w_max = 5`. Both queue side-by-side; H-153 runs first
  because it's strictly cheaper.
- **H-160 / H-161** lift H-022 / H-023 from `queued` to `active` (the override
  named these). H-160 is first-touch label retrain (was H-023); H-161 is
  meta-labeling (was H-022).
- **H-170 / H-171** are NEW. H-204 (River ARF/SRP/HAT comparison) feeds them
  with the K-expert shortlist. H-305 (heterogeneous ensemble + Hedge) is
  conceptually adjacent; H-170 is *online stacking* via a streaming logistic
  meta-learner over the four base online predictors' probabilities, distinct
  from H-305's regime-conditional Hedge weights. H-171 is online BMA — the
  fast-fail comparator to H-170.
- **H-180 / H-181 / H-182** decompose H-306 (trade postmortem). H-180 = per-trade
  attribution substrate; H-181 = SHAP-at-loss; H-182 = conditional-Sharpe by
  feature-regime cell. H-306's spawning rule absorbs H-180's output.
- **H-190 / H-191 / H-193** decompose H-320-a (per-metric bootstrap library).
  H-320-a is umbrella; H-190 = stationary-block subset; H-191 = rank-based
  subset; H-193 = trade-overlap-aware subset. Share H-320-a's accept-gate.
- **H-209 / H-210** are NEW. H-302 was abstention based on
  `p_offline − k·σ_epistemic`; H-209 is *Cantelli-bound sized* entries via the
  same σ but as size, not gate; H-210 is virtual-ensemble *disagreement* as
  an abstention rule (orthogonal axis from σ-magnitude).
- **H-192** = H-113 in BACKLOG (CSCV / PBO machinery). H-113 already accepted
  through round-012 → round-015 cleanup; H-192 is a refinement card naming
  the explicit S=16 partitions and the per-strategy reporting protocol.

---

## 4. Three sub-agent corrections folded in

LITERATURE-SCOUT and THEORIST sub-agents flagged three substantive corrections
that this round bakes into the cards. Failing to fold them in would leave
false citations or architectural errors in the BACKLOG.

### 4.1 Lekeufack §V-C miscitation in H-25 references

The override and prior BACKLOG entries cite Lekeufack et al. *Conformal
Decision Theory* §V-C as the source of confidence-margin sizing
`size = (p_online − q_t) / (1 − q_t) · max_size`. **This is wrong.**
Lekeufack §V-C is the Stock Trading Agent example with a buy/short/abstain
rule based on whether the prediction interval straddles zero
(Eq. 14, p. 6 — `D_λ_t = 1 if min(Ĉ_λ) > 0; −1 if max(Ĉ_λ) < 0; 0 otherwise`).
The `(p − q̂)/(1 − q̂)` confidence-margin formula comes from
Vovk-Gammerman-Shafer (2005) *Algorithmic Learning in a Random World*
Ch. 3 and is reused in Angelopoulos & Bates (2021) "A Gentle Introduction
to Conformal Prediction" §2.2 (arXiv 2107.07511). Inverse-set-width
sizing `1/|C(x)|` ties to Geifman & El-Yaniv (2017) selective classification
(already in INDEX). H-25's refinement and H-209 cards cite these correctly.

### 4.2 Multiscale Stochastic Volatility miscoverage

The book *Multiscale Stochastic Volatility for Equity, Interest Rate, and
Credit Derivatives* (Fouque-Papanicolaou-Sircar-Sølna 2011, local Downloads)
is cited in INDEX `[VOL]` for HAR-RV / multi-scale variance. Sub-agent
inspection: the book has **no BPV / realized-variance chapter**. Ch. 3
"Volatility Time Scales" pp. 86–118 is multi-scale OU mean-reversion under
risk-neutral pricing, NOT Barndorff-Nielsen jump-robust statistics. H-131's
references cannot rely on this book for BPV; web-fetch
**Barndorff-Nielsen & Shephard (2004)** "Power and bipower variation with
stochastic volatility and jumps", *J. Financial Econometrics* 2(1):1-37,
DOI `10.1093/jjfinec/nbh001` and **Patton & Sheppard (2015)** "Good volatility,
bad volatility: signed jumps and the persistence of volatility",
*Rev. Econ. Stat.* 97(3):683-697 are both added to `INDEX.md` `[VOL]` this
round.

### 4.3 Meta-labeling input set (architectural)

THEORIST flagged: H-161's secondary CatBoost on
`(features ⊕ p_offline ⊕ p_online)` predicting `1[trade_was_profitable_after_costs]`
**re-introduces the round-015 forbidden combiner pathology** if `p_offline`
is included. The architecture is hierarchical: `p_online` already conditions
on `p_offline`. Feeding both to the meta-model regresses the child onto
its parent's parent. H-161's mechanism MUST specify the meta-model input as
`(features ⊕ p_online)` only, with `p_offline` strictly absent. The card
text below pins this.

---

## 5. Anti-slop check — concrete numbers from rounds 015 (and 009 baseline)

Per the override, no card drafted before this table. Round-015 superseded
rounds 010–013; canonical numbers below.

| Round | Hypothesis | Knob | Test Sharpe | n_trades | p_boot | DSR | Notes |
|---|---|---|---|---|---|---|---|
| 009 | H-005b val-τ | τ\*=0.44 | **−0.089** | 3,427 | 0.000 | 0.000 (26-τ deflation) | sanity floor |
| 015 | Phase-A corrective (5 strategies) | various | **−0.089 / −11.59 / −11.87 / −5.67 / −3.96** | 3,427 / 289 / 288 / 2,372 / 3,448 | 0.000 / 1.000 / 0.735 / 0.505 / 1.000 | **0 for all 5** | n_trials=85, var_trial=8.36; CSCV PBO=0.000 cross-strategy; modal IS-best=`baseline_offline_tau` |

**Diagnosis (override §3, restated)**: three drift modes shaped the loop.
(a) Coverage-layer overfitting — eight of nine rounds were coverage refinement.
Mondrian-ACI ≤ 0.6pp gap is achieved (round-008); further pure-coverage rounds
are a noise tax. (b) Label/backtest mismatch — label is max-excursion;
backtest is symmetric triple-barrier. H-024 (no-stop) is the cheapest first
probe; H-160 (first-touch) is the deepest fix. (c) The conformal layer never
reached the strategy — `p_online` and Mondrian-ACI `q_t` exist, are
well-calibrated, never entered a backtest as anything other than
`p_online > τ`. H-25 sized + H-208 streaming gate close this gap.

**The model has discrimination** (offline `p_boot = 0.000` vs shuffled-signal
null at every τ — beats random-entry-at-same-rate). **It has no tradable edge**
at this label/cost/barrier-strategy combination after honest selection.

---

## 6. Recommended round ordering — four-phase plan

Per override §5. Halt conditions in LOOP_DISCIPLINE.md still apply (three
consecutive killed rounds → halt).

### Phase A — Honest baseline + strategy escalation (rounds 017–019)
- **Round 017**: H-024 unified no-stop variant (`c_stop=∞`) on the round-015
  harness. The cheapest test of label/backtest mismatch (5-min config flip).
  Decides whether sizing work has any base.
- **Round 018**: branches on round 017.
  - If 017 positive: H-25 Mondrian-ACI sized entries (variant a:
    `(p_online − q_t)/(1 − q_t)`, citation Vovk-Gammerman-Shafer 2005 / 
    Angelopoulos-Bates 2021).
  - If 017 negative: H-160 first-touch label retrain (full retrain;
    `M_horizon` matches backtest `c_stop`).
- **Round 019**: if 018 = sized: H-209 Cantelli-bound sized entries via
  virtual-ensemble σ. If 018 = first-touch: re-run round-015 unified harness
  with new labels.

### Phase B — Bootstrap suite + production refactor (rounds 020–023)
- **Round 020**: H-190 stationary block bootstrap library
  (`src/bootstrap.py`, Politis-Romano 1994 + Politis-White 2004 plug-in).
  Subsumes H-320-a's Sharpe/Sortino/CDaR scope.
- **Round 021**: H-191 per-metric bootstrap (ROC-AUC DeLong, PR-AUC stratified,
  per-bin Wilson) + H-193 per-trade bootstrap with overlap embargo
  (LdP §4.5 sequential bootstrap on uniqueness weights). Subsumes H-320-a's
  classification-metric scope.
- **Round 022**: H-192 CSCV / PBO refinement — explicit S=16 partitions,
  per-strategy histogram + scatter, `cscv_pbo` tag standardisation.
- **Round 023**: production refactor — promote `notebooks/*.ipynb` to `src/`
  modules with CLI entry points + pydantic config + deterministic re-runs.
  Notebooks become smoke-test harnesses; all rounds after Phase B run via
  `scripts/`. Discipline: business logic in notebooks fails CRITIC after
  Phase B.

### Phase C — Online ensembling + drift-triggered retraining (rounds 024–028)
- **Round 024**: H-204 base-learner comparison (River ARF vs SRP vs HAT vs
  online LR) under coverage-as-metric.
- **Round 025**: H-170 online stacking via streaming logistic meta-learner over
  the K-expert shortlist from H-204. The meta-learner sees only
  *base-probabilities*, NOT raw features (Wolpert 1992 / Ting & Witten 1999).
- **Round 026**: H-150 walk-forward retraining schedule (`refit_cadence_bars
  = 1500`, `refit_window` ∈ {expanding, sliding}). Removes the static-fit
  artefact in every PnL number. Pairs with H-310-a engineering work.
- **Round 027**: H-151 drift-triggered retraining via PageHinkley + KSWIN on
  the running LAC score. Pair with H-207 (ADWIN-triggered q_t soft-reset).
- **Round 028**: H-161 meta-labeling — secondary CatBoost on
  `(features ⊕ p_online)` predicting post-cost profitability.
  **Architectural pin**: `p_offline` strictly absent from meta-model input
  (round-015 contract).

### Phase D — Feature research + per-trade postmortem (rounds 029–034)
- **Round 029**: H-180 per-trade attribution dataframe (substrate). H-306
  spawning rule absorbs H-180's output.
- **Round 030**: H-181 SHAP-at-loss diagnosis on losing trades.
- **Rounds 031–034**: H-130 / H-131 / H-132 / H-040 / H-115 / H-153 in priority
  order from the SHAP-at-loss findings. H-301 taxonomy synthesis informs the
  cell ordering.

---

## 7. Round-016 self-audit (CRITIC checklist mirror)

- [x] Phase 0 read confirmed in §1; concrete numbers from round-015 in §5.
- [x] 19 new cards drafted in BACKLOG.md per ROUND_TEMPLATE.md format
  (Owner / Asks / Mechanism / Predicted effect / Falsification / References /
  Status / Cost / Dependencies — all 8+ fields).
- [x] No new card duplicates a queued card without explicit "supersedes" /
  "extends" language (§3 reconciliation table).
- [x] Every literature claim cites a local PDF + page range OR a newly catalogued
  INDEX entry with URL. New entries this round: Künsch (1989) JSTOR;
  Angelopoulos & Bates (2021) arXiv 2107.07511; Liu & Tsyvinski (2021)
  *RFS* 34(6); Alexander-Heck-Kaeck (2022) *AMF* 29(1); Barndorff-Nielsen &
  Shephard (2004) *JFE* 2(1); Patton & Sheppard (2015) *RES* 97(3); Bieganowski
  & Slepaczuk (2026) arXiv 2602.00776; mlfinlab GitHub; Hudson-Thames blog.
  New `[CROSS-ASSET]` tag.
- [x] CONSTITUTION V.b honored — every strategy card (H-024 refine, H-160,
  H-161, H-25 refine, H-209) names val-knob selection + inventory cap +
  costs + stop or explicit no-stop + max-DD circuit-breaker.
- [x] Multiple-comparisons correction named per sweep card: H-209 (Cantelli
  k-grid + Bailey-Borwein DSR), H-150 (per-refit MLflow tagging + DSR
  across refits), H-151 (paired Brier degradation post-trigger),
  H-160 (re-deflation under new label), H-170 (paired BSS over base set),
  H-190/191/193 (per-metric scheme).
- [x] Predicted effects above seed-noise band for every card claiming
  effect (H-130 +0.002–0.005 BSS vs σ_seed≈0.0008; H-131 +0.005–0.010 BSS;
  H-132 +0.001–0.004 BSS; H-209 ΔSharpe +0.05–0.20 vs σ≈0.02; H-160 ΔSharpe
  +0.10–0.30; H-161 hit-rate +2–5pp; H-170 BSS +0.01 over best single base).
- [x] §6 ends with concrete round ordering through round 034 with phase
  branching rules.
- [x] One commit, scope `loop-infra:`, no `round-NNN-accepted` tag.
- [x] No notebook touched, no `compute_*` added, no model trained.
- [x] No card averages or stacks `p_offline` and `p_online` (round-015
  contract). H-161 architectural pin in §4.3 makes this explicit.

The round is self-contained. CRITIC's substantive review must verify
(a) no algorithm code added, (b) no causality invariants touched,
(c) all 19 new cards conform to ROUND_TEMPLATE.md, (d) no card averages or
stacks `p_offline` and `p_online`. Any violation → kill round, KILL_LIST
entry, working tree reset.
