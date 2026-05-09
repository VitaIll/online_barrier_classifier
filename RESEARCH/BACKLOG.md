# Hypothesis Backlog

The autonomous loop targets `online_barrier_classifier`. The architecture is:

```
offline CatBoost  →  p_offline
                     +
              streaming conformal calibration layer  →  P(y=1 | x_k)
                                                          (conditional coverage)
```

The "online stage" (currently River `ARFClassifier`, fed `selected_features + p_offline`) is a **streaming conformal calibration / coverage layer**, not a separate classifier. The differentiator vs the sibling `barrier_classifier` (offline-only) is exactly this layer, and most "online improvements" are framed as conformal-coverage improvements (ACI, Mondrian-ACI, locally-weighted conformal, etc.).

User asks 2 (online) and 6 (UQ) collapse onto one axis: **improve the streaming conformal coverage layer.**

Status legend: `queued` / `in_progress` / `iterate` / `blocked`. Priority P1–P5 = expected α-on-user-goal / cost.

Top of file = highest priority.

---

## Tier 0 — Bootstrap to operational state (after these, the loop is fully self-driving)

### H-101 [P5] Build label + split utilities matching this project's actual flow
- **Owner**: IMPLEMENTER
- **Asks**: ask 5 (engineering hygiene)
- **Mechanism**: extract the label construction (`y_k = 1[ ln(H_{k+1}/C_k) ≥ α ]` with α = 90%-train-quantile) and the chronological split (`train_fraction=0.6`, `val_fraction (of train)=0.2`) from `notebooks/feature_build.ipynb` and `notebooks/offline_train.ipynb` into `src/utils.py` as reusable callables. **Important**: this project has NO embargo, NO walk-forward CV in production — don't blindly port the sibling's utilities, only the parts that match this project's contract. Re-enable `tests/_pending/test_causality.py` and `test_splits.py` after adapting the tests to the actual semantics (no embargo expectations).
- **Falsification**: round-trip — the new utilities reproduce the existing `dataset.parquet` labels and the existing `train/val/test` splits exactly when called with the current config.
- **Status**: ACCEPTED round-004 — `src/utils.py` adds `compute_log_excursion`, `calibrate_alpha`, `construct_labels`, `chronological_split`, `chronological_split_indices`. 21 tests in `tests/test_label_split_utils.py` pass; round-trip on the persisted `bars_20m_features.parquet` (n=78,714) matches notebook cell-3 boundaries exactly (train_end=37,783; val_end=47,228). Parked sibling tests `test_causality.py`/`test_splits.py` retired — semantics didn't match (sibling had embargo + walk-forward; this project has neither). Diagnostic plot `RESEARCH/diagrams/round_004/split_window_visualization.png` shows train/val/test windows on the date axis with per-window positive rates (train p+=0.097, val p+=0.113, test p+=0.097). Unblocks H-105 (NSGA-II HPO needs these splits).
- **Cost**: medium (~45 min, requires careful adaptation)

### H-102 [P4] Add sample-weighting utility *adapted* from sibling, with this project's primary metric in mind
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 3 (risk-aware weighting — this project currently has none)
- **Mechanism**: import `compute_barrier_distance_weight` and `compute_time_discount_weight` from sibling (already known correct), but design the integration carefully: the offline CatBoost currently uses `auto_class_weights="SqrtBalanced"` which is what causes the over-prediction problem the online layer is correcting. Adding sample weights ON TOP of SqrtBalanced may double-count. Test paths: (a) keep SqrtBalanced + add only time-discount; (b) drop SqrtBalanced, use barrier-distance weighting only; (c) drop SqrtBalanced, use both. Compare offline calibration outcomes.
- **Falsification**: at least one path produces an offline model whose mean predicted p is closer to the base rate (0.097) than the current 0.205.
- **Status**: queued
- **Cost**: medium

### H-103 [P4] Adopt undef-flag pattern (Section I.4)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5
- **Mechanism**: this project's current feature pipeline silently fills NaNs inside `safe_divide`-style helpers; the sibling's `undef__*` flag-as-input pattern is a strict improvement (preserves the missingness signal). Adapt `feature_pipeline.py` to emit per-undefined-condition flags alongside the value. Re-enable `tests/_pending/test_features.py` once the pattern is in place.
- **Status**: queued
- **Cost**: medium

### H-104 [P3] Investigate adding an explicit embargo between val and test (sibling has it; this project doesn't)
- **Owner**: THEORIST + IMPLEMENTER
- **Asks**: ask 4 (rigorous evaluation)
- **Mechanism**: with M=20 and label horizon = 1 decision bar, val→test boundary leakage is bounded (the last val bar's label uses the first test bar's high — that's a single-bar contamination, not a horizon-spanning one). Quantify the actual leakage by comparing test metrics with embargo ∈ {0, 1, 5, 30, 60} bars. If embargo ≥ 1 changes test ROC by < 0.005, document and accept "embargo=0 is fine for this project". Otherwise add to CONSTITUTION I.2.
- **Status**: queued
- **Cost**: medium

### H-005 [P5] Run inventory-aware backtest on the existing offline+online stack
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 4 — **first PnL number**
- **Mechanism**: load `artifacts/offline_model/model.cbm` and the persisted online predictions in `artifacts/online_eval/predictions.parquet` (already on disk from a previous run). Feed `p_offline` and `p_final` columns into `src/backtest.py::simulate_inventory_aware` with M=20, φ=0.00411 (matched to label α), c_stop ≈ φ initially (symmetric — see strategy-realism CONSTITUTION V.b). Compare offline-only-decisions vs offline+online-decisions vs always-on null under realistic costs. Tag round `backtest` (30-min budget).
- **Falsification**: harness emits finite metrics on real predictions; null Sharpe is approximately 0 (with symmetric barriers) or negative (with c_stop < φ); offline+online Sharpe matches or exceeds offline-only.
- **Status**: ACCEPTED round-001 — finite metrics confirmed on real stream; p_offline τ=0.30 Sharpe=+1.50 PSR=0.995 with bootstrap p<0.001 vs shuffled-signal null. Online layer (p_final=p_online) does NOT add Sharpe at any τ tested — consistent with CONSTITUTION I (online optimizes coverage not ranking). Spawned H-005b + H-005c.
- **Cost**: 30 min (actual: ~7 min wall-clock)

### H-024 [P5] (spawned by round 009) Strategy/backtest re-evaluation post-H-005b
- **Owner**: THEORIST + IMPLEMENTER
- **Asks**: ask 4 (alpha discovery)
- **Mechanism**: round-009 showed that under the current label (`max excursion ≥ α`) + triple-barrier backtest (φ=c_stop=α, cost 1bp/side), the validation-chosen τ produces test Sharpe ≈ 0. Two label/strategy variations are worth measuring on the same harness: (a) **no-stop backtest** (`c_stop`=∞ or large) — matches the implicit "open long, take profit at +α, exit at close otherwise" strategy of the current upper-barrier label; this is the most direct fix to the label/backtest mismatch we surfaced earlier in conversation; (b) **first-touch label retraining** (LdP triple-barrier label `1[upper hit before lower]`) — full alignment of label and backtest, but invalidates accumulated positives. Run (a) first because cheap (no retrain): just re-run round 009 with `c_stop=10*α`, see whether the val-chosen τ now produces real test Sharpe.
- **Falsification**: under (a), val-chosen τ produces test Sharpe > 0.5 with p_boot < 0.05 → label/backtest mismatch was the binding constraint. Otherwise the model genuinely has no edge at this horizon.
- **Status**: KILLED (round-010 in_progress, never completed; see KILL_LIST). *Refinement (round-015)*: H-024 should now run against the round-015 harness (`scripts/phase_A_round_015.py`) under the corrected two-layer architecture, with `c_stop=∞` swept against all 5 valid strategies (`baseline_offline_tau`, `baseline_online_tau`, `conformal_gate_tau`, `mondrian_aci_size`, `null_random_at_rate`). The `simulate_inventory_aware_sized` harness already supports `c_stop=float("inf")` (tested). Multi-strategy DSR with n_trials=85 (5 strategies × their respective grids) decides whether the no-stop variant rescues *any* of the 5. The H-304 prioritisation card ranks this as the cheapest test of the binding-constraint hypothesis. **Forbidden**: any reintroduction of sibling-style `(p_offline, p_online)` combiners (round-015 contract).
- **Cost**: low (re-run round 009 with one parameter change). *Refinement*: under the unified Phase A harness it's a 2-line config change + one Phase-A re-run, ~5 minutes wall.

### H-304 [P5] Strategy-axis prioritisation — 2×2 decision matrix with breakeven calc (Ask 5 synthesis)
- **Owner**: THEORIST + IMPLEMENTER
- **Asks**: ask 4 — stop generating untested strategy ideas; close the strategy-axis decision matrix.
- **Mechanism**: at honest val-chosen τ, round-009/Phase A test Sharpe = -0.089 (`baseline_offline_tau`) and ALL 7 Phase-A strategies have multi-strategy DSR=0. Strategy improvement therefore = changing one of four axes: **label / cost / barriers / sizing**. The 2×2 axes-vs-fixedness matrix:

  | Axis | Already-queued H | Fixedness | Effort | Expected ΔSharpe | Priority |
  |---|---|---|---|---|---|
  | **Label** | H-023 (first-touch) | invalidates legacy ROC=0.813 baseline | high (full retrain) | +0.3 to +1.0 if label was the binding constraint | 2 |
  | **Cost** | (none — fixed by venue ≈ 1bp/side on Binance spot) | venue-fixed | n/a | n/a | n/a |
  | **Barriers** | H-024 (no-stop, c_stop=∞) | strategy-only | low (config flip) | +0.05 to +0.5 if SL was killing winners early | 1 (run first) |
  | **Sizing** | H-025 (Mondrian-ACI sized), H-022 (meta-labeling), H-302 (σ-abstention) | requires strategy with non-zero base | medium | +0.0 to +0.3 only if base strategy has signal | 3-5 (after 1, 2) |

  **Breakeven hit-rate calculation (load-bearing)**: for the no-stop variant at α = 0.0041113, round-trip cost c = 2·1bp = 0.0002 in log-return. Expected per-trade gross log return:
  `E[PnL_gross] = p_TP · α + (1 - p_TP) · E[X | timeout, model selects entry]`

  Three scenarios bracket the timeout payoff `ε := E[X | timeout, model selects]`:
  - **Skill in selection only** (model picks bullish setups; ε = 0): breakeven `p_TP ≥ c/α = 0.0002/0.00411 = **4.87%**`. The base rate is 9.7%; the bar is *trivially low*. The model would need to be only marginally informative to break even.
  - **No-skill on timeouts** (selection improves p_TP but conditional drift is unbiased; ε = -p·α/(1-p) = -0.000442 at p=0.097): breakeven `p_TP ≥ (c + 0.000442)/(α + 0.000442) = 0.000642/0.004552 = **14.1%**`.
  - **Worst case** (all timeouts hit −α equivalent; ε = -α): breakeven `p_TP ≥ (1 + c/α)/2 = (1 + 0.0487)/2 = **52.4%**`.

  Round-009 empirical hit_rate=0.525 at val-tau=0.44 under SYMMETRIC barriers (TP=+α, SL=−α, timeout mixed). The symmetric hit_rate fraction *includes* SL losses, so the equivalent **p_TP under symmetric** is approximately `n_tp / n_trades`, which round-015's `RESEARCH/diagrams/phase_A/round_015/headline.json` exposes per strategy. **Empirically, if `p_TP_symmetric` ≥ ~30% at val-τ=0.44**, the no-stop variant comfortably exceeds the 14.1% bar and should produce positive test Sharpe; if `p_TP_symmetric` < 14% the no-stop variant still loses. The ratio between symmetric `p_TP` and no-stop `p_TP` is ≈ 1 (both measure "did upper barrier hit within M=20 bars"); H-024's run will produce the headline number directly.

  **Order of operations**: (1) H-024 no-stop unified Phase A run — cheapest, decides whether label/backtest mismatch was binding. (2) If H-024 fails (multi-DSR < 0.95 still), H-023 first-touch label retraining — expensive but most aligned. (3) Sized variants H-302/H-025 after a positive base strategy emerges. **Forbidden until H-304 is closed**: any new strategy variant card. The 2×2 above is the gating doc.
- **Predicted effect**: zero (synthesis card; the value is the gating discipline).
- **Falsification**: any future strategy round (Phase A round 4+) is queued without referencing this card's prioritisation OR shows test Sharpe > 0.5 on an axis NOT in the 2×2 → matrix is incomplete; revisit.
- **References**: López de Prado *AFML* (2018), local Ch. 3 (triple-barrier) + Ch. 14 (backtest stats). Kaufman, *Trading Systems and Methods*, local INDEX [BACKTEST] reference — stop-loss vs no-stop trade quality discussion.
- **Status**: queued (this card supersedes ad-hoc strategy-card generation).
- **Cost**: zero (already done above; the card is the deliverable).

*Refinement (round-014, one-line addenda)*:
- **H-022** (meta-labeling) — gated by H-304 prioritisation: only run after H-024 establishes a positive base; meta-labeling of a no-edge strategy amplifies noise.
- **H-023** (first-touch labels) — H-304 ranks as priority-2; only run if H-024 fails to deliver ΔSharpe > +0.5.
- **H-025** (sized) — see H-025's existing addendum (above) on σ-sizing vs σ-abstention.

### H-025 [P4] (spawned by round 009) Sized entries via Mondrian-ACI confidence
- **Owner**: IMPLEMENTER
- **Asks**: ask 4 + ask 6 — using the conformal layer to drive economic decisions
- **Mechanism**: Mondrian-ACI (round 008) gives a per-regime adaptive q_t — effectively a per-regime confidence bar. Open long when p ≥ q_t (i.e., when the prediction set is the singleton {1}); size proportional to a measure of confidence (e.g., `max(0, p - q_t)` truncated). Test on the validation window, pick a sizing rule, evaluate on test. The hypothesis is that the binary p ≥ τ rule is throwing away information that q_t has already extracted.
- **Falsification**: sized variant beats binary at val-chosen-rule (under the same tau* → q_t mapping) by Sharpe > 0.3 on test, with p_boot < 0.10.
- **Status**: queued (depends on round 009 H-024 to first establish whether ANY binary edge exists; if H-024 says no, sizing is a tax on noise).
- **Cost**: medium

*Refinement (round-014/015, addendum from Ask 2)*: H-025 (sized via Mondrian-ACI on `p_online`) and H-302 (σ-epistemic *abstention* on top of `p_online`) are NOT the same hypothesis and must not be conflated. **H-025** *scales* position size by the per-regime conformal confidence on `p_online` (`size = clip(k · max(0, p_online − (1 − q_lo_α)), 0, 1)`) — every above-threshold opportunity is taken, only at differing magnitude. **H-302** *gates* trade entry by a CatBoost-virtual-ensemble σ_epistemic on top of the chosen base strategy — when σ is above a val-chosen threshold, the trade is skipped entirely (size = 0). The two interact: H-025 uses the conformal q_t (a coverage-derived signal that includes both aleatoric and epistemic noise); H-302 isolates the epistemic component. Round-015 (corrected two-layer architecture) showed `mondrian_aci_size` at k\*=20 producing test Sharpe=−5.67 with p_boot=0.505 (no edge after deflation; cross-strategy DSR=0). Sizing on a no-edge base actively destroys value. H-302's separate σ-abstention test is essential before any joint H-025+σ variant. Run H-302 first; only if it shows σ-conditional Sharpe lift does the H-025+H-302 combination become a separate card. **Forbidden**: any framing that averages or stacks `p_offline` and `p_online` (round-015 contract).

### H-005b [P5] (spawned by round 001) Validation-split τ_open selection for the H-005 sweep
- **Owner**: IMPLEMENTER
- **Asks**: ask 4 (rigorous evaluation per CONSTITUTION V.b)
- **Mechanism**: round 001 picked τ_open as a post-hoc grid because the existing `artifacts/online_eval/` only persists test predictions. Re-run `notebooks/offline_train.ipynb` + `notebooks/online_eval.ipynb` to additionally persist a `predictions_val.parquet` (the chronological val window that was already used for early-stopping). Sweep τ_open on val Sharpe + PR-AUC, pick winner per CONSTITUTION V.b, then evaluate at fixed τ on test. Tag `backtest` (30-min budget). Compare round-001's post-hoc τ=0.30 to the val-chosen τ — if they coincide, the round-001 result holds; if they diverge, the round-001 numbers are demoted to "exploratory only" in REPORT.
- **Falsification**: val-chosen τ for p_offline lands inside the round-001 post-hoc grid {0.10..0.50}; if the val-Sharpe-grid optimum is outside this range, the round-001 sweep was insufficiently wide.
- **Status**: ACCEPTED round-009 — val tau\* = 0.44 (inside the round-001 grid {0.10..0.50}, falsifier-1 PASSES). Test Sharpe at val-chosen tau\* = -0.089 (PSR=0.458, p_boot=0.000 vs shuffled-signal null, DSR=0.000 after 26-tau-grid deflation). Round-001's PSR=0.995 at τ=0.30 demoted to "exploratory only" in REPORT.md headline. Strategy still beats random-entry-at-same-rate (model has SOME information) but does not produce tradable alpha after honest selection. Also: built `scripts/build_report.py` and `RESEARCH/REPORT.md` (canonical living dashboard, trading section leads, regenerates from per-round headline.json each round).
- **Cost**: medium (notebook surgery + 1 retrain)

### H-005c [P3] (spawned by round 001) CSCV PBO deflation of the H-005 τ sweep
- **Owner**: IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 4 — overfitting check on the post-hoc τ grid
- **Mechanism**: H-113 lands the CSCV / PBO machinery (combinatorially symmetric CV PBO per Bailey et al. 2014). Once H-113 ships, apply it to the round-001 τ_sweep_metrics.csv to compute the PBO of the best post-hoc τ. PBO < 0.5 means the τ=0.30 winner generalizes; PBO ≥ 0.5 means the post-hoc grid was overfit to test. This is the deflation step that justifies promoting round-001 numbers to a "headline" status.
- **Status**: blocked on H-113. *Refinement (round-014/015)*: `cscv_pbo` landed in `src/backtest.py` (committed round-012, preserved through round-015 cleanup; architecture-agnostic). H-005c now collapses to "use the new helper to recompute the round-001 PBO", purely a regression check.
- **Cost**: low (post-H-113)

### H-310-a [P5] Rolling-origin retraining harness (GENUINE GAP — Ask 3)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 3 — train-once → rolling-retrain to remove the static-model artefact in every PnL number to date.
- **Mechanism**: today `artifacts/offline_model/model.cbm` is fit once on the 60% train slice and applied statically to val + test (round-001 / round-009 / Phase A all rest on this single fit). A production system retrains on a rolling cadence; without it every Sharpe number is an upper bound on a stale model. Engineering scope: refit-cadence config (proposal: `refit_cadence_bars = 2160` ≈ 30 days at M=20; configurable) and `refit_window` ∈ {`expanding`, `sliding`}; CONSTITUTION I.1 holds at every refit (data ≤ T only; no future leakage); the `selected_features.json` top-120 must be **re-derived per refit**, not frozen; ensemble of 3 seeds per refit (CONSTITUTION V); MLflow tag schema = one run per refit with `parent_run_id` linking to the rolling experiment. Artifact layout: `artifacts/offline_model/rolling/refit_NNNN/{model.cbm, model.{0..2}.cbm, selected_features.json, config_snapshot.json, metadata.json}` plus one consolidated `rolling/manifest.parquet` keyed by `(refit_id, t_start_ms, t_end_ms)` and the corresponding test-slice predictions parquet `rolling/predictions.parquet` so downstream rounds (H-310-b, H-005b extensions, Phase B refactor) can reuse without re-running. **Hard subtlety**: the conformal calibration window must roll with the model — `n_cal=9,446` of round-008 was the chronological val slice; under rolling retrain, the conformal layer's per-regime q is warmed at each refit from the previous fold's terminal q (not re-warmed from scratch), preserving the streaming contract while avoiding the per-refit cold start. Regime cuts on `parkinson_var_rolling_mean_24` are **fit on the very first refit's train+cal only and frozen** for all subsequent refits, to avoid drift in tercile boundaries leaking future state. A new module `src/rolling.py` exposes `RollingTrainer(config, schedule)` with a `.run() -> RollingArtifacts` method.
- **Predicted effect**: per-refit BSS Δ on the test slice ranges between +0.005 (small) and +0.02 (large) vs the static fit, with the gain concentrated in the high-vol tercile (where round-005 showed offline ECE 0.171 — most stale). Test Sharpe under rolling-retrain expected to shift by ≥ +0.05 with high variance across refits; confidence interval of the mean shift requires the per-metric bootstrap (H-320-a). Seed-noise band ≈ 0.003 BSS / 0.02 Sharpe (from Phase A's offline-only re-runs being bit-identical at the same seed).
- **Falsification**: rolling-retrain test Sharpe is **NOT** statistically distinguishable from static-train test Sharpe (block-bootstrap p_boot ≥ 0.10 on the paired difference, using H-320-a's stationary block bootstrap with Politis-White block length). If the falsifier triggers, drop the rolling-retrain hypothesis: the model's stale-fit drag is below the noise floor on this dataset.
- **References**: López de Prado, *AFML* (2018), local PDF `Downloads\Advances in Financial Machine Learning ... 2018 ... .pdf` Ch. 7 (purged CV / embargo) and Ch. 11 (backtesting). Sugiyama & Kawanabe, *Machine Learning in Non-Stationary Environments* (MIT 2012), local PDF `Downloads\(Adaptive Computation and Machine Learning series) ... 2012 .pdf` Ch. 1–3 (covariate-shift adaptation under refit). Hansen *Econometrics* (2022) Ch. 14 (rolling-origin evaluation).
- **Status**: queued (P5 — single highest-leverage operational gap on this loop's roadmap).
- **Cost**: medium-high (engineering ~3 hours; per-refit train ~2 minutes wall on 30-day windows × ~21 refits = ~45 min wall). Tag round `backtest` (30-min budget for the runner; the engineering is its own commit).
- **Dependencies**: must follow H-101 (chronological_split utility — accepted round-004) and H-102/H-103 (weighting / undef-flag) ideally land first so each refit uses the production weighting and feature pipeline. CSCV PBO across refits as a check is downstream (uses H-005c-style logic on the new manifest).

### H-310-b [P5] Rolling-retrain backtest measurement (GENUINE GAP — Ask 3, paired with H-310-a)
- **Owner**: IMPLEMENTER + CRITIC
- **Asks**: ask 3 — re-run round-009 (val-tau backtest) under rolling-retrain.
- **Mechanism**: load `artifacts/offline_model/rolling/predictions.parquet` (built by H-310-a); re-run the round-015 unified harness (`scripts/phase_A_round_015.py` adapted) producing the 5 valid two-layer-architecture strategies (`baseline_offline_tau`, `baseline_online_tau`, `conformal_gate_tau`, `mondrian_aci_size`, `null_random_at_rate`) under rolling-retrain. Diff against the static-train numbers in `RESEARCH/diagrams/phase_A/round_015/phase_A_table.csv` row-by-row. Report per-strategy ΔSharpe, ΔBSS, Δper-regime-ECE, paired bootstrap p (H-320-a stationary block bootstrap on per-bar net log returns, block length via Politis-White). Multi-strategy DSR (Bailey-LdP) with `n_trials = 5 strategies × their respective grids = 85` deflation (matching round-015 exactly), applied to the rolling-retrain run as a self-contained Phase-A reproduction.
- **Predicted effect**: ΔSharpe ∈ [-0.05, +0.20] for `baseline_offline_tau` vs static; ΔBSS ∈ [+0.005, +0.02]; the high-vol tercile's ΔECE expected to be the largest single gain. Round-015's null result (multi-strategy DSR=0 for all 5) might or might not flip — the falsifier specifies the threshold.
- **Falsification**: at least one Phase-A strategy under rolling-retrain has multi-strategy DSR > 0.95 AND CSCV PBO < 0.5 → rolling retrain rescues tradable alpha. Otherwise the model has no edge regardless of refit cadence; the binding constraint is label/strategy/feature, not staleness. If even ΔBSS < +0.003 (below seed-noise band) on every strategy, the gap was an artefact of the loop's framing.
- **References**: AFML Ch. 14 (deflated Sharpe under rolling). Politis & Romano (1994) JASA, *The Stationary Bootstrap* — for the paired CI on ΔSharpe (web URL `https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870` — INDEX gap; added by this round). Bailey & López de Prado (2014) deflated Sharpe — already cited.
- **Status**: blocked on H-310-a. Once unblocked, this becomes the most consequential measurement round of the next quarter.
- **Cost**: medium (~30 min wall once -a is done; Phase-A harness is fast).
- **Dependencies**: H-310-a (predictions); H-320-a (paired block-bootstrap CI).

### H-320-a [P5] Per-metric bootstrap CI library (GENUINE GAP — Ask 8)
- **Owner**: IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 8 — every "headline number" reported to date lacks an honest CI.
- **Mechanism**: a single new module `src/bootstrap.py` exposes one function per metric, each returning `(point_estimate, ci_lo, ci_hi, B, scheme_name, block_length_or_none)`. The scheme matrix is fixed by the metric's sampling distribution, not chosen — see card body in `RESEARCH/research_plan_round_014.md` §Ask 8 Matrix. Concretely:

  | Metric | Function | Scheme |
  |---|---|---|
  | ROC-AUC | `bootstrap_roc_auc` | DeLong (1988) closed-form variance via Sun-Xu (2014) O((n+m) log(n+m)) implementation; bootstrap fallback stratified by class label |
  | PR-AUC | `bootstrap_pr_auc` | Stratified bootstrap (positives + negatives separately), B≥1000 (Boyd-Eng-Page 2013) |
  | Brier | `bootstrap_brier` | Stationary block bootstrap (Politis-Romano 1994), block length via Politis-White (2004) plug-in |
  | Calibration curve (per-bin) | `bootstrap_calibration_curve` | Wilson score interval per bin for the binomial CI; outer bootstrap for bin-edge sensitivity if requested |
  | ECE | `bootstrap_ece` | Debiased ECE_sweep estimator (Roelofs et al. 2022) with stratified bootstrap on samples within predicted-prob bins; report both naïve and debiased CIs |
  | Sharpe (trade returns) | `bootstrap_sharpe` | Stationary block bootstrap on per-trade returns, B≥5000; Bailey-LdP probabilistic Sharpe as closed-form alternative |
  | Sortino / Calmar | `bootstrap_sortino`, `bootstrap_calmar` | Stationary block bootstrap; B≥5000 (López de Prado AFML Ch. 14 — no closed form) |
  | Hit rate | `bootstrap_hit_rate` | Wilson within-strategy; McNemar paired sign test for between-strategy comparison |
  | Cross-strategy SR | `cross_strategy_sr` | Already implemented: `cscv_pbo` + `deflated_sharpe` in `src/backtest.py` (architecture-agnostic; preserved through round-015 cleanup) |

  Tests pin (Hypothesis property tests where applicable): scheme reduces to plain bootstrap on IID synthetic; CI width shrinks at √n; DeLong matches the bootstrap CI on a large balanced sample; block-length plug-in returns sane numbers on a known AR(1) process; Wilson interval matches `statsmodels.stats.proportion.proportion_confint` to 1e-12; debiased ECE on a known calibrated synthetic returns ECE ≈ 0 within MC error. **Strict load-bearing assumption check**: each function returns the scheme name and any tuned parameter (block length, B); the caller can audit. No silent fallback to plain bootstrap if a scheme's assumption is violated — raise.
- **Predicted effect**: the CIs around every existing accepted result should mostly be wide enough to contain zero (consistent with Phase A's "no edge survives" finding); per-bin calibration CIs will surface bins where the visual reliability curve is statistically indistinguishable from diagonal — unblocking H-320-b's audit. No effect on point estimates; the deliverable is the CI itself.
- **Falsification**: if the ROC-AUC DeLong CI does NOT match the empirical bootstrap CI to within 1% width on a synthetic balanced 50k-sample experiment, the implementation is broken. If the block-length plug-in returns block_length > n/4 (Politis-White instability under non-stationarity), the function must raise rather than silently return a bad CI — pinned by test.
- **References**:
  - DeLong, DeLong, Clarke-Pearson (1988) *Biometrics* 44:837 — ROC-AUC closed-form variance. URL: `https://pubmed.ncbi.nlm.nih.gov/3203132/` (INDEX gap, added).
  - Sun & Xu (2014) *IEEE SPL* 21:1389 — fast DeLong O((n+m) log(n+m)). URL: `https://ieeexplore.ieee.org/document/6851192/` (INDEX gap).
  - Boyd, Eng, Page (2013) *ECML PKDD*, LNAI 8190:451 — stratified bootstrap PR-AUC. URL: `https://pages.cs.wisc.edu/~boyd/aucpr_final.pdf` (INDEX gap).
  - Politis & Romano (1994) *JASA* 89:1303 — stationary block bootstrap. URL: `https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870` (INDEX gap).
  - Politis & White (2004) *Econometric Reviews* 23:53 — block-length plug-in. URL: `https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf` (INDEX gap; correction: Patton-Politis-White 2009 same authors).
  - Niculescu-Mizil & Caruana (2005) *ICML* — calibration evaluation; per-bin Wilson is the standard external recommendation. URL: `https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf` (INDEX gap).
  - Roelofs et al. (2022) *AISTATS* — debiased ECE. URL: `https://proceedings.mlr.press/v151/roelofs22a.html` (INDEX gap).
  - López de Prado, *AFML* (2018) Ch. 14 — deflated Sharpe + bootstrap (already in INDEX).
- **Status**: queued. New tag `[STATS-CI]` proposed for INDEX — adds 6 references in one round.
- **Cost**: medium (~3 hours engineering; tests are the bulk of the work).
- **Dependencies**: none load-bearing. Unblocks H-310-b, H-005c (replaces ad-hoc CI), H-320-b.

### H-320-b [P4] Re-render existing accepted-round numbers with proper CIs (GENUINE GAP — Ask 8 paired)
- **Owner**: IMPLEMENTER + CRITIC
- **Asks**: ask 8 paired with -a — close the audit loop.
- **Mechanism**: re-render every metric in `RESEARCH/REPORT.md` with the corresponding `src/bootstrap.py` CI: round-005 calibration plot per regime (per-bin Wilson + outer bootstrap); round-001/round-009 Sharpe (stationary block bootstrap CI vs the existing point estimate); round-007/round-008 coverage gap (per-regime stratified bootstrap on the (predicted, observed) pair); round-015 multi-strategy DSR (recomputed with paired block-bootstrap on per-strategy per-bar returns). Diff against the bare numbers; flag any "accepted" claim whose 95% CI contains the null. The deliverable is one updated REPORT.md rendering pass + one `RESEARCH/diagrams/round_NNN/audit_diff.csv` with one row per audited metric: `(round_id, metric, point, ci_lo, ci_hi, scheme, contains_null)`.
- **Predicted effect**: round-008 Mondrian-ACI per-regime gap CIs likely tight (≤0.6pp gap is small enough that bootstrap CI is in the [0, 1pp] range — clean accept). Round-009 Sharpe CI almost certainly contains zero (point estimate −0.089, n=3,427 trades, expected stationary-block-bootstrap CI on Sharpe ≈ ±0.5 at this n). Round-005 per-regime ECE CIs in the high-vol tercile may be wide enough that the offline 0.171 vs online 0.016 is still a clear gap, but the absolute uncertainty becomes visible.
- **Falsification**: the audit reveals an "accepted" result whose 95% CI contains the null AND whose effect size was below the seed-noise band. Such a row triggers a REPORT downgrade ("exploratory only", same treatment as round-001's PSR=0.995 received in round-009).
- **References**: same as H-320-a; AFML Ch. 14; Bailey-LdP 2014 deflated Sharpe.
- **Status**: blocked on H-320-a.
- **Cost**: low-medium (~90 min once -a is done; mostly script-level wiring).
- **Dependencies**: H-320-a.

**Round ordering (post-bootstrap)**:
1. ~~**H-005**~~ — ACCEPTED round-001.
2. ~~**H-201**~~ — ACCEPTED round-002. Coverage baseline established; α=0.20 low-vol gap = -7.9pp is what H-202..H-208 must close.
3. ~~**H-108**~~ — ACCEPTED round-003. `src/ensemble.py` lands; unblocks H-206.
4. ~~**H-101**~~ — ACCEPTED round-004. Label + split utilities in `src/utils.py`; round-trip validated on persisted parquet; unblocks H-105.
5. ~~**H-106**~~ — ACCEPTED round-005. Calibration metrics in `src/utils.py`. Per-regime visual proves online's regime-flat ECE thesis empirically.
6. ~~**H-107**~~ — ACCEPTED round-006. Plot helpers in `src/plotting.py`; weight plots deferred to H-102.
7. ~~**H-202**~~ — ACCEPTED round-007. ACI in `src/conformal.py`; marginal coverage hits target within 1.5σ on real stream; per-regime gap motivates H-203.
8. ~~**H-203**~~ — ACCEPTED round-008. Mondrian-ACI collapses per-regime gap to ≤ 0.6pp on every (regime, α, predictor); beats batch Mondrian-LAC and closes the α=0.20 low-vol p_online gap (-7.9pp → +0.04pp).
9. ~~**H-005b**~~ — ACCEPTED round-009. Val-chosen tau* = 0.44 → test Sharpe -0.089. Round-001's PSR=0.995 was a post-hoc artifact. Plus: canonical living REPORT.md scaffolding (trading section leads).
5. **H-106** — regime-stratified calibration helpers (primary metric per CONSTITUTION IV).
6. **H-107** — plot helpers + threshold-analysis CSV (visual-first reporting parity with sibling).
7. **H-202** — Adaptive Conformal Inference on offline output.
8. **H-105** — NSGA-II HPO setup (after H-101 establishes split utilities).
9. **H-102** — sample weighting (carefully, given SqrtBalanced double-count concern).
10. **H-103** — undef-flag pattern.
11. **H-111 / H-112 / H-114 / H-115** — sibling-borne feature groups.
12. **H-203 / H-204 / H-205 / H-206 / H-207 / H-208** — online-stage refinements.

(H-201b, H-201c are P4/P3 follow-ups inserted after current ordering; pick when natural.)

---

## Tier 1 — Streaming conformal coverage layer (asks 2 + 6, the project's differentiator)

The online stage is a streaming conformal layer providing conditional coverage. These rounds explicitly improve coverage validity and tightness, NOT raw ranking.

### H-201 [P5] Baseline coverage diagnostic on the existing online ARF
- **Owner**: IMPLEMENTER + THEORIST
- **Mechanism**: treat the existing River `ARFClassifier` output as if it were a conformal predictor at varying confidence levels (set construction via `predict_proba_one`). Compute marginal empirical coverage AND per-regime coverage on the test split. Quantify *coverage gap* = 1 - α - empirical_coverage by regime. This becomes the baseline every online-stage hypothesis must beat.
- **Falsification**: report finite numbers; a > 5% absolute coverage gap in any regime is the gap to close.
- **Status**: ACCEPTED round-002 — chronological 30/70 cal/eval split on predictions.parquet with parkinson_var_rolling_mean_24 terciles. Finite numbers everywhere; constant-predictor sanity passes (α=0.05 cov=1.0000, α=0.20 cov=0.9181=1-eval_base_rate). Naive `p≥α` over-covers by 8-10pp at α=0.10 → ARF probs not directly usable as conformal thresholds. Mondrian LAC compresses α=0.10 gap to ±3.4pp; α=0.20 still has -7.9pp on low-vol/p_online and -5.8pp on low-vol/p_offline → THE gap H-202..H-208 must close. Spawned H-201b (finer-bin regime), H-201c (cal-fraction sensitivity).
- **Cost**: low (no model changes; just measurement)

### H-201b [P4] (spawned by round 002) Finer regime binning to expose what Mondrian-LAC misses at α=0.20
- **Owner**: IMPLEMENTER
- **Asks**: ask 2 (online coverage)
- **Mechanism**: round 002 used 3-bucket parkinson_var terciles. The persistent low-vol gap at α=0.20 (-7.9pp over-coverage) suggests the low-vol bucket contains a sub-mode that Mondrian-LAC's single q_hat for the whole tercile cannot adapt to. Re-run `scripts/round_002_coverage_baseline.py` with `N_TERCILES=5` (quintiles) and check whether the low-quintile gap at α=0.20 shrinks below ±5pp. If it does, the ARF coverage gap is fundamentally a *resolution* problem (more buckets fix it); if it doesn't, the gap is a probability-quality problem (Mondrian can't fix it; needs ACI / locally-weighted conformal — H-202..H-205).
- **Falsification**: low-quintile gap at α=0.20 stays > 5pp under quintile binning → resolution alone won't help; H-202 work justified.
- **Status**: queued
- **Cost**: low (script tweak + rerun)

### H-201c [P3] (spawned by round 002) Sensitivity of coverage gap to calibration-fraction
- **Owner**: IMPLEMENTER
- **Asks**: ask 2 + ask 4 (rigorous evaluation)
- **Mechanism**: round 002 used a fixed 30/70 chronological cal/eval split. Sweep `CAL_FRAC ∈ {0.20, 0.30, 0.40, 0.50}` and report how max |gap| (Mondrian, α=0.10) and max |gap| (Mondrian, α=0.20) vary. If the gap is monotone-decreasing in cal_frac, the round-002 gap was an n_cal-power problem; if invariant, it's a real conditional-miscalibration signal.
- **Falsification**: gap collapses to within ±0.5pp at cal_frac=0.50 → the round-002 finding is an n_cal-power artifact, not a real gap.
- **Status**: queued
- **Cost**: low (script-level loop + 4× rerun)

### H-202 [P5] Adaptive Conformal Inference (Gibbs & Candès 2021)
- **Owner**: LITERATURE-SCOUT + IMPLEMENTER + CRITIC
- **Mechanism**: implement online ACI on top of the offline CatBoost. The threshold `q_t` evolves: `q_{t+1} = q_t + γ(α - 1{y_t ∈ C_t(x_t)})`. Compare marginal + per-regime coverage and set tightness vs the current ARF baseline (H-201). The hypothesis: ACI matches or beats the ARF's de-facto coverage with simpler, theoretically grounded calibration.
- **Falsification**: marginal empirical coverage must converge to 1-α (within 2σ over the stream); per-regime gap must not be worse than ARF baseline.
- **Status**: ACCEPTED round-007 — `aci_step` + `aci_stream` in `src/conformal.py`. 14/14 ACI tests pass. On real stream (n_eval=22,040, γ=0.01, q-warmed on n_cal=9,446): marginal coverage @α∈{0.05,0.10,0.20} is 0.9508/0.9018/0.8023 for p_offline and 0.9505/0.9023/0.8023 for p_online — gap to target ≤ 0.003 (≤ 1.5σ where σ ≈ 0.002). G&C 2021 Thm 1 marginal-coverage falsifier PASSES. Per-regime gap matches ARF/lac_marginal (better than naive_threshold's 8-10pp) but is worse than round-002's Mondrian-LAC by ~3-6pp at the high-vol tercile — this is the expected price of using a single global q_t and is exactly what H-203 (Mondrian-ACI) closes.
- **Cost**: medium

### H-203 [P5] Mondrian-ACI hybrid for regime-conditional coverage
- **Mechanism**: extend H-202 to maintain a separate `q_t` per volatility tercile (regime). Closes the per-regime coverage gap that plain ACI may leave open under regime drift.
- **Status**: ACCEPTED round-008 — `aci_mondrian_step` + `aci_mondrian_stream` in `src/conformal.py`. 9/9 Mondrian-ACI tests pass (incl. bit-exact reduction to plain ACI under single regime). On real stream: per-regime gap collapses from plain ACI's ±5–10pp to **≤ 0.6pp on every regime / α / predictor combination**. **Beats round-002 batch Mondrian-LAC** (LAC was ≤ 3.4pp at α=0.10 / -7.9pp on low-vol p_online at α=0.20; Mondrian-ACI is ≤ 0.21pp / -0.54pp). The α=0.20 low-vol gap that round-002 LEDGER explicitly named as "the gap H-202..H-208 must close" is now **closed to 0.04pp**. Marginal coverage stays on target.
- **Cost**: medium

### H-204 [P4] River ARF vs SRP vs HAT under coverage-as-metric
- **Mechanism**: paired prequential test of `ARFClassifier`, `SRPClassifier`, `HoeffdingAdaptiveTreeClassifier`. Primary metric: marginal + per-regime coverage at α=0.1, plus set tightness. Secondary: Brier.
- **Status**: queued
- **Cost**: medium

### H-205 [P4] Locally-weighted conformal (kernel-local validity)
- **Mechanism**: at test point x, weight calibration scores by kernel similarity to x. Provides conditional-coverage-by-feature-similarity. Reference: Manokhin Ch. 9.
- **Status**: queued
- **Cost**: medium

### H-206 [P4] CatBoost virtual-ensemble σ_epistemic as a conformal feature
- **Mechanism**: add `sigma_epistemic` from `src/uncertainty.py` to the conformal layer's score function. Tests whether epistemic uncertainty improves coverage tightness.
- **Status**: queued; subsumes the prior H-010.
- **Cost**: medium

### H-207 [P3] ADWIN drift detector triggers ACI threshold reset
- **Mechanism**: feed prediction errors into ADWIN; on detected drift, soft-reset `q_t` toward the prior. Hardens ACI against regime breaks.
- **Status**: queued
- **Cost**: low

---

## Tier 2 — Conformal coverage applied to backtest decisions (ask 4)

### H-011 [P5] Conformal LAC + Mondrian for trade abstention (offline)
- **Mechanism**: use the *batch* split-conformal in `src/conformal.py` (already ported) to gate trade entries — abstain when prediction set is full {0,1}. Combined with H-005's backtest harness for an abstention-aware Sharpe.
- **Status**: queued (depends on H-005)
- **Cost**: medium

### H-208 [P4] Streaming conformal trade gate (online ACI variant)
- **Mechanism**: same as H-011 but using the online ACI threshold from H-202; coverage adapts as the stream evolves.
- **Status**: blocked on H-011 + H-202.
- **Cost**: medium

### H-302 [P4] σ_epistemic-conditioned trade abstention (Ask 2 — distinct from H-025)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 2 + ask 6 — turning epistemic uncertainty into a hard go/no-go gate.
- **Mechanism**: σ_epistemic is the CatBoost virtual-ensemble uncertainty on the OFFLINE layer's output (`predict_with_decomposed_uq` in `src/uncertainty.py`). Under the two-layer architecture, the natural abstention rule is to skip trades when the offline input to the online layer is *epistemically* unconfident — these are exactly the points where the online layer's correction is operating on a noisy upstream signal. Two variants are tested in parallel:
  - **H-302a (gate `baseline_offline_tau`)**: drop trades where `p_offline − k·σ_epistemic < τ`. The offline raw signal is gated by its own confidence band — the simplest possible UQ-aware filter.
  - **H-302b (gate `baseline_online_tau`)**: drop trades where `p_online > τ` AND `p_offline − k·σ_epistemic < τ_off`, with `τ_off` chosen jointly with `(τ, k)` on val. The online output is taken only when the offline input was confident enough; this enforces an architectural "safety belt" on the input layer.
  Round-015's `baseline_offline_tau` is the offline base; round-015's `baseline_online_tau` is the online base. Sweep `k ∈ {0, 0.5, 1.0, 1.5, 2.0}` on **validation Sharpe** (n_trades ≥ 50 stability filter); apply the val-chosen knobs to test once. Multi-comparison deflation (Bailey-Borwein DSR) over the joint grid; CSCV PBO over the k-grid using H-320-a's bootstrap routines. The hypothesis: σ_epistemic is informative for *abstention* — high σ marks model leverage where the offline calibration is least trustworthy (round-005 high-vol ECE = 0.171). A disagreement diagnostic: scatter `(p_offline, σ_epistemic)` colored by realized y on test and check whether the abstained region has higher Brier than the kept region (kept-vs-abstained Brier ratio < 0.85 = signal). The offline model currently lives as a single `model.cbm`; H-302 requires materialising a 3-seed CatBoostEnsemble (`src/ensemble.py`, accepted round-003) trained under `langevin=True` (as the persisted offline already is), then running `predict_with_decomposed_uq` on val + test.
- **Predicted effect**: kept-vs-abstained Brier ratio in [0.80, 0.95] in the high-vol tercile; kept-region test Sharpe shift of +0.05 to +0.20 vs the chosen base (`baseline_offline_tau` for H-302a, `baseline_online_tau` for H-302b) if σ is informative; multi-strategy DSR > 0.95 only if k\* ≠ 0 and the lift survives the (τ × k) deflation. Seed-noise band ≈ 0.02 Sharpe (CatBoost virtual-ensemble σ has its own seed sensitivity).
- **Falsification**: at val-chosen `(τ\*, k\*)`, test Sharpe lift over the matching round-015 base < 0.05 AND kept-vs-abstained Brier ratio in (0.95, 1.05) → σ_epistemic is uninformative; H-302 is null. Additionally if k\*=0 wins the val grid, σ carries no actionable signal at any threshold → null.
- **References**:
  - Lekeufack et al. (2024) *Conformal Decision Theory*, local PDF `Downloads\conformal_decision_theory.pdf` pp. 1-7 (Theorem 1 controller dynamics; σ as decision-bound vs σ as feature; the case `f(λ)` non-monotone is exactly the BTC drift case).
  - Malinin, Prokhorenkova, Ustimenko (2021) ICLR, "Uncertainty in Gradient Boosting via Ensembles", arXiv `https://arxiv.org/abs/2006.10562` (INDEX gap, added by this round). Total / data / knowledge decomposition; SGLB virtual-ensemble draws.
  - Geifman & El-Yaniv (2017) NeurIPS, "Selective Classification for Deep Neural Networks", arXiv `https://arxiv.org/abs/1705.08500` (INDEX gap, added). Risk-Coverage curve + AURC; recommended as a secondary diagnostic alongside the Sharpe lift.
  - Manokhin (2024) *Practical Conformal Prediction*, local Ch. 2 pp. 16-17 — aleatoric vs epistemic framing.
- **Status**: queued. **Distinct** from H-025 (sized) and H-206 (σ as conformal feature) — see H-025 refinement above.
- **Cost**: medium (~2 hours wall: re-train 3-seed CatBoost ensemble with `langevin=True` + run virtual-ensemble decomposition on val/test + sweep k on val + bootstrap on test).
- **Dependencies**: H-108 (CatBoostEnsemble — accepted round-003); H-320-a (bootstrap CIs) ideally; runs without H-320-a but reports CIs once -a lands.

### H-305 [P4] Heterogeneous online ensemble with regime-aware Hedge (Ask 6)
- **Owner**: IMPLEMENTER + THEORIST + LITERATURE-SCOUT
- **Asks**: ask 2 + ask 6 — generalising the streaming conformal layer beyond a single ARF.
- **Mechanism**: today the online stage is one `ARFClassifier(n_models=100, ...)`. This card adds K=5 streaming experts at a level above ARF and routes between them via a regime-aware Hedge update. Concretely:
  - Experts (K=5): `ARFClassifier` (current), `SRPClassifier` (Streaming Random Patches), `HoeffdingAdaptiveTreeClassifier` (HAT), an `ARFClassifier` with shifted hyperparameters (`n_models=50, max_depth=10, lambda=4` — diversification), and a calibrated logistic regression on `p_offline` as the lightweight baseline.
  - Each expert k consumes the same `(selected_features, p_offline)` per step and outputs `p_k`.
  - Combiner: per-regime exponential weights `w_{r,t}(k) ∝ exp(−η · sum_{s≤t, regime_s=r} ℓ_s(k))` where `ℓ_s(k) = brier(p_k_s, y_s)` is the per-step Brier of expert k. The combined prediction is the regime-conditional weighted mean: `p_combined_t = Σ_k w_{r_t, t}(k) · p_k_t / Σ_k w_{r_t, t}(k)`. η is val-chosen on a coarse grid `{0.5, 1.0, 2.0} · sqrt(2 ln K / T_val)` per Hazan §1.3 / Freund-Schapire 1997.
  - The combined `p_combined` feeds Mondrian-ACI exactly as today (`src/conformal.py::aci_mondrian_stream`); coverage validity is not affected (conformal validity is post-hoc on any score function — Manokhin Ch. 9).
  - Compute budget: K=5 streaming models on 31,486 test bars ≈ 5× round-002 wall-clock = ~5 min, fits the `backtest`-tag 30-min budget.
- **Predicted effect**: per-regime Brier of the combined predictor lower than the min over individual experts' per-regime Brier on test, by Δ ∈ [−0.001, +0.005] in expectation (Hedge regret bound: `R_T ≤ √(T ln K / 2) ≈ √(31486 · ln 5 / 2) ≈ 159` cumulative Brier units, divided by T = 5e-3 per-step regret). Effect is **per-regime Brier**, not Sharpe — Sharpe lift requires the H-302/H-025 sizing/abstention layer above this. Per-regime coverage gap should remain ≤ 0.6pp (round-008 floor preserved by post-hoc Mondrian-ACI). Seed-noise band: ≈ 0.001 Brier per regime (single-seed re-runs of round-002).
- **Falsification**: per-regime Brier of `p_combined` > min{per-regime Brier of expert k} − 1.5σ_seed for any regime on test → ensemble is compute waste, kill. Also: cross-strategy DSR (when the combined predictor is added as an 8th strategy in Phase A) does not exceed 0.50 on the new pooled grid → no economic edge added by the ensemble.
- **References**:
  - Montiel et al. (2021) JMLR, "River: machine learning for streaming data in Python", `https://www.jmlr.org/papers/v22/20-1380.html` (INDEX gap, added). Framework + `predict_proba_one`/`learn_one` contract — already used by this project.
  - Gomes et al. (2017) *Machine Learning* 106:1469, "Adaptive Random Forest for evolving data stream classification", DOI `10.1007/s10994-017-5642-8` (INDEX gap, added). ARF + ADWIN drift detector.
  - Gomes, Read, Bifet (2019) ICDM, "Streaming Random Patches", `https://albertbifet.com/streaming-random-patches/` (INDEX gap, added). Global-subspace resampling vs ARF's per-leaf — recommended on heavy-tailed streams.
  - Bifet & Gavaldà (2009) IDA LNCS 5772:249, "Adaptive Learning from Evolving Data Streams" — HAT, DOI `10.1007/978-3-642-03915-7_22` (INDEX gap, added). Per-node ADWIN; alternate subtree replacement.
  - Hazan, *Introduction to Online Convex Optimization* (2e), local PDF `Downloads\online-convext-optimization-book.pdf` Ch. 1.3 / Ch. 5 (Hedge), Ch. 3 pp. 41-52 (OGD regret 3/2·GD√T). Hedge update + Freund-Schapire regret bound `R_T ≤ √(T ln N / 2)` with `η = √(8 ln N / T)`.
- **Status**: queued. **Critical guard**: the ensemble's primary metric is per-regime *coverage* gap (CONSTITUTION IV.1) and per-regime Brier; raw ROC-AUC alone CANNOT accept the ensemble (CONSTITUTION IV.5).
- **Cost**: medium-high (~5 min wall per run; engineering ~3 hours; `backtest`-tagged 30-min budget for the per-regime ensemble round).
- **Dependencies**: H-204 (River expert comparison) provides the K-expert shortlist — once H-204 ranks the best 4-5 experts on Brier, H-305 takes the top K. Streaming-coverage harness in `src/conformal.py` is already in place.

## Tier 3 — Risk-aware weighting beyond what's in the sibling (ask 3)

### H-020 [P4] CVaR-conditional tail upweighting beyond `w_max`
- **Mechanism**: extend `compute_barrier_distance_weight` with a CVaR-α tail multiplier. Concentrates capacity on the worst losses.
- **Status**: blocked on H-102.

### H-021 [P3] Asymmetric Huber-like custom CatBoost loss
- **Status**: queued (medium-high cost; risky with `boosting_type="Ordered"`).

### H-022 [P3] López de Prado meta-labeling layer
- **Mechanism**: second classifier on offline-positive subset, predicts whether the trade actually realizes net-positive payoff.
- **Status**: queued.

### H-023 [P3] First-touch (true triple-barrier) labels
- **Status**: queued; high impact, may invalidate accumulated positives. *Refinement (round-014)*: H-304 prioritisation card identifies first-touch as the highest-leverage *unfixed* item on the strategy axis, but cost is high (full retrain + invalidates legacy ROC=0.813 baseline). Run H-024 (no-stop, cheap) first; if no-stop alone delivers ΔSharpe > +0.5 with p_boot < 0.05, the label/backtest mismatch was the binding constraint and H-023 becomes optional. Otherwise H-023 is mandatory before any further sized-strategy work.

### H-303 [P4] Saturated asymmetric weighting — barrier-distance × time-discount with hard cap (Ask 4)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 3 — risk-aware weighting on top of H-102's sibling import.
- **Mechanism**: implement `compute_training_weights(bars, alpha, M, *, w_max, lambda_decay, time_decay_floor)` returning `w_i = min(w_max, w_dist(d_i)) · max(time_decay_floor, w_time(t_i))` where:
  - `d_i` is the **vol-normalised barrier distance** at sample `i` (loss proxy that is NOT the realised PnL; PnL would be label leakage). Concretely `d_i = (m_k − α) / σ_k_local` where `m_k = ln(H_{k+1}/C_k)` is the realized log-excursion and `σ_k_local` is `sqrt(parkinson_var_rolling_mean_24)` at boundary k. Wait — `m_k` IS the label-defining quantity; using it as the loss proxy IS leakage. **Corrected**: `d_i = barrier_distance_proxy(features_only_i)`, computed from past-only features (e.g., `α / σ_k_local` from H-111 — the "tightness" feature). `w_dist(d) = exp(λ · |d|)` saturating at `w_max` (Huber-style cap).
  - `t_i` is the chronological position; `w_time(t) = δ^(T_train − t)` with `δ ∈ [0.99, 0.999]` and a floor `time_decay_floor` (default 0.1) preventing oldest bars from going to zero (information loss).
  - Effective sample size `n_eff = (Σ w_i)^2 / Σ w_i²` reported per fit; if `n_eff / n_train < 0.6` reject the (w_max, λ, δ) combination as over-collapsed.
  - **Ablation grid**: `(w_max, λ, δ) ∈ {3, 5, 10} × {0.5, 1, 2} × {0.99, 0.995, 0.999}` = 27 cells. Multi-comparison correction: Bonferroni at the BSS accept-gate (α=0.05/27 ≈ 0.0019) AND CSCV PBO over the 27 cells using H-320-a's helper.
  - **SqrtBalanced interaction (load-bearing)**: ablate against three CatBoost class-weight settings: (i) `auto_class_weights="SqrtBalanced"` + sample weights; (ii) `auto_class_weights=None` + sample weights only; (iii) `auto_class_weights="SqrtBalanced"` no sample weights (status quo baseline). The H-102 LEDGER concern is the (i)/(iii) double-count; (ii) is the disentangling control.
- **Predicted effect**: with `(w_max, λ, δ) = (5, 1.0, 0.995)` (centre of grid), expected ΔBSS ∈ [+0.005, +0.012] over status-quo SqrtBalanced baseline; `mean_p` shifts from 0.205 toward 0.097 (the base rate) by Δ ∈ [−0.04, −0.10]. ECE in high-vol tercile expected to drop from 0.171 toward 0.10. Seed-noise band ≈ 0.003 BSS.
- **Falsification**: best-of-grid ΔBSS < +0.003 (within seed-noise band) AND `mean_p` movement < 0.02 → weighting carries no marginal information; kill. Additionally, if any cell with `n_eff / n_train < 0.6` enters the grid winner, the saturation cap was set too low — re-grid up.
- **References**:
  - Hansen *Econometrics* (2022), local citation in INDEX [MISC]; weighted M-estimation consistency under heavy-tailed weights.
  - van der Vaart & Wellner, *Weak Convergence and Empirical Processes*, local `Desktop\paper_editing\sources\_all_pdfs\vanDerVaartEtAl - Weak Convergence and Empirical Processes.pdf` Ch. 2 (bracketing under weighted empirical measure).
  - Bailey & López de Prado (2014) deflated Sharpe — for the deflation across the 27-cell grid.
  - López de Prado *AFML* (2018), local Ch. 4 (sample weighting in finance ML; the "uniqueness" weight).
- **Status**: queued. **Prerequisite chain (load-bearing)**: H-103 (undef-flag pattern) → H-102 (sibling weight import) → H-303 (this card). H-303 cannot run before H-102 because the sibling's `compute_barrier_distance_weight` and `compute_time_discount_weight` are the inputs; H-102 cannot run cleanly without H-103 because the undef-flag pattern is what allows the weights to be inspected at boundary cases without silent NaN-handling.
- **Cost**: medium (grid is 27 × 3 ablations × ensemble-of-3-seeds = 243 train-evals; in FAST_MODE on year-2024 only this is ~3 hours wall; full mode ~12 hours — plan as multi-round).
- **Dependencies**: H-103, H-102. Pairs with H-310 (rolling retrain) ideally — weighted retrain under rolling cadence is the production target.

*Refinement (round-014, addenda to existing weighting cards)*:
- **H-020** (CVaR tail upweight) — once H-303 lands, H-020 becomes incremental: add the CVaR-α multiplier on top of H-303's `(w_dist, w_time)` only if H-303 fails to close the high-vol tercile ECE below 0.10.
- **H-021** (asymmetric Huber-like custom CatBoost loss) — high-risk variant; only attempt if H-303's saturated linear weighting plateaus.

---

## Tier 4 — Features (research-driven, ask 1)

### H-040 [P4] Hurst / DFA / sample entropy (time-invariant memory descriptors)
- **Mechanism**: rolling R/S Hurst, DFA exponent, sample entropy on returns. Scale-free under Brownian rescaling. Reference: `Multiscale Stochastic Volatility` (local Downloads).
- **Status**: queued
- **Cost**: medium

### H-041 [P3] Wavelet decomposition energy ratios
- **Status**: queued

### H-042 [P3] HAR-RV residuals as features
- **Mechanism**: fit Heterogeneous Autoregressive Realized Volatility (Corsi 2009) on rolling windows; residual is forward-looking surprise.
- **Status**: queued

### H-044 [P3] Realized higher moments (bipower-corrected skew/kurtosis)
- **Status**: queued

### H-111 [P4] Group N — barrier-aware features (ported from sibling)
- **Owner**: IMPLEMENTER
- **Asks**: ask 1
- **Mechanism**: this project's barrier construction `α = 0.00411` is a fixed economic threshold, exactly the regime where barrier-aware features (`barrier__z_tight = α / (σ·√M)`, `barrier__emax_ratio = σ·√(2 ln M) / α`, `vol__ratio_short_long`) carry information. Sibling implements them in `barrier_classifier/src/utils.py::compute_barrier_aware_features` (16 windows + 12 vol pairs). Port adapted to this project's `M=20` and `windows=[1,2,4,8,12,24,48,96]`.
- **Status**: queued (high information density; adds ~50 features)
- **Cost**: medium

### H-112 [P4] Group O — excursion features (max drawup/drawdown, max-N-bar returns)
- **Owner**: IMPLEMENTER
- **Asks**: ask 1 — **label-aligned** features
- **Mechanism**: rolling max drawup, max drawdown, max 1-bar / 2-bar returns. Label-aligned because `y_k` itself is a max-future excursion. Sibling: `compute_excursion_features` with chunked stride-trick implementation (avoid 9-window naive O(n·W)).
- **Status**: queued
- **Cost**: medium

### H-114 [P4] Group C+ — volatility decomposition (bipower variation ratio, semivariance up/down/ratio, vol-of-vol)
- **Owner**: IMPLEMENTER
- **Asks**: ask 1
- **Mechanism**: bipower variation = jumpiness proxy (`RV / BPV`); semivariance asymmetry (`SV_down / SV_up` semivolatility ratio); vol-of-vol (rolling std of rolling vol). Sibling: `compute_volatility_decomposition`. The semivariance ratio in particular is well-suited to this project's positive-skew label (only upper barrier).
- **Status**: queued
- **Cost**: medium

### H-115 [P3] Permutation entropy (m=3, τ=1) — complexity / predictability
- **Owner**: IMPLEMENTER
- **Asks**: ask 1
- **Mechanism**: Bandt-Pompe normalized permutation entropy on returns. Sibling: `compute_permutation_entropy` with stable mergesort tie-breaking.
- **Status**: queued
- **Cost**: low

*Refinement (round-014)* — failure modes for the existing feature group, captured in one place per H-301 taxonomy:
- **H-040** Hurst/DFA/sample entropy — **breaks under jumps**; jumps create singularities the rolling estimators absorb as drift, biasing Hurst toward 0.5; sample entropy is small-sample-biased below n ≈ 100, so windows shorter than 8 (160-min) are unreliable. Online-compatibility: DFA needs slope on log-aggregated variance — **needs streaming approximation** (river has no native DFA). Hurst is online-compatible via R/S statistic recursive update.
- **H-041** wavelet energy ratios — **assumes stationarity within wavelet window**; under regime change within the window the ratios mis-attribute energy. Decimated DWT is **offline-only**; undecimated MODWT is online-compatible at higher cost.
- **H-042** HAR-RV residuals — **assumes stable variance-of-variance**; under regime change in vov, the HAR fit drifts within-window and residuals mix signal+drift. Online-compatible via recursive-LS HAR (Corsi 2009 supplement).
- **H-044** realized higher moments (bipower-corrected skew/kurt) — **bipower correction fails under simultaneous price+volume jumps** (the jump-correction assumes price-only jumps). Online-compatible via running bipower stat.
- **H-111** barrier-aware (`α/(σ√M)`, `σ√(2 ln M)/α`, `vol__ratio_short_long`) — **σ-proxy stability assumption**: features assume the rolling σ used is a stable estimator of local-regime volatility; under regime drift (the very thing this round wants to capture), the σ proxy lags. Mitigation: pair with H-114's bipower-derived robust σ. Online-compatible.
- **H-112** excursion features (drawup/drawdown, max-N-bar returns) — **label-aligned ⇒ tight correlation with label**; selection bias if used naively (the same DGP that produces the future-window label produces the past-window excursion proxy). H-112 is therefore high-information but at risk of leakage if the rolling window inadvertently includes lookahead. Online-compatible (rolling-min/max trivial). Mandatory pairing with H-103 (undef-flag pattern) so the boundary cases (segment edges, gap windows) are flagged not silently filled.
- **H-114** vol decomposition (BPV ratio, semivar up/down, vov) — **bipower fails under simultaneous jumps**; semivar asymmetry assumes upper/lower barrier asymmetry is stable across regimes; vov is itself heavy-tailed, calibration on training data may not generalise. Online-compatible.
- **H-115** permutation entropy — **small-sample-biased below n ≈ 100**; m=3 ordinal patterns saturate fast on noisy data. Online-compatible via running ordinal-pattern histogram.

The taxonomy (H-301) will fold these into one comparison table.

### H-301 [P4] Feature taxonomy and prioritisation (Ask 1 synthesis card)
- **Owner**: THEORIST
- **Asks**: ask 1 — taxonomy and pruning, not new compute_*.
- **Mechanism**: collate H-040, H-041, H-042, H-044, H-111, H-112, H-114, H-115, H-311, H-312, H-313 into one table indexed by family / online-compatibility / expected ΔBSS / failure mode / dependency on H-103. The table is the synthesis artefact; no new code is shipped under H-301. The card is what determines which feature card runs first under Phase D — sequencing is by `expected_ΔBSS / cost`. Pre-computed taxonomy lives in `RESEARCH/research_plan_round_014.md` §Ask 1; H-301's accept-gate is "does Phase D's first feature round pick the cell with highest expected ΔBSS that is also online-compatible (since the offline+online stack stays parallel)".
- **Predicted effect**: zero (synthesis card; no metric movement). The card's value is operational — it cuts the BACKLOG churn by 1-2 rounds where we otherwise would have explored an offline-only feature first that the online layer cannot consume.
- **Falsification**: a future feature accept-round (Phase D) shifts headline DSR by > 0.10 with a feature group that the taxonomy classified as "low expected ΔBSS" → taxonomy mis-prioritised, re-rank.
- **References**: López de Prado, *AFML* (2018), local PDF Ch. 5 (entropy features) + Ch. 8 (feature importance). Cover & Thomas, *Elements of Information Theory*, local `Desktop\paper_editing\sources\_all_pdfs\Cover - Elements of information theory.pdf` Ch. 8 (Asymptotic Equipartition + entropy bounds — backs H-040/H-115). Vershynin, *High-Dimensional Probability* — concentration bounds on entropy estimators (footnote citation only).
- **Status**: queued (this card is the gating doc for Phase D ordering).
- **Cost**: low (pure synthesis; no compute).
- **Dependencies**: H-103 (undef-flag pattern) — every feature group must conform to this before being measured; H-105 (NSGA-II HPO) — taxonomy informs the search space for HPO.

### H-311 [P3] Rolling Pearson(intra-bar return, signed volume) — microstructure imbalance, scale-invariant
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 1 — invariance under multiplicative volatility scaling.
- **Mechanism**: at each minute, signed volume = `volume · sign(close − open)`. Compute the rolling Pearson correlation between minute-resolution returns and signed volumes over windows {12, 24, 96} (in 1-min bars; corresponds to 12 min, 24 min, 96 min). Aggregate to decision-bar cadence (M=20) by taking the value at `n_k`. The feature is **scale-invariant**: under multiplicative volatility scaling `(ret, vol) → (c·ret, c·vol)`, the Pearson correlation is unchanged. The economic interpretation is **order-flow imbalance proxy**: positive correlation = aggressive buying (informed flow), negative = aggressive selling. Distinct from H-114's vol decomposition (which is variance, not directional).
- **Predicted effect**: ΔBSS ∈ [+0.002, +0.007] over the existing feature set, concentrated in the high-vol tercile (where directional flow is most informative). CatBoost importance expected to rank in the top 50 of 726 features. Seed-noise band ≈ 0.0008 BSS (from existing offline reseed runs).
- **Falsification**: ΔBSS < +0.002 (within 2.5σ noise) on the accept-gate run AND CatBoost importance rank > 200 → feature carries no marginal information; kill.
- **References**: López de Prado, *AFML* (2018), local Ch. 17 (microstructural features). Hasbrouck (1991) — VAR-impulse decomposition for trade-vs-price (cited via Cont & Tankov, web-fetch deferred — not in local INDEX). The Binance data does have `taker_buy_base` and `taker_buy_quote`, exactly the signed-flow proxies — no new data needed.
- **Status**: queued. Online-compatible (rolling Pearson is streaming via running sums of `(x, y, x², y², xy)`).
- **Cost**: low (~1 hour engineering + tests). New compute_* function: `compute_signed_volume_correlation(bars, windows)`. Multiple-comparison correction over the (3 windows × 3 ρ-stats) grid via Bonferroni at the feature-importance accept gate.

### H-312 [P3] Bipower-variation ratio across windows — vol-regime invariant under multiplicative scaling
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 1.
- **Mechanism**: bipower variation BPV(window) = `(π/2) · Σ |r_t| · |r_{t-1}|` (Barndorff-Nielsen & Shephard 2004) — a jump-robust integrated-variance estimator. The **ratio** BPV(short)/BPV(long) for short ∈ {2, 4, 8} and long ∈ {24, 48, 96} (in decision bars) is dimensionally a vol-regime indicator: > 1 = recent vol spike, < 1 = recent calm relative to baseline. Scale-invariant under multiplicative vol scaling (cancels in numerator and denominator). **Different from H-114's plain BPV** — H-312 is the *ratio*, not the level; `vol__ratio_short_long` in H-111 is window-pair driven by RV not BPV.
- **Predicted effect**: ΔBSS ∈ [+0.001, +0.005] when added on top of H-111+H-114 (incremental over those). The bipower-vs-RV decomposition is informative when jumps are common (BTCUSDT 1m has frequent jumps); under jump-light periods the BPV ratio collapses to the RV ratio and adds little.
- **Falsification**: ΔBSS < +0.001 over (H-111 + H-114) accept-gate baseline within 1.5σ → kill. Also: if importance is concentrated only in low-vol tercile (where jumps are less common), the feature is operating opposite to its expected mechanism — investigate or kill.
- **References**: Barndorff-Nielsen & Shephard (2004) JFE, "Power and bipower variation with stochastic volatility and jumps" (web-fetch needed — not in local INDEX). *Multiscale Stochastic Volatility*, local PDF — multi-window BPV decomposition. Online-compatible (BPV is rolling sum of `|r| · |r_prev|`).
- **Status**: queued. Pair with H-114 for ablation discipline.
- **Cost**: low.

### H-313 [P3] DFA slope on signed-return series — range-invariant memory
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 1.
- **Mechanism**: detrended fluctuation analysis on the cumulative signed-return series (Peng et al. 1994). Compute the rolling DFA exponent α(window=W) on the signed return series for W ∈ {96, 192, 288} (decision bars). α ≈ 0.5 = uncorrelated; α > 0.5 = persistent (trend-following); α < 0.5 = mean-reverting. **Range-invariant**: DFA is invariant under affine transformation of the series (subtract local trend before computing fluctuation). **Different from H-040's plain Hurst**: DFA is more robust under non-stationary trends (Hurst R/S is biased under drift; DFA actively detrends).
- **Predicted effect**: ΔBSS ∈ [+0.001, +0.006] when added on top of H-040 (incremental); the DFA-vs-Hurst distinction is informative when there is non-zero drift in the analysis window — exactly the BTCUSDT 2024-2025 case (+120% drift). High-vol regime expected to contain the bulk of the signal.
- **Falsification**: ΔBSS < +0.001 over (H-040) accept-gate baseline within 1.5σ → kill. Also: if α is monotone in the window length (no scaling regime change), DFA carries no scale-free signal — kill.
- **References**: Peng et al. (1994) Phys Rev E, "Mosaic organization of DNA nucleotides" — original DFA definition (web-fetch). *Multiscale Stochastic Volatility*, local PDF — DFA in finance context. Cover & Thomas Ch. 8 — entropy bound on memory. **Online-compatibility**: DFA's standard estimator is offline-only (needs full series + multi-scale fits); online approximation via the recursive cumulative-DFA (Hartmann et al. 2013) is the implementable variant — adds a 1-line note to H-301's table.
- **Status**: queued.
- **Cost**: medium (the online approximation is non-trivial; tests need a known α process).

---

---

## Tier 5 — Production engineering (ask 5)

### H-050 [P4] Feature pipeline as hash-keyed cache
- **Mechanism**: every `compute_*` keyed by `(name, version, config_hash, data_hash)`; cache to `experiments/feature_cache/`.
- **Status**: queued

### H-051 [P4] `compute_*` registry — replace imperative chain in `feature_build` notebook
- **Status**: queued

### H-054 [P3] Pre-commit hook running fast tests
- **Status**: queued

### H-105 [P5] NSGA-II multi-objective Optuna HPO with walk-forward CV (sibling import)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5 — this project currently has NO hyperparameter optimization (`ENABLE_HPO=False` everywhere implicitly).
- **Mechanism**: port the sibling's HPO pattern from `barrier_classifier/notebooks/03_model_training.ipynb`: NSGA-II sampler (objectives = minimize logloss, maximize PR-AUC), walk-forward CV across N folds (each fold's train precedes its val chronologically + embargo), per-trial seed variation (CB_SEED + trial.number), Pareto-frontier visualization, "best trial" selection rule (within 5% of min-logloss, take best PR-AUC), `HPO_DROP_OLDEST_FRAC` speed knob, ordered Pool with timestamp.
- **Falsification**: a fresh HPO run + retrain produces metrics within ±5% of the legacy ROC=0.813 / Brier=0.090 (or beats them).
- **Status**: queued
- **Cost**: medium-high (45+ min wall-clock per HPO run; plan as `backtest`-budget round)

### H-106 [P5] Regime-stratified calibration reporting (sibling import + adaptation)
- **Owner**: IMPLEMENTER
- **Asks**: ask 4 — primary metric per CONSTITUTION IV
- **Mechanism**: port `compute_all_metrics`, `expected_calibration_error`, `calibration_by_regime`, `threshold_analysis` from sibling `src/utils.py`. Wire into `notebooks/offline_train.ipynb` and `notebooks/online_eval.ipynb`. Pick a fixed regime signal (proposal: `parkinson_var_rolling_mean_24` from this project's existing features) and run `pd.qcut(_, 3)` to get terciles. Output `RESEARCH/diagrams/round_NNN/calibration_by_regime.png` per round.
- **Status**: ACCEPTED round-005 — four helpers in `src/utils.py`. 13 tests pass. First per-regime calibration plot on real test stream produces a striking diagnostic: offline ECE scales 0.05/0.10/0.17 (low/med/high vol — overpredicts everywhere, severity rises with vol); online ARF ECE ≈ 0.015 uniformly across regimes (mean_p ≈ base_rate per regime). This is direct empirical proof of the project's design thesis (online sacrifices ranking for regime-conditional calibration) and identifies high-vol as where offline calibration breaks worst — pointing H-202..H-208 at vol-conditional improvement.
- **Cost**: medium

### H-107 [P4] Plot helpers + threshold-analysis CSV (sibling import for visual-first reporting)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5 + visual-first CONSTITUTION rule
- **Mechanism**: port `plot_calibration_curve`, `plot_calibration_by_regime`, `plot_feature_importance`, `plot_threshold_curves`, `plot_weight_profiles`, `plot_weight_distributions` from sibling. The `threshold_analysis.csv` output (precision, recall, trade-rate at sweep of thresholds) is missing from this project's eval artifacts.
- **Status**: ACCEPTED round-006 — `src/plotting.py` lands four helpers (calibration_curve, calibration_by_regime, feature_importance, threshold_curves). 13 smoke tests pass; round-006 visual report card on real data reproduces round-005 ECE numbers exactly (offline 0.108, online 0.015). `plot_weight_*` explicitly deferred to H-102 round. `feature_importance` API improved over sibling: takes `importances` array directly (works with `CatBoostEnsemble` / sklearn / anything), not a model object.
- **Cost**: low

### H-108 [P4] CatBoostEnsemble wrapper class (sibling import)
- **Owner**: IMPLEMENTER
- **Asks**: ask 5
- **Mechanism**: port the `CatBoostEnsemble` class (averages predictions / feature importances / best iterations across N seed-varied CatBoost models). Drop-in replacement for the ad-hoc model handling currently in `notebooks/offline_train.ipynb`. Saves cleanly + reloads via base+`.{i}.cbm` pattern.
- **Status**: ACCEPTED round-003 — `src/ensemble.py` with `CatBoostEnsemble` (predict_proba/feature_importance/best_iteration averaging + save_model + new `load_ensemble` classmethod). 10 tests in `tests/test_ensemble.py` pass. Diagnostic plot `RESEARCH/diagrams/round_003/ensemble_dispersion.png` shows non-trivial seed disagreement (max single-model deviation from ensemble = 0.070). Unblocks H-206.
- **Cost**: low (~20 min)

### H-113 [P3] CSCV — Probability of Backtest Overfit (López de Prado Ch. 11)
- **Owner**: LITERATURE-SCOUT + IMPLEMENTER
- **Asks**: ask 4 — overfitting check for any backtest-tagged round
- **Mechanism**: combinatorially symmetric CV: split test into S equal slices, take all `(S choose S/2)` partitions, rank in-sample SR vs out-of-sample SR per partition; PBO = Pr[best IS ranks below median OOS]. Loop must report PBO whenever a round changes the strategy or hyperparameters. Reference: Bailey et al. (2014) `pseudo-mathematics-and-financial-charlatanism`.
- **Status**: ACCEPTED (round-012 → round-015 cleanup). The `cscv_pbo` function in `src/backtest.py` survived the round-015 cleanup (architecture-agnostic; 4 tests still pass including pure-noise PBO ∈ [0.35, 0.75] and clear-winner PBO < 0.05; 12,870 combinations on n_chunks=16). Round-015 is where the cross-strategy + per-strategy CSCV passes are run against the corrected harness — round-015's CSCV PBO outputs (`cscv_per_strategy.csv`, `cscv_cross_strategy.json`) are the canonical reference, not the deleted round-012/013 artefacts.
- **Cost**: medium

### H-306 [P4] Backtest-trade postmortem schema and round (GENUINE GAP — Ask 7)
- **Owner**: THEORIST + IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 4 — surface conditional alpha (or its absence) so future strategy rounds know where to look.
- **Mechanism**: `src/backtest.py::Trade` already tracks `(k_open, k_close, n_open, n_close, entry_price, exit_price, exit_reason ∈ {tp,sl,timeout}, pnl_log_gross, pnl_log_net, p_signal, bars_held)`. No round to date has *postmortemed* this. This card defines the canonical postmortem schema and an artefact contract every future strategy round must diff against.

  **Conditional slices** (each with a numerical falsifier):
  1. **Exit-reason mix** — TP / SL / timeout fraction breakdown; per-reason mean PnL. Falsifier: TP fraction < 0.25 → upper-barrier rarely hit (model over-trades or α too aggressive).
  2. **Hold-time distribution** — bars-held histogram per exit-reason. Falsifier: median TP hold > 12 bars (60% of M) → TPs are not "fast wins"; the strategy is functionally a hold-to-close (timeout-equivalent).
  3. **p_signal decile** — for each `p_signal` decile, hit-rate + Sharpe. Falsifier: top-decile hit-rate ≤ bottom-decile + 5pp → predictor uninformative at the margin.
  4. **Realized vol regime at entry** — slice trades by `parkinson_var_rolling_mean_24` tercile at entry. Falsifier: high-vol Sharpe ≤ low-vol Sharpe within seed-noise band → vol is not a discriminator.
  5. **σ_epistemic decile (when CatBoost virtual ensemble is available)** — low-σ trades should have higher hit-rate. Falsifier: bottom-σ-decile hit-rate ≤ top-σ-decile + 3pp → σ uninformative for trade selection (kills H-302 directly).
  6. **Intraday hour (UTC)** — Sharpe by entry hour (24 buckets). Falsifier: no hour cluster has Sharpe > median + 2σ_seed_noise → no intraday timing edge.
  7. **Day-of-week** — Sharpe by Mon..Sun. Falsifier: range across 7 days < 1.5σ → no DoW edge.

  **Aggregate falsifier (load-bearing)**: NO slice shows hit-rate deviation > 1.5σ AND no slice's Sharpe is positive at p_boot < 0.10 (using H-320-a's stratified bootstrap) → model has **no exploitable conditional alpha**; kill all sized-strategy work (H-025, H-302). This is the canonical "the model has no edge anywhere" decision.

  **Persisted artefacts** (one round, persisted forever):
  - `RESEARCH/diagrams/round_NNN/trade_postmortem.parquet` — per-trade with all slice indicators precomputed.
  - One PNG per slice (7 slices ⇒ 7 PNGs).
  - `RESEARCH/diagrams/round_NNN/postmortem_summary.csv` — one row per slice × bucket with point + bootstrap CI.
  - `headline.json` containing the aggregate falsifier verdict.

  **Spawning rule**: any future strategy round must run this postmortem on its trades and diff against the round-NNN baseline. The diff (cells with statistically-significant change vs prior round) is what the next strategy iteration uses.
- **Predicted effect**: zero (postmortem card; the value is the diagnostic, not a metric improvement). Likely outcome on round-015's `baseline_offline_tau` trades (modal IS-best across CSCV; the canonical Phase-A starting point): top-decile p_signal hit-rate ~0.55 vs base 0.52 (5pp lift but within noise); high-vol Sharpe negative-but-larger-magnitude than low-vol; 02:00–06:00 UTC hour cluster (low-liquidity regime) has worst Sharpe. These are educated guesses — the postmortem produces the truth.
- **Falsification of the postmortem ITSELF**: if any slice shows a lift > 3σ and the strategy round-NNN-1 didn't capitalise on it → the postmortem schema is missing a slice; add it.
- **References**: López de Prado *AFML* (2018), local Ch. 14 (backtest statistics — t-stat per regime, attribution). Bailey & López de Prado deflated-Sharpe paper (already cited; the regime-decomposition statistic). Kaufman, *Trading Systems and Methods* — postmortem chapter.
- **Status**: queued (P4). **Spawning rule (operational)**: future strategy round MUST re-run + diff. Add to `LOOP_DISCIPLINE.md` under "Standing checks per round" once H-306 ships.
- **Cost**: medium (~3 hours engineering for the schema; per-round application is ~5 min).
- **Dependencies**: H-320-a (per-slice bootstrap CI). Pairs with H-310-b (rolling retrain measurement) — postmortem applied to rolling-retrain trades.

---

## Tier 6 — Sibling reference index

The sibling `C:\Users\vitil\OneDrive\Desktop\barrier_classifier\` is the **read-only reference** for offline-stage practices. Specific practices already queued above as hypotheses:

| Sibling artifact | Imported as | Status |
|---|---|---|
| `compute_barrier_distance_weight` (deep-loss exp upweight, soft cap) | H-102 | queued |
| `compute_time_discount_weight` (geometric δ-decay with floor + cutoff) | H-102 | queued |
| `compute_training_weights` (combined w_dist · w_time, effective-N) | H-102 | queued |
| `chronological_split_with_embargo`, `walk_forward_cv` | H-101 + H-104 | queued |
| `get_imputation_value`, `create_undef_flags_and_impute` (regex lookup) | H-103 | queued |
| `CatBoostEnsemble` class | H-108 | queued |
| NSGA-II Optuna HPO with walk-forward CV (notebook 03) | H-105 | queued |
| `compute_all_metrics`, `expected_calibration_error`, `calibration_by_regime`, `threshold_analysis` | H-106 | queued |
| `plot_calibration_curve`, `plot_feature_importance`, `plot_threshold_curves`, weight plots | H-107 | queued |
| `compute_barrier_aware_features` (Group N) | H-111 | queued |
| `compute_excursion_features` (Group O drawup/drawdown + maxret) | H-112 | queued |
| `compute_volatility_decomposition` (bipower, semivar, vov) | H-114 | queued |
| `compute_permutation_entropy` (Bandt-Pompe) | H-115 | queued |
| `bootstrap_no_skill_pvalue`, `deflated_sharpe` (already in `src/backtest.py`) | done at bootstrap | accept |
| `predict_with_decomposed_uq`, `predictive_intervals` (already in `src/uncertainty.py`) | done at bootstrap | accept |
| `fit_conformal`, `predict_set`, `coverage_by_regime` (already in `src/conformal.py`) | done at bootstrap | accept |

**Rule for sibling-import rounds**: don't blindly copy. Adapt to this project's:
- `M=20` (sibling uses `M=10`); window sets differ
- No EMBARGO (sibling has 60); per-segment burn-in uses `burn_in_bars=96` (sibling uses `K_WARMUP=144`)
- Online stage = streaming conformal coverage (sibling has none of this)
- `auto_class_weights=SqrtBalanced` interaction with sample weighting (H-102)
- Existing notebook flow (sibling has `01_data_download / 02_feature_building / 03_model_training`, this project has `data_download / feature_build / offline_train / online_eval`)

---

## Notes
- Items at `[P5]` are unconditionally scheduled before `[P3]`.
- "BLOCKED" items wait until the unblock-er is `accept`ed.
- New ideas append to the appropriate tier section, NOT the top — priority discipline is enforced.
- Items spawned mid-round are marked with `(spawned by round NNN)` for traceability.
