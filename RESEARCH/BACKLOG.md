# Hypothesis Backlog

> **POST-CONFORMAL-REMOVAL BANNER (2026-05-10)**: H-202..H-209 (the Mondrian-ACI
> variants and adaptive-conformal-inference cards) are SUPERSEDED. The Mondrian-ACI
> conformal layer has been removed in this commit's wave (sibling ARCH owns the
> cleanup). The new strategy is `ThresholdGate(p_online >= tau)`. See
> [`docs/concepts.md`](../docs/concepts.md) for the new architecture and
> [`RESEARCH/REPORT.md`](REPORT.md) for the headline. Cards labelled SUPERSEDED
> below should not be activated by the loop without first re-introducing the
> conformal layer through a research round.
>
> **All cards that referenced deleted scripts (`scripts/phase_A_round_*.py`,
> `scripts/round_010_no_stop_backtest.py`, etc.) have been pruned or updated to
> reference `wagie experiment run <spec.yaml>` instead.** Reproduction now goes
> through the wagie harness (`experiments/baseline.yaml`,
> `experiments/replay_r031_low_vol_gate.yaml`, etc.).

The autonomous loop targets `online_barrier_classifier`. The architecture is:

```
offline CatBoost  ->  p_offline
                       +
              River ARF (bagging-calibrated streaming online layer)  ->  p_online
                                                                          (calibrated probability;
                                                                           system output)
                       +
              ThresholdGate(p_online >= tau)  ->  decisions
```

The "online stage" (River `ARFClassifier`, fed `selected_features + p_offline`)
is a **bagging-calibrated streaming layer with ADWIN drift surfacing** — NOT
a separate classifier and (post 2026-05-10) NOT a conformal layer. Its calibration
properties (regime-flat ECE, low Brier vs offline) come from the bagging across
ARF members; the ADWIN signal flags drift but does not trigger a coverage update.

User asks 2 (online) and 6 (UQ) collapse onto one axis: **improve `p_online`
calibration (Brier / ECE with bootstrap CI), surface drift cleanly, gate
intelligently** — NOT improve the (removed) conformal coverage layer.

Status legend: `actionable` / `blocked` / `superseded`. Priority P1-P5 = expected
α-on-user-goal / cost. Top of file = highest priority.

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
- **Status**: KILLED (round-010 in_progress, never completed; see KILL_LIST). *Post-conformal-removal refinement*: H-024 should now run as a wagie spec (`experiments/baseline_with_offline_no_stop.yaml`, NEW spec — caller adds it) with `wagie.broker.stop_loss_log: 1.0e9` (effectively c_stop=∞), `threshold_gate(p_offline)`, and the accept-gate fields. The `conformal_gate_tau` and `mondrian_aci_size` strategies referenced in the round-015 framing are GONE; only `threshold_gate` and `regime_gated` survive (and `null_random_at_rate` as a comparator if sibling TRADING ports it). **Forbidden**: any reintroduction of sibling-style `(p_offline, p_online)` combiners.
- **Cost**: low — one new YAML spec + `wagie experiment run` (~5 minutes wall).

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

  Round-009 empirical hit_rate=0.525 at val-tau=0.44 under SYMMETRIC barriers (TP=+α, SL=−α, timeout mixed). The symmetric hit_rate fraction *includes* SL losses, so the equivalent **p_TP under symmetric** is approximately `n_tp / n_trades`, which the H-024 wagie spec's `metrics.json::trading.tp_count / n_trades` will expose directly. **Empirically, if `p_TP_symmetric` ≥ ~30% at val-τ=0.44**, the no-stop variant comfortably exceeds the 14.1% bar and should produce positive test Sharpe; if `p_TP_symmetric` < 14% the no-stop variant still loses. The H-024 run produces the headline number directly. (The round-015 `headline.json` referenced in pre-wagie versions of this card no longer exists; that script was deleted in commit `61ce420`.)

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
- **Status: SUPERSEDED**. The Mondrian-ACI per-regime `q_t` no longer exists. Replacement: `EvCalibratedSize(k * margin)` where `margin = max(0, p_online - tau)` — a direct margin-on-the-bagging-calibrated-probability sizing rule. The "sized vs binary" question collapses to a sweep on `EvCalibratedSize.k`, expressible as a 5-spec multi-doc YAML similar to `experiments/sweep_alpha.yaml`.

*Refinement (round-014/015, addendum from Ask 2)*: H-025 (sized via Mondrian-ACI on `p_online`) and H-302 (σ-epistemic *abstention* on top of `p_online`) are NOT the same hypothesis and must not be conflated. **H-025** *scales* position size by the per-regime conformal confidence on `p_online` (`size = clip(k · max(0, p_online − (1 − q_lo_α)), 0, 1)`) — every above-threshold opportunity is taken, only at differing magnitude. **H-302** *gates* trade entry by a CatBoost-virtual-ensemble σ_epistemic on top of the chosen base strategy — when σ is above a val-chosen threshold, the trade is skipped entirely (size = 0). The two interact: H-025 uses the conformal q_t (a coverage-derived signal that includes both aleatoric and epistemic noise); H-302 isolates the epistemic component. Round-015 (corrected two-layer architecture) showed `mondrian_aci_size` at k\*=20 producing test Sharpe=−5.67 with p_boot=0.505 (no edge after deflation; cross-strategy DSR=0). Sizing on a no-edge base actively destroys value. H-302's separate σ-abstention test is essential before any joint H-025+σ variant. Run H-302 first; only if it shows σ-conditional Sharpe lift does the H-025+H-302 combination become a separate card. **Forbidden**: any framing that averages or stacks `p_offline` and `p_online` (round-015 contract).

### H-005b [P5] (spawned by round 001) Validation-split τ_open selection for the H-005 sweep
- **Owner**: IMPLEMENTER
- **Asks**: ask 4 (rigorous evaluation per CONSTITUTION V.b)
- **Mechanism**: round 001 picked τ_open as a post-hoc grid because the existing `artifacts/online_eval/` only persists test predictions. Re-run `notebooks/offline_train.ipynb` + `notebooks/online_eval.ipynb` to additionally persist a `predictions_val.parquet` (the chronological val window that was already used for early-stopping). Sweep τ_open on val Sharpe + PR-AUC, pick winner per CONSTITUTION V.b, then evaluate at fixed τ on test. Tag `backtest` (30-min budget). Compare round-001's post-hoc τ=0.30 to the val-chosen τ — if they coincide, the round-001 result holds; if they diverge, the round-001 numbers are demoted to "exploratory only" in REPORT.
- **Falsification**: val-chosen τ for p_offline lands inside the round-001 post-hoc grid {0.10..0.50}; if the val-Sharpe-grid optimum is outside this range, the round-001 sweep was insufficiently wide.
- **Status**: ACCEPTED round-009 — val tau\* = 0.44 (inside the round-001 grid {0.10..0.50}, falsifier-1 PASSES). Test Sharpe at val-chosen tau\* = -0.089 (PSR=0.458, p_boot=0.000 vs shuffled-signal null, DSR=0.000 after 26-tau-grid deflation). Round-001's PSR=0.995 at τ=0.30 demoted to "exploratory only" in REPORT.md headline. Strategy still beats random-entry-at-same-rate (model has SOME information) but does not produce tradable alpha after honest selection. Also: built `scripts/build_report.py` and `RESEARCH/REPORT.md` (canonical living dashboard, trading section leads, regenerates from per-round headline.json each round).
- **Cost**: medium (notebook surgery + 1 retrain)

### H-005c [P3] (spawned by round 001) CSCV PBO deflation of the H-005 τ sweep
- **STATUS: blocked / SUPERSEDED**. Original framing depended on `simulate_inventory_aware_sized` outputs from rounds that have been REVOKED (see HEALTH_OF_RESULTS). Re-framed: any `wagie experiment run` with `bootstrap.scheme: cscv_pbo` (sibling RIGOR — assumed merged post this commit's wave) gets PBO out of the box. Standalone H-005c is no longer needed.
- **Cost**: zero (CSCV is a built-in scheme via the spec).

### H-310-a [P5] Rolling-origin retraining harness (GENUINE GAP — Ask 3)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 3 — train-once -> rolling-retrain to remove the static-model artefact in every PnL number to date.
- **Mechanism**: today `artifacts/offline_model/model.cbm` is fit once on the 60% train slice and applied statically to val + test. A production system retrains on a rolling cadence; without it every Sharpe number is an upper bound on a stale model. Engineering scope: refit-cadence config in `wagie.training` (proposal: `refit_cadence_bars = 2160` ≈ 30 days at M=20; configurable) and `refit_window` ∈ {`expanding`, `sliding`}; CONSTITUTION I.1 holds at every refit (data ≤ T only; no future leakage); the `selected_features.json` top-120 must be **re-derived per refit**, not frozen; ensemble of 3 seeds per refit; one run per refit with `parent_run_id` linking to the rolling experiment. Artifact layout: `artifacts/offline_model/rolling/refit_NNNN/{model.cbm, model.{0..2}.cbm, selected_features.json, config_snapshot.json, metadata.json}` plus one consolidated `rolling/manifest.parquet` keyed by `(refit_id, t_start_ms, t_end_ms)` and the corresponding test-slice predictions parquet `rolling/predictions.parquet` so downstream rounds (H-310-b, etc) can reuse without re-running. Regime cuts on `parkinson_var_rolling_mean_24` are **fit on the very first refit's train+cal only and frozen** for all subsequent refits, to avoid drift in tercile boundaries leaking future state. A new module `wagie.training.rolling` exposes `RollingTrainer(config, schedule)` with a `.run() -> RollingArtifacts` method. **Note (post-conformal-removal)**: the original "conformal calibration window must roll with the model" subtlety is gone — there is no Mondrian-ACI `q` to warm. The ARF state itself rolls forward unbroken (the streaming contract holds across refits as long as the offline `p_offline` input is from the latest refit).
- **Predicted effect**: per-refit BSS Δ on the test slice ranges between +0.005 (small) and +0.02 (large) vs the static fit, with the gain concentrated in the high-vol tercile (where round-005 showed offline ECE 0.171 — most stale). Test Sharpe under rolling-retrain expected to shift by ≥ +0.05 with high variance across refits; confidence interval of the mean shift requires the per-metric bootstrap (H-320-a). Seed-noise band ≈ 0.003 BSS / 0.02 Sharpe (from Phase A's offline-only re-runs being bit-identical at the same seed).
- **Falsification**: rolling-retrain test Sharpe is **NOT** statistically distinguishable from static-train test Sharpe (block-bootstrap p_boot ≥ 0.10 on the paired difference, using H-320-a's stationary block bootstrap with Politis-White block length). If the falsifier triggers, drop the rolling-retrain hypothesis: the model's stale-fit drag is below the noise floor on this dataset.
- **References**: López de Prado, *AFML* (2018), local PDF `Downloads\Advances in Financial Machine Learning ... 2018 ... .pdf` Ch. 7 (purged CV / embargo) and Ch. 11 (backtesting). Sugiyama & Kawanabe, *Machine Learning in Non-Stationary Environments* (MIT 2012), local PDF `Downloads\(Adaptive Computation and Machine Learning series) ... 2012 .pdf` Ch. 1–3 (covariate-shift adaptation under refit). Hansen *Econometrics* (2022) Ch. 14 (rolling-origin evaluation).
- **Status**: queued (P5 — single highest-leverage operational gap on this loop's roadmap).
- **Cost**: medium-high (engineering ~3 hours; per-refit train ~2 minutes wall on 30-day windows × ~21 refits = ~45 min wall). Tag round `backtest` (30-min budget for the runner; the engineering is its own commit).
- **Dependencies**: must follow H-101 (chronological_split utility — accepted round-004) and H-102/H-103 (weighting / undef-flag) ideally land first so each refit uses the production weighting and feature pipeline. CSCV PBO across refits as a check is downstream (uses H-005c-style logic on the new manifest).

### H-310-b [P5] Rolling-retrain backtest measurement (GENUINE GAP — Ask 3, paired with H-310-a)
- **Owner**: IMPLEMENTER + CRITIC
- **Asks**: ask 3 — re-run the baseline (val-tau backtest) under rolling-retrain.
- **Mechanism**: load `artifacts/offline_model/rolling/predictions.parquet` (built by H-310-a); run a new YAML spec `experiments/baseline_with_offline_rolling.yaml` that points the engine at the rolling predictions. Three strategies under rolling-retrain: `threshold_gate(p_online)`, `threshold_gate(p_offline)`, `regime_gated`. Diff against the static-train numbers from `wagie experiment run experiments/baseline_with_offline.yaml` row-by-row (paired bootstrap p via sibling RIGOR's stationary-block-bootstrap on per-bar net log returns).
- **Predicted effect**: ΔSharpe ∈ [-0.05, +0.20] for `threshold_gate(p_offline)` vs static; ΔBrier ∈ [-0.005, -0.001] (i.e. small Brier improvement). The high-vol tercile's ΔECE expected to be the largest single calibration gain.
- **Falsification**: at least one rolling-retrain spec satisfies the accept gate (`predicted_effect_min` exceeded; `n_trades >= min_n_trades`) under bootstrap CI -> rolling retrain rescues edge. Otherwise the binding constraint is label/strategy/feature, not staleness.
- **References**: AFML Ch. 14 (deflated Sharpe under rolling). Politis & Romano (1994) — paired CI on ΔSharpe.
- **Status**: blocked on H-310-a. Once unblocked, this becomes the most consequential measurement round of the next quarter.
- **Cost**: medium (~30 min wall once -a is done).
- **Dependencies**: H-310-a (rolling predictions); sibling RIGOR (paired bootstrap CI).

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

## Tier 1 — Streaming online layer (asks 2 + 6) [post-conformal-removal]

> **TIER STATUS (2026-05-10)**: H-201..H-209 in this tier are SUPERSEDED. The
> conformal layer they refined has been removed. Calibration of `p_online` is now
> the single primary metric. H-201 (coverage diagnostic) and the ACCEPTED ACI cards
> (H-202, H-203) are kept for historical traceability; the rest are archived.
>
> Replacement actionable cards live in Tier 7 / Ask 2 (drift surfacing,
> `ThresholdGate(p_online)` calibration audits) and in the new spec catalog
> (`experiments/sweep_alpha.yaml`, `experiments/replay_r031_low_vol_gate.yaml`).

The online stage is a bagging-calibrated streaming layer with ADWIN drift
surfacing. Cards below were originally framed as conformal-layer refinements
and are SUPERSEDED unless explicitly noted otherwise.

### H-201 [P5] Baseline coverage diagnostic on the existing online ARF
- **Owner**: IMPLEMENTER + THEORIST
- **Mechanism**: treat the existing River `ARFClassifier` output as if it were a conformal predictor at varying confidence levels (set construction via `predict_proba_one`). Compute marginal empirical coverage AND per-regime coverage on the test split. Quantify *coverage gap* = 1 - α - empirical_coverage by regime. This becomes the baseline every online-stage hypothesis must beat.
- **Falsification**: report finite numbers; a > 5% absolute coverage gap in any regime is the gap to close.
- **Status**: ACCEPTED round-002 — chronological 30/70 cal/eval split on predictions.parquet with parkinson_var_rolling_mean_24 terciles. Finite numbers everywhere; constant-predictor sanity passes (α=0.05 cov=1.0000, α=0.20 cov=0.9181=1-eval_base_rate). Naive `p≥α` over-covers by 8-10pp at α=0.10 → ARF probs not directly usable as conformal thresholds. Mondrian LAC compresses α=0.10 gap to ±3.4pp; α=0.20 still has -7.9pp on low-vol/p_online and -5.8pp on low-vol/p_offline → THE gap H-202..H-208 must close. Spawned H-201b (finer-bin regime), H-201c (cal-fraction sensitivity).
- **Cost**: low (no model changes; just measurement)

### H-201b [P4] (spawned by round 002) Finer regime binning to expose what Mondrian-LAC misses at α=0.20
- **STATUS: SUPERSEDED** (post-conformal-removal). The script this card targeted (`scripts/round_002_coverage_baseline.py`) is gone, and the Mondrian-ACI/LAC layer it would refine has been removed. Replacement: a per-regime calibration audit on the wagie ARF output, expressed as a research round on `experiments/baseline.yaml` outputs (no separate spec needed; `metrics.json` already breaks down by regime).

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

### H-204 [P4] River ARF vs SRP vs HAT (refit as a calibration audit, not a coverage audit)
- **Status: actionable (REFRAMED)**. Original framing was conformal-coverage; under the post-conformal-removal architecture the comparison reduces to: which streaming learner gives the best per-regime Brier on `p_online` at fixed compute budget? Falsifier becomes "ARF Brier - HAT Brier > seed-noise band". The round-024 result (ARF dominates HAT on Brier and ECE; HAT 2.8x faster) under the legacy harness is keepable in principle. Re-run via a new YAML spec that selects the streaming kind (default ARF; HAT if `wagie.model.streaming_kind: hat` — sibling STREAMING owns this knob).
- **Cost**: medium (when a `streaming_kind` knob exists).

### H-205 [P4] Locally-weighted conformal (kernel-local validity)
- **Status: SUPERSEDED** (no conformal layer to weight).

### H-206 [P4] CatBoost virtual-ensemble σ_epistemic as a conformal feature
- **Status: SUPERSEDED** as written (depends on the conformal layer). Reframed as H-302 (σ-epistemic *abstention*) and H-209 (Cantelli sized entries) below; both gate decisions, not conformal sets. The σ_epistemic feature itself remains useful — see those replacement cards.

### H-207 [P3] ADWIN drift detector triggers ACI threshold reset
- **Status: SUPERSEDED**. ARF already wraps ADWIN per-tree; sibling STREAMING surfaces the top-level ADWIN signal on the LAC score `s_t = 1 - p_online(y_t | x_t)` and emits a `DRIFT` event in `metrics.json`. There is no `q_t` to reset; the drift signal goes to logging + human review (no auto-retrain in v1).

---

## Tier 2 — Conformal coverage applied to backtest decisions (ask 4)

### H-011 [P5] Conformal LAC + Mondrian for trade abstention (offline)
- **Status: SUPERSEDED**. The conformal layer is gone; abstention is now controlled by `ThresholdGate(p_online >= tau)` and (optionally) `RegimeGated`. See `experiments/sweep_alpha.yaml` for the τ sweep that replaces this card's accept question.

### H-208 [P4] Streaming conformal trade gate (online ACI variant)
- **Status: SUPERSEDED**. Replaced by `ThresholdGate(p_online >= tau)` + sibling RIGOR's accept-gate. The "trade gate" responsibility lives in the strategy registry, not the (removed) conformal layer.

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

## Tier 7 — Round-016 mandate cards (override §4, 8-axis re-anchor)

Round-016 (research-planning) added the H-1xx cards below per user override.
Cards are cross-referenced into the H-3xx series from round-014; relationships
documented in `RESEARCH/research_plan_round_016.md` §3. The four-phase ordering
(Phase A rounds 017–019 strategy escalation; Phase B rounds 020–023 bootstrap +
production refactor; Phase C rounds 024–028 online ensembling + drift retrain;
Phase D rounds 029–034 features + postmortem) supersedes the round-014 next-8
ordering.

**Forbidden across all cards in this Tier (and elsewhere)**: averaging or
convex-blending `p_offline` and `p_online`; training a linear regression on
`(p_offline, p_online)`; driving Mondrian-ACI off `p_offline`. Round-015
contract; round-010..013 KILL_LIST history.

### Ask 1 — Invariant features

#### H-130 [P4] Derived flow features at 20m bars (NEW)
- **Owner**: IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 1
- **Mechanism**: Binance kline already exposes `taker_buy_base` and `taker_buy_quote` (in `data_download.ipynb` output). At 20m cadence construct: `taker_buy_ratio_20m = sum(taker_buy_base) / sum(volume)` (the bar-aggregated VPIN imbalance per AFML 18.8.4 form `|2v^B − 1|`); `signed_dollar_flow = sum(taker_buy_quote) − sum(quote_volume − taker_buy_quote)`; `signed_vwap_dev = signed_dollar_flow / sum(quote_volume)`. Derived flow features over rolling windows `{2,4,8,12,24,48,96}` decision bars. Strict CONSTITUTION I.6 — value at boundary k uses minute bars `n ≤ n_k`. The optional later upgrade — top-of-book snapshots from Binance Futures REST — is a separate `data_download` round and explicitly NOT in scope here.
- **Predicted effect**: ΔBSS ∈ [+0.002, +0.005], concentrated in mid-vol parkinson-tercile (offline ECE 0.10) where directional flow is most informative; high-vol tercile dominated by jump component already; low-vol gain ≈ 0. Seed-noise band ≈ 0.0008 BSS.
- **Falsification**: paired offline retrain with-vs-without H-130 features; require ΔBSS ≥ 0.0008 (1.5σ_seed) AND CatBoost importance rank top-50 of 726 features. Below threshold → kill.
- **References**: López de Prado, *AFML* (2018) Ch. 17–19, local PDF `Downloads\Advances in Financial Machine Learning ... Anna's Archive.pdf` pp. 281-292 (tick rule p. 281-282, Kyle's λ p. 287-288, Hasbrouck p. 289, VPIN/OFI p. 291-292; VPIN form §18.8.4 p. 276). `Downloads\Binance BTC_USDT Derivatives Data & Feature Engineering.pdf` pp. 4-6 (explicit `flow__taker_buy_ratio`, `flow__net_vol_btcs`, `flow__net_vol_csum_5m` schema). Bieganowski & Slepaczuk (2026) *Explainable Patterns in Cryptocurrency Microstructure* arXiv 2602.00776 (web; INDEX add `[MICROSTRUCT]`) — OFI / spread / VWAP-to-mid as portable cross-crypto SHAP-leading features.
- **Status**: queued.
- **Cost**: medium (~1.5 hours engineering + tests + paired offline retrain ~30 min).
- **Dependencies**: H-103 undef-flag pattern; new `compute_derived_flow_features` registered after H-103 lands.

#### H-131 [P4] BPV/RV jump-detector + signed-semivariance asymmetry + vov (REFINES H-114, H-312)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 1
- **Mechanism**: refine H-114 (vol decomposition at level) and H-312 (BPV ratio across windows) into one card framing: (i) BPV/RV ∈ (0,1] as continuous-vs-jump regime — `BPV = (π/2)·Σ |r_i|·|r_{i-1}|` (Barndorff-Nielsen & Shephard 2004); 1−BPV/RV is jump-share. (ii) `SV_up − SV_down` as direction-aware variance, mechanically aligned with one-sided upper-barrier label (Patton & Sheppard 2015) — strict t ≤ k−1 to prevent leakage (the same bar contributes to both label and SV otherwise). (iii) vol-of-vol (rolling std of rolling vol). The **ratio** BPV/RV is dimensionless and regime-invariant under multiplicative SV — different from H-114's level. The label one-sidedness gives `SV_up − SV_down` mechanical alignment with label sign — informative but partly tautological; AFML Ch. 5 fractional differentiation residualises the contemporaneous coupling.
- **Predicted effect**: SV asymmetry ΔBSS +0.003 to +0.008 (mechanically aligned, concentrated in high-vol tercile); BPV ratio ΔBSS +0.001 to +0.003 marginal over level; vov +0.001. Total ~0.005–0.010 net of redundancy with H-114. Seed-noise ≈ 0.0008 BSS.
- **Falsification**: SV asymmetry partial-dependence flat OR sign opposite to forward excursion in ≥1 of 3 vol terciles → leakage suspect, kill. BPV ratio ΔBSS over BPV-level baseline ≤ 0.0008 → drop ratio. Failure mode under simultaneous price+volume jumps (BTC liquidation cascades): BPV staggered-product `|r_i||r_{i-1}|` itself spikes; use truncated-power variation (Mancini 2009) or impose lag ≥ 2 minutes. **Multiscale Stochastic Volatility book has no BPV chapter** — sub-agent verified; do NOT cite for BPV.
- **References**: Barndorff-Nielsen & Shephard (2004) "Power and bipower variation with stochastic volatility and jumps" *J. Financial Econometrics* 2(1):1-37 DOI `10.1093/jjfinec/nbh001` (web; INDEX add `[VOL]`). Patton & Sheppard (2015) *Rev. Econ. Stat.* 97(3):683-697 (web; INDEX add `[VOL]`). Barndorff-Nielsen-Kinnebrock-Shephard (2010) realized semivariance (web; INDEX add). AFML Ch. 5 (frac-diff) local PDF.
- **Status**: queued; primary card; H-114 + H-312 kept as queued for granular ablation if H-131 fails.
- **Cost**: medium (~2 hours engineering + tests + paired retrain).
- **Dependencies**: H-103 undef-flag pattern.

#### H-132 [P3] Cross-asset (ETH/BTC) features (NEW)
- **Owner**: IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 1
- **Mechanism**: separate `data_download` round fetches ETHUSDT 1m for the same date range. Construct at 20m cadence: `eth_btc_logret_corr_24` (rolling Pearson over 24 1m bars), `eth_btc_basis_proxy = log(ETH_close / BTC_close) − rolling_mean_96`, `eth_resid_20m` (residualised ETH return after single-factor projection on BTC per Liu & Tsyvinski 2021 RFS — the residual carries asset-specific information beyond the common factor). **Hard requirement**: drop the bar entirely if `ETH.close_time > BTC.close_time` for that bar; do NOT forward-fill (would silently introduce lag-1 staleness violating CONSTITUTION I.6). Cost-flag: triggers feature realignment per CONSTITUTION I.6.
- **Predicted effect**: ΔBSS ∈ [+0.001, +0.004], concentrated in high-vol tercile (where ECE = 0.17 hurts most) via correlation-spike feature; marginal over BTC autocorrelation small (BTC absorbs ~70% of ETH 1m signal). Lead-lag direction is regime-dependent (ETH leads in risk-off; BTC leads in 2024 ETF-era rallies) — non-stationary.
- **Falsification**: paired offline retrain require ΔBSS ≥ 0.005 lift AND positive lift in ≥2 of 3 vol terciles (per AFML Ch. 8 MDA on residualised feature). Below threshold → drop entire block (parsimony beats marginal gain when introducing observation-time risk).
- **References**: Liu & Tsyvinski (2021) "Risks and Returns of Cryptocurrency" *Rev. Fin. Stud.* 34(6):2689-2727 DOI `10.1093/rfs/hhaa113` (web; INDEX add `[CROSS-ASSET]` new tag). Alexander, Heck, Kaeck (2022) "The role of binance in bitcoin volatility transmission" *Applied Mathematical Finance* 29(1) (web; INDEX add `[CROSS-ASSET]`). AFML Ch. 8 (feature importance — local PDF).
- **Status**: queued. New INDEX tag `[CROSS-ASSET]`.
- **Cost**: medium-high (separate `data_download` round + offline retrain ~1.5 hours total).
- **Dependencies**: separate ETHUSDT data download.

### Ask 2 — σ_epistemic → decisions

#### H-209 [P4] Cantelli-bound sized entries via virtual-ensemble σ (NEW)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 2
- **Mechanism**: `cantelli_decision_rule` is already implemented in `src/uncertainty.py`. Wire into the backtest: open size = `clip(p_lower_cantelli − τ, 0, max_size)` where `p_lower = max(0, p̂ − k·σ)` from Cantelli's inequality at confidence `1 − 1/(1+k²)`. Sweep `k ∈ {0, 0.5, 1.0, 1.5, 2.0}` on validation Sharpe with `n_trades ≥ 50` stability filter; apply val-chosen `k*` to test once. Bailey-Borwein DSR over the joint `(τ, k)` grid; CSCV PBO via H-192 helper. **CITATION CORRECTION** (round-016 §4.1): the `(p − q̂)/(1 − q̂)` confidence-margin sizing referenced elsewhere comes from Vovk-Gammerman-Shafer (2005) Ch. 3 / Angelopoulos & Bates (2021) §2.2 (arXiv 2107.07511) — NOT Lekeufack §V-C (which is buy/short/abstain on interval-straddle).
- **Predicted effect**: at val-chosen `k*` ≠ 0, ΔSharpe +0.05 to +0.20 vs binary `baseline_offline_tau` if σ is informative; multi-strategy DSR > 0.95 only if k\* ≠ 0 and lift survives `(τ × k)` deflation. Seed-noise ≈ 0.02 Sharpe.
- **Falsification**: `k* = 0` wins the val grid (no σ-dependence) OR ΔSharpe at val-chosen `k*` ≤ +0.05 → sizing is a tax on a non-edge; kill.
- **References**: Cantelli's inequality (textbook, Wainwright High-Dim Stats Ch. 1 — local Desktop). Malinin, Prokhorenkova, Ustimenko (2021) ICLR arXiv 2006.10562 (INDEX). Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction" arXiv 2107.07511 (web; INDEX add). `src/uncertainty.py::cantelli_decision_rule` already implemented.
- **Status: actionable** (P4). Distinct from H-302 (σ-abstention). H-25's "Mondrian-ACI margin sizing" axis is SUPERSEDED (no Mondrian-ACI). Sizing axis under the new architecture: `EvCalibratedSize(k * margin)` using bagging-derived `p_online` directly OR the Cantelli p_lower margin via H-209.
- **Cost**: medium (~2 hours; no retrain — uses persisted offline model).
- **Dependencies**: H-108 (CatBoostEnsemble — accepted round-003); H-190 (block-bootstrap CI on Sharpe lift).

#### H-210 [P3] Virtual-ensemble agreement gate (NEW)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 2
- **Mechanism**: when n_virtual virtual-ensemble predictions disagree, abstain. Specifically: for each test bar compute `n_above_tau = sum(virtuals > τ)`; abstain if `n_above_tau ∈ [0.25 · n_virtual, 0.75 · n_virtual]` (the "virtuals straddle τ" zone). CatBoost-side analog of the Mondrian-ACI prediction-set abstention. Orthogonal to σ-magnitude — measures *disagreement rate*, not σ.
- **Predicted effect**: paired Brier improvement on the gated subset > seed-noise band (σ_seed ≈ 0.001 Brier per regime). At fixed trade rate, expect Brier reduction ≈ 0.003.
- **Falsification**: gated-subset Brier ≤ ungated-subset Brier within seed-noise band → agreement is uninformative; kill.
- **References**: Geifman & El-Yaniv (2017) selective classification arXiv 1705.08500 (INDEX). Malinin et al. 2021 (INDEX).
- **Status**: queued (P3). Pair with H-209 once H-209 picks `k*`.
- **Cost**: low (~1 hour; uses persisted virtual-ensemble outputs from H-108).
- **Dependencies**: H-108.

### Ask 3 — Walk-forward retraining

#### H-150 [P5] Walk-forward retraining schedule (REFINES H-310-a)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 3
- **Mechanism**: every K = 1500 decision bars (≈ 20 days at M=20; the override §4 tightening of H-310-a's K=2160), retrain CatBoost from scratch on `[t0, t-1]` with same hyperparameters (no HPO until H-105 lands), persist as `model_window_NNN.cbm`, generate `p_offline` forward through next K bars. Online ARF does not reset; bookmarked "model changed at bar N" event lets the streaming layer read the boundary. Engineering harness lives in H-310-a; H-150 specifies the cadence and falsifier.
- **Predicted effect**: per-refit ΔBSS ∈ [+0.005, +0.02] vs static fit; gain concentrated in high-vol tercile (round-005 offline ECE 0.171); test Sharpe shift +0.05 to +0.20 across refits with high variance. Seed-noise band ≈ 0.003 BSS / 0.02 Sharpe.
- **Falsification**: walk-forward test Brier ≥ frozen-model test Brier minus seed-noise band (i.e., retraining must not cost calibration globally; if it does, K is too small — re-grid up).
- **References**: López de Prado, *AFML* (2018) Ch. 7 (purged CV) — local PDF. Sugiyama & Kawanabe (2012) Ch. 1–3 (covariate-shift adaptation) — local PDF. Microsoft Qlib walk-forward template `https://github.com/microsoft/qlib` (web; engineering pattern reference).
- **Status**: queued (P5). Refines H-310-a's `refit_cadence_bars`.
- **Cost**: medium-high (per-refit train ~2 min × ~21 refits ≈ 45 min wall; engineering in H-310-a).
- **Dependencies**: H-310-a engineering harness; H-320-a / H-190 paired CI on ΔBSS.

#### H-151 [P4] Drift-triggered retrain via PageHinkley + KSWIN on running LAC score (NEW)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 3
- **Mechanism**: schedule-based (H-150) is necessary but not sufficient. Stream the running LAC score `s_t = 1 − p̂(y_t | x_t)` on the labeled stream into a PageHinkley detector AND a KSWIN detector (Kolmogorov-Smirnov windowing — captures distributional shift, not just mean shift). On either signal, force ahead-of-schedule retrain. ADWIN already in River and used inside the ARF — that monitors per-tree error; this card adds a top-level monitor on the LAC score directly.
- **Predicted effect**: PageHinkley + KSWIN signals correlate (Spearman > 0.4) with subsequent paired Brier degradation on the next 100 bars.
- **Falsification**: Spearman < 0.4 → trigger is a noise generator; kill.
- **References**: Bifet & Gavaldà (2007) ADWIN (INDEX). Raab et al. (2020) "Reactive Soft Prototype Computing for Concept Drift Streams" (KSWIN) Neurocomputing (web; INDEX add). Mansour Zoubeirou A Mayaki (2022) autoregressive drift (web).
- **Status**: queued (P4).
- **Cost**: medium (~3 hours engineering + tests).
- **Dependencies**: H-150 as on-schedule complement; H-207 (ADWIN-triggered q_t reset) pairs.

#### H-152 [P3] Post-retrain feature-importance audit (HOUSEKEEPER-grade)
- **Owner**: HOUSEKEEPER
- **Asks**: ask 3
- **Mechanism**: when a retrain fires (H-150 or H-151), compare top-50 feature importances against the previous retrain. If ≥ 5 features rank-flip out of the top 50, log a `FEATURE_IMPORTANCE_DRIFT` event. Feeds into pruning decisions; does not gate retraining.
- **Predicted effect**: zero (audit card; the value is the diagnostic).
- **Falsification**: drift events do not correlate (Spearman > 0.3) with subsequent test Brier degradation → audit signal is noise; demote.
- **References**: AFML Ch. 8 (feature importance — local PDF).
- **Status**: queued (P3).
- **Cost**: low.
- **Dependencies**: H-150 / H-151.

### Ask 4 — Asymmetric weighting

#### H-153 [P4] Tail-truncated Huber-saturated asymmetric weighting (REFINES H-303)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 4
- **Mechanism**: define `w_i = min(w_max, (loss_i / median_loss)^β)` with `w_max = 5` (CRITIC-checkable cap, prevents tail from dominating) and `β = 1.0` (linear above median). On the upper-barrier label, `loss_i` for false negatives = foregone PnL; for false positives = realized stop-loss-minus-cost. Strict `w_max = 5` cap per Cont (2001) stylized facts: BTC log-returns have α ≈ 3-4 power-law tail; uncapped tail upweighting blows up generalization. Pair with H-020 (CVaR-α tail) only if H-303's saturated linear weighting plateaus.
- **Predicted effect**: paired offline test ECE drops ≥ 0.005 in high-vol tercile; mean p shifts from 0.205 toward 0.097 (base rate). NOT just paired ROC-AUC rise (CONSTITUTION IV: ROC-AUC alone never accepts).
- **Falsification**: paired offline test ECE shift ≤ 0.001 (within 0.5σ_seed) AND mean p movement ≤ 0.02 → weighting carries no marginal information; kill.
- **References**: Huber (1964) "Robust estimation of a location parameter" *Ann. Math. Stat.* 35:73 (web). Wainwright *High-Dim Stats* Ch. 1-2 (M-estimation under heavy tails) local Desktop. Cont (2001) "Empirical properties of asset returns: stylized facts and statistical issues" *Quantitative Finance* 1:223 (web).
- **Status**: queued (P4); narrows H-303 to one specific saturated form. H-303's 27-cell ablation grid kept as queued.
- **Cost**: medium (FAST_MODE ~1 hour).
- **Dependencies**: H-103 undef-flag, H-102 sibling weight import.

### Ask 5 — Strategy improvement

#### H-160 [P5] First-touch (true triple-barrier) label retrain (LIFTS H-023)
- **Owner**: IMPLEMENTER + THEORIST + LITERATURE-SCOUT
- **Asks**: ask 5
- **Mechanism**: re-run `feature_build.ipynb` and `offline_train.ipynb` with LdP triple-barrier label `1[upper_barrier_hit_first within M_horizon]` instead of current `1[max excursion ≥ α]`. Horizon `M_horizon` matches backtest's `c_stop` lookahead. Reference implementation: AFML Snippets 3.3-3.5 pp. 48-50 (`getEvents`, `applyPtSlOnT1`, `getBins`); symmetric horizontal barriers (`ptSl=[ptSl,ptSl]`) match this project's symmetric backtest. THEORIST estimate: at α=0.00411 with 20-min horizon, BTC 1m σ̂ ≈ 4-8bp → barriers near ±1σ of horizon RV; expected label flip rate 15-25% of bars, mostly upper-then-lower paths currently y=1 should-be y=0 under first-touch. AFML §3.9 dropLabels procedure (p. 54) drops minPct < 5% classes if vertical-barrier hits dominate.
- **Predicted effect**: ΔSharpe +0.10 to +0.30 on test, mostly via reduced FP density at operating threshold. At val-chosen τ\* under triple-barrier label, test Sharpe > 0.5 with stationary-block-bootstrap p_boot < 0.05 (block ≈ 24 bars per H-190) AND CSCV PBO < 0.5 (per H-192).
- **Falsification**: at val-chosen τ\* under triple-barrier label, test Sharpe ≤ 0 OR p_boot ≥ 0.10 OR CSCV PBO ≥ 0.5 → model genuinely has no edge at this horizon; Phase D pivots to feature research (H-130/131/132). Or: retrained model Brier improvement ≤ 0.5% → path-asymmetry signal isn't there.
- **References**: AFML Ch. 3.4-3.6 pp. 45-51 (local PDF — Snippets 3.3 + 3.4 + 3.5). Hudson & Thames blog `https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/` (web; INDEX add). mlfinlab `https://github.com/hudson-and-thames/mlfinlab` (web; INDEX add — `mlfinlab.labeling.labeling.get_events` matches Snippets 3.3-3.4 verbatim). nkonts/barrier-method `https://github.com/nkonts/barrier-method` for vectorized variant.
- **Status**: queued (P5; lifts H-023). Runs only if H-024 (no-stop) does NOT deliver ΔSharpe > +0.5 with p_boot < 0.05.
- **Cost**: high (full retrain ~45 min budget; tag `backtest`).
- **Dependencies**: H-024 first-pass result; H-190 stationary block bootstrap; H-192 CSCV.

#### H-161 [P5] Meta-labeling per LdP §3.6 (LIFTS H-022; CRITICAL ARCHITECTURAL PIN)
- **Owner**: IMPLEMENTER + THEORIST + LITERATURE-SCOUT
- **Asks**: ask 5
- **Mechanism**: secondary CatBoost on `(features ⊕ p_online)` predicting `1[trade_was_profitable_after_costs]` on the subset where `p_online > τ_lower`. Primary model emits the side; meta-model emits bet/no-bet. **CRITICAL ARCHITECTURAL PIN**: meta-model input is `(features ⊕ p_online)` ONLY — `p_offline` is strictly absent. Reason: round-015 contract — `p_online` already conditions on `p_offline` by construction; feeding both regresses child onto parent's parent, recreating the forbidden combiner pathology (round-014 §4.3 / round-016 §4.3 sub-agent correction). AFML Snippet 3.7 pattern: when `side` provided, `out['ret'] *= side`, `bin = 0 if ret ≤ 0`. Probability of secondary class-1 → bet **size** (AFML §3.7 pp. 52-53). Meta-train and meta-eval temporal splits with purging (AFML Ch. 7).
- **Predicted effect**: hit-rate +2-5pp; ΔSharpe +0.15 to +0.40. Subset-size floor: ~150 positive trades (below this, secondary classifier overfits) — round-015 had ~700 trades total at τ\*=0.44 with ~300 positives; n_trades floor ≈ 500 (≥150 positives) over the meta-training window.
- **Falsification**: 5×2 nested CV meta-Brier no better than `p_online`-only logistic baseline (Δ ≤ 0.005) → meta-labeling adds friction; kill. Also: gated trade subset precision NOT higher than ungated → meta-model is uninformative; kill. n_positives < 150 in any meta-train fold → statistically inadmissible.
- **References**: AFML Ch. 3.6-3.8 pp. 50-53 (local PDF; Snippets 3.7 + 3.8). Hudson & Thames blog (web; INDEX add). mlfinlab labeling docs (web; INDEX add). Wikipedia "Meta-Labeling".
- **Status**: queued (P5; lifts H-022). Runs only after a positive base strategy emerges from H-024 / H-160.
- **Cost**: high (full secondary-model train; tag `backtest`).
- **Dependencies**: H-024 OR H-160 producing positive base; H-190 / H-192.

### Ask 6 — Online ensembling

#### H-170 [P4] Online stacking via streaming logistic-regression meta-learner (NEW)
- **Owner**: IMPLEMENTER + THEORIST + LITERATURE-SCOUT
- **Asks**: ask 6
- **Mechanism**: per-bar, four base online learners produce probabilities — `[ARFClassifier (current), SRPClassifier, HoeffdingAdaptiveTreeClassifier, river.linear_model.LogisticRegression]`, each consuming `selected_features ⊕ p_offline`. A fifth-stage River `LogisticRegression` (Adam optimizer or plain SGD) consumes the **four base probabilities AS its input features** (NOT raw features — Wolpert 1992 standard) and learns to combine them. Online stacking; meta-learner is itself online; combination weights adapt as bases drift. Mondrian-ACI on top of the meta-output preserves coverage (post-hoc on any score function). **Distinct from H-305**: H-170 is a learned meta-learner; H-305 is a Hedge-update controller with regime-conditional weights.
- **Predicted effect**: paired BSS on test ≥ +0.01 over best single base. Per-regime Brier of meta lower than min over individual bases by Δ ∈ [-0.001, +0.005] (Hedge-style regret bound `R_T ≤ √(T ln K / 2)`).
- **Falsification**: paired BSS Δ ≤ +0.005 (within 6× seed-noise σ_seed ≈ 0.0008) → bases too correlated, online stacking adds nothing; kill.
- **References**: Wolpert (1992) "Stacked generalization" *Neural Networks* 5:241 (web; INDEX add). Ting & Witten (1999) "Issues in Stacked Generalization" *JAIR* 10:271 (web; INDEX add). Pesaranghader (2017) arXiv 1709.02457 (web). Wozniak et al. (2014) Bayesian online ensembles (web). abuyukcakir/gooweml `https://github.com/abuyukcakir/gooweml` (web; GOOWE-ML pattern).
- **Status**: queued (P4).
- **Cost**: medium (~3 hours engineering + tests; per-run wall ~5 min).
- **Dependencies**: H-204 (River expert comparison) shortlists the K=4 bases.

#### H-171 [P3] Bayesian Model Averaging weights, online (NEW)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 6
- **Mechanism**: maintain online posterior weights `w_t ∝ exp(−cumulative_log_loss_t)` per base learner with exponential forgetting `γ ∈ [0.95, 0.999]`. Simpler-but-weaker companion to H-170; if H-170 ships, H-171 is a fast-fail comparator deciding whether the learned meta is worth its complexity. **Note explicitly**: BMA collapses to the best single model under non-stationarity — this is a feature, not a bug, when one base genuinely dominates.
- **Predicted effect**: BMA performs equal to or worse than H-170 by paired BSS Δ ∈ [-0.005, 0]. Gives a clean "is the meta-learner adding non-trivial structure" diagnostic.
- **Falsification**: BMA performance within 1σ of H-170 → meta-learner adds no non-linear structure; H-170 is overfitting.
- **References**: Raftery, Gneiting et al. (2005) "Using Bayesian Model Averaging to Calibrate Forecast Ensembles" *Monthly Weather Rev.* 133:1155 (web; INDEX add). Sloughter et al. (2007) (web). "Bayesian Ensembling: Insights from Online Optimization and Empirical Bayes" arXiv 2505.15638 (web).
- **Status**: queued (P3); fast-fail comparator to H-170.
- **Cost**: low (~1 hour after H-170 lands).
- **Dependencies**: H-170 (runs immediately after as comparator).

### Ask 7 — Per-trade postmortem

#### H-180 [P4] Per-trade attribution dataframe (NEW; substrate for Ask 7)
- **Owner**: IMPLEMENTER
- **Asks**: ask 7
- **Mechanism**: extend the wagie `SimBroker` / `Portfolio` to emit a per-trade `parquet` (`artifacts/runs/<run_id>/per_trade.parquet`) with columns: `entry_bar`, `exit_bar`, `entry_p_offline`, `entry_p_online`, `entry_sigma_epistemic`, `entry_regime`, `realized_pnl_bp`, `bars_held`, `exit_reason ∈ {profit_barrier, stop_barrier, timeout}`, `cost_paid_bp`. Substrate for everything else in Ask 7. The H-306 spawning rule (round-014) ingests this output. `entry_q_t` is no longer applicable (no Mondrian-ACI layer); the column is dropped.
- **Predicted effect**: zero (data-emission round; the value is the substrate).
- **Falsification**: trivial; accepts on schema correctness + smoke test (every column populated, finite, `exit_reason` ∈ valid set, n_trades matches `metrics.json::trading.n_trades`).
- **References**: AFML Ch. 14 (backtest statistics — local PDF); existing `Trade` dataclass in `src/backtest.py`.
- **Status**: queued (P4).
- **Cost**: low (~1 hour engineering).
- **Dependencies**: none.

#### H-181 [P4] SHAP attribution on losing trades (NEW; Ask 7 diagnosis)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 7
- **Mechanism**: on the losing-trade subset (per-trade `pnl_bp < 0` from H-180), compute SHAP values for the offline CatBoost prediction at entry. Report top-10 features whose SHAP-at-loss differs from SHAP-at-win by Mann-Whitney U p < 0.01 (Bonferroni or BHY-FDR over the 726 features). Identifies features the offline model is most miscalibrated on. Feeds into H-130 / H-131 / H-132 prioritization for Phase D.
- **Predicted effect**: at least three features show p < 0.01 with effect sign consistent across two non-overlapping windows.
- **Falsification**: zero features pass p < 0.01 OR effect sign inconsistent across windows → SHAP-at-loss is noise; kill.
- **References**: SHAP `https://shap.readthedocs.io/` (web; INDEX add). Bieganowski & Slepaczuk (2026) arXiv 2602.00776 (web; INDEX add) for the SHAP-on-microstructure pattern.
- **Status**: queued (P4).
- **Cost**: medium (SHAP on 31,486 test predictions ~10 min on CPU).
- **Dependencies**: H-180.

#### H-182 [P3] Conditional Sharpe by feature regime (NEW; Ask 7)
- **Owner**: IMPLEMENTER
- **Asks**: ask 7
- **Mechanism**: per `parkinson_var` tercile × `Hurst` tercile (when H-040 lands), compute Sharpe with stationary-block-bootstrap CI (using H-190). Strategy-side analog of `calibration_by_regime`. 9 cells × Sharpe + CI per cell.
- **Predicted effect**: at least one regime cell has Sharpe CI not crossing zero (after Bonferroni or BHY-FDR over 9 cells) → that cell becomes a candidate for regime-conditional sizing.
- **Falsification**: every cell's Sharpe CI crosses zero after BHY-FDR → no regime-conditional alpha; kill regime-conditional sizing axis.
- **References**: AFML Ch. 14 (local). Hurst/DFA in Multiscale Stochastic Volatility (local — multi-scale OU only, NOT BPV per round-016 §4.2 correction).
- **Status**: queued (P3).
- **Cost**: low.
- **Dependencies**: H-180, H-190, H-040.

### Ask 8 — Bootstrap suite

#### H-190 [P5] Stationary block bootstrap for Sharpe / Sortino / max DD (NEW; subsumes part of H-320-a)
- **Owner**: IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 8
- **Mechanism**: implement stationary-block-bootstrap with auto-block-length per Politis & White (2004) plug-in: `b_opt = ((2 ĝ²(0)) / G²)^(1/3) N^(1/3)` where ĝ uses the flat-top window `λ(x) = 1 for |x| ≤ 1/2; 2(1−|x|) for 1/2 < |x| ≤ 1; 0 else`; M chosen by Politis-White rule. Produces valid CIs for ratio-of-means trading metrics under weak dependence + Hadamard-differentiable functionals (delta-method composition: vdV&W §3.9 Thm 3.9.11 + Künsch 1989 Thm 3.1). **No silent fallback** to plain bootstrap if assumptions violated — raise (pinned by test).
- **Predicted effect**: on shuffled-iid data, CI matches IID bootstrap to within 5%; on autocorrelated synthetic, widens correctly with ACF. On real BTCUSDT 1m PnL with heavy tails + regime breaks, CI may under-cover by 10-25% (THEORIST). Max-DD CI under-covers more severely (~30%) — path functional, not Hadamard-differentiable.
- **Falsification**: on synthetic IID, CI half-width differs from IID-bootstrap by > 5% → broken. On synthetic GARCH(1,1) + Student-t(ν=4) calibrated to BTCUSDT, empirical 95% coverage < 0.85 → block-length plug-in unstable; raise rather than silently return.
- **References**: Politis & Romano (1994) JASA 89:1303 (INDEX). Politis & White (2004) Econometric Reviews 23:53 (INDEX); Patton-Politis-White (2009) correction (INDEX). Künsch (1989) "The jackknife and the bootstrap for general stationary observations" *Annals of Statistics* 17:1217 (web; INDEX add — block-bootstrap consistency theorem). van der Vaart & Wellner Ch. 3.6 + 3.9 (local Desktop pp. 345-358 + 372-387; delta-method composition). AFML Ch. 14 pp. 195-210 (local).
- **Status**: queued (P5); subsumes Sharpe / Sortino / CDaR scope of H-320-a.
- **Cost**: medium (~3 hours engineering; tests are bulk).

#### H-191 [P4] Per-metric bootstrap for ROC-AUC / PR-AUC / calibration-curve (NEW)
- **Owner**: IMPLEMENTER + LITERATURE-SCOUT
- **Asks**: ask 8
- **Mechanism**: ROC-AUC: DeLong (1988) closed-form variance via Sun & Xu (2014) O((n+m) log(n+m)) — asymptotically valid even under temporal dependence (rank statistic concentrates at O(n^{-1}) under stationary mixing per Hoeffding 1948 + Yoshihara 1976). PR-AUC: stratified bootstrap (positives + negatives separately) per Boyd-Eng-Page 2013, B ≥ 1000. Calibration: per-bin Wilson-score interval (NOT bootstrap — per-bin n is small; Wilson is closed-form on binomial). ECE: debiased ECE_sweep estimator per Roelofs et al. 2022 with stratified bootstrap on samples within predicted-prob bins.
- **Predicted effect**: synthetic-iid ROC-bootstrap-CI matches DeLong CI within 0.005. PR-AUC naive bootstrap under-covers ~40% at n_pos:n_neg = 1:9; stratified fixes to <10% miscoverage. Calibration bootstrap CI is non-monotone across bins (jagged) and 10-20% wider than Wilson at small n_bin.
- **Falsification**: synthetic-iid ROC-bootstrap-CI half-width differs from DeLong by > 0.005 → broken. Wilson interval mismatches `statsmodels.stats.proportion.proportion_confint` to 1e-12 → numerical bug.
- **References**: DeLong et al. (1988) Biometrics 44:837 (INDEX). Sun & Xu (2014) IEEE SPL 21:1389 (INDEX). Boyd-Eng-Page (2013) ECML PKDD LNAI 8190:451 (INDEX). Niculescu-Mizil & Caruana (2005) ICML (INDEX). Roelofs et al. (2022) AISTATS PMLR 151 (INDEX).
- **Status**: queued (P4); subsumes ROC / PR / calibration scope of H-320-a.
- **Cost**: medium (~2 hours).

#### H-192 [P3] CSCV PBO refinement (REFINES H-113)
- **Owner**: IMPLEMENTER + CRITIC
- **Asks**: ask 8
- **Mechanism**: refine H-113 (already accepted via round-012 → round-015 cleanup; `cscv_pbo` in `src/backtest.py`). Pin S = 16 partitions (Bailey et al. default), report PBO histogram + per-partition (IS-rank, OOS-rank) scatter, tag any backtest round with `cscv_pbo` for traceability. Round-015 outputs (`cscv_per_strategy.csv`, `cscv_cross_strategy.json`) are canonical reference.
- **Predicted effect**: zero (refinement card; the value is procedural standardisation).
- **Falsification**: existing `cscv_pbo` helper produces PBO ≠ 0.000 cross-strategy on round-015 inputs → regression bug.
- **References**: Bailey, Borwein, López de Prado, Zhu (2014) `https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf` (web; INDEX confirm). pypbo `https://github.com/esvhd/pypbo` (web).
- **Status**: queued (P3); H-113's refinement.
- **Cost**: low.

#### H-193 [P4] Per-trade-level bootstrap with overlap embargo (NEW)
- **Owner**: IMPLEMENTER + THEORIST
- **Asks**: ask 8
- **Mechanism**: trades that overlap in time have correlated returns (LdP "trade uniqueness"). At val-τ=0.44 with mean bars_held ≈ 12 (60% of M=20) and trade rate 0.108/bar, overlap rate ≈ 0.74 (THEORIST). Implement (a) sequential bootstrap per AFML Snippet 4.5 — draws made with updated probabilities `δ_j^(2) = ū_j^(2) (Σ_k ū_k^(2))^{-1}` to prevent picking already-overlapping trades; (b) average-uniqueness weighting per AFML Ch. 4.4 p. 61: `ū_i = (Σ u_t,i) / (Σ 1_t,i)`. Sequential bootstrap is preferred (no information loss).
- **Predicted effect**: plain trade-bootstrap under-covers Sharpe-CI by ~30% at overlap rate ≈ 0.74; LdP-weighted (1/c_t) bootstrap recovers nominal coverage. CI half-width ratio plain/dedup ≈ √ū ≈ 0.88; the 5% match threshold violated at this overlap level.
- **Falsification**: subsample to non-overlapping trades only (~280 trades) — if dedup'd CI matches plain trade-bootstrap CI within 5% on full set, overlap estimate wrong; if dedup'd CI ≥ 12% wider, H-193 confirmed.
- **References**: AFML Ch. 4.4-4.7 pp. 61-69 (local PDF; Snippets 4.1 / 4.2 / 4.5). Rao-Pathak-Koltchinskii (1997) JSPI 64:257 (web).
- **Status**: queued (P4).
- **Cost**: medium (~2 hours).
- **Dependencies**: H-180 (per-trade dataframe).

### Round-016 refinements to existing cards

These are summary refinements applied in-place above for the named existing cards. The full text of each is preserved in its existing tier; this section pins the mandate's specific addenda for traceability.

- **H-040** (Hurst / DFA / sample entropy) — addendum confirmed: tie to regime via 2D cube `Hurst tercile × parkinson_var tercile`; CRITIC threshold for keeping Hurst as primary regime signal is any `(H_tercile, p_tercile)` cell with `|ECE − overall_ECE| > 0.02` on offline.
- **H-115** (permutation entropy m=3, τ=1) — addendum: confirm Bandt-Pompe normalization and stable mergesort tie-breaking; cite Cover & Thomas Ch. 8 (local Desktop) for the entropy floor.
- **H-114** → SUPERSEDED in primary sequencing by **H-131**; kept queued for granular ablation.
- **H-102** (sample weighting) — addendum: pair the existing 4-cell ablation matrix with H-153's Huber-saturated form; H-303 27-cell grid stays queued for full sweep.
- **H-020** (CVaR-α tail) — addendum: gated by H-303 / H-153; explicit cap `5 × mean_loss` (Cont 2001 stylized facts).
- **H-024** (no-stop variant) — addendum stands; runs against round-015 unified harness with `c_stop=∞` (round 017).
- **H-025** (Mondrian-ACI sized) — refinement: `(p − q̂)/(1 − q̂)` confidence-margin sizing — **CITATION CORRECTION** per round-016 §4.1: cite Vovk-Gammerman-Shafer (2005) Ch. 3 and Angelopoulos & Bates (2021) §2.2 (arXiv 2107.07511, web; INDEX add); NOT Lekeufack §V-C (sub-agent verified — Lekeufack §V-C is buy/short/abstain on interval-straddle).
- **H-204** (River ARF vs SRP vs HAT) — refinement: blocking-precondition for H-170; ships first, shortlists K=4 bases for online stacking.
- **H-206** (σ_epistemic as conformal feature) — refinement: two specific score-function variants — (a) `s_i = (1 − p̂_i(y_i)) / σ_epistemic_i` (uncertainty-normalized LAC); (b) `s_i = (1 − p̂_i(y_i)) + λ · σ_epistemic_i` (additive penalty, λ tuned on cal). Falsification: monotone rank correlation > 0.6 between set width and σ_epistemic on eval; otherwise the score function isn't extracting epistemic information.
- **H-207** (ADWIN-triggered q_t reset) — refinement: soft-reset `q_t` halfway toward prior `α` rather than fully (preserve adaptation history). Pair with H-151.
- **H-208** (streaming conformal trade gate) — UNBLOCK: H-011 + H-202 dependencies are accepted. Mechanism: abstain when `predict_set` returns `{0,1}` (full set = no information at given α). Pair with H-25 — gate decides whether to trade; sizing decides how big.
- **H-022** → **H-161** (lifted; H-161 is the active card under the round-016 mandate).
- **H-023** → **H-160** (lifted).
- **H-113** → **H-192** (refined; protocol pinned).

---

## Notes
- Items at `[P5]` are unconditionally scheduled before `[P3]`.
- "BLOCKED" items wait until the unblock-er is `accept`ed.
- New ideas append to the appropriate tier section, NOT the top — priority discipline is enforced.
- Items spawned mid-round are marked with `(spawned by round NNN)` for traceability.
- Round-016 mandate cards (H-1xx series) live in Tier 7 above; H-3xx series from round-014 lives in Tiers 0/4/5 unchanged.
