# H-201: Coverage baseline on existing ARF + offline predictions (round 002)

## Claim
When `artifacts/online_eval/predictions.parquet` is treated as a conformal
predictor (split-conformal LAC, finite-sample-corrected quantile per Manokhin
Ch.3 p.37), the **marginal** empirical coverage at α∈{0.05, 0.10, 0.20} will
be near-nominal for `p_online` (which is the streaming-ARF-calibrated output)
but the **per-volatility-regime coverage gap** will exceed the 5% threshold
the BACKLOG entry calls out, in at least one parkinson-variance tercile. This
is the gap every subsequent online-stage hypothesis (H-202 ACI, H-203
Mondrian-ACI, …) must close. No model retrains; pure measurement round.

## Mechanism
Two-mode evaluation per Manokhin Ch.6 + Ch.11, plus controls:

1. **Marginal LAC** (Ch.3): fit `q_hat = ceil((n_cal+1)(1-α))/n_cal` quantile
   of LAC scores `s = 1 - p̂(y|x)` on a chronological calibration slice
   (first 30% of the 31486-bar test split = n_cal≈9446); evaluate marginal
   empirical coverage on the held-out 70% eval slice.
2. **Mondrian LAC** (Ch.11 p.188): fit one `q_hat_r` per
   `parkinson_var_rolling_mean_24` tercile (fixed regime signal per
   CONSTITUTION IV); evaluate per-regime coverage. `coverage_by_regime` from
   `src/conformal.py` is already implemented.
3. **Naive-threshold control** — the H-201 wording's "treat as if it were a
   conformal predictor at varying confidence levels" baseline: set =
   `{y : p̂(y|x) ≥ α}` with NO calibration. Isolates whether observed gaps
   are calibration problems (LAC fixes them) or fundamental probability-
   quality problems (LAC doesn't fix them).
4. **Constant-predictor sanity** (p ≡ base_rate = 0.0971): at α=0.05 the
   positive class set is always full (cov→1); at α=0.20 the positive class
   is covered exactly at the base rate. Anything else → harness is broken.
   This is the harness-falsifier the THEORIST asked for.

Run on three predictors: `p_offline`, `p_online`, and `p_const = 0.0971`.

## Predicted effect (THEORIST)
- `p_offline` is over-confident (mean ≈ 0.20 vs base rate 0.097) → marginal
  Mondrian coverage will be near-nominal because LAC self-corrects;
  **per-regime gap widest in high-vol tercile, positive class** because the
  offline model under-weights the rare high-vol regime.
- `p_online` is closer to diagonal → smaller per-regime gap; the headline
  benefit of the online layer is exactly this conditional calibration.
- At α=0.05 (deeper tail) gaps amplify vs α=0.10.

## Seed-noise band
N/A. This round trains nothing. The diagnostic is deterministic given fixed
predictions + fixed regime cuts. Required ≥2-rerun band per CRITIC checklist
is satisfied trivially: invoking the script twice produces byte-identical
CSV. Recorded as `seed_noise_band = 0.0`.

## Falsification (substantive)
H-201 dies if the marginal Mondrian-LAC empirical coverage at α=0.10 is
within ±0.5pp of nominal (0.90) in **every** parkinson_var tercile for
**both** `p_offline` and `p_online`. That would mean there is no coverage gap
to close and the online layer has no calibration job.

## Falsification (harness-internal)
H-201's harness is broken if the constant-predictor sanity check fails:
specifically, at α=0.05 the constant-predictor `p≡base_rate` MUST yield
empirical marginal coverage in [0.999, 1.000] (set is always {0,1}); at
α=0.20 it MUST yield exactly the base rate ≈ 0.097 for class=1 (because
the singleton-1 set is empty and the all-zero set never includes y=1).
Anything outside these bands → harness bug.

## Decision rule for round 002
- `accept` if (a) all marginal coverages finite, (b) constant-predictor
  sanity passes, (c) at least ONE per-regime coverage gap exceeds 5% in
  the Mondrian-LAC table for at least one of `p_offline` / `p_online` —
  i.e., a gap exists for H-202..H-208 to close.
- `iterate` if (a)+(b) hold but (c) fails — harness works but online layer
  is already well-calibrated regime-conditionally; queue refinement that
  uses tighter tercile binning (quintile? halibut? finer regime signal).
- `kill` only if (a) or (b) fails — implies a wiring or harness bug.

## What this round is NOT
- Not an Adaptive Conformal Inference (ACI) round — H-202.
- Not a model-quality round; primary metric is coverage NOT Brier/ROC per
  CONSTITUTION IV.1.

## References
- Local: Manokhin (2024) Ch.3 pp.30, 34, 37–38; Ch.6 pp.69–84; Ch.9
  pp.158–166; Ch.11 pp.187–192. Path:
  `Downloads/Valeri Manokhin - Practical Guide to Applied Conformal
  Prediction in Python-Packt (2024).pdf`.
- Local: Lekeufack et al. (2024) *Conformal Decision Theory* §III + §V-C —
  controller benchmark. Path: `Downloads/conformal_decision_theory.pdf`
  pp.2–3, 6–7.
- Local: Angelopoulos & Bates et al. (2022/2025) *Conformal Risk Control*
  §1, §2.3, §4.1 — non-exchangeable extension `Σ TV(Z_i, Z_{n+1})` bounding
  the expected coverage gap. Path: `Downloads/conformal_control.pdf`
  pp.1–4, 10.
- Vovk 2012 *Conditional validity of inductive conformal predictors* —
  cited for object-conditional impossibility result, NOT in local corpus;
  flagged for HOUSEKEEPER to acquire.

## Sub-agent IDs
- LITERATURE-SCOUT: add76b51f25b39577
- CODE-SCOUT: a0c1ad043d80fdeb0
- THEORIST: a87b9e4999021938e
