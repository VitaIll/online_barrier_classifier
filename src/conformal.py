"""Split-conformal prediction for binary barrier classification.

Implements the LAC (Least Ambiguous Classifier) non-conformity score with
Mondrian (regime-stratified) calibration, per Manokhin (2024) *Practical
Conformal Prediction in Python* Ch. 1, 3, 9, 11. Decision rationale:

- **LAC over APS**: For binary classification with ~20% positive rate, LAC
  produces sharper sets than APS (Adaptive Prediction Sets). APS's benefit
  (multi-step cumulative mass) collapses at K=2; the "vast set" failure mode
  of APS is exactly what we want to avoid (Manokhin Ch. 9 p. 164).
- **`p_mean` over `(p_mean, sigma_epistemic)`**: Conformal validity holds for
  *any* score function — wrapping epistemic UQ inside the conformal score
  changes only efficiency, never the 1-α coverage guarantee. Per Lekeufack
  et al., *Conformal Decision Theory* (2024), directly calibrating the
  decision is more efficient than wrapping prediction sets around UQ. We
  therefore use `p_mean` as the conformal score and route `sigma_epistemic`
  to the downstream sizing/decision layer (Cantelli rule in `uncertainty.py`).
- **Mondrian over plain LAC**: Per-volatility-regime calibration restores
  conditional coverage in a regime-drift setting (Vovk 2003 / Manokhin Ch. 11
  p. 188) without leaking across the chronological split or violating
  EMBARGO_K. Adaptive Conformal Inference is reserved for online deployment
  (a future round); for batch eval Mondrian is the right tool.
- **Finite-sample-corrected quantile**: `q_hat = ceil((n_cal + 1) * (1 - α)) / n_cal`
  derived from the exchangeability bag including the test point (Manokhin
  Ch. 9 p. 161, Ch. 3 p. 37 step 5). Plain `(1 - α)` quantile undercovers.

References (local corpus):
- Manokhin (2024), Practical Conformal Prediction in Python, Packt — Ch. 1
  pp. 6-8, Ch. 3 pp. 30/34/37-38, Ch. 9 pp. 161-165, Ch. 11 pp. 187-192.
- Lekeufack et al. (2024), Conformal Decision Theory, arXiv:2310.05921 — p. 1-2
  (decision-direct calibration), p. 7 (binary-classification + UQ pattern).
- Vovk, Gammerman, Shafer (2005), Algorithmic Learning in a Random World.
- Romano, Sesia, Candès (2020), APS — referenced but not adopted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import math
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Score function (LAC = Least Ambiguous Classifier)
# ----------------------------------------------------------------------------

def lac_score(p_proba_class1: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """LAC non-conformity score: `s_i = 1 - p_hat(y_i | x_i)`.

    For binary classification:
        - if y_i = 1, s_i = 1 - p_proba_class1[i]
        - if y_i = 0, s_i = 1 - (1 - p_proba_class1[i]) = p_proba_class1[i]
    """
    p1 = np.asarray(p_proba_class1, dtype=float)
    y = np.asarray(y_true).astype(int)
    return np.where(y == 1, 1.0 - p1, p1)


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-α) quantile of `scores`.

    `q_hat = ceil((n + 1) * (1 - α)) / n` of the sorted scores; equivalent to
    `np.quantile(scores, q, method='higher')` with `q = ceil((n+1)(1-α))/n`.
    Required for finite-`n` exchangeability validity (Manokhin Ch. 3 p. 37).
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute quantile of empty scores.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    q_level = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


# ----------------------------------------------------------------------------
# Calibrator
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ConformalCalibrator:
    """Stores fit-time state for a split-conformal procedure.

    For plain LAC: `q_hat` is a scalar.
    For Mondrian (regime-stratified) LAC: `q_hat_by_regime` maps each
    `regime_id` to its own q_hat.
    """

    alpha: float
    n_cal: int
    q_hat: Optional[float] = None
    q_hat_by_regime: Optional[Mapping[str, float]] = None
    mode: str = "lac"
    regime_n: Optional[Mapping[str, int]] = None

    def is_mondrian(self) -> bool:
        return self.q_hat_by_regime is not None


def fit_conformal(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float,
    *,
    regime_cal: Optional[np.ndarray] = None,
    min_per_regime: int = 50,
) -> ConformalCalibrator:
    """Fit the split-conformal calibrator on a held-out calibration set.

    Parameters
    ----------
    p_cal : (n_cal,)  Predicted P(y=1) on the calibration set.
    y_cal : (n_cal,)  True binary labels.
    alpha : float   Miscoverage level; coverage = 1 - alpha.
    regime_cal : (n_cal,) array-like, optional
        If provided, fit a Mondrian LAC stratified by the regime label.
        Regimes with fewer than `min_per_regime` calibration samples fall back
        to the global q_hat to avoid underpowered per-bucket quantiles.
    min_per_regime : int, default 50
        Minimum samples per regime for a regime-specific q_hat.
    """
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal).astype(int)
    if p_cal.shape != y_cal.shape:
        raise ValueError("p_cal and y_cal must have the same shape.")

    scores = lac_score(p_cal, y_cal)

    if regime_cal is None:
        return ConformalCalibrator(
            alpha=float(alpha),
            n_cal=len(scores),
            q_hat=finite_sample_quantile(scores, alpha),
            mode="lac",
        )

    regime_cal = np.asarray(regime_cal)
    global_q = finite_sample_quantile(scores, alpha)
    q_by_regime: dict[str, float] = {}
    n_by_regime: dict[str, int] = {}
    for regime in np.unique(regime_cal):
        mask = regime_cal == regime
        n = int(mask.sum())
        n_by_regime[str(regime)] = n
        if n >= min_per_regime:
            q_by_regime[str(regime)] = finite_sample_quantile(scores[mask], alpha)
        else:
            # Fallback to global q_hat for under-sampled regimes.
            q_by_regime[str(regime)] = global_q

    return ConformalCalibrator(
        alpha=float(alpha),
        n_cal=len(scores),
        q_hat=global_q,
        q_hat_by_regime=q_by_regime,
        mode="mondrian",
        regime_n=n_by_regime,
    )


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def predict_set(
    calibrator: ConformalCalibrator,
    p_test: np.ndarray,
    *,
    regime_test: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute prediction sets at test time.

    For each test sample, the prediction set is
        C(x) = { y : 1 - p_hat(y | x) <= q_hat(regime(x)) }
             = { y : p_hat(y | x) >= 1 - q_hat(regime(x)) }

    Returns
    -------
    np.ndarray of shape (n_test, 2), dtype=bool. Column 0 = "y=0 in set",
    column 1 = "y=1 in set". Possible row patterns:
        [True, False]  -> singleton {0}
        [False, True]  -> singleton {1}
        [True, True]   -> full set {0, 1} (genuinely uncertain)
        [False, False] -> empty set (model rejects both classes; "abstain")
    """
    p_test = np.asarray(p_test, dtype=float)
    n_test = len(p_test)
    sets = np.zeros((n_test, 2), dtype=bool)

    if calibrator.is_mondrian():
        if regime_test is None:
            raise ValueError("Mondrian calibrator requires regime_test.")
        regime_test = np.asarray(regime_test)
        for i in range(n_test):
            r = str(regime_test[i])
            q_hat = calibrator.q_hat_by_regime.get(r, calibrator.q_hat)
            sets[i, 0] = (1.0 - p_test[i]) >= 1.0 - q_hat
            sets[i, 1] = p_test[i] >= 1.0 - q_hat
    else:
        q_hat = float(calibrator.q_hat)
        sets[:, 0] = (1.0 - p_test) >= 1.0 - q_hat
        sets[:, 1] = p_test >= 1.0 - q_hat

    return sets


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def empirical_coverage(pred_sets: np.ndarray, y_test: np.ndarray) -> float:
    """Fraction of samples whose true label is in the prediction set."""
    pred_sets = np.asarray(pred_sets, dtype=bool)
    y = np.asarray(y_test).astype(int)
    if pred_sets.shape != (len(y), 2):
        raise ValueError(f"pred_sets must be (n, 2); got {pred_sets.shape}.")
    return float(np.mean([pred_sets[i, y[i]] for i in range(len(y))]))


def set_size_distribution(pred_sets: np.ndarray) -> dict[str, float]:
    """Fraction of test samples in each set-size bucket."""
    pred_sets = np.asarray(pred_sets, dtype=bool)
    sizes = pred_sets.sum(axis=1)
    n = len(sizes)
    return {
        "empty": float(np.mean(sizes == 0)),
        "singleton_0": float(np.mean((sizes == 1) & ~pred_sets[:, 1])),
        "singleton_1": float(np.mean((sizes == 1) & pred_sets[:, 1])),
        "full": float(np.mean(sizes == 2)),
    }


# ----------------------------------------------------------------------------
# Adaptive Conformal Inference (Gibbs & Candès 2021)
# ----------------------------------------------------------------------------
#
# ACI dispenses with the held-out calibration set: the threshold q_t is
# updated online via a single stochastic-approximation step per observation:
#
#     q_{t+1} = q_t + γ * (err_t - α)
#
# where err_t = 1{ y_t ∉ C_t(x_t) } is the per-step miscoverage indicator
# and α ∈ (0,1) is the target miscoverage level. The set C_t(x_t) is built
# from the SAME LAC score used by the batch path (`lac_score`), so the two
# routes share the set-construction semantics:
#
#     C_t(x) = { y : 1 - p(y|x) <= q_t }
#
# Asymptotic guarantee (G&C 2021 Thm 1): for any γ > 0 and any data sequence
# (no exchangeability assumption), |emp_coverage - (1-α)| → 0 at rate
# O(1/(γT)) under bounded scores. γ trades off convergence speed vs noise:
# small γ = slower but tighter; large γ = faster but choppier.

def aci_step(
    p_t: float,
    y_t: int,
    q_t: float,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, float, int]:
    """One ACI step: predict set at threshold q_t, observe y_t, update q.

    Returns
    -------
    set_t : (2,) bool array — same convention as ``predict_set``.
    q_next : float — updated threshold for the next step.
    err_t : int — 1 if y_t miscovered (y_t ∉ C_t), else 0.

    The threshold is clamped to [0, 1] after the update; without the clamp
    a long miscoverage run on a degenerate stream could push q_t outside the
    score range and freeze the dynamics.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if gamma < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}.")

    # Set construction — mirrors predict_set() for batch consistency.
    set_t = np.array(
        [
            (1.0 - p_t) >= 1.0 - q_t,  # class 0 in set
            p_t >= 1.0 - q_t,          # class 1 in set
        ],
        dtype=bool,
    )
    y_int = int(y_t)
    if y_int not in (0, 1):
        raise ValueError(f"y_t must be 0 or 1, got {y_t}.")
    err_t = 0 if set_t[y_int] else 1
    q_next = float(np.clip(q_t + gamma * (err_t - alpha), 0.0, 1.0))
    return set_t, q_next, err_t


def aci_stream(
    p_stream: np.ndarray,
    y_stream: np.ndarray,
    alpha: float,
    *,
    gamma: float = 0.01,
    q_init: float = 0.5,
) -> dict[str, Any]:
    """Run ACI over a binary classification stream.

    Parameters
    ----------
    p_stream : (T,) float — predicted P(y=1) at each step.
    y_stream : (T,) int — observed labels (0/1).
    alpha    : float — target miscoverage.
    gamma    : float, default 0.01 — learning rate.
    q_init   : float, default 0.5 — initial threshold; 0.5 ≈ neutral.

    Returns dict with:
        sets       : (T, 2) bool — prediction sets per step
        q_history  : (T,) float — q_t at the START of each step
        err_history: (T,) int   — 1{miscovered} per step
        coverage   : float — empirical marginal coverage over the stream
    """
    p = np.asarray(p_stream, dtype=float)
    y = np.asarray(y_stream).astype(int)
    if p.shape != y.shape:
        raise ValueError(
            f"p_stream and y_stream must have the same shape, got {p.shape} vs {y.shape}."
        )

    T = len(p)
    sets = np.zeros((T, 2), dtype=bool)
    q_history = np.zeros(T, dtype=float)
    err_history = np.zeros(T, dtype=int)

    q_t = float(q_init)
    for t in range(T):
        q_history[t] = q_t
        set_t, q_t, err_t = aci_step(p[t], y[t], q_t, alpha, gamma)
        sets[t] = set_t
        err_history[t] = err_t

    return {
        "sets": sets,
        "q_history": q_history,
        "err_history": err_history,
        "coverage": float(1.0 - err_history.mean()) if T > 0 else float("nan"),
    }


# ----------------------------------------------------------------------------
# Mondrian-ACI: per-regime q_t with the same online update rule
# ----------------------------------------------------------------------------
#
# Plain ACI (above) maintains a single global q_t. Under regime drift the
# global q_t can't track simultaneous shifts in score distribution across
# regimes — round-007's diagnostic showed this empirically: at alpha=0.10,
# plain ACI over-covered low-vol by 6pp and under-covered high-vol by 6pp
# while marginal coverage was on target. Mondrian-ACI fixes this by binding
# one q_t per regime label and updating ONLY the active regime's q at each
# step.
#
# This generalises plain ACI: with a single regime label, Mondrian-ACI is
# bit-equivalent to aci_step / aci_stream. The test suite pins this.

def aci_mondrian_step(
    p_t: float,
    y_t: int,
    regime_t: Any,
    q_t_by_regime: Mapping[Any, float],
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, dict[Any, float], int]:
    """One Mondrian-ACI step.

    Looks up the active regime's threshold, builds the LAC set, observes y_t,
    and returns the predicted set, the updated regime->q mapping (only the
    active regime's q changed), and the miscoverage indicator.

    ``regime_t`` must be a key already present in ``q_t_by_regime``; the
    caller is responsible for initializing q for every regime that will
    appear in the stream.
    """
    if regime_t not in q_t_by_regime:
        raise KeyError(
            f"regime {regime_t!r} not in q_t_by_regime "
            f"(known: {list(q_t_by_regime.keys())})"
        )
    q_t = float(q_t_by_regime[regime_t])
    set_t, q_next, err_t = aci_step(p_t, y_t, q_t, alpha, gamma)
    out = dict(q_t_by_regime)
    out[regime_t] = q_next
    return set_t, out, err_t


def aci_mondrian_stream(
    p_stream: np.ndarray,
    y_stream: np.ndarray,
    regime_stream: np.ndarray,
    alpha: float,
    *,
    gamma: float = 0.01,
    q_init: float = 0.5,
    q_init_by_regime: Mapping[Any, float] | None = None,
) -> dict[str, Any]:
    """Run Mondrian-ACI over a binary stream with regime labels.

    ``regime_stream[t]`` is the regime label active at step t. Each regime
    keeps its own q_t; only the active regime's q is updated per step.

    Parameters
    ----------
    p_stream, y_stream : (T,) — same as aci_stream.
    regime_stream : (T,) array-like — regime label per step; any hashable.
    alpha, gamma  : same as aci_stream.
    q_init        : float, default 0.5 — used for any regime whose initial q
                    is not in ``q_init_by_regime``.
    q_init_by_regime : optional override; useful when warming q from a
                       calibration window per regime.

    Returns dict with:
        sets            : (T, 2) bool — prediction sets per step.
        q_history       : (T,) float — q_t at the START of each step (active regime).
        q_history_by_regime : dict regime -> (T_regime,) float trajectory of
                              that regime's q over its active steps.
        err_history     : (T,) int — 1{miscovered} per step.
        coverage        : float — empirical marginal coverage.
        regimes_seen    : list — distinct regime labels encountered.
    """
    p = np.asarray(p_stream, dtype=float)
    y = np.asarray(y_stream).astype(int)
    regimes = np.asarray(regime_stream)
    if not (p.shape == y.shape == regimes.shape):
        raise ValueError(
            f"p_stream/y_stream/regime_stream must have the same shape; "
            f"got {p.shape}/{y.shape}/{regimes.shape}"
        )

    T = len(p)
    sets = np.zeros((T, 2), dtype=bool)
    q_history = np.zeros(T, dtype=float)
    err_history = np.zeros(T, dtype=int)

    # Initialise q per regime.
    seen_regimes = list(np.unique(regimes).tolist())
    overrides = dict(q_init_by_regime or {})
    q_by_regime: dict[Any, float] = {
        r: float(overrides.get(r, q_init)) for r in seen_regimes
    }
    # Also accept overrides for regimes not yet seen (defensive).
    for r, v in overrides.items():
        if r not in q_by_regime:
            q_by_regime[r] = float(v)

    q_history_per_regime: dict[Any, list[float]] = {r: [] for r in q_by_regime}

    for t in range(T):
        r_t = regimes[t]
        # Defensive: regime not in q_by_regime yet (shouldn't happen because
        # np.unique enumerates all up-front, but if regime_stream uses
        # different dtype semantics this catches it).
        if r_t not in q_by_regime:
            q_by_regime[r_t] = float(q_init)
            q_history_per_regime[r_t] = []
        q_t_active = q_by_regime[r_t]
        q_history[t] = q_t_active
        q_history_per_regime[r_t].append(q_t_active)

        set_t, q_by_regime, err_t = aci_mondrian_step(
            p[t], y[t], r_t, q_by_regime, alpha, gamma,
        )
        sets[t] = set_t
        err_history[t] = err_t

    return {
        "sets": sets,
        "q_history": q_history,
        "q_history_by_regime": {
            r: np.asarray(traj, dtype=float) for r, traj in q_history_per_regime.items()
        },
        "err_history": err_history,
        "coverage": float(1.0 - err_history.mean()) if T > 0 else float("nan"),
        "regimes_seen": list(q_by_regime.keys()),
    }


def coverage_by_regime(
    pred_sets: np.ndarray, y_test: np.ndarray, regime_test: np.ndarray
) -> pd.DataFrame:
    """Per-regime empirical coverage and set-size diagnostics."""
    pred_sets = np.asarray(pred_sets, dtype=bool)
    y = np.asarray(y_test).astype(int)
    regime_test = np.asarray(regime_test)

    rows = []
    for r in np.unique(regime_test):
        mask = regime_test == r
        if mask.sum() == 0:
            continue
        sub_sets = pred_sets[mask]
        sub_y = y[mask]
        cov = float(np.mean([sub_sets[i, sub_y[i]] for i in range(len(sub_y))]))
        sizes = sub_sets.sum(axis=1)
        rows.append({
            "regime": str(r),
            "n": int(mask.sum()),
            "coverage": cov,
            "frac_empty": float(np.mean(sizes == 0)),
            "frac_singleton": float(np.mean(sizes == 1)),
            "frac_full": float(np.mean(sizes == 2)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# High-level: trade gating with conformal + UQ
# ----------------------------------------------------------------------------

def conformal_trade_signal(
    pred_sets: np.ndarray,
    p_mean: np.ndarray,
    sigma_epistemic: np.ndarray,
    *,
    tau_open: float,
    k: float = 1.0,
) -> np.ndarray:
    """Combine conformal abstention with the UQ-aware decision rule.

    Trade only if:
    1. The prediction set is the singleton {1} (model commits to "barrier hit").
    2. AND the Cantelli lower bound clears `tau_open`:
       `p_mean - k * sigma_epistemic > tau_open`.

    This is the doubly-gated rule: conformal handles type-I error
    (abstain on uncertainty); Cantelli handles confidence-magnitude.
    Per CDT (Lekeufack et al. 2024), conformal stays single-purpose
    (validity) while sigma_epistemic governs the *decision* (efficiency).
    """
    pred_sets = np.asarray(pred_sets, dtype=bool)
    is_singleton_one = pred_sets[:, 1] & ~pred_sets[:, 0]
    cantelli_pass = (np.asarray(p_mean) - k * np.asarray(sigma_epistemic)) > tau_open
    return is_singleton_one & cantelli_pass
