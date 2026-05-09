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
from typing import Mapping, Optional

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
