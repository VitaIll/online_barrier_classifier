"""Warm q_init_by_regime for the streaming Mondrian-ACI calibrator.

Per F0 spec A.x: in prod the BinanceBroker is cold-start; we persist
q_init_by_regime to disk on engine.on_stop, reload on engine.on_start.
This module computes those warm-up values offline from a calibration set.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def lac_score(p_class1: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAC non-conformity: s = (1 - p) if y=1 else p."""
    p = np.asarray(p_class1, dtype=float)
    y = np.asarray(y, dtype=int)
    return np.where(y == 1, 1.0 - p, p)


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal finite-sample-corrected (1-α) quantile."""
    n = len(scores)
    if n == 0:
        return 0.5
    q = math.ceil((n + 1) * (1.0 - alpha)) / n
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores, q))


def warm_q_init_by_regime(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    regime_cal: np.ndarray,
    alpha: float,
    *,
    min_per_regime: int = 50,
    fallback: float = 0.5,
) -> dict[int, float]:
    """Per-regime LAC quantile; falls back if regime is too thin."""
    p = np.asarray(p_cal, dtype=float)
    y = np.asarray(y_cal, dtype=int)
    r = np.asarray(regime_cal, dtype=int)
    if not (p.shape == y.shape == r.shape):
        raise ValueError("p_cal / y_cal / regime_cal must be same length")
    scores = lac_score(p, y)
    out: dict[int, float] = {}
    for reg in np.unique(r):
        mask = r == reg
        if int(mask.sum()) >= min_per_regime:
            out[int(reg)] = finite_sample_quantile(scores[mask], alpha)
        else:
            out[int(reg)] = float(fallback)
    return out


__all__ = ["lac_score", "finite_sample_quantile", "warm_q_init_by_regime"]
