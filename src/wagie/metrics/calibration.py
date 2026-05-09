"""Calibration metrics — Brier, ECE, reliability bins.

Calibration is the LEADING quality metric in this repo. The online layer's
contribution shows here, not in ranking metrics like ROC/PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class Brier:
    """Streaming Brier score: mean((p - y)^2)."""

    def __init__(self) -> None:
        self._sum_sq = 0.0
        self._n = 0

    def update(self, y_true: int, p_pred: float) -> "Brier":
        self._sum_sq += (float(p_pred) - float(int(y_true))) ** 2
        self._n += 1
        return self

    def get(self) -> float:
        return float(self._sum_sq / self._n) if self._n else 0.0


def brier_score(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: Sequence[int], p_pred: Sequence[float], n_bins: int = 10,
) -> float:
    """ECE: mean over bins of |bin_p_mean - bin_y_mean|, weighted by bin size."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0:
        return 0.0
    qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    e = 0.0
    n = len(p)
    for i in range(n_bins):
        m = (p >= qs[i]) & (p < qs[i + 1])
        if m.sum() > 0:
            e += abs(p[m].mean() - y[m].mean()) * (m.sum() / n)
    return float(e)


@dataclass
class ReliabilityBin:
    p_mean: float
    y_mean: float
    n: int


def reliability_bins(
    y_true: Sequence[int], p_pred: Sequence[float], n_bins: int = 10,
) -> list[ReliabilityBin]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0:
        return []
    qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        m = (p >= qs[i]) & (p < qs[i + 1])
        nm = int(m.sum())
        if nm == 0:
            continue
        out.append(ReliabilityBin(
            p_mean=float(p[m].mean()),
            y_mean=float(y[m].mean()),
            n=nm,
        ))
    return out


__all__ = [
    "Brier", "brier_score", "expected_calibration_error",
    "ReliabilityBin", "reliability_bins",
]
