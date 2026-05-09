"""Conformal coverage metrics for the ACI calibrator.

For each level α: empirical_coverage = P(y in Ĉ_α). Should track 1-α.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CoverageStats:
    alpha: float
    empirical_coverage: float
    target_coverage: float
    n: int

    @property
    def gap(self) -> float:
        return self.empirical_coverage - self.target_coverage


def empirical_coverage(
    in_set: Sequence[int], y_true: Sequence[int], alpha: float,
) -> CoverageStats:
    n = len(in_set)
    if n == 0:
        return CoverageStats(alpha=alpha, empirical_coverage=0.0,
                             target_coverage=1.0 - alpha, n=0)
    n_covered = sum(1 for s, y in zip(in_set, y_true) if int(s) == int(y))
    return CoverageStats(
        alpha=alpha,
        empirical_coverage=float(n_covered) / n,
        target_coverage=1.0 - alpha,
        n=n,
    )


__all__ = ["CoverageStats", "empirical_coverage"]
