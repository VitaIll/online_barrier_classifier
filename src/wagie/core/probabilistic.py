"""Probabilistic primitives: Quantile, PredictionInterval, CoverageGuarantee."""

from __future__ import annotations

from dataclasses import dataclass

from wagie.core.numeric import Probability


@dataclass(frozen=True, slots=True)
class Quantile:
    """A (level, value) pair: F^{-1}(level) = value."""

    level: Probability
    value: float


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    """A prediction interval with two quantiles. Closed under intersect."""

    lower: Quantile
    upper: Quantile

    def __post_init__(self):
        if self.lower.level >= self.upper.level:
            raise ValueError(
                f"PredictionInterval lower.level must be < upper.level; "
                f"got {self.lower.level} >= {self.upper.level}"
            )

    @property
    def width(self) -> float:
        return self.upper.value - self.lower.value

    @property
    def nominal_coverage(self) -> Probability:
        return Probability(float(self.upper.level) - float(self.lower.level))

    def contains(self, x: float) -> bool:
        return self.lower.value <= x <= self.upper.value


@dataclass(frozen=True, slots=True)
class CoverageGuarantee:
    """Empirical vs nominal coverage. is_valid() is the conformal validity test."""

    nominal: Probability
    empirical: Probability
    n_samples: int

    @property
    def gap(self) -> float:
        return float(self.empirical) - float(self.nominal)

    def is_valid(self, tol: float = 0.02) -> bool:
        """Within `tol` of nominal coverage."""
        return abs(self.gap) <= tol


__all__ = ["Quantile", "PredictionInterval", "CoverageGuarantee"]
