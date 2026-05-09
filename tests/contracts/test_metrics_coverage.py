"""Contract tests for `wagie.metrics.coverage`.

Covers:
  - Empty input branch (n=0).
  - target = 1 - alpha invariant.
  - empirical_coverage gap sign (over- and under-coverage).
  - Perfect / zero coverage edge cases.
  - CoverageStats.gap property arithmetic.
"""

from __future__ import annotations

import pytest

from wagie.metrics import CoverageStats, empirical_coverage


# ---------- empty input -----------------------------------------------------


def test_empty_inputs_return_zero_coverage() -> None:
    cs = empirical_coverage([], [], alpha=0.1)
    assert cs.n == 0
    assert cs.empirical_coverage == 0.0
    assert cs.target_coverage == pytest.approx(0.9)
    assert cs.alpha == 0.1


# ---------- target = 1 - alpha ---------------------------------------------


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20, 0.50])
def test_target_is_one_minus_alpha(alpha: float) -> None:
    cs = empirical_coverage([1, 1, 0], [1, 0, 0], alpha=alpha)
    assert cs.target_coverage == pytest.approx(1.0 - alpha)


# ---------- empirical math --------------------------------------------------


def test_perfect_coverage() -> None:
    in_set = [1, 1, 1, 1, 1]
    y_true = [1, 1, 1, 1, 1]
    cs = empirical_coverage(in_set, y_true, alpha=0.1)
    assert cs.empirical_coverage == 1.0
    assert cs.n == 5
    assert cs.gap == pytest.approx(1.0 - 0.9)


def test_zero_coverage() -> None:
    in_set = [0, 0, 0, 0]
    y_true = [1, 1, 1, 1]
    cs = empirical_coverage(in_set, y_true, alpha=0.1)
    assert cs.empirical_coverage == 0.0
    assert cs.gap == pytest.approx(-0.9)


def test_partial_coverage_known_value() -> None:
    """3 of 5 samples covered → empirical = 0.6, target = 0.8 → gap = -0.2."""
    in_set = [1, 0, 1, 0, 1]
    y_true = [1, 1, 1, 1, 1]
    cs = empirical_coverage(in_set, y_true, alpha=0.2)
    assert cs.empirical_coverage == pytest.approx(0.6)
    assert cs.target_coverage == pytest.approx(0.8)
    assert cs.gap == pytest.approx(-0.2)


# ---------- gap sign convention --------------------------------------------


def test_undercoverage_gives_negative_gap() -> None:
    """Empirical 0.5 < target 0.9 → gap < 0."""
    in_set = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cs = empirical_coverage(in_set, y_true, alpha=0.1)
    assert cs.empirical_coverage == 0.5
    assert cs.gap < 0.0
    assert cs.gap == pytest.approx(0.5 - 0.9)


def test_overcoverage_gives_positive_gap() -> None:
    """Cover 9/10 with target 0.5 → gap = +0.4."""
    in_set = [1] * 9 + [0]
    y_true = [1] * 10
    cs = empirical_coverage(in_set, y_true, alpha=0.5)
    assert cs.empirical_coverage == pytest.approx(0.9)
    assert cs.gap == pytest.approx(0.4)


# ---------- counting matches int(s) == int(y) ------------------------------


def test_coverage_counts_predicate_matching_y() -> None:
    """`in_set` of 0 with y_true 0 → counts as covered (set contains 0)."""
    in_set = [0, 0, 0, 1]
    y_true = [0, 0, 0, 1]
    cs = empirical_coverage(in_set, y_true, alpha=0.1)
    assert cs.empirical_coverage == 1.0


def test_coverage_floats_are_cast_to_int() -> None:
    """Implementation casts via int(s)/int(y); 1.0 → 1."""
    in_set = [1.0, 0.0]
    y_true = [1.0, 0.0]
    cs = empirical_coverage(in_set, y_true, alpha=0.1)
    assert cs.empirical_coverage == 1.0


# ---------- CoverageStats dataclass ----------------------------------------


def test_coverage_stats_dataclass_fields() -> None:
    cs = CoverageStats(alpha=0.1, empirical_coverage=0.85, target_coverage=0.9, n=100)
    assert cs.alpha == 0.1
    assert cs.empirical_coverage == 0.85
    assert cs.target_coverage == 0.9
    assert cs.n == 100
    assert cs.gap == pytest.approx(-0.05)


def test_coverage_stats_gap_zero_when_perfect() -> None:
    cs = CoverageStats(alpha=0.2, empirical_coverage=0.8, target_coverage=0.8, n=10)
    assert cs.gap == pytest.approx(0.0)
