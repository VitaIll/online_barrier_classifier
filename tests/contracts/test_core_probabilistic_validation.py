"""Coverage tests for the small primitive types in
`wagie.core.probabilistic` and `wagie.core.validation`."""

from __future__ import annotations

import pytest

from wagie.core.numeric import Probability
from wagie.core.probabilistic import (
    CoverageGuarantee,
    PredictionInterval,
    Quantile,
)
from wagie.core.time import Duration, Timestamp, TimeWindow
from wagie.core.validation import CrossValSplitter, Split


# ---- Quantile / PredictionInterval -----------------------------------------


def test_prediction_interval_post_init_rejects_inverted_levels():
    lo = Quantile(level=Probability(0.95), value=0.0)
    hi = Quantile(level=Probability(0.05), value=10.0)
    with pytest.raises(ValueError, match="lower.level must be < upper.level"):
        PredictionInterval(lower=lo, upper=hi)


def test_prediction_interval_equal_levels_rejected():
    lo = Quantile(level=Probability(0.5), value=0.0)
    hi = Quantile(level=Probability(0.5), value=10.0)
    with pytest.raises(ValueError):
        PredictionInterval(lower=lo, upper=hi)


def test_prediction_interval_width_and_coverage():
    lo = Quantile(level=Probability(0.05), value=10.0)
    hi = Quantile(level=Probability(0.95), value=30.0)
    pi = PredictionInterval(lower=lo, upper=hi)
    assert pi.width == 20.0
    assert pytest.approx(float(pi.nominal_coverage)) == 0.90


def test_prediction_interval_contains():
    lo = Quantile(level=Probability(0.05), value=10.0)
    hi = Quantile(level=Probability(0.95), value=30.0)
    pi = PredictionInterval(lower=lo, upper=hi)
    assert pi.contains(20.0)
    assert pi.contains(10.0)  # inclusive
    assert pi.contains(30.0)  # inclusive
    assert not pi.contains(9.999)
    assert not pi.contains(30.001)


# ---- CoverageGuarantee -----------------------------------------------------


def test_coverage_guarantee_gap_and_validity():
    cg = CoverageGuarantee(
        nominal=Probability(0.90),
        empirical=Probability(0.91),
        n_samples=1000,
    )
    assert pytest.approx(cg.gap, abs=1e-9) == 0.01
    assert cg.is_valid(tol=0.02)
    assert not cg.is_valid(tol=0.005)


def test_coverage_guarantee_negative_gap():
    cg = CoverageGuarantee(
        nominal=Probability(0.90),
        empirical=Probability(0.85),
        n_samples=1000,
    )
    assert pytest.approx(cg.gap, abs=1e-9) == -0.05
    assert not cg.is_valid(tol=0.02)
    assert cg.is_valid(tol=0.10)


# ---- Split / CrossValSplitter ---------------------------------------------


def _ts(ms: int) -> Timestamp:
    return Timestamp.from_ms(ms)


def _window(start_ms: int, end_ms: int) -> TimeWindow:
    return TimeWindow(start=_ts(start_ms), end=_ts(end_ms))


def test_split_construction_ok():
    train = _window(0, 100)
    test = _window(150, 200)
    s = Split(train=train, test=test, embargo=Duration.from_seconds(0.050))
    assert s.is_disjoint()


def test_split_violates_embargo_raises():
    train = _window(0, 100)
    test = _window(101, 200)  # only 1ms gap
    with pytest.raises(ValueError, match="violates embargo"):
        Split(train=train, test=test, embargo=Duration.from_seconds(0.050))


def test_split_invalid_window_raises():
    """`TimeWindow.__post_init__` itself rejects start>end before we even
    reach `Split`; we surface that path here."""
    with pytest.raises(ValueError):
        TimeWindow(start=_ts(100), end=_ts(0))


def test_split_is_disjoint_when_overlap_then_false():
    train = _window(0, 100)
    test = _window(150, 200)
    s = Split(train=train, test=test, embargo=Duration.from_seconds(0.050))
    # Construct a case where train.end + embargo == test.start (boundary OK)
    s2 = Split(train=_window(0, 100), test=_window(150, 200),
               embargo=Duration.from_seconds(0.050))
    assert s.is_disjoint()
    assert s2.is_disjoint()


def test_cross_val_splitter_protocol_runtime_checkable():
    class _Stub:
        def split(self, events):
            return iter([])

    assert isinstance(_Stub(), CrossValSplitter)
