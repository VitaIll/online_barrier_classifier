"""Hypothesis property tests for wagie.core.numeric primitives.

Algebraic invariants only — no I/O, no engine. Each test bounds the input
strategy to a finite range so we don't drift into NaN/inf weirdness.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from wagie.core.numeric import Bps, LogReturn, Price, Probability, Quantity


# --------- Bounded float strategy: avoids NaN/inf and overflow ---------------
finite = st.floats(min_value=-100.0, max_value=100.0,
                   allow_nan=False, allow_infinity=False)
small_finite = st.floats(min_value=-1.0, max_value=1.0,
                         allow_nan=False, allow_infinity=False)
positive_finite = st.floats(min_value=1e-9, max_value=1e6,
                            allow_nan=False, allow_infinity=False)
nonneg_finite = st.floats(min_value=0.0, max_value=1e6,
                          allow_nan=False, allow_infinity=False)
unit_finite = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# --------------------------- Probability ------------------------------------

@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(v=unit_finite)
def test_probability_in_unit_interval(v):
    """Any float in [0, 1] constructs a valid Probability."""
    p = Probability(v)
    assert 0.0 <= float(p) <= 1.0


@settings(max_examples=50, deadline=None)
@given(v=st.floats(allow_nan=False, allow_infinity=False))
def test_probability_rejects_outside_unit(v):
    """Construction is closed: any float outside [0, 1] raises."""
    if 0.0 <= v <= 1.0:
        Probability(v)  # no raise expected
    else:
        with pytest.raises(ValueError):
            Probability(v)


@pytest.mark.parametrize("invalid", [-0.0001, 1.0001, float("nan")])
def test_probability_rejects_invalid_explicit(invalid):
    with pytest.raises(ValueError):
        Probability(invalid)


@settings(max_examples=50, deadline=None)
@given(p1=unit_finite, p2=unit_finite)
def test_probability_average_stays_in_unit(p1, p2):
    """A pointwise mean of two probabilities is itself a probability.
    (Probability does not define + as a closed op; we test the convex
    combination, which IS algebraically closed in [0,1].)"""
    a, b = Probability(p1), Probability(p2)
    avg = (float(a) + float(b)) / 2.0
    out = Probability(avg)
    assert 0.0 <= float(out) <= 1.0


# --------------------------- Price ------------------------------------------

@settings(max_examples=50, deadline=None)
@given(v=positive_finite)
def test_price_construction_positive(v):
    Price(v)


@pytest.mark.parametrize("invalid", [0.0, -1e-9, -1.0, float("inf"), float("nan")])
def test_price_rejects_non_positive(invalid):
    with pytest.raises(ValueError):
        Price(invalid)


@settings(max_examples=50, deadline=None)
@given(p1=positive_finite, p2=positive_finite)
def test_price_log_return_antisymmetric(p1, p2):
    """log(p2/p1) == -log(p1/p2) up to numerical tolerance."""
    a = Price(p1).log_return_to(Price(p2))
    b = Price(p2).log_return_to(Price(p1))
    assert math.isclose(float(a), -float(b), abs_tol=1e-10)


# --------------------------- Quantity ---------------------------------------

@settings(max_examples=50, deadline=None)
@given(v=nonneg_finite)
def test_quantity_construction_nonneg(v):
    Quantity(v)


@pytest.mark.parametrize("invalid", [-1e-9, -1.0, float("inf"), float("nan")])
def test_quantity_rejects_invalid(invalid):
    with pytest.raises(ValueError):
        Quantity(invalid)


# --------------------------- LogReturn: abelian group -----------------------

@settings(max_examples=50, deadline=None)
@given(a=small_finite, b=small_finite, c=small_finite)
def test_logreturn_associativity(a, b, c):
    """(a + b) + c == a + (b + c) within float tolerance."""
    ra, rb, rc = LogReturn(a), LogReturn(b), LogReturn(c)
    left = (ra + rb) + rc
    right = ra + (rb + rc)
    assert math.isclose(float(left), float(right), abs_tol=1e-12)


@settings(max_examples=50, deadline=None)
@given(a=small_finite, b=small_finite)
def test_logreturn_commutativity(a, b):
    ra, rb = LogReturn(a), LogReturn(b)
    assert math.isclose(float(ra + rb), float(rb + ra), abs_tol=1e-12)


@settings(max_examples=50, deadline=None)
@given(a=small_finite)
def test_logreturn_identity(a):
    """a + 0 == a and 0 + a == a."""
    ra = LogReturn(a)
    assert math.isclose(float(ra + LogReturn(0.0)), float(ra), abs_tol=1e-12)
    assert math.isclose(float(LogReturn(0.0) + ra), float(ra), abs_tol=1e-12)


@settings(max_examples=50, deadline=None)
@given(a=small_finite)
def test_logreturn_inverse(a):
    """a + (-a) == 0."""
    ra = LogReturn(a)
    s = ra + (-ra)
    assert math.isclose(float(s), 0.0, abs_tol=1e-12)


@settings(max_examples=50, deadline=None)
@given(a=small_finite, b=small_finite)
def test_logreturn_subtraction_consistent(a, b):
    """a - b == a + (-b)."""
    ra, rb = LogReturn(a), LogReturn(b)
    assert math.isclose(float(ra - rb), float(ra + (-rb)), abs_tol=1e-12)


# --------------------------- LogReturn ↔ Bps round-trip ---------------------

@settings(max_examples=50, deadline=None)
@given(bps=st.floats(min_value=-10000.0, max_value=10000.0,
                     allow_nan=False, allow_infinity=False))
def test_logreturn_from_bps_roundtrip(bps):
    """LogReturn.from_bps(N).to_bps() == N within tolerance."""
    lr = LogReturn.from_bps(bps)
    assert math.isclose(lr.to_bps(), bps, abs_tol=1e-9)


@settings(max_examples=50, deadline=None)
@given(b=st.floats(min_value=-10000.0, max_value=10000.0,
                   allow_nan=False, allow_infinity=False))
def test_bps_logreturn_roundtrip(b):
    """Bps -> LogReturn -> Bps is identity."""
    lr = Bps(b).to_log_return()
    bk = Bps.from_log_return(lr)
    assert math.isclose(float(bk), b, abs_tol=1e-9)


@settings(max_examples=50, deadline=None)
@given(pct=st.floats(min_value=-0.5, max_value=0.5,
                     allow_nan=False, allow_infinity=False))
def test_logreturn_pct_roundtrip(pct):
    """from_pct(x).to_pct() == x."""
    assume(pct > -1.0)  # log1p(-1) is -inf
    lr = LogReturn.from_pct(pct)
    assert math.isclose(lr.to_pct(), pct, abs_tol=1e-12)
