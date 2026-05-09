"""Algebraic-law tests for wagie.core primitives.

These verify the mathematical structure of the foundation types via Hypothesis.
Failure of any of these is a regression of a fundamental invariant.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from wagie.core import (
    Bps, Duration, LogReturn, Observation, Price, Probability, Quantity,
    Timestamp, TimeWindow,
)


# -------------------------- strategies --------------------------------------

@st.composite
def durations(draw):
    return Duration(draw(st.integers(min_value=-1_000_000_000_000, max_value=1_000_000_000_000)))


@st.composite
def positive_durations(draw):
    return Duration(draw(st.integers(min_value=1, max_value=1_000_000_000_000)))


@st.composite
def timestamps(draw):
    return Timestamp(draw(st.integers(min_value=0, max_value=10_000_000_000_000_000)))


@st.composite
def log_returns(draw):
    return LogReturn(draw(st.floats(min_value=-1.0, max_value=1.0,
                                      allow_nan=False, allow_infinity=False)))


@st.composite
def probabilities(draw):
    return Probability(draw(st.floats(min_value=0.0, max_value=1.0,
                                       allow_nan=False, allow_infinity=False)))


@st.composite
def time_windows(draw):
    s = draw(timestamps())
    delta = draw(positive_durations())
    return TimeWindow(s, s + delta)


# ------------------------ Duration: abelian group ----------------------------

@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(d1=durations(), d2=durations(), d3=durations())
def test_duration_associativity(d1, d2, d3):
    assert (d1 + d2) + d3 == d1 + (d2 + d3)


@given(d=durations())
def test_duration_identity(d):
    assert d + Duration.zero() == d
    assert Duration.zero() + d == d


@given(d=durations())
def test_duration_inverse(d):
    assert d + (-d) == Duration.zero()


@given(d1=durations(), d2=durations())
def test_duration_commutativity(d1, d2):
    assert d1 + d2 == d2 + d1


# ------------------------ LogReturn: abelian group ---------------------------

@given(r1=log_returns(), r2=log_returns(), r3=log_returns())
def test_log_return_associativity(r1, r2, r3):
    a = (r1 + r2) + r3
    b = r1 + (r2 + r3)
    assert math.isclose(float(a), float(b), abs_tol=1e-12)


@given(r=log_returns())
def test_log_return_identity(r):
    assert math.isclose(float(r + LogReturn.zero()), float(r), abs_tol=1e-12)


@given(r=log_returns())
def test_log_return_inverse(r):
    assert math.isclose(float(r + (-r)), 0.0, abs_tol=1e-12)


@given(bps=st.floats(min_value=-10000, max_value=10000,
                     allow_nan=False, allow_infinity=False))
def test_log_return_bps_roundtrip(bps):
    lr = LogReturn.from_bps(bps)
    assert math.isclose(lr.to_bps(), bps, abs_tol=1e-9)


# ------------------------ Timestamp: affine space ----------------------------

@given(t1=timestamps(), t2=timestamps())
def test_timestamp_subtraction_yields_duration(t1, t2):
    d = t2 - t1
    assert isinstance(d, Duration)


@given(t1=timestamps(), t2=timestamps())
def test_timestamp_subtraction_inverts_addition(t1, t2):
    d = t2 - t1
    assert t1 + d == t2


@given(t=timestamps())
def test_timestamp_subtract_self_is_zero(t):
    assert t - t == Duration.zero()


# ------------------------ Probability constraint ----------------------------

@given(v=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_probability_construction_valid(v):
    p = Probability(v)
    assert 0.0 <= float(p) <= 1.0


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("inf"), float("nan")])
def test_probability_rejects_out_of_range(invalid):
    with pytest.raises(ValueError):
        Probability(invalid)


# ------------------------ Price / Quantity invariants -----------------------

@given(v=st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False))
def test_price_construction_positive(v):
    Price(v)


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_price_rejects_non_positive(invalid):
    with pytest.raises(ValueError):
        Price(invalid)


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan")])
def test_quantity_rejects_invalid(invalid):
    with pytest.raises(ValueError):
        Quantity(invalid)


@given(p1=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
       p2=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_price_log_return_to_inverse(p1, p2):
    """Going from p1 to p2 then back should be exact log."""
    a = Price(p1).log_return_to(Price(p2))
    b = Price(p2).log_return_to(Price(p1))
    assert math.isclose(float(a), -float(b), abs_tol=1e-12)


# ------------------------ TimeWindow: lattice -------------------------------

@given(w=time_windows())
def test_time_window_self_intersection_is_self(w):
    assert (w & w) == w


@given(w1=time_windows(), w2=time_windows())
def test_time_window_intersection_commutes(w1, w2):
    assert (w1 & w2) == (w2 & w1)


@given(w=time_windows(), t=timestamps())
def test_time_window_contains_consistent(w, t):
    if w.contains(t):
        assert w.start <= t < w.end


# ------------------------ Observation: with_* commutes ----------------------

def _mk_obs():
    from wagie.core.event import DecisionBar
    bar = DecisionBar(
        ts_init=Timestamp(1_700_000_000_000_000_000),
        open=Price(100.0), high=Price(101.0), low=Price(99.0), close=Price(100.5),
        volume=Quantity(1.0),
    )
    return Observation(bar=bar)


@given(p1=probabilities(), p2=probabilities())
def test_observation_p_offline_p_online_independent(p1, p2):
    obs = _mk_obs()
    a = obs.with_p_offline(p1).with_p_online(p2)
    b = obs.with_p_online(p2).with_p_offline(p1)
    assert a == b


@given(p=probabilities(), r=st.integers(min_value=0, max_value=2))
def test_observation_p_regime_independent(p, r):
    obs = _mk_obs()
    a = obs.with_p_offline(p).with_regime(r)
    b = obs.with_regime(r).with_p_offline(p)
    assert a == b


def test_observation_with_features_immutable():
    obs = _mk_obs()
    obs2 = obs.with_features(x=1.0)
    assert "x" not in obs.features
    assert obs2.features["x"] == 1.0


# ------------------------ Bps <-> LogReturn roundtrip ------------------------

@given(b=st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False))
def test_bps_log_return_roundtrip(b):
    lr = Bps(b).to_log_return()
    bk = Bps.from_log_return(lr)
    assert math.isclose(float(bk), b, abs_tol=1e-9)
