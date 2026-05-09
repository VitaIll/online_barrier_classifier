"""Hypothesis property tests for wagie.core.time primitives.

Affine-space and lattice invariants:
    - Duration: abelian group (assoc, commut, identity, inverse)
    - Timestamp: affine (T - T = D, T + D = T)
    - TimeWindow: lattice under ∩ (commut, idempotent)
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies as st

from wagie.core.time import Duration, Timestamp, TimeWindow


# --------- Strategies — bounded to keep int64 well within bounds ------------

# Up to 10^16 ns ≈ 116 days; comfortably below int64 max.
ns_strategy = st.integers(min_value=-10**15, max_value=10**15)
ts_ns_strategy = st.integers(min_value=0, max_value=10**16)
positive_ns_strategy = st.integers(min_value=1, max_value=10**12)


@st.composite
def durations(draw):
    return Duration(draw(ns_strategy))


@st.composite
def timestamps(draw):
    return Timestamp(draw(ts_ns_strategy))


@st.composite
def time_windows(draw):
    s = draw(timestamps())
    delta = Duration(draw(positive_ns_strategy))
    return TimeWindow(s, s + delta)


# ----------------------- Duration: abelian group ---------------------------

@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(d1=durations(), d2=durations(), d3=durations())
def test_duration_associativity(d1, d2, d3):
    """(d1 + d2) + d3 == d1 + (d2 + d3) — exact for ns ints."""
    assert (d1 + d2) + d3 == d1 + (d2 + d3)


@settings(max_examples=50, deadline=None)
@given(d1=durations(), d2=durations())
def test_duration_commutativity(d1, d2):
    assert d1 + d2 == d2 + d1


@settings(max_examples=50, deadline=None)
@given(d=durations())
def test_duration_identity(d):
    """d + Duration(0) == d == Duration(0) + d."""
    assert d + Duration.zero() == d
    assert Duration.zero() + d == d


@settings(max_examples=50, deadline=None)
@given(d=durations())
def test_duration_inverse(d):
    """d + (-d) == 0 — exact for ns ints."""
    assert d + (-d) == Duration.zero()
    assert (-d) + d == Duration.zero()


@settings(max_examples=50, deadline=None)
@given(d1=durations(), d2=durations())
def test_duration_subtraction_consistent(d1, d2):
    """d1 - d2 == d1 + (-d2)."""
    assert d1 - d2 == d1 + (-d2)


@settings(max_examples=50, deadline=None)
@given(m=st.integers(min_value=-10_000, max_value=10_000))
def test_duration_from_minutes_roundtrip(m):
    """Duration.from_minutes(N).minutes == N for integer minutes."""
    d = Duration.from_minutes(m)
    assert d.minutes == float(m)


# --------------------- Timestamp: affine space -----------------------------

@settings(max_examples=50, deadline=None)
@given(t1=timestamps(), t2=timestamps())
def test_timestamp_minus_timestamp_is_duration(t1, t2):
    d = t2 - t1
    assert isinstance(d, Duration)


@settings(max_examples=50, deadline=None)
@given(t1=timestamps(), t2=timestamps())
def test_timestamp_subtraction_inverts_addition(t1, t2):
    """t1 + (t2 - t1) == t2 — fundamental affine invariant."""
    d = t2 - t1
    assert t1 + d == t2


@settings(max_examples=50, deadline=None)
@given(t=timestamps())
def test_timestamp_self_subtract_is_zero(t):
    assert t - t == Duration.zero()


@settings(max_examples=50, deadline=None)
@given(t=timestamps(), d=durations())
def test_timestamp_add_subtract_roundtrip(t, d):
    assert (t + d) - d == t


# --------------------- TimeWindow: lattice ---------------------------------

@settings(max_examples=50, deadline=None)
@given(w=time_windows())
def test_time_window_intersection_idempotent(w):
    """W ∩ W == W."""
    assert (w & w) == w


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(w1=time_windows(), w2=time_windows())
def test_time_window_intersection_commutes(w1, w2):
    """W1 ∩ W2 == W2 ∩ W1."""
    assert (w1 & w2) == (w2 & w1)


@settings(max_examples=50, deadline=None)
@given(w=time_windows())
def test_time_window_length_is_positive(w):
    """end > start ⇒ length > 0."""
    assert w.length().ns > 0


@settings(max_examples=50, deadline=None)
@given(w=time_windows())
def test_time_window_contains_endpoints_half_open(w):
    """Half-open interval: contains(start) is True, contains(end) is False."""
    assert w.contains(w.start)
    assert not w.contains(w.end)
