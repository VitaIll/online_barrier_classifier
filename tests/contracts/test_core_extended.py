"""Extra coverage for the small accessors / NotImplemented branches in
`wagie.core.numeric` and `wagie.core.time`. The big algebraic-law tests live
in test_core_algebra.py; this file targets the leftover lines."""

from __future__ import annotations

import math

import pytest

from wagie.core.numeric import Bps, LogReturn, Price, Probability, Quantity
from wagie.core.time import Duration, Timestamp, TimeWindow


# ---- Price / Quantity validation ------------------------------------------


def test_price_rejects_zero():
    with pytest.raises(ValueError):
        Price(0.0)


def test_price_rejects_negative():
    with pytest.raises(ValueError):
        Price(-1.0)


def test_price_rejects_nan():
    with pytest.raises(ValueError):
        Price(float("nan"))


def test_price_repr():
    assert "Price(" in repr(Price(100.0))


def test_price_log_return_to_other_price():
    p1 = Price(100.0)
    p2 = Price(110.0)
    assert isinstance(p1.log_return_to(p2), LogReturn)
    assert p1.log_return_to(p2) == pytest.approx(math.log(1.10))


def test_quantity_rejects_negative():
    with pytest.raises(ValueError):
        Quantity(-0.001)


def test_quantity_rejects_nan():
    with pytest.raises(ValueError):
        Quantity(float("nan"))


def test_quantity_zero_ok():
    assert float(Quantity(0.0)) == 0.0


def test_quantity_repr():
    assert "Quantity(" in repr(Quantity(1.5))


# ---- Probability ----------------------------------------------------------


def test_probability_rejects_above_one():
    with pytest.raises(ValueError):
        Probability(1.0001)


def test_probability_rejects_negative():
    with pytest.raises(ValueError):
        Probability(-1e-9)


def test_probability_rejects_nan():
    with pytest.raises(ValueError):
        Probability(float("nan"))


def test_probability_half():
    assert float(Probability.half()) == 0.5


def test_probability_repr():
    assert "Probability(" in repr(Probability(0.7))


# ---- LogReturn ------------------------------------------------------------


def test_log_return_rejects_nan():
    with pytest.raises(ValueError):
        LogReturn(float("nan"))


def test_log_return_zero():
    assert float(LogReturn.zero()) == 0.0


def test_log_return_radd():
    a = LogReturn(0.01)
    b = LogReturn(0.02)
    # __radd__ path
    assert (b.__radd__(a)) == pytest.approx(0.03)


def test_log_return_sub():
    a = LogReturn(0.05)
    b = LogReturn(0.02)
    assert float(a - b) == pytest.approx(0.03)


def test_log_return_add_returns_notimplemented_for_non_logreturn():
    a = LogReturn(0.01)
    assert a.__add__(1.0) is NotImplemented
    assert a.__sub__(1.0) is NotImplemented
    assert a.__radd__(1.0) is NotImplemented


def test_log_return_to_pct_round_trip():
    pct = 0.05
    r = LogReturn.from_pct(pct)
    assert r.to_pct() == pytest.approx(pct)


def test_log_return_to_bps_round_trip():
    bps = 41.11
    assert LogReturn.from_bps(bps).to_bps() == pytest.approx(bps)


def test_log_return_neg():
    a = LogReturn(0.01)
    assert float(-a) == -0.01


def test_log_return_repr():
    assert "LogReturn(" in repr(LogReturn(0.0041113))


# ---- Bps ------------------------------------------------------------------


def test_bps_rejects_nan():
    with pytest.raises(ValueError):
        Bps(float("nan"))


def test_bps_to_and_from_log_return():
    b = Bps(41.11)
    r = b.to_log_return()
    assert isinstance(r, LogReturn)
    assert r.to_bps() == pytest.approx(41.11)
    b2 = Bps.from_log_return(r)
    assert float(b2) == pytest.approx(41.11)


def test_bps_repr():
    assert "Bps(" in repr(Bps(10.0))


# ---- Timestamp ------------------------------------------------------------


def test_timestamp_post_init_coerces_non_int_to_int():
    t = Timestamp(1.7e18)
    assert isinstance(t.ns, int)


def test_timestamp_from_seconds():
    t = Timestamp.from_seconds(1.5)
    assert t.ns == int(1.5 * 1_000_000_000)


def test_timestamp_from_iso_naive_assumed_utc():
    t = Timestamp.from_iso("2024-01-01T00:00:00")
    assert isinstance(t, Timestamp)
    assert t.isoformat().endswith("Z") or "+00:00" in t.isoformat()


def test_timestamp_from_iso_with_tz():
    t = Timestamp.from_iso("2024-01-01T00:00:00+00:00")
    assert isinstance(t, Timestamp)


def test_timestamp_seconds_property():
    t = Timestamp(1_000_000_000)  # 1 second
    assert t.seconds == 1.0


def test_timestamp_sub_duration():
    t = Timestamp(2_000_000_000)
    d = Duration.from_seconds(0.5)
    res = t - d
    assert isinstance(res, Timestamp)
    assert res.ns == 1_500_000_000


def test_timestamp_add_returns_notimplemented_for_non_duration():
    t = Timestamp(0)
    assert t.__add__(5) is NotImplemented


def test_timestamp_sub_returns_notimplemented_for_non_ts_or_dur():
    t = Timestamp(0)
    assert t.__sub__("five") is NotImplemented


# ---- Duration -------------------------------------------------------------


def test_duration_post_init_coerces_non_int():
    d = Duration(1.5)
    assert isinstance(d.ns, int)


def test_duration_from_hours():
    d = Duration.from_hours(2)
    assert d.ns == 2 * 3600 * 1_000_000_000


def test_duration_from_days():
    d = Duration.from_days(1)
    assert d.ns == 24 * 3600 * 1_000_000_000


def test_duration_seconds_minutes_hours_properties():
    d = Duration.from_hours(1.5)
    assert d.seconds == pytest.approx(5400.0)
    assert d.minutes == pytest.approx(90.0)
    assert d.hours == pytest.approx(1.5)


def test_duration_sub_returns_duration():
    a = Duration.from_minutes(10)
    b = Duration.from_minutes(3)
    assert float((a - b).minutes) == pytest.approx(7.0)


def test_duration_add_returns_notimplemented_for_non_duration():
    d = Duration.from_seconds(1)
    assert d.__add__(1) is NotImplemented
    assert d.__sub__(1) is NotImplemented


def test_duration_mul_scalar():
    d = Duration.from_minutes(10)
    assert (d * 2).minutes == pytest.approx(20.0)
    assert (3 * d).minutes == pytest.approx(30.0)
    assert d.__mul__("nope") is NotImplemented


def test_duration_repr_minutes_branch():
    d = Duration.from_minutes(15)
    assert "Duration(minutes=" in repr(d)


def test_duration_repr_ns_branch():
    d = Duration(123)
    assert "Duration(ns=123)" in repr(d)


# ---- TimeWindow -----------------------------------------------------------


def test_time_window_shift():
    w = TimeWindow(start=Timestamp(0), end=Timestamp(1_000_000_000))
    w2 = w.shift(Duration.from_seconds(1))
    assert w2.start.seconds == pytest.approx(1.0)
    assert w2.end.seconds == pytest.approx(2.0)


def test_time_window_repr_includes_iso_arrow():
    w = TimeWindow(start=Timestamp.from_seconds(0), end=Timestamp.from_seconds(1))
    s = repr(w)
    assert "TimeWindow(" in s and "→" in s
