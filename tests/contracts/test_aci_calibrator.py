"""Contract tests for MondrianACICalibrator (per-regime adaptive conformal q_t)."""

from __future__ import annotations

import math

import pytest

from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.time import Duration, Timestamp
from wagie.pipeline.mondrian_aci import MondrianACICalibrator


def _bar(close: float = 100.0, segment_id: int = 0, ts_ns: int = 1_000_000_000) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(ts_ns),
        instrument=DEFAULT_INSTRUMENT,
        open=Price(close),
        high=Price(close * 1.01),
        low=Price(close * 0.99),
        close=Price(close),
        volume=Quantity(1.0),
        duration=Duration.from_minutes(20),
        segment_id=segment_id,
    )


def _obs(p_online: float | None = 0.5, regime_id: int | None = 0) -> Observation:
    o = Observation(bar=_bar())
    if regime_id is not None:
        o = o.with_regime(regime_id)
    if p_online is not None:
        o = o.with_p_online(Probability(p_online))
    else:
        # NaN p_online: bypass Probability (which rejects NaN) by setting field directly
        import dataclasses
        o = dataclasses.replace(o, p_online=float("nan"))
    return o


# ---------------------------------------------------------------------------
# Direction of the ACI step
# ---------------------------------------------------------------------------

def test_q_grows_when_all_miscover():
    """LAC score s = 1 - p when y=1. With p_online=0.0, s=1.0 > q_init=0.5,
    so err=1 each step. q_next = q + γ(1 - α) > q."""
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=0.0, regime_id=0)
    q_before = cal._q[0][0.1]
    for _ in range(5):
        cal.update(obs, label=1)
    q_after = cal._q[0][0.1]
    # Expected: q monotonically increases by gamma*(1-alpha) = 0.045 per step.
    assert q_after > q_before
    # Should be capped at 1.0 eventually but not exceed it.
    assert q_after <= 1.0
    # Direction check: roughly q_init + n*gamma*(1-alpha) for small n
    expected = min(1.0, q_before + 5 * 0.05 * (1.0 - 0.1))
    assert math.isclose(q_after, expected, abs_tol=1e-9)


def test_q_shrinks_when_no_miscover():
    """With p_online=1.0 and y=1, s=0.0 < q_init=0.5: err=0 each step.
    q_next = q - γα < q."""
    cal = MondrianACICalibrator(alphas=(0.2,), gamma=0.05, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=1.0, regime_id=0)
    q_before = cal._q[0][0.2]
    for _ in range(5):
        cal.update(obs, label=1)
    q_after = cal._q[0][0.2]
    assert q_after < q_before
    assert q_after >= 0.0
    expected = max(0.0, q_before - 5 * 0.05 * 0.2)
    assert math.isclose(q_after, expected, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Per-regime independence
# ---------------------------------------------------------------------------

def test_per_regime_independence():
    """Updating regime A's q must not move regime B's q."""
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=3, q_init=0.5)
    q_b_before = cal._q[1][0.1]
    obs_a = _obs(p_online=0.0, regime_id=0)
    for _ in range(10):
        cal.update(obs_a, label=1)
    q_a_after = cal._q[0][0.1]
    q_b_after = cal._q[1][0.1]
    assert q_a_after != q_b_before
    assert q_b_after == q_b_before


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_p_online_yields_nan_qlo_and_false_inset():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=None, regime_id=0)  # NaN p_online
    out = cal.transform(obs)
    assert math.isnan(out.q_lo[0.1])
    assert out.in_set[0.1] is False


def test_update_with_nan_p_online_is_noop():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=None, regime_id=0)
    h_before = cal.state_hash()
    cal.update(obs, label=1)
    h_after = cal.state_hash()
    # n_seen unchanged because update() bails before incrementing — but update() does NOT
    # increment n_seen anyway (only transform does). So state_hash must match exactly.
    assert h_after == h_before


def test_update_label_none_is_noop():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=0.5, regime_id=0)
    h_before = cal.state_hash()
    cal.update(obs, label=None)
    assert cal.state_hash() == h_before


def test_update_invalid_label_is_noop():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=0.5, regime_id=0)
    h_before = cal.state_hash()
    cal.update(obs, label="not_an_int")  # type: ignore[arg-type]
    assert cal.state_hash() == h_before


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------

def test_state_dict_round_trip_preserves_hash():
    cal = MondrianACICalibrator(alphas=(0.1, 0.2), gamma=0.05, n_regimes=2, q_init=0.5)
    obs = _obs(p_online=0.0, regime_id=1)
    for _ in range(3):
        cal.update(obs, label=1)
        cal.transform(obs)
    sd = cal.state_dict()
    h_orig = cal.state_hash()

    cal2 = MondrianACICalibrator(alphas=(0.1, 0.2), gamma=0.05, n_regimes=2, q_init=0.5)
    cal2.load_state_dict(sd)
    assert cal2.state_hash() == h_orig
    assert cal2.n_seen == cal.n_seen


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_restores_q_init():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=2, q_init=0.7)
    obs = _obs(p_online=0.0, regime_id=0)
    for _ in range(5):
        cal.update(obs, label=1)
    assert cal._q[0][0.1] != 0.7
    cal.reset()
    assert cal._q[0][0.1] == 0.7
    assert cal._q[1][0.1] == 0.7
    assert cal.n_seen == 0


def test_reset_restores_q_init_by_regime():
    init = {0.1: {0: 0.3, 1: 0.6}}
    cal = MondrianACICalibrator(
        alphas=(0.1,), gamma=0.05, n_regimes=2, q_init=0.5,
        q_init_by_regime=init,
    )
    assert cal._q[0][0.1] == 0.3
    assert cal._q[1][0.1] == 0.6
    obs = _obs(p_online=0.0, regime_id=0)
    for _ in range(3):
        cal.update(obs, label=1)
    cal.reset()
    assert cal._q[0][0.1] == 0.3
    assert cal._q[1][0.1] == 0.6


# ---------------------------------------------------------------------------
# Constructor with q_init_by_regime
# ---------------------------------------------------------------------------

def test_constructor_q_init_by_regime():
    init = {0.1: {0: 0.2, 1: 0.4, 2: 0.6}, 0.2: {0: 0.1}}
    cal = MondrianACICalibrator(
        alphas=(0.1, 0.2), gamma=0.05, n_regimes=3, q_init=0.5,
        q_init_by_regime=init,
    )
    assert cal._q[0][0.1] == 0.2
    assert cal._q[1][0.1] == 0.4
    assert cal._q[2][0.1] == 0.6
    # Missing entries fall back to q_init
    assert cal._q[1][0.2] == 0.5
    assert cal._q[0][0.2] == 0.1


# ---------------------------------------------------------------------------
# Equilibrium under stationary err rate alpha
# ---------------------------------------------------------------------------

def test_equilibrium_stable_under_target_err_rate():
    """If err_t ~ Bernoulli(alpha) i.i.d., E[Δq] = γ(α - α) = 0.
    Drive a deterministic stream with empirical err frequency exactly α and
    confirm q does not explode."""
    cal = MondrianACICalibrator(alphas=(0.2,), gamma=0.05, n_regimes=1, q_init=0.5)
    # Pattern: 1 miscover, 4 covers (rate = 0.2 = alpha) repeated.
    miscover = _obs(p_online=0.0, regime_id=0)   # s=1.0 > q → err=1
    cover = _obs(p_online=1.0, regime_id=0)      # s=0.0 < q → err=0
    pattern = [miscover] + [cover] * 4
    for _ in range(50):
        for o in pattern:
            label = 1
            cal.update(o, label=label)
    q_final = cal._q[0][0.2]
    # Should remain near q_init=0.5 (within a few gamma steps).
    assert 0.0 <= q_final <= 1.0
    assert abs(q_final - 0.5) < 5 * 0.05  # within 5 step-sizes


# ---------------------------------------------------------------------------
# Auxiliary coverage of transform & regime-of helpers
# ---------------------------------------------------------------------------

def test_transform_populates_q_lo_and_in_set():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.01, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=0.6, regime_id=0)
    out = cal.transform(obs)
    assert 0.1 in out.q_lo
    assert 0.1 in out.in_set
    # in_set: p (0.6) >= 1 - q (0.5) → 0.6 >= 0.5 → True
    assert out.in_set[0.1] is True


def test_transform_increments_n_seen():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.01, n_regimes=1, q_init=0.5)
    obs = _obs(p_online=0.5, regime_id=0)
    n0 = cal.n_seen
    cal.transform(obs)
    cal.transform(obs)
    assert cal.n_seen == n0 + 2


def test_regime_id_out_of_range_falls_back_to_zero():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=2, q_init=0.5)
    obs = _obs(p_online=0.0, regime_id=99)  # out of range
    q_zero_before = cal._q[0][0.1]
    cal.update(obs, label=1)
    # update() routed to regime 0 since 99 >= n_regimes
    assert cal._q[0][0.1] != q_zero_before


def test_regime_id_none_falls_back_to_zero():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=2, q_init=0.5)
    obs = Observation(bar=_bar()).with_p_online(Probability(0.0))
    # regime_id is None
    q_zero_before = cal._q[0][0.1]
    cal.update(obs, label=1)
    assert cal._q[0][0.1] != q_zero_before


def test_load_state_dict_partial():
    """load_state_dict tolerates missing regimes/alphas (only updates known)."""
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=2, q_init=0.5)
    cal.load_state_dict({"q": {"0": {"0.1": 0.7}}, "n_seen": 11})
    assert cal._q[0][0.1] == 0.7
    assert cal._q[1][0.1] == 0.5  # untouched
    assert cal.n_seen == 11


def test_load_state_dict_ignores_unknown_keys():
    cal = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    cal.load_state_dict({"q": {"99": {"0.1": 0.9}, "0": {"0.99": 0.9}}, "n_seen": 0})
    assert cal._q[0][0.1] == 0.5
