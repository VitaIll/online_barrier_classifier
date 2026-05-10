"""Tests for AdaptiveThresholdController and CompositeAdaptiveStrategy.

Covers:
  - EWMA correctness vs numpy reference
  - dtau_max clip enforcement
  - P-controller convergence on a constant-rate stream
  - Pause hysteresis (drop below r_min, recover above r_resume)
  - should_enter gates: paused, sigma_ve, p_online thresholds
  - Strategy.get_state() shape
  - Strategy hold-age close on bars_held >= t_max
  - Round-trip state_dict / load_state_dict
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wagie.core.action import Action, ActionKind, Side
from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT, PositionId
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.portfolio import Portfolio
from wagie.core.position import Position
from wagie.core.time import Duration, Timestamp
from wagie.strategy import (
    AdaptiveThresholdController,
    CompositeAdaptiveStrategy,
    StrategyContext,
)


_NS_PER_MIN = 60_000_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(close: float = 100.0, ts_ns: int = _NS_PER_MIN) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(ts_ns), instrument=DEFAULT_INSTRUMENT,
        open=Price(close), high=Price(close * 1.01), low=Price(close * 0.99),
        close=Price(close), volume=Quantity(1.0),
        duration=Duration.from_minutes(20), segment_id=0,
    )


def _obs(p_online: float | None = None,
         sigma_ve: float | None = None,
         ts_ns: int = _NS_PER_MIN) -> Observation:
    obs = Observation(bar=_bar(ts_ns=ts_ns))
    if p_online is not None:
        obs = obs.with_p_online(Probability(p_online))
    if sigma_ve is not None:
        obs = obs.with_sigma_ve(sigma_ve)
    return obs


def _make_controller(**overrides) -> AdaptiveThresholdController:
    defaults = dict(
        tau=0.55, sigma_max=0.30, r_star_init=0.10,
        tau_floor=0.524, tau_ceil=0.95,
        gamma=0.005, lambda_decay=2e-3, dtau_max=0.001,
        r_min=0.02, r_resume=0.05,
    )
    defaults.update(overrides)
    return AdaptiveThresholdController(**defaults)


def _open_position(open_ts: Timestamp,
                   pid_int: int = 1) -> Position:
    return Position.open_at(
        position_id=PositionId(pid_int),
        instrument=DEFAULT_INSTRUMENT,
        side=Side.LONG,
        size=Quantity(1.0),
        entry_price=Price(100.0),
        ts=open_ts,
        take_profit_log=LogReturn.from_bps(41),
        stop_loss_log=LogReturn.from_bps(41),
        expiry=Duration.from_minutes(20 * 100),
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_controller_init_warm_starts_ewmas_at_target() -> None:
    c = _make_controller(r_star_init=0.10)
    assert c.r_hat_ewma == pytest.approx(0.10)
    assert c.r_star_ewma == pytest.approx(0.10)
    assert c.paused is False
    assert c.last_sigma_ve is None


def test_controller_rejects_resume_below_min() -> None:
    with pytest.raises(ValueError, match="r_resume"):
        _make_controller(r_min=0.05, r_resume=0.05)
    with pytest.raises(ValueError, match="r_resume"):
        _make_controller(r_min=0.10, r_resume=0.05)


def test_controller_rejects_invalid_tau_bounds() -> None:
    with pytest.raises(ValueError, match="tau_floor"):
        _make_controller(tau_floor=0.6, tau_ceil=0.5)


# ---------------------------------------------------------------------------
# EWMA correctness
# ---------------------------------------------------------------------------


def test_ewma_matches_numpy_reference() -> None:
    """Compare controller's EWMA to a pure numpy implementation."""
    c = _make_controller(r_star_init=0.10, lambda_decay=0.01)
    rng = np.random.default_rng(seed=42)
    n = 500
    entered_seq = (rng.uniform(0.0, 1.0, n) < 0.30).astype(int)
    label_seq = (rng.uniform(0.0, 1.0, n) < 0.15).astype(int)

    # Reference EWMA (matches the controller's recurrence: x_{t} = (1-λ)x_{t-1} + λ y_t).
    lam = c.lambda_decay
    ref_r_hat = c.r_hat_ewma
    ref_r_star = c.r_star_ewma
    for e, y in zip(entered_seq, label_seq):
        ref_r_hat = (1.0 - lam) * ref_r_hat + lam * float(e)
        ref_r_star = (1.0 - lam) * ref_r_star + lam * float(y)
        c.observe(entered=bool(e), label_matured=int(y))

    assert c.r_hat_ewma == pytest.approx(ref_r_hat, abs=1e-9)
    assert c.r_star_ewma == pytest.approx(ref_r_star, abs=1e-9)


def test_observe_label_none_skips_r_star_update() -> None:
    c = _make_controller(r_star_init=0.10)
    r_star_before = c.r_star_ewma
    c.observe(entered=True, label_matured=None)
    # r_hat moves toward 1.0 (from 0.10), r_star unchanged.
    assert c.r_hat_ewma > 0.10
    assert c.r_star_ewma == pytest.approx(r_star_before, abs=1e-12)


# ---------------------------------------------------------------------------
# dtau_max clip
# ---------------------------------------------------------------------------


def test_dtau_max_clip_respected_on_shock() -> None:
    """Shock the controller with entered=True for 100 bars, label=0 — every
    single update must have |Δτ| ≤ dtau_max."""
    c = _make_controller(dtau_max=1e-3, gamma=1.0,  # huge gain to provoke clip
                         tau=0.60, r_star_init=0.10)
    prev_tau = c.tau
    for _ in range(100):
        c.observe(entered=True, label_matured=0)
        delta = c.tau - prev_tau
        assert abs(delta) <= c.dtau_max + 1e-12
        prev_tau = c.tau


def test_tau_clipped_to_floor_and_ceil() -> None:
    """τ must never escape [tau_floor, tau_ceil]."""
    # Push τ down: entered=False, label=1 → err < 0 → Δτ < 0.
    c_down = _make_controller(tau=0.55, tau_floor=0.524, gamma=10.0,
                              dtau_max=0.05)
    for _ in range(2000):
        c_down.observe(entered=False, label_matured=1)
    assert c_down.tau >= 0.524 - 1e-12

    # Push τ up: entered=True, label=0 → err > 0 → Δτ > 0.
    c_up = _make_controller(tau=0.60, tau_ceil=0.95, gamma=10.0,
                            dtau_max=0.05)
    for _ in range(2000):
        c_up.observe(entered=True, label_matured=0)
    assert c_up.tau <= 0.95 + 1e-12


# ---------------------------------------------------------------------------
# P-controller convergence
# ---------------------------------------------------------------------------


def test_p_controller_pushes_tau_up_when_overentering() -> None:
    """When entered_rate (0.20) is higher than label_rate (0.10) the
    controller's positive error should drive τ monotonically up — until it
    saturates at tau_ceil if integration is long enough."""
    rng = np.random.default_rng(seed=7)
    c = _make_controller(tau=0.55, gamma=0.005, lambda_decay=4e-3,
                         r_star_init=0.10, tau_ceil=0.95, dtau_max=0.001)
    tau_init = c.tau
    n = 5000
    entered_seq = (rng.uniform(0.0, 1.0, n) < 0.20).astype(int)
    label_seq = (rng.uniform(0.0, 1.0, n) < 0.10).astype(int)
    for e, y in zip(entered_seq, label_seq):
        c.observe(entered=bool(e), label_matured=int(y))
    # τ rose substantially under the constant pressure.
    assert c.tau > tau_init + 0.05
    # Each EWMA tracks its own forcing distribution (not each other), but the
    # signed error r_hat - r_star stayed positive throughout — that's what
    # drives τ up.
    assert c.r_hat_ewma > c.r_star_ewma


def test_p_controller_pushes_tau_down_when_underentering() -> None:
    """Symmetric: entered_rate=0.05, label_rate=0.20 → τ drops to floor."""
    rng = np.random.default_rng(seed=11)
    c = _make_controller(tau=0.80, gamma=0.005, lambda_decay=4e-3,
                         r_star_init=0.20, tau_floor=0.524, dtau_max=0.001)
    tau_init = c.tau
    n = 5000
    entered_seq = (rng.uniform(0.0, 1.0, n) < 0.05).astype(int)
    label_seq = (rng.uniform(0.0, 1.0, n) < 0.20).astype(int)
    for e, y in zip(entered_seq, label_seq):
        c.observe(entered=bool(e), label_matured=int(y))
    assert c.tau < tau_init - 0.05


# ---------------------------------------------------------------------------
# Pause hysteresis
# ---------------------------------------------------------------------------


def test_pause_triggers_below_r_min() -> None:
    """Constant label=0 drives r_star_ewma below r_min → paused=True."""
    c = _make_controller(r_star_init=0.10, lambda_decay=0.05,
                         r_min=0.02, r_resume=0.05)
    for _ in range(1000):
        c.observe(entered=False, label_matured=0)
    assert c.paused is True
    assert c.r_star_ewma < c.r_min


def test_resume_above_r_resume_not_at_r_min() -> None:
    """Hysteresis: once paused, must climb above r_resume (> r_min) to resume."""
    c = _make_controller(r_star_init=0.10, lambda_decay=0.05,
                         r_min=0.02, r_resume=0.05)
    # Drop into pause.
    for _ in range(1000):
        c.observe(entered=False, label_matured=0)
    assert c.paused is True

    # Feed labels=1 — r_star_ewma climbs.
    for _ in range(1000):
        c.observe(entered=False, label_matured=1)
    assert c.paused is False
    assert c.r_star_ewma > c.r_resume


def test_pause_state_does_not_flip_without_label_observation() -> None:
    """Updates to ``paused`` only fire when ``label_matured`` is not None."""
    c = _make_controller(r_star_init=0.10, lambda_decay=0.05)
    # Force r_star into pause via labels.
    for _ in range(1000):
        c.observe(entered=False, label_matured=0)
    assert c.paused is True
    # Now feed a long stretch with label=None — paused must stay True.
    for _ in range(1000):
        c.observe(entered=True, label_matured=None)
    assert c.paused is True


def test_tau_frozen_while_paused() -> None:
    """When paused, observe() must not advance τ (no point chasing error)."""
    c = _make_controller(r_star_init=0.10, lambda_decay=0.05,
                         r_min=0.02, r_resume=0.05, tau=0.60)
    # Drop into pause.
    for _ in range(1000):
        c.observe(entered=False, label_matured=0)
    assert c.paused is True
    tau_at_pause = c.tau
    for _ in range(500):
        c.observe(entered=True, label_matured=0)  # Would push tau up if active.
    assert c.tau == pytest.approx(tau_at_pause, abs=1e-12)


# ---------------------------------------------------------------------------
# should_enter gating
# ---------------------------------------------------------------------------


def test_should_enter_false_when_paused() -> None:
    c = _make_controller(tau=0.50, sigma_max=1.0)
    c.paused = True
    assert c.should_enter(0.99, 0.01) is False
    # last_sigma_ve still cached.
    assert c.last_sigma_ve == pytest.approx(0.01)


def test_should_enter_false_when_sigma_ve_above_max() -> None:
    c = _make_controller(tau=0.50, sigma_max=0.30)
    assert c.should_enter(0.99, 0.31) is False
    assert c.last_sigma_ve == pytest.approx(0.31)


def test_should_enter_true_at_threshold_with_low_sigma() -> None:
    c = _make_controller(tau=0.60, sigma_max=0.30)
    assert c.should_enter(0.60, 0.10) is True
    assert c.should_enter(0.61, 0.10) is True
    assert c.should_enter(0.59, 0.10) is False


def test_should_enter_sigma_none_passes_gate() -> None:
    """sigma_ve=None means the upstream stage didn't expose dispersion;
    treat as 'no info', do not block."""
    c = _make_controller(tau=0.50, sigma_max=0.30)
    assert c.should_enter(0.55, None) is True
    assert c.last_sigma_ve is None


# ---------------------------------------------------------------------------
# Controller get_state
# ---------------------------------------------------------------------------


def test_controller_get_state_keys() -> None:
    c = _make_controller()
    c.last_sigma_ve = 0.42
    state = c.get_state()
    assert set(state) == {"tau", "r_hat", "r_star", "paused", "sigma_ve"}
    assert state["tau"] == pytest.approx(c.tau)
    assert state["r_hat"] == pytest.approx(c.r_hat_ewma)
    assert state["r_star"] == pytest.approx(c.r_star_ewma)
    assert state["paused"] is False
    assert state["sigma_ve"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# CompositeAdaptiveStrategy
# ---------------------------------------------------------------------------


def test_strategy_requires_controller() -> None:
    with pytest.raises(ValueError, match="controller"):
        CompositeAdaptiveStrategy(controller=None)


def test_strategy_get_state_includes_required_keys() -> None:
    strat = CompositeAdaptiveStrategy(controller=_make_controller(), t_max=50)
    state = strat.get_state()
    expected = {"tau", "r_hat", "r_star", "paused", "sigma_ve",
                "inv_size", "hold_age_max"}
    assert expected.issubset(state.keys())


def test_strategy_decide_observes_pending_label_then_resets() -> None:
    """update() stashes label, decide() consumes it once + clears."""
    c = _make_controller(r_star_init=0.10, lambda_decay=0.5)  # large lambda for visible move
    strat = CompositeAdaptiveStrategy(controller=c, t_max=50)
    # Stash a positive label via update().
    strat.update(_obs(p_online=0.4), label=1)
    assert strat._pending_label == 1
    r_star_before = c.r_star_ewma
    # Drive a decide — controller observes the label.
    actions = strat.decide(_obs(p_online=0.4), StrategyContext())
    assert strat._pending_label is None
    # r_star moved toward 1.0.
    assert c.r_star_ewma > r_star_before


def test_strategy_emits_open_when_p_above_tau_and_no_position() -> None:
    c = _make_controller(tau=0.50, sigma_max=0.30, r_star_init=0.10)
    strat = CompositeAdaptiveStrategy(controller=c, t_max=50)
    actions = strat.decide(_obs(p_online=0.8, sigma_ve=0.10),
                           StrategyContext())
    assert len(actions) == 1
    a = actions[0]
    assert a.kind == ActionKind.OPEN
    assert a.side == Side.LONG


def test_strategy_no_open_when_already_holding() -> None:
    c = _make_controller(tau=0.50, sigma_max=0.30, r_star_init=0.10)
    strat = CompositeAdaptiveStrategy(controller=c, t_max=50)
    pos = _open_position(Timestamp(_NS_PER_MIN))
    portfolio = Portfolio().with_position(pos)
    actions = strat.decide(_obs(p_online=0.9), StrategyContext(portfolio=portfolio,
                                                                clock_ns=2 * _NS_PER_MIN))
    # Open suppressed because n_open_orders > 0.
    open_actions = [a for a in actions if a.kind == ActionKind.OPEN]
    assert open_actions == []


def test_strategy_no_open_when_paused() -> None:
    c = _make_controller(tau=0.50, sigma_max=0.30, r_star_init=0.10)
    c.paused = True
    strat = CompositeAdaptiveStrategy(controller=c, t_max=50)
    actions = strat.decide(_obs(p_online=0.99, sigma_ve=0.05),
                           StrategyContext())
    open_actions = [a for a in actions if a.kind == ActionKind.OPEN]
    assert open_actions == []


def test_strategy_emits_close_when_bars_held_reaches_t_max() -> None:
    """Position opened at bar 0, current bar at bar t_max => close emitted."""
    c = _make_controller(tau=0.50, sigma_max=0.30, r_star_init=0.10)
    t_max = 5
    strat = CompositeAdaptiveStrategy(controller=c, t_max=t_max, bar_minutes=20)
    open_ts = Timestamp(0)
    pos = _open_position(open_ts)
    portfolio = Portfolio().with_position(pos)
    # Current clock at t_max bars after open.
    current_ns = t_max * 20 * _NS_PER_MIN
    ctx = StrategyContext(portfolio=portfolio, clock_ns=current_ns)
    actions = strat.decide(_obs(p_online=0.4, ts_ns=current_ns), ctx)
    close_actions = [a for a in actions if a.kind == ActionKind.CLOSE]
    assert len(close_actions) == 1
    assert close_actions[0].target_position_id == pos.position_id


def test_strategy_no_close_before_t_max() -> None:
    c = _make_controller(tau=0.50, sigma_max=0.30, r_star_init=0.10)
    t_max = 10
    strat = CompositeAdaptiveStrategy(controller=c, t_max=t_max, bar_minutes=20)
    open_ts = Timestamp(0)
    pos = _open_position(open_ts)
    portfolio = Portfolio().with_position(pos)
    current_ns = (t_max - 1) * 20 * _NS_PER_MIN
    ctx = StrategyContext(portfolio=portfolio, clock_ns=current_ns)
    actions = strat.decide(_obs(p_online=0.4, ts_ns=current_ns), ctx)
    close_actions = [a for a in actions if a.kind == ActionKind.CLOSE]
    assert close_actions == []


def test_strategy_get_state_reflects_inventory_and_age_after_decide() -> None:
    c = _make_controller(tau=0.50, sigma_max=0.30, r_star_init=0.10)
    strat = CompositeAdaptiveStrategy(controller=c, t_max=100, bar_minutes=20)
    pos = _open_position(Timestamp(0))
    portfolio = Portfolio().with_position(pos)
    current_ns = 3 * 20 * _NS_PER_MIN
    ctx = StrategyContext(portfolio=portfolio, clock_ns=current_ns)
    strat.decide(_obs(p_online=0.4, ts_ns=current_ns), ctx)
    state = strat.get_state()
    assert state["inv_size"] == 1
    assert state["hold_age_max"] == 3


def test_strategy_state_dict_round_trip() -> None:
    c = _make_controller(tau=0.55, r_star_init=0.10, lambda_decay=0.1)
    strat = CompositeAdaptiveStrategy(controller=c, t_max=20, bar_minutes=20)
    # Drive some state.
    strat.update(_obs(p_online=0.4), label=1)
    strat.decide(_obs(p_online=0.7, sigma_ve=0.05), StrategyContext())
    snapshot = strat.state_dict()

    # Apply to a fresh strategy.
    c2 = _make_controller(tau=0.10, r_star_init=0.99, lambda_decay=0.1)
    strat2 = CompositeAdaptiveStrategy(controller=c2, t_max=20, bar_minutes=20)
    strat2.load_state_dict(snapshot)
    assert strat2.controller.tau == pytest.approx(snapshot["tau"])
    assert strat2.controller.r_hat_ewma == pytest.approx(snapshot["r_hat_ewma"])
    assert strat2.controller.r_star_ewma == pytest.approx(snapshot["r_star_ewma"])
    assert strat2.controller.paused == snapshot["paused"]


def test_strategy_state_hash_changes_with_tau() -> None:
    c1 = _make_controller(tau=0.55)
    c2 = _make_controller(tau=0.65)
    s1 = CompositeAdaptiveStrategy(controller=c1)
    s2 = CompositeAdaptiveStrategy(controller=c2)
    assert s1.state_hash() != s2.state_hash()


def test_strategy_reset_clears_state() -> None:
    c = _make_controller(r_star_init=0.10, lambda_decay=0.5)
    strat = CompositeAdaptiveStrategy(controller=c, t_max=20, bar_minutes=20)
    strat.update(_obs(), label=1)
    strat.decide(_obs(p_online=0.9, sigma_ve=0.01), StrategyContext())
    assert c.r_hat_ewma != pytest.approx(0.10) or strat._last_entered  # state moved
    strat.reset()
    assert c.r_hat_ewma == pytest.approx(0.10)
    assert c.r_star_ewma == pytest.approx(0.10)
    assert c.paused is False
    assert strat._last_entered is False
    assert strat._pending_label is None


def test_strategy_transform_calls_decide_when_no_ctx_provided() -> None:
    """Stage Protocol: transform() with ctx=None falls through to decide."""
    c = _make_controller(tau=0.50, sigma_max=0.30, r_star_init=0.10)
    strat = CompositeAdaptiveStrategy(controller=c, t_max=20)
    obs = _obs(p_online=0.9, sigma_ve=0.05)
    obs2 = strat.transform(obs)
    assert isinstance(obs2, Observation)
    # Action attached.
    assert any(a.kind == ActionKind.OPEN for a in obs2.actions)
