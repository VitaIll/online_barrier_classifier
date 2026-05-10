"""G9 contracts: RiskEngine + policies."""

from __future__ import annotations

import dataclasses

from wagie.core import (
    DEFAULT_INSTRUMENT, Action, Duration, ExitReason, LogReturn, Portfolio,
    Position, Price, Probability, Quantity, Side, Timestamp,
)
from wagie.core.identity import PositionId
from wagie.risk import (
    KillSwitchPolicy, MaxDrawdownPolicy, MaxLossPerPositionPolicy,
    MaxOrderRatePolicy, MaxPositionsPolicy, RiskEngine,
)


def _mk_open_action():
    return Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(40),
        stop_loss=LogReturn.from_bps(40),
        expiry=Duration.from_minutes(20),
    )


def _mk_position(pid=1):
    return Position.open_at(
        position_id=PositionId(pid),
        instrument=DEFAULT_INSTRUMENT,
        side=Side.LONG,
        size=Quantity(1.0),
        entry_price=Price(100.0),
        ts=Timestamp(1),
        take_profit_log=LogReturn.from_bps(40),
        stop_loss_log=LogReturn.from_bps(40),
        expiry=Duration.from_minutes(20),
    )


def test_max_positions_blocks_when_full():
    pf = Portfolio()
    for i in range(3):
        pf = pf.with_position(_mk_position(pid=i + 1))
    p = MaxPositionsPolicy(max_open=3)
    r = p.check(_mk_open_action(), pf, Timestamp(1))
    assert not r.approved
    assert "max_open" in r.reason


def test_max_positions_allows_close_when_full():
    pf = Portfolio()
    for i in range(3):
        pf = pf.with_position(_mk_position(pid=i + 1))
    p = MaxPositionsPolicy(max_open=3)
    r = p.check(Action.close(PositionId(1)), pf, Timestamp(1))
    assert r.approved


def test_max_drawdown_blocks_open_when_exceeded():
    pf = dataclasses.replace(
        Portfolio(),
        realized_pnl_log=LogReturn(-0.05),
        high_water_mark_log=LogReturn(0.02),  # peak was +2%, now -5% from peak
    )
    p = MaxDrawdownPolicy(max_dd=-0.04)
    r = p.check(_mk_open_action(), pf, Timestamp(1))
    assert not r.approved


def test_max_drawdown_allows_close():
    pf = dataclasses.replace(
        Portfolio(),
        realized_pnl_log=LogReturn(-0.05),
        high_water_mark_log=LogReturn(0.02),
    )
    p = MaxDrawdownPolicy(max_dd=-0.04)
    r = p.check(Action.close(PositionId(1)), pf, Timestamp(1))
    assert r.approved


def test_max_loss_per_position_suggests_scaled_stop():
    p = MaxLossPerPositionPolicy(max_loss=0.005)
    a = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(40),
        stop_loss=LogReturn.from_bps(80),    # 80bps > 50bps cap
        expiry=Duration.from_minutes(20),
    )
    r = p.check(a, Portfolio(), Timestamp(1))
    assert not r.approved
    assert r.suggested_action is not None
    assert abs(float(r.suggested_action.stop_loss) - 0.005) < 1e-9


def test_max_order_rate_blocks_after_threshold():
    p = MaxOrderRatePolicy(max_per_minute=2)
    pf = Portfolio()
    # Three opens within the same second
    ts = Timestamp(1_000_000_000)
    assert p.check(_mk_open_action(), pf, ts).approved
    assert p.check(_mk_open_action(), pf, ts).approved
    r = p.check(_mk_open_action(), pf, ts)
    assert not r.approved


def test_kill_switch_blocks_open_only():
    ks = KillSwitchPolicy()
    ks.engage("test")
    pf = Portfolio()
    ts = Timestamp(1)
    assert not ks.check(_mk_open_action(), pf, ts).approved
    assert ks.check(Action.close(PositionId(1)), pf, ts).approved
    assert ks.check(Action.cancel(PositionId(1)), pf, ts).approved
    assert ks.check(Action.modify_stop(PositionId(1),
                                        new_stop_loss=LogReturn.from_bps(20)),
                    pf, ts).approved


def test_risk_engine_first_rejection_wins():
    eng = RiskEngine([
        MaxPositionsPolicy(max_open=10),
        KillSwitchPolicy(engaged=True, reason="test"),
    ])
    r = eng.check(_mk_open_action(), Portfolio(), Timestamp(1))
    assert not r.approved
    assert r.policy_name == "kill_switch"


def test_risk_engine_hot_swap():
    eng = RiskEngine([MaxPositionsPolicy(max_open=10)])
    assert "max_positions" in eng.policy_names
    eng.add(KillSwitchPolicy())
    assert "kill_switch" in eng.policy_names
    eng.remove("max_positions")
    assert "max_positions" not in eng.policy_names
    assert any(c["op"] == "remove" and c["name"] == "max_positions"
               for c in eng.changes)


# ---- RiskEngine.default() now wires four policies for realistic backtests ----


def test_default_engine_wires_four_policies():
    """Default engine must include MaxPositions + MaxDrawdown + MaxLossPerPosition +
    KillSwitch — the audit recommendation for default-safe backtests."""
    eng = RiskEngine.default()
    names = set(eng.policy_names)
    assert {
        "max_positions",
        "max_drawdown",
        "max_loss_per_position",
        "kill_switch",
    }.issubset(names)
    assert len(eng.policies) >= 4


def test_default_engine_max_dd_threshold_is_30pct():
    """Audit recommendation: default max_dd is -0.30 (30% peak-to-trough)."""
    eng = RiskEngine.default()
    p = eng.get("max_drawdown")
    assert p is not None
    assert abs(p.max_dd - (-0.30)) < 1e-9


def test_default_engine_rejects_open_at_drawdown_breach():
    """An open at deep drawdown must be rejected by the wired MaxDrawdownPolicy."""
    eng = RiskEngine.default()
    pf = dataclasses.replace(
        Portfolio(),
        realized_pnl_log=LogReturn(-0.40),    # -40% from 0 peak
        high_water_mark_log=LogReturn(0.0),
    )
    r = eng.check(_mk_open_action(), pf, Timestamp(1))
    assert not r.approved
    assert r.policy_name == "max_drawdown"


def test_default_engine_max_loss_per_position_default_is_5pct():
    """Default cap on per-position SL is 5% (0.05 log-units)."""
    eng = RiskEngine.default()
    p = eng.get("max_loss_per_position")
    assert p is not None
    assert abs(p.max_loss - 0.05) < 1e-9


def test_default_engine_kill_switch_starts_disengaged():
    """default_armed() returns a wired-but-NOT-firing kill switch — operator
    must explicitly engage."""
    eng = RiskEngine.default()
    ks = eng.get("kill_switch")
    assert ks is not None
    assert ks.engaged is False
    # Verify operator can engage it and it then blocks opens.
    ks.engage("manual halt")
    r = eng.check(_mk_open_action(), Portfolio(), Timestamp(1))
    assert not r.approved
    assert r.policy_name == "kill_switch"


def test_kill_switch_default_armed_factory():
    """KillSwitchPolicy.default_armed(armed=False) returns a fully-disengaged
    policy with no operator intent metadata."""
    ks = KillSwitchPolicy.default_armed(armed=False)
    assert ks.engaged is False
    assert ks.reason == ""

    ks_armed = KillSwitchPolicy.default_armed(armed=True)
    assert ks_armed.engaged is False         # armed != engaged
    assert ks_armed.reason != ""
