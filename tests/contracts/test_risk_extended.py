"""Extra coverage for `wagie.risk`: happy paths, reset() no-ops, and
the operator-mutation surface (add/remove/replace/reset_all)."""

from __future__ import annotations

import dataclasses

from wagie.core import Action, Duration, LogReturn, Portfolio, Probability, Side, Timestamp
from wagie.core.identity import PositionId
from wagie.risk import (
    KillSwitchPolicy, MaxDrawdownPolicy, MaxLossPerPositionPolicy,
    MaxOrderRatePolicy, MaxPositionsPolicy, RiskEngine,
)


def _open() -> Action:
    return Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(40),
        stop_loss=LogReturn.from_bps(40),
        expiry=Duration.from_minutes(20),
    )


# ---- happy paths (the "approved" branch in each policy) -------------------


def test_max_positions_approves_open_when_under_cap():
    p = MaxPositionsPolicy(max_open=5)
    r = p.check(_open(), Portfolio(), Timestamp(1))
    assert r.approved


def test_max_drawdown_approves_close():
    p = MaxDrawdownPolicy(max_dd=-0.04)
    r = p.check(Action.close(PositionId(1)), Portfolio(), Timestamp(1))
    assert r.approved


def test_max_drawdown_approves_open_when_dd_within_limit():
    p = MaxDrawdownPolicy(max_dd=-0.10)
    r = p.check(_open(), Portfolio(), Timestamp(1))  # dd=0
    assert r.approved


def test_max_loss_per_position_approves_when_stop_within_cap():
    p = MaxLossPerPositionPolicy(max_loss=0.005)
    a = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(40),
        stop_loss=LogReturn.from_bps(40),  # 40 bps < 50 bps cap
        expiry=Duration.from_minutes(20),
    )
    assert p.check(a, Portfolio(), Timestamp(1)).approved


def test_max_loss_per_position_approves_non_open():
    p = MaxLossPerPositionPolicy(max_loss=0.005)
    assert p.check(Action.hold(), Portfolio(), Timestamp(1)).approved


def test_max_order_rate_approves_close():
    p = MaxOrderRatePolicy(max_per_minute=2)
    assert p.check(Action.close(PositionId(1)), Portfolio(), Timestamp(1)).approved


def test_max_order_rate_window_evicts_old_entries():
    p = MaxOrderRatePolicy(max_per_minute=2)
    pf = Portfolio()
    p.check(_open(), pf, Timestamp(0))                          # ts=0s
    p.check(_open(), pf, Timestamp(0))                          # ts=0s
    # 2 minutes later — old entries evicted, new approval succeeds
    assert p.check(_open(), pf, Timestamp(120_000_000_000)).approved


def test_kill_switch_approves_when_disengaged():
    ks = KillSwitchPolicy()
    assert ks.check(_open(), Portfolio(), Timestamp(1)).approved


def test_kill_switch_disengage_clears_reason():
    ks = KillSwitchPolicy()
    ks.engage("vol")
    assert ks.engaged and ks.reason == "vol"
    ks.disengage()
    assert not ks.engaged and ks.reason == ""


def test_kill_switch_allows_scale_out_when_engaged():
    ks = KillSwitchPolicy()
    ks.engage("test")
    a = Action.scale_out(PositionId(1), fraction=Probability(0.5))
    assert ks.check(a, Portfolio(), Timestamp(1)).approved


# ---- reset() no-ops (cover the bodies) ------------------------------------


def test_policy_reset_methods_run():
    for p in [
        MaxPositionsPolicy(),
        MaxDrawdownPolicy(),
        MaxLossPerPositionPolicy(),
    ]:
        p.reset()  # no-op, must not raise


def test_max_order_rate_reset_clears_recent():
    p = MaxOrderRatePolicy(max_per_minute=2)
    p.check(_open(), Portfolio(), Timestamp(0))
    assert len(p._recent) == 1
    p.reset()
    assert len(p._recent) == 0


def test_kill_switch_reset_disengages():
    ks = KillSwitchPolicy()
    ks.engage("x")
    ks.reset()
    assert not ks.engaged


# ---- RiskEngine operator-mutation surface ---------------------------------


def test_risk_engine_add_replaces_when_same_name():
    eng = RiskEngine([MaxPositionsPolicy(max_open=5)])
    eng.add(MaxPositionsPolicy(max_open=10))
    # Still one policy by that name; max_open is now 10
    assert len(eng.policies) == 1
    assert eng.get("max_positions").max_open == 10
    assert any(c["op"] == "replace" for c in eng.changes)


def test_risk_engine_replace_returns_false_when_not_found():
    eng = RiskEngine([MaxPositionsPolicy()])
    assert eng.replace("nope", KillSwitchPolicy()) is False


def test_risk_engine_remove_returns_false_when_not_found():
    eng = RiskEngine([MaxPositionsPolicy()])
    assert eng.remove("nope") is False


def test_risk_engine_reset_all_disengages_kill():
    ks = KillSwitchPolicy()
    ks.engage("x")
    eng = RiskEngine([ks, MaxOrderRatePolicy(max_per_minute=2)])
    eng.policies[1].check(_open(), Portfolio(), Timestamp(1))
    eng.reset_all()
    assert not ks.engaged
    assert len(eng.policies[1]._recent) == 0


def test_risk_engine_default_includes_max_positions_and_kill_switch():
    eng = RiskEngine.default()
    names = eng.policy_names
    assert "max_positions" in names
    assert "kill_switch" in names


def test_risk_engine_get_returns_none_when_missing():
    eng = RiskEngine([])
    assert eng.get("anything") is None
