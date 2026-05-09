"""Strategy is a pure callable returning Sequence[Action] (P4 breaking)."""

from __future__ import annotations

from wagie.core import (
    Action, ActionKind, DecisionBar, Duration, LogReturn, Observation,
    Portfolio, Price, Probability, Quantity, Side, Timestamp,
)
from wagie.strategy import EvCalibratedSize, PureConformalGate, StrategyContext


def _mk_obs(p_online=None, in_set_10=False, q_lo_10=0.5, regime_id=0) -> Observation:
    bar = DecisionBar(
        ts_init=Timestamp(1_700_000_000_000_000_000),
        open=Price(100.0), high=Price(101.0), low=Price(99.0),
        close=Price(100.5), volume=Quantity(1.0),
    )
    obs = Observation(bar=bar)
    if p_online is not None:
        obs = obs.with_p_online(Probability(p_online))
    obs = obs.with_calibration(0.10, q_lo_10, in_set_10)
    obs = obs.with_regime(regime_id)
    return obs


def _ctx(n_open: int = 0) -> StrategyContext:
    return StrategyContext(portfolio=Portfolio())


def test_pure_conformal_gate_returns_empty_when_not_in_set():
    s = PureConformalGate(name="s", alpha=0.10)
    obs = _mk_obs(p_online=0.7, in_set_10=False)
    actions = s.decide(obs, _ctx())
    assert tuple(actions) == ()


def test_pure_conformal_gate_signals_when_in_set():
    s = PureConformalGate(name="s", alpha=0.10)
    obs = _mk_obs(p_online=0.7, in_set_10=True)
    actions = list(s.decide(obs, _ctx()))
    assert len(actions) == 1
    a = actions[0]
    assert isinstance(a, Action)
    assert a.kind == ActionKind.OPEN
    assert a.side == Side.LONG
    assert float(a.size) == 1.0


def test_ev_calibrated_size_proportional():
    s = EvCalibratedSize(name="s", alpha=0.10, k=5.0)
    obs = _mk_obs(p_online=0.7, in_set_10=True, q_lo_10=0.4)
    actions = list(s.decide(obs, _ctx()))
    assert len(actions) == 1
    assert actions[0].kind == ActionKind.OPEN
    assert abs(float(actions[0].size) - 0.5) < 1e-9


def test_strategy_can_emit_multiple_actions():
    """A strategy MAY return multiple actions in one tick — open + scale + modify."""
    from wagie.core.identity import PositionId

    class MultiAction(PureConformalGate):
        def decide(self, frame, ctx):
            return [
                Action.open(side=Side.LONG, size=Probability(0.5),
                            take_profit=LogReturn.from_bps(40),
                            stop_loss=LogReturn.from_bps(40),
                            expiry=Duration.from_minutes(20)),
                Action.modify_stop(PositionId(7),
                                    new_stop_loss=LogReturn.from_bps(20)),
                Action.close(PositionId(3)),
            ]

    s = MultiAction(name="multi")
    actions = list(s.decide(_mk_obs(in_set_10=True), _ctx()))
    assert len(actions) == 3
    assert {a.kind for a in actions} == {ActionKind.OPEN, ActionKind.MODIFY_STOP,
                                           ActionKind.CLOSE}
