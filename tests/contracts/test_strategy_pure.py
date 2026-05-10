"""Strategy is a pure callable returning Sequence[Action] (P4 breaking).

The Mondrian-ACI conformal layer is gone — strategies gate on the calibrated
``p_online`` directly via a ThresholdGate (``p_online >= tau``).  The legacy
``pure_conformal`` registry name is kept as an alias for back-compat YAMLs.
"""

from __future__ import annotations

from wagie.core import (
    Action, ActionKind, DecisionBar, Duration, LogReturn, Observation,
    Portfolio, Price, Probability, Quantity, Side, Timestamp,
)
from wagie.strategy import (
    STRATEGY_REGISTRY,
    EvCalibratedSize,
    PureConformalGate,
    StrategyContext,
    ThresholdGate,
    build_strategy,
)


def _mk_obs(p_online=None, regime_id=0) -> Observation:
    bar = DecisionBar(
        ts_init=Timestamp(1_700_000_000_000_000_000),
        open=Price(100.0), high=Price(101.0), low=Price(99.0),
        close=Price(100.5), volume=Quantity(1.0),
    )
    obs = Observation(bar=bar)
    if p_online is not None:
        obs = obs.with_p_online(Probability(p_online))
    obs = obs.with_regime(regime_id)
    return obs


def _ctx(n_open: int = 0) -> StrategyContext:
    return StrategyContext(portfolio=Portfolio())


# -----------------------------------------------------------------------------
# ThresholdGate
# -----------------------------------------------------------------------------


def test_threshold_gate_returns_empty_when_below_tau():
    s = ThresholdGate(name="s", tau=0.5)
    obs = _mk_obs(p_online=0.4)
    assert tuple(s.decide(obs, _ctx())) == ()


def test_threshold_gate_signals_when_above_tau():
    s = ThresholdGate(name="s", tau=0.5)
    obs = _mk_obs(p_online=0.7)
    actions = list(s.decide(obs, _ctx()))
    assert len(actions) == 1
    a = actions[0]
    assert isinstance(a, Action)
    assert a.kind == ActionKind.OPEN
    assert a.side == Side.LONG
    assert float(a.size) == 1.0


def test_threshold_gate_returns_empty_when_no_p_online():
    s = ThresholdGate(name="s", tau=0.5)
    obs = _mk_obs(p_online=None)
    assert tuple(s.decide(obs, _ctx())) == ()


def test_threshold_gate_at_exact_tau_signals():
    """tau is inclusive — p_online == tau opens."""
    s = ThresholdGate(name="s", tau=0.5)
    obs = _mk_obs(p_online=0.5)
    actions = list(s.decide(obs, _ctx()))
    assert len(actions) == 1


# -----------------------------------------------------------------------------
# EvCalibratedSize (rebased on p_online directly)
# -----------------------------------------------------------------------------


def test_ev_calibrated_size_proportional():
    s = EvCalibratedSize(name="s", tau=0.5, k=5.0)
    obs = _mk_obs(p_online=0.7)
    actions = list(s.decide(obs, _ctx()))
    assert len(actions) == 1
    assert actions[0].kind == ActionKind.OPEN
    # margin = 0.7 - 0.5 = 0.2; size = clip(5 * 0.2, 0, 1) = 1.0
    assert abs(float(actions[0].size) - 1.0) < 1e-9


def test_ev_calibrated_size_partial_size():
    s = EvCalibratedSize(name="s", tau=0.5, k=2.0)
    obs = _mk_obs(p_online=0.7)
    actions = list(s.decide(obs, _ctx()))
    # margin = 0.2, size = clip(2*0.2, 0, 1) = 0.4
    assert len(actions) == 1
    assert abs(float(actions[0].size) - 0.4) < 1e-9


def test_ev_calibrated_size_below_tau_empty():
    s = EvCalibratedSize(name="s", tau=0.5, k=5.0)
    obs = _mk_obs(p_online=0.4)
    assert tuple(s.decide(obs, _ctx())) == ()


def test_strategy_can_emit_multiple_actions():
    """A strategy MAY return multiple actions in one tick — open + scale + modify."""
    from wagie.core.identity import PositionId

    class MultiAction(ThresholdGate):
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
    actions = list(s.decide(_mk_obs(p_online=0.9), _ctx()))
    assert len(actions) == 3
    assert {a.kind for a in actions} == {ActionKind.OPEN, ActionKind.MODIFY_STOP,
                                           ActionKind.CLOSE}


# -----------------------------------------------------------------------------
# Registry: both alias names
# -----------------------------------------------------------------------------


def test_registry_threshold_gate_alias():
    assert STRATEGY_REGISTRY["threshold_gate"] is ThresholdGate


def test_registry_pure_conformal_alias_resolves_to_threshold():
    """Back-compat: legacy YAMLs say `pure_conformal`; we now build a
    ThresholdGate underneath."""
    assert STRATEGY_REGISTRY["pure_conformal"] is ThresholdGate


def test_pure_conformal_gate_export_is_threshold_gate():
    """The PureConformalGate symbol still exists for legacy import paths but
    aliases to ThresholdGate."""
    assert PureConformalGate is ThresholdGate


def test_build_strategy_threshold_gate_kwarg():
    s = build_strategy("threshold_gate", tau=0.3)
    assert isinstance(s, ThresholdGate)
    assert s.tau == 0.3


def test_build_strategy_pure_conformal_alias():
    """Legacy `pure_conformal` kind still works; tau is honored."""
    s = build_strategy("pure_conformal", tau=0.2)
    assert isinstance(s, ThresholdGate)
    assert s.tau == 0.2


def test_build_strategy_drops_legacy_alpha_kwarg():
    """If a legacy YAML or test passes alpha=, build_strategy quietly drops it."""
    s = build_strategy("threshold_gate", tau=0.5, alpha=0.10)
    assert isinstance(s, ThresholdGate)


def test_build_strategy_ev_calibrated_size_kwargs():
    s = build_strategy("ev_calibrated_size", tau=0.4, k=3.0)
    assert isinstance(s, EvCalibratedSize)
    assert s.tau == 0.4
    assert s.k == 3.0
