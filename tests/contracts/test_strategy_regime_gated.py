"""RegimeGatedStrategy contracts.

Asserts that the regime gate suppresses OPEN/SCALE_IN actions when the bar's
regime_id is not in the allow-list, while leaving inventory-management actions
(CLOSE / SCALE_OUT / MODIFY_STOP / CANCEL / HOLD) untouched.

Also tests integration with the registry and a base strategy stub that
mimics the threshold-gate signal surface (so this test passes whether ARCH's
ThresholdGate is in tree or not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from wagie.core.action import Action, ActionKind, Side
from wagie.core.event import DecisionBar
from wagie.core.identity import PositionId
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.portfolio import Portfolio
from wagie.core.time import Duration, Timestamp
from wagie.strategy import (
    PureConformalGate,
    RegimeGatedStrategy,
    STRATEGY_REGISTRY,
    StrategyBase,
    StrategyContext,
    build_strategy,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mk_obs(*, regime_id, p_online=0.7, **_unused):
    """_unused absorbs legacy `in_set_alpha=` / `in_set=` kwargs from the
    pre-conformal-removal API; the gate now reads p_online directly."""
    bar = DecisionBar(
        ts_init=Timestamp(1_700_000_000_000_000_000),
        open=Price(100.0), high=Price(101.0), low=Price(99.0),
        close=Price(100.5), volume=Quantity(1.0),
    )
    obs = Observation(bar=bar)
    obs = obs.with_p_online(Probability(p_online))
    if regime_id is not None:
        obs = obs.with_regime(regime_id)
    return obs


def _ctx() -> StrategyContext:
    return StrategyContext(portfolio=Portfolio())


# A minimal stub that always emits an OPEN action — useful so this test
# exercises gating without depending on ARCH's ThresholdGate.
@dataclass(frozen=True, slots=True)
class _AlwaysOpen(StrategyBase):
    name: str = "always_open"

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        return (self._open(1.0),)


# A stub that returns a mix of OPEN + management actions to verify selective gating.
@dataclass(frozen=True, slots=True)
class _OpenAndManage(StrategyBase):
    name: str = "open_and_manage"

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        return (
            self._open(0.5),                                  # OPEN — gate-able
            Action.close(PositionId(7)),                      # always allowed
            Action.modify_stop(PositionId(3),
                                new_stop_loss=LogReturn.from_bps(20)),  # always allowed
            Action.scale_out(PositionId(4),
                             fraction=Probability(0.5)),     # always allowed
            Action.cancel(PositionId(5)),                     # always allowed
        )


# ---------------------------------------------------------------------------
# core gating
# ---------------------------------------------------------------------------


def test_suppresses_open_in_disallowed_regime():
    inner = _AlwaysOpen()
    g = RegimeGatedStrategy(
        name="g", base=inner, allowed_regimes=(0,),
        take_profit=inner.take_profit, stop_loss=inner.stop_loss, expiry=inner.expiry,
    )
    obs = _mk_obs(regime_id=2)
    actions = list(g.decide(obs, _ctx()))
    assert actions == []


def test_passes_open_in_allowed_regime():
    inner = _AlwaysOpen()
    g = RegimeGatedStrategy(
        name="g", base=inner, allowed_regimes=(0, 1),
        take_profit=inner.take_profit, stop_loss=inner.stop_loss, expiry=inner.expiry,
    )
    for r in (0, 1):
        obs = _mk_obs(regime_id=r)
        actions = list(g.decide(obs, _ctx()))
        assert len(actions) == 1
        assert actions[0].kind == ActionKind.OPEN


def test_management_actions_pass_through_in_disallowed_regime():
    """CLOSE / SCALE_OUT / MODIFY_STOP / CANCEL must always be allowed —
    you must always be able to manage existing inventory regardless of regime.
    """
    inner = _OpenAndManage()
    g = RegimeGatedStrategy(
        name="g", base=inner, allowed_regimes=(0,),
        take_profit=inner.take_profit, stop_loss=inner.stop_loss, expiry=inner.expiry,
    )
    obs = _mk_obs(regime_id=99)  # not in allow-list
    actions = list(g.decide(obs, _ctx()))
    kinds = {a.kind for a in actions}
    # OPEN suppressed; the four management actions remain
    assert ActionKind.OPEN not in kinds
    assert kinds == {
        ActionKind.CLOSE, ActionKind.MODIFY_STOP,
        ActionKind.SCALE_OUT, ActionKind.CANCEL,
    }


def test_missing_regime_id_treated_as_disallowed():
    """If regime_id is None we err on the side of suppression."""
    inner = _AlwaysOpen()
    g = RegimeGatedStrategy(
        name="g", base=inner, allowed_regimes=(0,),
        take_profit=inner.take_profit, stop_loss=inner.stop_loss, expiry=inner.expiry,
    )
    obs = _mk_obs(regime_id=None)
    actions = list(g.decide(obs, _ctx()))
    assert actions == []


def test_construction_requires_base():
    with pytest.raises(ValueError, match="base"):
        RegimeGatedStrategy(name="g", base=None, allowed_regimes=(0,))


# ---------------------------------------------------------------------------
# registry / yaml-style construction
# ---------------------------------------------------------------------------


def test_registry_includes_regime_gated():
    assert "regime_gated" in STRATEGY_REGISTRY


def test_build_strategy_regime_gated_wraps_pure_conformal():
    """YAML kind=regime_gated, base_kind=pure_conformal must work today (does
    not require ARCH's threshold_gate to be present yet)."""
    g = build_strategy(
        "regime_gated",
        base_kind="pure_conformal",
        allowed_regimes=[0, 1],
        alpha=0.10,
    )
    assert isinstance(g, RegimeGatedStrategy)
    assert g.allowed_regimes == (0, 1)
    # disallowed regime ⇒ no actions
    obs = _mk_obs(regime_id=5, in_set=True)
    assert list(g.decide(obs, _ctx())) == []
    # allowed regime + in_set ⇒ open
    obs = _mk_obs(regime_id=0, in_set=True)
    out = list(g.decide(obs, _ctx()))
    assert len(out) == 1
    assert out[0].kind == ActionKind.OPEN


def test_build_strategy_regime_gated_rejects_recursive_wrap():
    with pytest.raises(ValueError, match="cannot wrap regime_gated"):
        build_strategy(
            "regime_gated",
            base_kind="regime_gated",
            allowed_regimes=[0],
        )


def test_build_strategy_regime_gated_rejects_unknown_base_kind():
    with pytest.raises(ValueError, match="unknown base_kind"):
        build_strategy(
            "regime_gated",
            base_kind="phony_strategy",
            allowed_regimes=[0],
        )


def test_state_hash_changes_with_allow_list():
    inner = _AlwaysOpen()
    g1 = RegimeGatedStrategy(
        name="g", base=inner, allowed_regimes=(0,),
        take_profit=inner.take_profit, stop_loss=inner.stop_loss, expiry=inner.expiry,
    )
    g2 = RegimeGatedStrategy(
        name="g", base=inner, allowed_regimes=(0, 1),
        take_profit=inner.take_profit, stop_loss=inner.stop_loss, expiry=inner.expiry,
    )
    assert g1.state_hash() != g2.state_hash()


# ---------------------------------------------------------------------------
# integration: wraps a real ThresholdGate when ARCH lands it
# ---------------------------------------------------------------------------


def test_wraps_threshold_gate_when_present():
    """If ARCH's ThresholdGate is registered as 'threshold_gate', verify the
    regime-gated decorator can wrap it. Skipped otherwise so this test is
    forwards-compatible across the parallel ARCH branch.
    """
    if "threshold_gate" not in STRATEGY_REGISTRY:
        pytest.skip("ThresholdGate not yet registered (ARCH branch unmerged)")
    g = build_strategy(
        "regime_gated",
        base_kind="threshold_gate",
        allowed_regimes=[0],
        tau=0.5,
    )
    assert isinstance(g, RegimeGatedStrategy)
    assert g.allowed_regimes == (0,)
