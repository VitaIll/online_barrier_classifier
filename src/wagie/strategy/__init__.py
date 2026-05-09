"""Strategy as a typed callable returning Sequence[Action].

Strategy.decide returns a sequence of Actions. A strategy can open a new
position, scale out an old one, and modify a stop on a third — all in one
tick.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

from wagie.core.action import Action, ActionKind, Side
from wagie.core.numeric import LogReturn, Probability
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.core.portfolio import Portfolio
from wagie.core.time import Duration


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Read-only view available to Strategy.decide. Carries the live Portfolio
    so strategies can inspect their own positions, equity, and drawdown."""

    portfolio: Portfolio = field(default_factory=Portfolio)
    clock_ns: int = 0
    risk_engine_view: tuple[str, ...] = ()    # active policy names

    # Legacy-compat aliases (G7 strategies expected these)
    @property
    def position(self) -> float:
        return sum(p.signed_size for p in self.portfolio.open_positions)

    @property
    def n_open_orders(self) -> int:
        return self.portfolio.n_open

    @property
    def realized_pnl_log(self) -> float:
        return float(self.portfolio.realized_pnl_log)


@runtime_checkable
class Strategy(Protocol):
    """Pure functional strategy. decide() returns a Sequence[Action]."""

    name: str

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]: ...


# -----------------------------------------------------------------------------
# StrategyBase
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyBase:
    """Shared init for concrete strategies. Subclasses override decide()."""

    name: str = "strategy"
    side: Side = Side.LONG
    take_profit: LogReturn = field(default_factory=lambda: LogReturn.from_bps(41.11))
    stop_loss: LogReturn = field(default_factory=lambda: LogReturn.from_bps(41.11))
    expiry: Duration = field(default_factory=lambda: Duration.from_minutes(20))

    kind: StageKind = StageKind.STRATEGY

    def transform(self, obs: Observation, ctx: Optional[StrategyContext] = None) -> Observation:
        """Stage Protocol adapter: store decided actions on the Observation."""
        if ctx is None:
            ctx = StrategyContext()
        actions = self.decide(obs, ctx)
        return obs.with_actions(actions)

    def update(self, obs: Observation, label=None) -> None:
        return None

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(b"|")
        h.update(repr((float(self.side), float(self.take_profit),
                       float(self.stop_loss), self.expiry.ns)).encode())
        return h.digest()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None

    def reset(self) -> None:
        return None

    @property
    def n_seen(self) -> int:
        return 0

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        """Subclasses override. Default: hold (empty sequence)."""
        return ()

    # Convenience for subclasses
    def _open(self, size: float = 1.0) -> Action:
        return Action.open(
            side=self.side,
            size=Probability(min(1.0, max(0.0, size))),
            take_profit=self.take_profit,
            stop_loss=self.stop_loss,
            expiry=self.expiry,
        )


# -----------------------------------------------------------------------------
# Concrete strategies
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PureConformalGate(StrategyBase):
    """Open when in_set_α == 1. Single open at a time (no scale)."""

    alpha: float = 0.10

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        if ctx.n_open_orders > 0:
            return ()
        if not frame.in_set.get(self.alpha, False):
            return ()
        return (self._open(1.0),)


@dataclass(frozen=True, slots=True)
class EvCalibratedSize(StrategyBase):
    """Size by per-regime conformal margin: size = clip(k * margin, 0, 1)."""

    alpha: float = 0.10
    k: float = 5.0

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        if ctx.n_open_orders > 0:
            return ()
        if not frame.in_set.get(self.alpha, False):
            return ()
        p = frame.p_online
        q = frame.q_lo.get(self.alpha)
        if p is None or q is None:
            return ()
        margin = max(0.0, float(p) - (1.0 - float(q)))
        size = max(0.0, min(1.0, self.k * margin))
        return (self._open(size),) if size > 0.0 else ()


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[StrategyBase]] = {
    "pure_conformal": PureConformalGate,
    "ev_calibrated_size": EvCalibratedSize,
}


def build_strategy(kind: str, **kwargs) -> StrategyBase:
    if kind not in STRATEGY_REGISTRY:
        raise ValueError(f"unknown strategy kind {kind!r}; valid: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[kind](**kwargs)


__all__ = [
    "Strategy", "StrategyContext", "StrategyBase",
    "PureConformalGate", "EvCalibratedSize",
    "STRATEGY_REGISTRY", "build_strategy",
]
