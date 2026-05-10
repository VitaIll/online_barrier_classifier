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
from wagie.strategy.adaptive_threshold import AdaptiveThresholdController


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
class ThresholdGate(StrategyBase):
    """Open when ``obs.p_online >= tau``.

    The ARF online layer already produces well-calibrated probabilities
    (per-regime ECE ~0.014–0.016 in published runs); a probability
    threshold is the cleanest gate — no conformal in_set predicate needed.

    Single open at a time (no scale).
    """

    tau: float = 0.50

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        if ctx.n_open_orders > 0:
            return ()
        p = frame.p_online
        if p is None:
            return ()
        if float(p) < float(self.tau):
            return ()
        return (self._open(1.0),)


@dataclass(frozen=True, slots=True)
class EvCalibratedSize(StrategyBase):
    """Size by calibrated probability margin: size = clip(k * (p - tau), 0, 1).

    This rebases on ``p_online`` directly: no conformal q_lo dependency.
    """

    tau: float = 0.50
    k: float = 5.0

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        if ctx.n_open_orders > 0:
            return ()
        p = frame.p_online
        if p is None:
            return ()
        margin = float(p) - float(self.tau)
        if margin <= 0.0:
            return ()
        size = max(0.0, min(1.0, self.k * margin))
        return (self._open(size),) if size > 0.0 else ()


# -----------------------------------------------------------------------------
# CompositeAdaptiveStrategy — wraps AdaptiveThresholdController + hold-age cap
# -----------------------------------------------------------------------------


@dataclass
class CompositeAdaptiveStrategy:
    """Strategy that gates on an :class:`AdaptiveThresholdController` and
    closes positions once their bar-age reaches ``t_max``.

    Mutable (carries a controller with EWMA state) — uses a non-frozen
    dataclass. Cannot inherit from the frozen ``StrategyBase`` (Python
    forbids non-frozen subclasses of a frozen parent), so the same Stage
    Protocol surface and barrier defaults are duplicated as fields and
    methods here.

    The controller is observed once per bar:

    - ``decide()`` runs first, drains any matured label that arrived via
      ``update()`` since the last decide, and calls ``controller.observe``
      with (previous-bar entered, matured label) — this advances both EWMAs
      in lock-step with bar boundaries.
    - The matured-label hand-off uses the existing Stage Protocol
      ``update(obs, label)`` contract — no engine-level hook is required.
      ``learn_one`` is invoked by the engine before ``transform_one`` per
      tick, so the label that mutates state in ``update`` is consumed by
      the very next ``decide`` call.
    - Hold-age exits compute bar-age from the position's ``open_ts`` and
      the current clock; the strategy issues ``Action.close`` on positions
      whose age has reached ``t_max``.
    """

    controller: Optional[AdaptiveThresholdController] = None
    name: str = "composite_adaptive"
    t_max: int = 200
    bar_minutes: int = 20
    side: Side = Side.LONG
    take_profit: LogReturn = field(default_factory=lambda: LogReturn.from_bps(41.11))
    stop_loss: LogReturn = field(default_factory=lambda: LogReturn.from_bps(41.11))
    expiry: Duration = field(default_factory=lambda: Duration.from_minutes(20))
    # Which probability the entry gate reads. "online" (default) uses the
    # ARF-corrected p_online; "offline" bypasses the ARF and reads the raw
    # p_offline from the offline ensemble. ARF still runs in the pipeline
    # (so calibration/drift sections stay meaningful) — only the strategy
    # signal source changes.
    signal_source: str = "online"
    kind: StageKind = StageKind.STRATEGY

    # Mutable state — initialised lazily; not constructor args.
    _last_entered: bool = field(init=False, default=False)
    _pending_label: Optional[int] = field(init=False, default=None)
    _last_inv_size: int = field(init=False, default=0)
    _last_hold_age_max: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.controller is None:
            raise ValueError("CompositeAdaptiveStrategy requires a controller")
        if int(self.t_max) <= 0:
            raise ValueError(f"t_max must be > 0; got {self.t_max}")
        if int(self.bar_minutes) <= 0:
            raise ValueError(f"bar_minutes must be > 0; got {self.bar_minutes}")

    # ---- Stage Protocol surface (mirrors StrategyBase) ---------------------

    def transform(self, obs: Observation, ctx: Optional[StrategyContext] = None) -> Observation:
        """Stage Protocol adapter: store decided actions on the Observation."""
        if ctx is None:
            ctx = StrategyContext()
        actions = self.decide(obs, ctx)
        return obs.with_actions(actions)

    def _open(self, size: float = 1.0) -> Action:
        return Action.open(
            side=self.side,
            size=Probability(min(1.0, max(0.0, size))),
            take_profit=self.take_profit,
            stop_loss=self.stop_loss,
            expiry=self.expiry,
        )

    @property
    def n_seen(self) -> int:
        return 0

    # ---- Stage Protocol: update consumes matured labels --------------------

    def update(self, obs: Observation, label: Optional[int] = None) -> None:
        """Stash the matured label so the next ``decide`` can feed it to the
        controller alongside the previous bar's entered flag."""
        if label is not None:
            try:
                self._pending_label = int(label)
            except (TypeError, ValueError):
                self._pending_label = None

    # ---- Helpers -----------------------------------------------------------

    def _bar_age(self, pos, clock_ns: int) -> int:
        """Number of bars elapsed since the position opened, floor-divided."""
        try:
            elapsed_ns = max(0, int(clock_ns) - int(pos.open_ts.ns))
        except (AttributeError, TypeError):
            return 0
        if self.bar_minutes <= 0:
            return 0
        bar_ns = int(self.bar_minutes) * 60_000_000_000
        return elapsed_ns // bar_ns if bar_ns > 0 else 0

    # ---- decide ------------------------------------------------------------

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        # 1. Advance the controller using the *previous* bar's entered flag and
        #    any matured label that arrived since the last decide.
        matured_label = self._pending_label
        self._pending_label = None
        self.controller.observe(
            entered=self._last_entered, label_matured=matured_label,
        )

        actions: list[Action] = []

        # 2. Hold-age exits — close positions that have reached t_max bars.
        open_positions = tuple(ctx.portfolio.open_positions)
        max_age = 0
        for pos in open_positions:
            age = self._bar_age(pos, ctx.clock_ns)
            if age > max_age:
                max_age = age
            if age >= int(self.t_max):
                actions.append(Action.close(pos.position_id))

        # 3. Entry decision — only when nothing already open (single-shot
        #    semantics, mirroring ThresholdGate).
        entered = False
        if ctx.n_open_orders == 0:
            p = (frame.p_offline if self.signal_source == "offline"
                 else frame.p_online)
            if p is not None:
                entered = self.controller.should_enter(
                    float(p),
                    float(frame.sigma_ve) if frame.sigma_ve is not None else None,
                )
                if entered:
                    actions.append(self._open(1.0))
            else:
                # No probability yet — record sigma_ve for inspection but
                # don't enter.
                self.controller.last_sigma_ve = (
                    float(frame.sigma_ve) if frame.sigma_ve is not None else None
                )

        # 4. Cache state for the engine's per-bar collector + next observe().
        self._last_entered = bool(entered)
        self._last_inv_size = len(open_positions)
        self._last_hold_age_max = int(max_age)

        return tuple(actions)

    # ---- get_state ---------------------------------------------------------

    def get_state(self) -> dict:
        s = self.controller.get_state()
        s["inv_size"] = int(self._last_inv_size)
        s["hold_age_max"] = int(self._last_hold_age_max)
        return s

    # ---- Stateful overrides ------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "tau": float(self.controller.tau),
            "r_hat_ewma": float(self.controller.r_hat_ewma),
            "r_star_ewma": float(self.controller.r_star_ewma),
            "paused": bool(self.controller.paused),
            "last_entered": bool(self._last_entered),
            "pending_label": (
                int(self._pending_label) if self._pending_label is not None else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        if "tau" in state:
            self.controller.tau = float(state["tau"])
        if "r_hat_ewma" in state:
            self.controller.r_hat_ewma = float(state["r_hat_ewma"])
        if "r_star_ewma" in state:
            self.controller.r_star_ewma = float(state["r_star_ewma"])
        if "paused" in state:
            self.controller.paused = bool(state["paused"])
        if "last_entered" in state:
            self._last_entered = bool(state["last_entered"])
        if "pending_label" in state:
            v = state["pending_label"]
            self._pending_label = int(v) if v is not None else None

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"composite_adaptive|")
        h.update(self.name.encode())
        h.update(b"|")
        h.update(repr((
            float(self.controller.tau),
            float(self.controller.r_hat_ewma),
            float(self.controller.r_star_ewma),
            bool(self.controller.paused),
            int(self.t_max),
            int(self.bar_minutes),
        )).encode())
        return h.digest()

    def reset(self) -> None:
        # Re-initialise EWMAs to the target (mirrors __post_init__ on the
        # controller). Pause flag clears too.
        self.controller.r_hat_ewma = float(self.controller.r_star_init)
        self.controller.r_star_ewma = float(self.controller.r_star_init)
        self.controller.paused = False
        self.controller.last_sigma_ve = None
        self._last_entered = False
        self._pending_label = None
        self._last_inv_size = 0
        self._last_hold_age_max = 0


# Back-compat alias — old code paths importing PureConformalGate get the
# threshold gate instead.
PureConformalGate = ThresholdGate


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type] = {
    "threshold_gate": ThresholdGate,
    "pure_conformal": ThresholdGate,   # alias for back-compat YAMLs
    "ev_calibrated_size": EvCalibratedSize,
    "composite_adaptive": CompositeAdaptiveStrategy,
}


def build_strategy(kind: str, **kwargs):
    if kind not in STRATEGY_REGISTRY:
        raise ValueError(f"unknown strategy kind {kind!r}; valid: {list(STRATEGY_REGISTRY)}")
    cls = STRATEGY_REGISTRY[kind]
    # Strip any leftover ACI-era kwargs we no longer accept.
    kwargs.pop("alpha", None)
    kwargs.pop("layer", None)
    # ThresholdGate / EvCalibratedSize don't take `k` unless they support it.
    if cls is ThresholdGate:
        kwargs.pop("k", None)
    # Special-case: regime_gated takes a base_kind to wrap.
    if kind == "regime_gated":
        return _build_regime_gated(**kwargs)
    # Special-case: composite_adaptive takes nested controller_kwargs.
    if kind == "composite_adaptive":
        ctrl_kwargs = kwargs.pop("controller_kwargs", None) or kwargs.pop("controller", None)
        if isinstance(ctrl_kwargs, dict):
            kwargs["controller"] = AdaptiveThresholdController(**ctrl_kwargs)
        elif isinstance(ctrl_kwargs, AdaptiveThresholdController):
            kwargs["controller"] = ctrl_kwargs
    return cls(**kwargs)


# -----------------------------------------------------------------------------
# RegimeGatedStrategy decorator
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegimeGatedStrategy(StrategyBase):
    """Decorator that suppresses any non-HOLD action whose underlying base
    strategy emitted them when ``obs.regime_id`` is not in ``allowed_regimes``.

    Allows actions through unchanged when the regime gate passes — including
    SCALE_OUT/CLOSE/MODIFY_STOP/CANCEL on existing positions, which we DO
    NOT suppress in any regime (you must always be allowed to manage
    existing inventory).

    Construct with ``RegimeGatedStrategy(base=ThresholdGate(...), allowed_regimes=(0,))``
    or via the registry: ``build_strategy("regime_gated", base_kind="threshold_gate",
    allowed_regimes=[0], ...)``.
    """

    base: Optional[StrategyBase] = None
    allowed_regimes: tuple[int, ...] = ()

    def __post_init__(self):
        if self.base is None:
            raise ValueError("RegimeGatedStrategy requires `base` strategy")

    def decide(self, frame: Observation, ctx: StrategyContext) -> Sequence[Action]:
        actions = self.base.decide(frame, ctx)
        # Always pass through inventory-management actions regardless of regime.
        always_allowed = {
            ActionKind.HOLD, ActionKind.CLOSE, ActionKind.SCALE_OUT,
            ActionKind.MODIFY_STOP, ActionKind.CANCEL,
        }
        regime = frame.regime_id
        if regime is None:
            # No regime yet ⇒ default to suppression of opens/scale-ins (conservative)
            return tuple(a for a in actions if a.kind in always_allowed)
        if int(regime) in self.allowed_regimes:
            return actions
        # Regime not allowed: suppress opens/scale-ins; allow management.
        return tuple(a for a in actions if a.kind in always_allowed)

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"regime_gated|")
        h.update(self.base.state_hash())
        h.update(b"|")
        h.update(repr(tuple(sorted(self.allowed_regimes))).encode())
        return h.digest()


def _build_regime_gated(
    *,
    base_kind: str,
    allowed_regimes,
    name: str = "regime_gated",
    **base_kwargs,
) -> "RegimeGatedStrategy":
    """Factory used by build_strategy('regime_gated', ...).

    Builds the inner base strategy by name then wraps it. ``base_kwargs`` are
    forwarded to the inner strategy constructor.
    """
    if base_kind == "regime_gated":
        raise ValueError("regime_gated cannot wrap regime_gated")
    if base_kind not in STRATEGY_REGISTRY:
        raise ValueError(
            f"unknown base_kind {base_kind!r}; valid: "
            f"{[k for k in STRATEGY_REGISTRY if k != 'regime_gated']}"
        )
    base_cls = STRATEGY_REGISTRY[base_kind]
    base_strategy = base_cls(**base_kwargs)
    allowed = tuple(int(r) for r in allowed_regimes)
    return RegimeGatedStrategy(
        name=name,
        base=base_strategy,
        allowed_regimes=allowed,
        # Inherit barrier defaults from the inner strategy so engine wiring
        # (which inspects take_profit/stop_loss/expiry on the outer object)
        # still gets sensible values.
        take_profit=base_strategy.take_profit,
        stop_loss=base_strategy.stop_loss,
        expiry=base_strategy.expiry,
        side=base_strategy.side,
    )


# Register after class definition so builder lookup resolves it.
STRATEGY_REGISTRY["regime_gated"] = RegimeGatedStrategy


__all__ = [
    "Strategy", "StrategyContext", "StrategyBase",
    "ThresholdGate", "PureConformalGate", "EvCalibratedSize",
    "CompositeAdaptiveStrategy", "AdaptiveThresholdController",
    "RegimeGatedStrategy",
    "STRATEGY_REGISTRY", "build_strategy",
]
