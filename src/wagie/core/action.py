"""Action vocabulary — the trading-floor verbs.

Strategy.decide() returns Sequence[Action]. The Engine routes each Action
through the RiskEngine, then dispatches to the Broker. ActionKind is the sum
type tag. Use `Action.long`/`Action.short`/`Action.open` to construct opens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Optional

from wagie.core.identity import PositionId
from wagie.core.numeric import LogReturn, Probability
from wagie.core.time import Duration


class Side(IntEnum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class ExitReason(StrEnum):
    """How a position was closed."""

    TP = "tp"
    SL = "sl"
    TIMEOUT = "timeout"
    MANUAL = "manual"        # closed via Action.close
    OPERATOR = "operator"    # closed via TradingConsole
    BATCHED_WINNER = "batched_winner"   # swept-out profitable position when a
                                         # sibling on the same instrument hit TP
    SAFETY_CAP = "safety_cap"            # broker hard hold-age safety net (NOT
                                         # a strategy decision; see SimBroker)


class ActionKind(StrEnum):
    """Tag for the Action sum type. Engine dispatches on this."""

    HOLD = "hold"
    OPEN = "open"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    CLOSE = "close"
    MODIFY_STOP = "modify_stop"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class Action:
    """A single trading-floor operation. Pure data; emitted by Strategy.decide().

    Different ActionKinds use different field subsets — the constructors below
    fill them correctly. Direct construction is fine but use the classmethods
    when in doubt.
    """

    kind: ActionKind = ActionKind.HOLD

    # Reference an existing position (for SCALE_IN/SCALE_OUT/CLOSE/MODIFY_STOP/CANCEL)
    target_position_id: Optional[PositionId] = None

    # OPEN / SCALE_IN
    side: Optional[Side] = None
    size: Optional[Probability] = None
    take_profit: Optional[LogReturn] = None
    stop_loss: Optional[LogReturn] = None
    expiry: Optional[Duration] = None

    # MODIFY_STOP
    new_take_profit: Optional[LogReturn] = None
    new_stop_loss: Optional[LogReturn] = None

    # SCALE_OUT — fraction of current_size to close
    fraction: Optional[Probability] = None

    # ---- constructors ----

    @classmethod
    def hold(cls) -> "Action":
        return cls(kind=ActionKind.HOLD)

    @classmethod
    def open(
        cls,
        *,
        side: Side = Side.LONG,
        size: Probability = Probability(1.0),
        take_profit: LogReturn,
        stop_loss: LogReturn,
        expiry: Duration,
    ) -> "Action":
        return cls(
            kind=ActionKind.OPEN,
            side=side, size=size,
            take_profit=take_profit, stop_loss=stop_loss, expiry=expiry,
        )

    @classmethod
    def long(cls, **kwargs) -> "Action":
        return cls.open(side=Side.LONG, **kwargs)

    @classmethod
    def short(cls, **kwargs) -> "Action":
        return cls.open(side=Side.SHORT, **kwargs)

    @classmethod
    def scale_in(
        cls,
        position_id: PositionId,
        *,
        size: Probability,
    ) -> "Action":
        return cls(kind=ActionKind.SCALE_IN, target_position_id=position_id, size=size)

    @classmethod
    def scale_out(
        cls,
        position_id: PositionId,
        *,
        fraction: Probability,
    ) -> "Action":
        return cls(kind=ActionKind.SCALE_OUT, target_position_id=position_id, fraction=fraction)

    @classmethod
    def close(cls, position_id: PositionId) -> "Action":
        return cls(kind=ActionKind.CLOSE, target_position_id=position_id)

    @classmethod
    def modify_stop(
        cls,
        position_id: PositionId,
        *,
        new_take_profit: Optional[LogReturn] = None,
        new_stop_loss: Optional[LogReturn] = None,
    ) -> "Action":
        if new_take_profit is None and new_stop_loss is None:
            raise ValueError("modify_stop requires new_take_profit or new_stop_loss")
        return cls(
            kind=ActionKind.MODIFY_STOP,
            target_position_id=position_id,
            new_take_profit=new_take_profit,
            new_stop_loss=new_stop_loss,
        )

    @classmethod
    def cancel(cls, position_id: PositionId) -> "Action":
        return cls(kind=ActionKind.CANCEL, target_position_id=position_id)

    # ---- helpers ----

    @property
    def is_active(self) -> bool:
        """True if this action would actually do something at the broker."""
        return self.kind != ActionKind.HOLD

    @property
    def is_opening(self) -> bool:
        return self.kind in (ActionKind.OPEN, ActionKind.SCALE_IN)


__all__ = ["Side", "ExitReason", "ActionKind", "Action"]
