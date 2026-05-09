"""Events — the algebraic data type at the heart.

Every "thing that happens" is a subclass of Event with a `ts_init: Timestamp`
and an `instrument: InstrumentId`. The Engine processes Events in ts_init order.

Pattern matching dispatches at the Engine level. New event types (QuoteTick,
TradeTick, FundingRateUpdate for live broker) extend Event without modifying
the loop — only adding a `case` branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from wagie.core.action import Action, ExitReason
from wagie.core.identity import DEFAULT_INSTRUMENT, InstrumentId, OrderId
from wagie.core.numeric import LogReturn, Price, Quantity
from wagie.core.time import Duration, Timestamp


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for everything that happens at a specific time on an instrument.

    Subclasses MUST be @dataclass(frozen=True, slots=True). Subclasses MAY add
    fields but MUST NOT remove ts_init or instrument.
    """

    ts_init: Timestamp
    instrument: InstrumentId = DEFAULT_INSTRUMENT


@dataclass(frozen=True, slots=True)
class MinuteBar(Event):
    """One 1-minute kline."""

    open: Price = field(default_factory=lambda: Price(1.0))
    high: Price = field(default_factory=lambda: Price(1.0))
    low: Price = field(default_factory=lambda: Price(1.0))
    close: Price = field(default_factory=lambda: Price(1.0))
    volume: Quantity = field(default_factory=lambda: Quantity(0.0))
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_base: float = 0.0
    taker_buy_quote: float = 0.0
    segment_id: int = 0
    ts_close: Optional[Timestamp] = None  # close-time of this minute (= open + 60s)

    @property
    def close_time(self) -> Timestamp:
        return self.ts_close or (self.ts_init + Duration.from_seconds(60))


@dataclass(frozen=True, slots=True)
class DecisionBar(Event):
    """An aggregated M-min bar. The Pipeline consumes these.

    `underlying` carries the M minute-bars for the broker's first-touch logic.
    The Strategy MUST NOT read `underlying` (only the broker does).
    """

    open: Price = field(default_factory=lambda: Price(1.0))
    high: Price = field(default_factory=lambda: Price(1.0))
    low: Price = field(default_factory=lambda: Price(1.0))
    close: Price = field(default_factory=lambda: Price(1.0))
    volume: Quantity = field(default_factory=lambda: Quantity(0.0))
    duration: Duration = field(default_factory=lambda: Duration.from_minutes(20))
    underlying: tuple = field(default_factory=tuple)
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_base: float = 0.0
    taker_buy_quote: float = 0.0
    segment_id: int = 0

    @property
    def close_time(self) -> Timestamp:
        return self.ts_init + self.duration


@dataclass(frozen=True, slots=True)
class OrderSubmitted(Event):
    """Strategy submitted an order. Broker turns this into a fill."""

    order_id: OrderId = OrderId(0)
    decision: Optional[Action] = None
    submit_ts: Optional[Timestamp] = None


@dataclass(frozen=True, slots=True)
class OrderFilled(Event):
    """An order was filled by the venue/broker (entry side)."""

    order_id: OrderId = OrderId(0)
    fill_price: Price = field(default_factory=lambda: Price(1.0))
    fill_size: Quantity = field(default_factory=lambda: Quantity(0.0))


@dataclass(frozen=True, slots=True)
class BarrierTouched(Event):
    """A position was closed by hitting TP, SL, or timeout."""

    order_id: OrderId = OrderId(0)
    entry_ts: Optional[Timestamp] = None
    exit_ts: Optional[Timestamp] = None
    entry_price: Price = field(default_factory=lambda: Price(1.0))
    exit_price: Price = field(default_factory=lambda: Price(1.0))
    fill_size: Quantity = field(default_factory=lambda: Quantity(0.0))
    side: int = 1
    reason: ExitReason = ExitReason.TIMEOUT
    pnl_log_gross: LogReturn = field(default_factory=LogReturn.zero)
    pnl_log_net: LogReturn = field(default_factory=LogReturn.zero)
    features_at_open: dict = field(default_factory=dict)
    y_matured: int = 0


@dataclass(frozen=True, slots=True)
class DriftDetected(Event):
    """A drift detector fired. The Pipeline can react (recalibrate, refit, etc.)."""

    detector_name: str = ""
    metric: str = ""
    value: float = 0.0


# -----------------------------------------------------------------------------
# Position lifecycle events (G8)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionOpened(Event):
    """Emitted when a new position is established (first fill)."""

    position_id: int = 0
    side: int = 1
    size: float = 0.0
    entry_price: float = 0.0


@dataclass(frozen=True, slots=True)
class PositionScaledIn(Event):
    """Position grew via Action.scale_in."""

    position_id: int = 0
    size_added: float = 0.0
    fill_price: float = 0.0
    new_avg_entry_price: float = 0.0


@dataclass(frozen=True, slots=True)
class PositionScaledOut(Event):
    """Position shrunk via Action.scale_out (partial close)."""

    position_id: int = 0
    size_removed: float = 0.0
    fill_price: float = 0.0
    realized_pnl_log: float = 0.0


@dataclass(frozen=True, slots=True)
class PositionStopModified(Event):
    """Stop loss or take profit changed."""

    position_id: int = 0
    old_stop_loss: float = 0.0
    new_stop_loss: float = 0.0
    old_take_profit: float = 0.0
    new_take_profit: float = 0.0
    actor: str = "strategy"   # "strategy" | "operator"


@dataclass(frozen=True, slots=True)
class PositionClosed(Event):
    """Emitted when a position fully closes (size goes to 0).

    `reason` carries TP/SL/TIMEOUT for triple-barrier closes, MANUAL for
    Action.close, OPERATOR for TradingConsole-driven closes.
    """

    position_id: int = 0
    side: int = 1
    fill_size: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_ts: Optional[Timestamp] = None
    exit_ts: Optional[Timestamp] = None
    reason: str = "tp"
    pnl_log_gross: float = 0.0
    pnl_log_net: float = 0.0
    features_at_open: dict = field(default_factory=dict)
    y_matured: int = 0


@dataclass(frozen=True, slots=True)
class RiskRejected(Event):
    """An action was rejected by the RiskEngine. Audit trail."""

    action_kind: str = "hold"
    reason: str = ""
    policy_name: str = ""


@dataclass(frozen=True, slots=True)
class RiskPolicyChanged(Event):
    """A risk policy was added/removed/replaced. Operator audit."""

    op: str = "add"          # "add" | "remove" | "replace"
    policy_name: str = ""
    actor: str = "operator"


__all__ = [
    "Event",
    "MinuteBar",
    "DecisionBar",
    "OrderSubmitted",
    "OrderFilled",
    "BarrierTouched",
    "DriftDetected",
    "PositionOpened",
    "PositionScaledIn",
    "PositionScaledOut",
    "PositionStopModified",
    "PositionClosed",
    "RiskRejected",
    "RiskPolicyChanged",
]
