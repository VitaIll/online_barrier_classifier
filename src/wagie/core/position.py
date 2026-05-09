"""Position — full lifecycle state of a single trade.

Immutable; transitions produce new Position instances. The Portfolio stores
the current Position per id; events are appended to fills history.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional

from wagie.core.action import ExitReason, Side
from wagie.core.identity import InstrumentId, PositionId
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.time import Duration, Timestamp


@dataclass(frozen=True, slots=True)
class Position:
    """A single open or recently-closed position with full state.

    Lifecycle: open → (optional scale_in/out, modify_stop) → close.
    Each transition produces a new Position via with_* methods.
    """

    position_id: PositionId
    instrument: InstrumentId
    side: Side

    # Sizing — current_size shrinks on scale_out
    open_size: Quantity              # cumulative size opened (incl. scale_ins)
    current_size: Quantity           # remaining open
    avg_entry_price: Price           # size-weighted across scale-ins

    # Times
    open_ts: Timestamp
    last_update_ts: Timestamp
    expiry_ts: Timestamp

    # Active stops (mutable via modify_stop → new Position)
    take_profit_log: LogReturn
    stop_loss_log: LogReturn

    # PnL state (last_mark_price drives unrealized; updated by Portfolio.with_mark)
    last_mark_price: Price
    realized_pnl_log: LogReturn = dataclasses.field(default_factory=LogReturn.zero)
    unrealized_pnl_log: LogReturn = dataclasses.field(default_factory=LogReturn.zero)
    mae_log: LogReturn = dataclasses.field(default_factory=LogReturn.zero)
    mfe_log: LogReturn = dataclasses.field(default_factory=LogReturn.zero)

    # Audit — every fill that touched this position
    fills: tuple = field(default_factory=tuple)

    # Closed flag
    closed: bool = False
    close_reason: Optional[ExitReason] = None

    # ---- derived ----

    @property
    def time_in_trade(self) -> Duration:
        return self.last_update_ts - self.open_ts

    @property
    def total_pnl_log(self) -> LogReturn:
        return LogReturn(float(self.realized_pnl_log) + float(self.unrealized_pnl_log))

    @property
    def is_winning(self) -> bool:
        return float(self.total_pnl_log) > 0.0

    @property
    def signed_size(self) -> float:
        return float(self.current_size) * (1 if self.side == Side.LONG else -1)

    # ---- transitions ----

    @classmethod
    def open_at(
        cls,
        *,
        position_id: PositionId,
        instrument: InstrumentId,
        side: Side,
        size: Quantity,
        entry_price: Price,
        ts: Timestamp,
        take_profit_log: LogReturn,
        stop_loss_log: LogReturn,
        expiry: Duration,
    ) -> "Position":
        return cls(
            position_id=position_id,
            instrument=instrument,
            side=side,
            open_size=size,
            current_size=size,
            avg_entry_price=entry_price,
            open_ts=ts,
            last_update_ts=ts,
            expiry_ts=ts + expiry,
            take_profit_log=take_profit_log,
            stop_loss_log=stop_loss_log,
            last_mark_price=entry_price,
        )

    def with_mark(self, price: Price, ts: Timestamp) -> "Position":
        """Refresh unrealized PnL / MAE / MFE at a new mark."""
        if self.closed or float(self.current_size) <= 0.0:
            return self
        if self.side == Side.LONG:
            unrealized = math.log(float(price) / float(self.avg_entry_price))
        else:
            unrealized = math.log(float(self.avg_entry_price) / float(price))
        unrealized *= float(self.current_size)
        new_mae = LogReturn(min(float(self.mae_log), unrealized))
        new_mfe = LogReturn(max(float(self.mfe_log), unrealized))
        return dataclasses.replace(
            self,
            last_mark_price=price,
            last_update_ts=ts,
            unrealized_pnl_log=LogReturn(unrealized),
            mae_log=new_mae,
            mfe_log=new_mfe,
        )

    def with_modified_stops(
        self,
        *,
        take_profit_log: Optional[LogReturn] = None,
        stop_loss_log: Optional[LogReturn] = None,
        ts: Optional[Timestamp] = None,
    ) -> "Position":
        return dataclasses.replace(
            self,
            take_profit_log=take_profit_log if take_profit_log is not None else self.take_profit_log,
            stop_loss_log=stop_loss_log if stop_loss_log is not None else self.stop_loss_log,
            last_update_ts=ts or self.last_update_ts,
        )

    def with_scale_in(
        self,
        *,
        size_added: Quantity,
        price: Price,
        ts: Timestamp,
    ) -> "Position":
        new_open = Quantity(float(self.open_size) + float(size_added))
        new_curr = Quantity(float(self.current_size) + float(size_added))
        new_avg = Price(
            (float(self.avg_entry_price) * float(self.current_size)
             + float(price) * float(size_added)) / new_curr
        )
        return dataclasses.replace(
            self,
            open_size=new_open,
            current_size=new_curr,
            avg_entry_price=new_avg,
            last_update_ts=ts,
        )

    def with_scale_out(
        self,
        *,
        size_removed: Quantity,
        exit_price: Price,
        ts: Timestamp,
        cost_log: LogReturn = LogReturn(0.0),
    ) -> "Position":
        size_f = float(size_removed)
        if self.side == Side.LONG:
            pnl_close = math.log(float(exit_price) / float(self.avg_entry_price))
        else:
            pnl_close = math.log(float(self.avg_entry_price) / float(exit_price))
        pnl_close *= size_f
        pnl_net = pnl_close - 2.0 * float(cost_log) * size_f

        new_curr = Quantity(max(0.0, float(self.current_size) - size_f))
        new_realized = LogReturn(float(self.realized_pnl_log) + pnl_net)
        is_closed = float(new_curr) <= 1e-12
        return dataclasses.replace(
            self,
            current_size=new_curr,
            last_update_ts=ts,
            realized_pnl_log=new_realized,
            unrealized_pnl_log=LogReturn(0.0) if is_closed else self.unrealized_pnl_log,
            closed=is_closed,
            close_reason=ExitReason.MANUAL if is_closed else None,
        )

    def with_close(
        self,
        *,
        exit_price: Price,
        ts: Timestamp,
        reason: ExitReason,
        cost_log: LogReturn = LogReturn(0.0),
    ) -> "Position":
        size_f = float(self.current_size)
        if self.side == Side.LONG:
            pnl_close = math.log(float(exit_price) / float(self.avg_entry_price))
        else:
            pnl_close = math.log(float(self.avg_entry_price) / float(exit_price))
        pnl_close *= size_f
        pnl_net = pnl_close - 2.0 * float(cost_log) * size_f
        return dataclasses.replace(
            self,
            current_size=Quantity(0.0),
            last_update_ts=ts,
            realized_pnl_log=LogReturn(float(self.realized_pnl_log) + pnl_net),
            unrealized_pnl_log=LogReturn(0.0),
            closed=True,
            close_reason=reason,
            last_mark_price=exit_price,
        )

    def with_appended_fill(self, fill) -> "Position":
        return dataclasses.replace(self, fills=self.fills + (fill,))


__all__ = ["Position"]
