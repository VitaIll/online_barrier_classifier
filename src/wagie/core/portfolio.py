"""Portfolio — aggregates open Positions + cash + equity + drawdown.

Immutable dataclass. The broker holds the live Portfolio; transitions produce
new Portfolio instances. Strategies see a snapshot via StrategyContext.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Optional

from wagie.core.identity import InstrumentId, PositionId
from wagie.core.numeric import LogReturn, Price, Quantity
from wagie.core.position import Position
from wagie.core.time import Timestamp


@dataclass(frozen=True, slots=True)
class Portfolio:
    """All open positions + realized PnL + drawdown tracking.

    Snapshot semantics: every modification returns a new Portfolio.
    Equity is in log-units (additive), to keep arithmetic simple while we're
    single-currency. Multi-currency Money type can come in v1.x.
    """

    # PositionId -> Position. Open AND just-closed (until next sweep).
    positions: Mapping[PositionId, Position] = field(default_factory=dict)
    realized_pnl_log: LogReturn = field(default_factory=LogReturn.zero)
    high_water_mark_log: LogReturn = field(default_factory=LogReturn.zero)
    starting_equity_log: LogReturn = field(default_factory=LogReturn.zero)

    # ---- aggregates ----

    @property
    def n_open(self) -> int:
        return sum(1 for p in self.positions.values() if not p.closed)

    @property
    def is_flat(self) -> bool:
        return self.n_open == 0

    @property
    def open_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions.values() if not p.closed)

    @property
    def closed_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions.values() if p.closed)

    @property
    def unrealized_pnl_log(self) -> LogReturn:
        return LogReturn(sum(float(p.unrealized_pnl_log) for p in self.open_positions))

    @property
    def equity_log(self) -> LogReturn:
        return LogReturn(
            float(self.starting_equity_log)
            + float(self.realized_pnl_log)
            + float(self.unrealized_pnl_log)
        )

    @property
    def drawdown_log(self) -> LogReturn:
        """Negative or zero. Distance from peak equity in log-units."""
        return LogReturn(float(self.equity_log) - float(self.high_water_mark_log))

    @property
    def total_pnl_log(self) -> LogReturn:
        return LogReturn(float(self.realized_pnl_log) + float(self.unrealized_pnl_log))

    # ---- queries ----

    def get(self, position_id: PositionId) -> Optional[Position]:
        return self.positions.get(position_id)

    def positions_for(self, instr: InstrumentId) -> tuple[Position, ...]:
        return tuple(p for p in self.open_positions if p.instrument == instr)

    def net_position(self, instr: InstrumentId) -> Quantity:
        net = sum(p.signed_size for p in self.open_positions if p.instrument == instr)
        return Quantity(abs(net))   # Quantity is non-negative; sign info lost
                                     # — net direction via sign of net itself

    def net_signed(self, instr: InstrumentId) -> float:
        return sum(p.signed_size for p in self.open_positions if p.instrument == instr)

    def winners(self) -> tuple[Position, ...]:
        return tuple(p for p in self.open_positions if p.is_winning)

    def losers(self) -> tuple[Position, ...]:
        return tuple(p for p in self.open_positions if not p.is_winning)

    def by_age(self) -> tuple[Position, ...]:
        return tuple(sorted(self.open_positions, key=lambda p: p.open_ts.ns))

    # ---- transitions ----

    def with_position(self, p: Position) -> "Portfolio":
        new_positions = dict(self.positions)
        new_positions[p.position_id] = p
        return dataclasses.replace(self, positions=new_positions)

    def with_removed(self, position_id: PositionId) -> "Portfolio":
        new_positions = {k: v for k, v in self.positions.items() if k != position_id}
        return dataclasses.replace(self, positions=new_positions)

    def with_realized(self, pnl_added: LogReturn) -> "Portfolio":
        new_realized = LogReturn(float(self.realized_pnl_log) + float(pnl_added))
        return dataclasses.replace(self, realized_pnl_log=new_realized)

    def with_mark(
        self,
        mark_prices: Mapping[InstrumentId, Price],
        ts: Timestamp,
    ) -> "Portfolio":
        """Mark-to-market all open positions; refresh high-water mark."""
        new_positions = dict(self.positions)
        for pid, p in self.positions.items():
            if p.closed:
                continue
            mp = mark_prices.get(p.instrument)
            if mp is None:
                continue
            new_positions[pid] = p.with_mark(mp, ts)
        out = dataclasses.replace(self, positions=new_positions)
        # Update high-water mark if equity moved up
        if float(out.equity_log) > float(out.high_water_mark_log):
            out = dataclasses.replace(out, high_water_mark_log=out.equity_log)
        return out

    def __repr__(self) -> str:
        return (f"Portfolio(n_open={self.n_open}, "
                f"realized={float(self.realized_pnl_log):+.4f}, "
                f"unrealized={float(self.unrealized_pnl_log):+.4f}, "
                f"dd={float(self.drawdown_log):+.4f})")


__all__ = ["Portfolio"]
