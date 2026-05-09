"""Identifiers — typed string/int wrappers that prevent passing a Venue
where a Symbol is expected. NewType erases at runtime; mypy keeps it honest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType


Symbol = NewType("Symbol", str)
Venue = NewType("Venue", str)
TraderId = NewType("TraderId", str)
OrderId = NewType("OrderId", int)
PositionId = NewType("PositionId", int)


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """Symbol + venue. Hashable, equality-typed."""

    symbol: Symbol
    venue: Venue = Venue("binance")

    def __str__(self) -> str:
        return f"{self.symbol}@{self.venue}"

    @classmethod
    def of(cls, symbol: str, venue: str = "binance") -> "InstrumentId":
        return cls(Symbol(symbol), Venue(venue))


# Sentinel default for single-symbol code paths. Multi-symbol future-proofing:
# when Engine grows multi-symbol routing it dispatches on InstrumentId.
DEFAULT_INSTRUMENT: InstrumentId = InstrumentId.of("DEFAULT", "default")


__all__ = [
    "Symbol", "Venue", "TraderId", "OrderId", "PositionId",
    "InstrumentId", "DEFAULT_INSTRUMENT",
]
