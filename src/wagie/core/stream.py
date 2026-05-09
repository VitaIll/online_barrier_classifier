"""Stream / Source / Sink Protocols.

The Engine consumes a Source[Event] and routes orders to a Sink. Both
ParquetReplaySource and BinanceLiveSource are Sources; both SimBroker and
BinanceBroker are Sinks. Engine code is identical for both.
"""

from __future__ import annotations

from typing import Iterator, Protocol, TypeVar, runtime_checkable

from wagie.core.event import BarrierTouched, Event, OrderSubmitted


T = TypeVar("T", covariant=True)


@runtime_checkable
class Stream(Protocol[T]):
    """A monotonically-ordered iterable of items with ts_init."""

    def __iter__(self) -> Iterator[T]: ...


@runtime_checkable
class Source(Protocol):
    """Yields Events in chronological order. Backtest: parquet replay.
    Live: websocket. Same Stream[Event] type from the Engine's POV."""

    def stream(self) -> Iterator[Event]: ...
    def close(self) -> None: ...


@runtime_checkable
class Sink(Protocol):
    """Consumes orders, produces fills.

    Engine submits OrderSubmitted events; Sink emits BarrierTouched events
    when positions close. Same shape for sim and live.
    """

    def submit(self, order: OrderSubmitted) -> None: ...
    def fills(self) -> Iterator[BarrierTouched]: ...


__all__ = ["Stream", "Source", "Sink"]
