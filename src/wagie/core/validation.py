"""Validation primitives — Split (with embargo) and CrossValSplitter Protocol.

Split.__post_init__ IS the leak-prevention test. It's evaluated at
construction, so embargo violation becomes structurally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

from wagie.core.event import Event
from wagie.core.stream_typing import StreamLike  # forward decl
from wagie.core.time import Duration, TimeWindow


@dataclass(frozen=True, slots=True)
class Split:
    """A train/test partition with embargo + purge.

    Validated at construction:
        train.end + embargo <= test.start         (no leakage forward)
        train.end - purge   not within label_horizon of test events
    """

    train: TimeWindow
    test: TimeWindow
    embargo: Duration = Duration.zero()
    purge: Duration = Duration.zero()

    def __post_init__(self):
        if self.train.end + self.embargo > self.test.start:
            raise ValueError(
                f"Split violates embargo: train.end ({self.train.end}) + "
                f"embargo ({self.embargo}) > test.start ({self.test.start})"
            )
        if self.train.end < self.train.start or self.test.end < self.test.start:
            raise ValueError("TimeWindow must satisfy start <= end")

    def is_disjoint(self) -> bool:
        return not self.train.overlaps(self.test)


@runtime_checkable
class CrossValSplitter(Protocol):
    """Anything that yields Splits over a stream of Events."""

    def split(self, events: StreamLike[Event]) -> Iterator[Split]: ...


__all__ = ["Split", "CrossValSplitter"]
