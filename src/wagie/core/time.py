"""Time primitives — affine arithmetic.

Mathematical structure:
    (Timestamp, Duration, +, -)  is an affine space:
        Timestamp - Timestamp = Duration
        Timestamp + Duration  = Timestamp
        Timestamp + Timestamp  is NOT defined
    (Duration, +, 0)             is an abelian group
    (TimeWindow, ∩, ∪)            is a lattice (over comparable windows)

All ns-resolution. Fits in int64 until year 2262 (NautilusTrader convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


_NS_PER_MS: int = 1_000_000
_NS_PER_S: int = 1_000_000_000
_NS_PER_MIN: int = 60 * _NS_PER_S
_NS_PER_HOUR: int = 60 * _NS_PER_MIN
_NS_PER_DAY: int = 24 * _NS_PER_HOUR


@dataclass(frozen=True, slots=True, order=True)
class Timestamp:
    """A point in time. Nanoseconds since Unix epoch. Ordered, hashable, immutable."""

    ns: int

    def __post_init__(self):
        if not isinstance(self.ns, int):
            object.__setattr__(self, "ns", int(self.ns))

    @classmethod
    def from_ms(cls, ms: int) -> "Timestamp":
        return cls(int(ms) * _NS_PER_MS)

    @classmethod
    def from_seconds(cls, s: float) -> "Timestamp":
        return cls(int(s * _NS_PER_S))

    @classmethod
    def from_iso(cls, iso: str) -> "Timestamp":
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return cls(int(dt.timestamp() * _NS_PER_S))

    @property
    def ms(self) -> int:
        return self.ns // _NS_PER_MS

    @property
    def seconds(self) -> float:
        return self.ns / _NS_PER_S

    def isoformat(self) -> str:
        dt = datetime.fromtimestamp(self.ns / _NS_PER_S, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    def __sub__(self, other):
        if isinstance(other, Timestamp):
            return Duration(self.ns - other.ns)
        if isinstance(other, Duration):
            return Timestamp(self.ns - other.ns)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Duration):
            return Timestamp(self.ns + other.ns)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Timestamp(ns={self.ns}, iso='{self.isoformat()}')"


@dataclass(frozen=True, slots=True, order=True)
class Duration:
    """A signed time interval. Group under + with identity Duration.zero()."""

    ns: int

    def __post_init__(self):
        if not isinstance(self.ns, int):
            object.__setattr__(self, "ns", int(self.ns))

    @classmethod
    def zero(cls) -> "Duration":
        return cls(0)

    @classmethod
    def from_seconds(cls, s: float) -> "Duration":
        return cls(int(s * _NS_PER_S))

    @classmethod
    def from_minutes(cls, m: float) -> "Duration":
        return cls(int(m * _NS_PER_MIN))

    @classmethod
    def from_hours(cls, h: float) -> "Duration":
        return cls(int(h * _NS_PER_HOUR))

    @classmethod
    def from_days(cls, d: float) -> "Duration":
        return cls(int(d * _NS_PER_DAY))

    @property
    def seconds(self) -> float:
        return self.ns / _NS_PER_S

    @property
    def minutes(self) -> float:
        return self.ns / _NS_PER_MIN

    @property
    def hours(self) -> float:
        return self.ns / _NS_PER_HOUR

    def __add__(self, other):
        if isinstance(other, Duration):
            return Duration(self.ns + other.ns)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Duration):
            return Duration(self.ns - other.ns)
        return NotImplemented

    def __neg__(self) -> "Duration":
        return Duration(-self.ns)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Duration(int(self.ns * scalar))
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        if abs(self.ns) >= _NS_PER_MIN and self.ns % _NS_PER_MIN == 0:
            return f"Duration(minutes={self.minutes})"
        return f"Duration(ns={self.ns})"


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Half-open interval [start, end). Lattice under ∩, ∪.

    Splits use this to validate non-overlap (purge/embargo). Used internally
    for any "this thing depends on data in this window" assertion.
    """

    start: Timestamp
    end: Timestamp

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError(
                f"TimeWindow needs start < end; got {self.start} >= {self.end}"
            )

    def length(self) -> Duration:
        return self.end - self.start

    def contains(self, ts: Timestamp) -> bool:
        return self.start <= ts < self.end

    def overlaps(self, other: "TimeWindow") -> bool:
        return self.start < other.end and other.start < self.end

    def shift(self, dur: Duration) -> "TimeWindow":
        return TimeWindow(self.start + dur, self.end + dur)

    def __and__(self, other: "TimeWindow") -> Optional["TimeWindow"]:
        s = max(self.start, other.start)
        e = min(self.end, other.end)
        return TimeWindow(s, e) if s < e else None

    def __repr__(self) -> str:
        return f"TimeWindow({self.start.isoformat()} → {self.end.isoformat()})"


__all__ = ["Timestamp", "Duration", "TimeWindow"]
