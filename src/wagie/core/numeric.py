"""Numeric primitives with units.

Subclasses of float so they work transparently with numpy/polars/pandas, but
constructors enforce constraints (Probability in [0,1], Price > 0, etc).

Mathematical structure:
    LogReturn  : abelian group under +, with zero, scalar mul
    Probability: closed [0, 1] subset of ℝ
    Price      : positive reals; Price - Price → LogReturn (via .log_return_to)
    Quantity   : non-negative reals
    Bps        : 1e-4 units; convertible to/from LogReturn
"""

from __future__ import annotations

import math


class Price(float):
    """Strictly positive monetary value."""

    __slots__ = ()

    def __new__(cls, value):
        v = float(value)
        if v <= 0.0 or not math.isfinite(v):
            raise ValueError(f"Price must be finite > 0; got {value!r}")
        return super().__new__(cls, v)

    def log_return_to(self, other: "Price") -> "LogReturn":
        return LogReturn(math.log(float(other) / float(self)))

    def __repr__(self) -> str:
        return f"Price({float(self)!r})"


class Quantity(float):
    """Non-negative quantity (e.g. trade size, volume)."""

    __slots__ = ()

    def __new__(cls, value):
        v = float(value)
        if v < 0.0 or not math.isfinite(v):
            raise ValueError(f"Quantity must be finite >= 0; got {value!r}")
        return super().__new__(cls, v)

    def __repr__(self) -> str:
        return f"Quantity({float(self)!r})"


class LogReturn(float):
    """A log-return. Group under addition.

    Examples:
        LogReturn.from_bps(41.11)  # 41.11 bps = 0.0041111
        r1 + r2                     # group operation
        LogReturn.zero()            # identity element
        -r                          # inverse
    """

    __slots__ = ()

    def __new__(cls, value):
        v = float(value)
        if not (math.isfinite(v) or math.isinf(v)):
            raise ValueError(f"LogReturn must not be NaN; got {value!r}")
        return super().__new__(cls, v)

    @classmethod
    def zero(cls) -> "LogReturn":
        return cls(0.0)

    @classmethod
    def from_bps(cls, bps: float) -> "LogReturn":
        return cls(float(bps) * 1e-4)

    @classmethod
    def from_pct(cls, pct: float) -> "LogReturn":
        return cls(math.log1p(float(pct)))

    def to_bps(self) -> float:
        return float(self) * 1e4

    def to_pct(self) -> float:
        return math.exp(float(self)) - 1.0

    def __add__(self, other) -> "LogReturn":
        if isinstance(other, LogReturn):
            return LogReturn(float(self) + float(other))
        return NotImplemented

    def __radd__(self, other) -> "LogReturn":
        if isinstance(other, LogReturn):
            return LogReturn(float(other) + float(self))
        return NotImplemented

    def __sub__(self, other) -> "LogReturn":
        if isinstance(other, LogReturn):
            return LogReturn(float(self) - float(other))
        return NotImplemented

    def __neg__(self) -> "LogReturn":
        return LogReturn(-float(self))

    def __repr__(self) -> str:
        return f"LogReturn({float(self)!r})"


class Probability(float):
    """A probability in [0, 1]. Construction validates; arithmetic is on float."""

    __slots__ = ()

    def __new__(cls, value):
        v = float(value)
        if not (0.0 <= v <= 1.0) or math.isnan(v):
            raise ValueError(f"Probability must be in [0,1]; got {value!r}")
        return super().__new__(cls, v)

    @classmethod
    def half(cls) -> "Probability":
        return cls(0.5)

    def __repr__(self) -> str:
        return f"Probability({float(self)!r})"


class Bps(float):
    """Basis points. 1 bps = 1e-4. Convertible to/from LogReturn."""

    __slots__ = ()

    def __new__(cls, value):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"Bps must be finite; got {value!r}")
        return super().__new__(cls, v)

    def to_log_return(self) -> LogReturn:
        return LogReturn.from_bps(float(self))

    @classmethod
    def from_log_return(cls, lr: LogReturn) -> "Bps":
        return cls(lr.to_bps())

    def __repr__(self) -> str:
        return f"Bps({float(self)!r})"


__all__ = ["Price", "Quantity", "LogReturn", "Probability", "Bps"]
