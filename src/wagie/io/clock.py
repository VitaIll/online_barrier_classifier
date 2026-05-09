"""Clock interface — NautilusTrader pattern.

The Pipeline and Strategy NEVER call time.time(). They ask self.clock.now_ns().
In backtest the engine calls clock.set_time_ns(bar.ts_close_ns) before dispatching;
in prod set_time_ns is a no-op and now_ns() reads the OS clock.

This is the highest-leverage prod ≡ backtest abstraction in the package.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Source of 'current' wall-clock for the harness."""

    def now_ns(self) -> int: ...
    def set_time_ns(self, ts_ns: int) -> None: ...


class TestClock:
    """Settable, deterministic clock for backtest. Refuses to run backwards."""

    # Tell pytest NOT to collect this as a test class.
    __test__ = False

    __slots__ = ("_ts",)

    def __init__(self, ts_ns: int = 0):
        self._ts = int(ts_ns)

    def now_ns(self) -> int:
        return self._ts

    def set_time_ns(self, ts_ns: int) -> None:
        ts_ns = int(ts_ns)
        if ts_ns < self._ts:
            raise ValueError(
                f"TestClock cannot run backwards: tried {ts_ns}, current {self._ts}"
            )
        self._ts = ts_ns


class LiveClock:
    """Production clock; set_time_ns is a no-op so engine code path is identical."""

    __slots__ = ()

    def now_ns(self) -> int:
        return time.time_ns()

    def set_time_ns(self, ts_ns: int) -> None:  # noqa: ARG002
        return None  # no-op in prod
