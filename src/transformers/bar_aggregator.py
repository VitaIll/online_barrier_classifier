"""
Decision bar aggregator for converting minute bars to N-minute bars.

This component aggregates streaming minute-bar observations into decision bars
of configurable length (default: 20 minutes).

This is not a River Transformer because it emits either None (while buffering)
or an aggregated bar dict (when ready).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional


class DecisionBarAggregator:
    """
    Aggregate minute bars into decision bars.

    Collects n strictly consecutive minute bars (in time) and emits a
    single decision bar with aggregated OHLCV statistics.

    Any missing-minute gap resets the internal buffer and starts a new
    in-memory segment. Segment boundaries should also be provided explicitly
    via segment_id (produced in data_download.ipynb).

    Parameters
    ----------
    n : int, default 20
        Number of minute bars per decision bar.
    expected_step_ms : int, default 60000
        Expected spacing between consecutive minute bars (milliseconds).
    """

    def __init__(self, n: int = 20, expected_step_ms: int = 60_000) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.expected_step_ms = expected_step_ms
        self.buffer: deque[dict[str, Any]] = deque()
        self.bar_count: int = 0
        self._prev_open_time: Optional[int] = None
        self._prev_segment_id: Optional[int] = None

    def reset(self) -> None:
        """Reset internal buffer and time/segment trackers."""
        self.buffer.clear()
        self._prev_open_time = None
        self._prev_segment_id = None

    def update(self, x: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Add a minute bar to the buffer and aggregate when full.

        Parameters
        ----------
        x : dict
            Minute bar with keys: open_time, open, high, low, close,
            volume, close_time, quote_volume, trades, taker_buy_base,
            taker_buy_quote, segment_id.

        Returns
        -------
        dict or None
            Aggregated decision bar when ready, else None while buffering.
        """
        open_time = int(x["open_time"])
        segment_id = x.get("segment_id")

        # Reset on explicit segment boundary.
        if (
            self._prev_segment_id is not None
            and segment_id is not None
            and int(segment_id) != self._prev_segment_id
        ):
            self.reset()

        # Reset on any missing-minute gap.
        if self._prev_open_time is not None:
            if open_time - self._prev_open_time != self.expected_step_ms:
                self.reset()

        self._prev_open_time = open_time
        if segment_id is not None:
            self._prev_segment_id = int(segment_id)

        self.buffer.append(x)
        if len(self.buffer) < self.n:
            return None

        bar = self._aggregate()
        self.buffer.clear()
        self.bar_count += 1
        return bar

    def _aggregate(self) -> dict[str, Any]:
        """Aggregate buffered minute bars into a single decision bar."""
        bars = list(self.buffer)
        segment_id = bars[0].get("segment_id")

        return {
            "open_time": bars[0]["open_time"],
            "close_time": bars[-1]["close_time"],
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
            "volume": sum(b["volume"] for b in bars),
            "quote_volume": sum(b["quote_volume"] for b in bars),
            "trades": sum(b["trades"] for b in bars),
            "taker_buy_base": sum(b["taker_buy_base"] for b in bars),
            "taker_buy_quote": sum(b["taker_buy_quote"] for b in bars),
            "segment_id": segment_id,
        }
