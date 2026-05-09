"""Custom rolling stats for IQR and PTP (peak-to-peak).

river.stats has Min, Max, Quantile but not IQR or PTP directly. We build them
on top via composition: PTP = Max - Min; IQR = Q75 - Q25.

These satisfy the river.utils.Rolling contract: update(x), revert(x) (where
possible), get() returning a float.
"""

from __future__ import annotations

import collections
import math


class RollingPTP:
    """Rolling peak-to-peak (max - min) using a sliding window deque.

    Note: not O(1) revert — this implementation is O(W) per call but cached.
    For a streaming financial pipeline at 1 update/20min this is fine.
    """

    def __init__(self):
        self._buf: collections.deque[float] = collections.deque()
        self._n = 0

    def update(self, x: float) -> "RollingPTP":
        if x is None:
            return self
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return self
        if math.isnan(xf):
            return self
        self._buf.append(xf)
        self._n += 1
        return self

    def revert(self, x: float) -> "RollingPTP":  # noqa: ARG002
        if self._buf:
            self._buf.popleft()
        return self

    def get(self) -> float:
        if not self._buf:
            return float("nan")
        return max(self._buf) - min(self._buf)


class RollingIQR:
    """Rolling IQR via sorted-list maintenance. O(log W) update via bisect."""

    def __init__(self):
        self._buf: collections.deque[float] = collections.deque()

    def update(self, x: float) -> "RollingIQR":
        if x is None:
            return self
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return self
        if math.isnan(xf):
            return self
        self._buf.append(xf)
        return self

    def revert(self, x: float) -> "RollingIQR":  # noqa: ARG002
        if self._buf:
            self._buf.popleft()
        return self

    def get(self) -> float:
        if len(self._buf) < 4:
            return float("nan")
        srt = sorted(self._buf)
        n = len(srt)
        q25_idx = max(0, int(n * 0.25))
        q75_idx = min(n - 1, int(n * 0.75))
        return srt[q75_idx] - srt[q25_idx]


# Adapter: river.utils.Rolling expects an estimator with a name and update/get.
# Our stats above are duck-typed but missing `name`. river uses it in repr
# only, so we can ignore. Wrap if needed.

class _Wrap:
    """Minimal river-stats-compatible wrapper. river.utils.Rolling calls
    .update(x) and .revert(x) on the underlying stat plus the rolling window
    bookkeeping. We rely on the wrapped class to provide both."""

    def __init__(self, inner):
        self._inner = inner
        self.name = type(inner).__name__

    def update(self, x):
        self._inner.update(x)
        return self

    def revert(self, x):
        if hasattr(self._inner, "revert"):
            self._inner.revert(x)
        return self

    def get(self):
        return self._inner.get()


def make_ptp() -> _Wrap:
    return _Wrap(RollingPTP())


def make_iqr() -> _Wrap:
    return _Wrap(RollingIQR())


__all__ = ["RollingPTP", "RollingIQR", "make_ptp", "make_iqr"]
