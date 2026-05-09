"""Streaming features — river.utils.Rolling + river.stats.X.

Bit-identical between batch and stream BY CONSTRUCTION because they implement
O(1) revertable updates (Welford for variance, monotonic deque for max/min, etc).

Use for: rolling mean, var, sum, EMA, max, min over decision-bar features
(returns, log-volume, etc.).
"""

from __future__ import annotations

import collections
import math
from typing import Callable

from river import stats, utils

from wagie.features.base import (
    FeatureKind,
    FeatureSpec,
    PastCovariateSpec,
    _hash_floats,
)


class RiverRollingFeature:
    """Wraps a river.utils.Rolling around a river.stats.Univariate.

    Implements the lag-binding-at-config-time contract: at update_one(x), we
    push x[source_col] into a lag buffer of size `spec.lag`; once the lag
    buffer is full, the oldest element is fed into the rolling stat. So the
    rolling window only sees values from at least `spec.lag` bars ago.

    Examples:
        spec = PastCovariateSpec("return_mean_24", FeatureKind.STREAMING,
                                 lag=1, window=24)
        f = RiverRollingFeature(spec, source_col="return", stat_factory=stats.Mean)
    """

    __slots__ = (
        "spec",
        "_source_col",
        "_stat_factory",
        "_rolling",
        "_lag_buf",
        "_n_seen",
    )

    def __init__(
        self,
        spec: FeatureSpec,
        source_col: str,
        stat_factory: Callable[..., object],
    ):
        if spec.kind is not FeatureKind.STREAMING:
            raise ValueError(
                f"RiverRollingFeature requires kind=STREAMING; got {spec.kind}"
            )
        self.spec = spec
        self._source_col = source_col
        self._stat_factory = stat_factory
        self._rolling = self._make_rolling_stat()
        # lag buffer: holds the last `lag` source values; element at index 0
        # is the oldest (about to be consumed by the rolling stat).
        self._lag_buf: collections.deque = collections.deque(maxlen=spec.lag)
        self._n_seen = 0

    def _make_rolling_stat(self):
        """Build a rolling stat. If factory is a rolling-* class (accepts
        window_size arg), use it directly. Otherwise wrap in utils.Rolling."""
        try:
            # Try as a rolling stat with built-in window
            return self._stat_factory(window_size=self.spec.window)
        except TypeError:
            # Fallback: wrap a non-windowed stat (Mean, Var) in utils.Rolling
            return utils.Rolling(self._stat_factory(), window_size=self.spec.window)

    def update_one(self, x: dict) -> dict:
        v = x.get(self._source_col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            # Don't poison the buffer with NaN; emit NaN this step.
            return {**x, self.spec.name: float("nan")}
        v = float(v)
        # Push newest value into the lag buffer; if full, the oldest is consumed
        # into the rolling stat (this realizes lag-binding-at-config-time).
        if len(self._lag_buf) == self._lag_buf.maxlen:
            oldest = self._lag_buf[0]
            self._rolling.update(oldest)
        self._lag_buf.append(v)
        self._n_seen += 1
        # Emit the rolling stat's current value, but only once we've seen
        # >= update_samples observations (warmup gate).
        if self._n_seen < self.spec.update_samples:
            return {**x, self.spec.name: float("nan")}
        result = self._rolling.get()
        return {**x, self.spec.name: float(result) if result is not None else float("nan")}

    def reset_state(self) -> None:
        self._rolling = self._make_rolling_stat()
        self._lag_buf.clear()
        self._n_seen = 0

    def state_hash(self) -> bytes:
        # Hash the lag buffer + the rolling stat's current value. river stats
        # don't expose internal Welford state cleanly; the get() value is a
        # proxy that's deterministic given the input sequence.
        try:
            cur = float(self._rolling.get()) if self._rolling.get() is not None else 0.0
        except Exception:
            cur = 0.0
        return _hash_floats(list(self._lag_buf) + [cur, float(self._n_seen)])


class LagFeature:
    """Pure lag — emits x[source_col] from `lag` bars ago.

    spec.window is forced to 1 (lag features are point lookups, not aggregates).
    """

    __slots__ = ("spec", "_source_col", "_buf", "_n_seen")

    def __init__(self, spec: FeatureSpec, source_col: str):
        if spec.window != 1:
            raise ValueError(
                f"LagFeature {spec.name}: window must be 1; got {spec.window}"
            )
        self.spec = spec
        self._source_col = source_col
        self._buf: collections.deque = collections.deque(maxlen=spec.lag)
        self._n_seen = 0

    def update_one(self, x: dict) -> dict:
        v = x.get(self._source_col)
        if v is None:
            v = float("nan")
        out_val: float
        if len(self._buf) == self._buf.maxlen:
            out_val = float(self._buf[0])
        else:
            out_val = float("nan")
        self._buf.append(float(v))
        self._n_seen += 1
        return {**x, self.spec.name: out_val}

    def reset_state(self) -> None:
        self._buf.clear()
        self._n_seen = 0

    def state_hash(self) -> bytes:
        return _hash_floats(list(self._buf) + [float(self._n_seen)])


class IdentityFeature:
    """Pass-through: emits x[source_col] under spec.name. lag=0 allowed only via
    FutureCovariateSpec; otherwise lag>=1 enforced at FeatureSpec level."""

    __slots__ = ("spec", "_source_col", "_buf")

    def __init__(self, spec: FeatureSpec, source_col: str):
        self.spec = spec
        self._source_col = source_col
        # If lag > 0, we still need a buffer to delay by `lag` bars.
        self._buf: collections.deque = collections.deque(
            maxlen=max(1, spec.lag)
        ) if spec.lag > 0 else None  # type: ignore[assignment]

    def update_one(self, x: dict) -> dict:
        v = x.get(self._source_col, float("nan"))
        v = float(v) if v is not None else float("nan")
        if self.spec.lag == 0:
            return {**x, self.spec.name: v}
        # lag > 0: emit the value from `lag` bars ago
        if len(self._buf) == self._buf.maxlen:
            out_val = self._buf[0]
        else:
            out_val = float("nan")
        self._buf.append(v)
        return {**x, self.spec.name: out_val}

    def reset_state(self) -> None:
        if self._buf is not None:
            self._buf.clear()

    def state_hash(self) -> bytes:
        if self._buf is None:
            return b"identity_nolag"
        return _hash_floats(list(self._buf))


def make_rolling_mean(name: str, source_col: str, window: int, lag: int = 1) -> RiverRollingFeature:
    """Convenience: rolling mean of a source column, lag-shifted."""
    spec = PastCovariateSpec(name=name, kind=FeatureKind.STREAMING, lag=lag, window=window)
    return RiverRollingFeature(spec, source_col=source_col, stat_factory=stats.Mean)


def make_rolling_var(name: str, source_col: str, window: int, lag: int = 1) -> RiverRollingFeature:
    """Convenience: rolling variance (Welford) of a source column, lag-shifted."""
    spec = PastCovariateSpec(name=name, kind=FeatureKind.STREAMING, lag=lag, window=window)
    return RiverRollingFeature(spec, source_col=source_col, stat_factory=stats.Var)


def make_lag(name: str, source_col: str, lag: int) -> LagFeature:
    """Convenience: simple lag-N feature."""
    spec = PastCovariateSpec(name=name, kind=FeatureKind.STREAMING, lag=lag, window=1)
    return LagFeature(spec, source_col=source_col)


__all__ = [
    "RiverRollingFeature",
    "LagFeature",
    "IdentityFeature",
    "make_rolling_mean",
    "make_rolling_var",
    "make_lag",
]
