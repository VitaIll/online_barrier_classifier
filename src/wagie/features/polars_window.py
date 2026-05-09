"""Polars batch features — causal-by-construction rolling expressions.

For OHLC volatility estimators that don't have streaming O(1) revertable forms:
Parkinson, Garman-Klass, Yang-Zhang, Rogers-Satchell. Implemented as polars
expressions (rolling_mean / rolling_sum on derived columns). Causal because
polars `rolling_*` are trailing by default.

Implementation: maintain a deque of size `update_samples` with the recent OHLC
rows. On each update_one, materialize a small polars frame from the deque,
apply the registered expression, and emit the latest value. Cost is O(W) per
call, but W is small (24-96 bars typical).
"""

from __future__ import annotations

import collections
import math
from typing import Callable

import polars as pl

from wagie.features.base import (
    FeatureKind,
    FeatureSpec,
    PastCovariateSpec,
    _hash_floats,
)


# Each expression must accept a polars DataFrame with columns (open, high, low,
# close, volume) and return a polars Expr that produces ONE column with the
# feature value at each row. The PolarsBatchFeature reads the LAST row's value.
PolarsExprFn = Callable[[], pl.Expr]


class PolarsBatchFeature:
    """Rolling-window feature computed via polars expression on a recent slice.

    The feature's `lag` and `window` together define how many bars of history
    are buffered (`update_samples = lag + window`). On each update_one(x), we
    push x's (o,h,l,c,v) into a deque of that size, materialize a polars frame,
    apply the user-supplied expression, and emit the latest value, then DROP
    the oldest `lag` bars from being read (to enforce lag-binding-at-config-time).
    """

    __slots__ = ("spec", "_expr_fn", "_buf", "_cols", "_n_seen")

    OHLCV_COLS = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        spec: FeatureSpec,
        expr_fn: PolarsExprFn,
        cols: tuple[str, ...] = OHLCV_COLS,
    ):
        if spec.kind is not FeatureKind.POLARS:
            raise ValueError(
                f"PolarsBatchFeature requires kind=POLARS; got {spec.kind}"
            )
        self.spec = spec
        self._expr_fn = expr_fn
        self._cols = cols
        # Deque holds tuples in column-order; size = update_samples
        self._buf: collections.deque = collections.deque(maxlen=spec.update_samples)
        self._n_seen = 0

    def update_one(self, x: dict) -> dict:
        # Buffer current row's OHLCV
        try:
            row = tuple(float(x[c]) for c in self._cols)
        except (KeyError, TypeError, ValueError):
            self._n_seen += 1
            return {**x, self.spec.name: float("nan")}
        self._buf.append(row)
        self._n_seen += 1

        if self._n_seen < self.spec.update_samples:
            return {**x, self.spec.name: float("nan")}

        # Slice off the most-recent `lag` rows (lag-binding-at-config-time):
        # the expression only sees rows from `lag` bars ago.
        usable = list(self._buf)
        if self.spec.lag > 0:
            usable = usable[: -self.spec.lag] if self.spec.lag <= len(usable) else []
        if len(usable) < self.spec.window:
            return {**x, self.spec.name: float("nan")}

        # Materialize a tiny polars frame and apply the expression.
        df = pl.DataFrame(
            {c: [row[i] for row in usable] for i, c in enumerate(self._cols)}
        )
        try:
            result = df.select(self._expr_fn().alias("_out"))["_out"]
            val = result[-1]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return {**x, self.spec.name: float("nan")}
            return {**x, self.spec.name: float(val)}
        except Exception:
            return {**x, self.spec.name: float("nan")}

    def reset_state(self) -> None:
        self._buf.clear()
        self._n_seen = 0

    def state_hash(self) -> bytes:
        flat: list[float] = []
        for row in self._buf:
            flat.extend(row)
        flat.append(float(self._n_seen))
        return _hash_floats(flat)


# -----------------------------------------------------------------------------
# Concrete expressions for OHLC volatility estimators.
# polars-ta does NOT have these (verified Wave 4); we hand-roll them.
# -----------------------------------------------------------------------------

def parkinson_var_expr(window: int) -> PolarsExprFn:
    """Parkinson variance estimator: var = (1 / (4 ln 2)) * mean((ln(H/L))^2)
    over a rolling window. Causal by polars rolling semantics."""
    def _fn() -> pl.Expr:
        return (
            ((pl.col("high") / pl.col("low")).log()) ** 2
        ).rolling_mean(window_size=window, min_samples=window) / (4.0 * math.log(2.0))
    return _fn


def garman_klass_var_expr(window: int) -> PolarsExprFn:
    """Garman-Klass: var = mean(0.5 * (ln H/L)^2 - (2 ln 2 - 1) * (ln C/O)^2)."""
    def _fn() -> pl.Expr:
        log_hl = (pl.col("high") / pl.col("low")).log()
        log_co = (pl.col("close") / pl.col("open")).log()
        per_bar = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
        return per_bar.rolling_mean(window_size=window, min_samples=window)
    return _fn


def rogers_satchell_var_expr(window: int) -> PolarsExprFn:
    """Rogers-Satchell drift-independent variance estimator."""
    def _fn() -> pl.Expr:
        log_ho = (pl.col("high") / pl.col("open")).log()
        log_hc = (pl.col("high") / pl.col("close")).log()
        log_lo = (pl.col("low") / pl.col("open")).log()
        log_lc = (pl.col("low") / pl.col("close")).log()
        per_bar = log_ho * log_hc + log_lo * log_lc
        return per_bar.rolling_mean(window_size=window, min_samples=window)
    return _fn


def make_parkinson_var(name: str, window: int, lag: int = 1) -> PolarsBatchFeature:
    spec = PastCovariateSpec(name=name, kind=FeatureKind.POLARS, lag=lag, window=window)
    return PolarsBatchFeature(spec, parkinson_var_expr(window))


def make_garman_klass_var(name: str, window: int, lag: int = 1) -> PolarsBatchFeature:
    spec = PastCovariateSpec(name=name, kind=FeatureKind.POLARS, lag=lag, window=window)
    return PolarsBatchFeature(spec, garman_klass_var_expr(window))


def make_rogers_satchell_var(name: str, window: int, lag: int = 1) -> PolarsBatchFeature:
    spec = PastCovariateSpec(name=name, kind=FeatureKind.POLARS, lag=lag, window=window)
    return PolarsBatchFeature(spec, rogers_satchell_var_expr(window))


__all__ = [
    "PolarsBatchFeature",
    "parkinson_var_expr",
    "garman_klass_var_expr",
    "rogers_satchell_var_expr",
    "make_parkinson_var",
    "make_garman_klass_var",
    "make_rogers_satchell_var",
]
