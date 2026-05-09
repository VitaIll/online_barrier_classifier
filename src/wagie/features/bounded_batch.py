"""Bounded-batch features — numba kernels over fixed backward window W.

For features that don't have streaming O(1) revertable forms AND aren't cleanly
expressible as polars rolling: Hurst (R/S), DFA, sample entropy, permutation
entropy, FFW fractional differentiation. Recompute over W samples on each call
(O(W) per bar).

Numba is a hard dep (D9). Kernels are written to be deterministic given the
same input array. Cross-machine bit equality is a NIGHTLY test (T19 — different
LLVM versions can change reduction order).
"""

from __future__ import annotations

import collections
import math
from typing import Callable

import numpy as np

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fall back to numpy; tests will still run, perf will degrade.

from wagie.features.base import (
    FeatureKind,
    FeatureSpec,
    PastCovariateSpec,
    _hash_floats,
)


KernelFn = Callable[[np.ndarray], float]


class BoundedBatchFeature:
    """Numba kernel over a circular buffer of size `update_samples`.

    Kernel signature: (np.ndarray of length window) -> float.
    The buffer holds the last `lag + window` source values; kernel sees only
    the oldest `window` of them (lag-binding-at-config-time).
    """

    __slots__ = ("spec", "_kernel", "_source_col", "_buf", "_n_seen")

    def __init__(
        self,
        spec: FeatureSpec,
        kernel: KernelFn,
        source_col: str,
    ):
        if spec.kind is not FeatureKind.NUMBA:
            raise ValueError(
                f"BoundedBatchFeature requires kind=NUMBA; got {spec.kind}"
            )
        self.spec = spec
        self._kernel = kernel
        self._source_col = source_col
        self._buf: collections.deque = collections.deque(maxlen=spec.update_samples)
        self._n_seen = 0

    def update_one(self, x: dict) -> dict:
        v = x.get(self._source_col)
        if v is None:
            v = float("nan")
        v = float(v)
        if math.isnan(v):
            self._n_seen += 1
            return {**x, self.spec.name: float("nan")}
        self._buf.append(v)
        self._n_seen += 1
        if self._n_seen < self.spec.update_samples:
            return {**x, self.spec.name: float("nan")}
        # Slice: kernel sees rows from `lag` bars ago (oldest `window` of buffer).
        arr = np.array(list(self._buf)[: -self.spec.lag] if self.spec.lag > 0 else list(self._buf),
                       dtype=np.float64)
        if len(arr) < self.spec.window:
            return {**x, self.spec.name: float("nan")}
        try:
            val = float(self._kernel(arr[-self.spec.window:]))
            if math.isnan(val) or math.isinf(val):
                return {**x, self.spec.name: float("nan")}
            return {**x, self.spec.name: val}
        except Exception:
            return {**x, self.spec.name: float("nan")}

    def reset_state(self) -> None:
        self._buf.clear()
        self._n_seen = 0

    def state_hash(self) -> bytes:
        return _hash_floats(list(self._buf) + [float(self._n_seen)])


# -----------------------------------------------------------------------------
# Kernels. Decorate with numba.njit if available; fall back to numpy.
# -----------------------------------------------------------------------------

def _maybe_njit(fn):
    if NUMBA_AVAILABLE:
        return numba.njit(cache=True, fastmath=False)(fn)
    return fn


@_maybe_njit
def _hurst_rs_kernel(x: np.ndarray) -> float:
    """Hurst exponent via Rescaled Range (R/S) analysis.
    Ported from src/features.py::compute_hurst_rs but on a fixed-size window."""
    n = x.shape[0]
    if n < 32:
        return float("nan")
    # Subseries lengths (geometric scales)
    n_scales = 6
    log_min = math.log(8.0)
    log_max = math.log(n / 2.0)
    if log_max <= log_min:
        return float("nan")
    log_n = np.empty(n_scales, dtype=np.float64)
    log_rs = np.empty(n_scales, dtype=np.float64)

    for i in range(n_scales):
        ll = log_min + i * (log_max - log_min) / (n_scales - 1)
        m = int(math.exp(ll))
        if m < 8 or m > n:
            return float("nan")
        n_chunks = n // m
        if n_chunks < 1:
            return float("nan")
        rs_sum = 0.0
        rs_count = 0
        for ck in range(n_chunks):
            chunk = x[ck * m : (ck + 1) * m]
            mu = np.mean(chunk)
            dev = chunk - mu
            cum = np.cumsum(dev)
            r = cum.max() - cum.min()
            s = np.std(chunk)
            if s > 0.0 and r > 0.0:
                rs_sum += r / s
                rs_count += 1
        if rs_count == 0:
            return float("nan")
        log_n[i] = math.log(float(m))
        log_rs[i] = math.log(rs_sum / rs_count)

    # Linear regression in log-space
    log_n_mean = log_n.mean()
    log_rs_mean = log_rs.mean()
    num = 0.0
    den = 0.0
    for i in range(n_scales):
        num += (log_n[i] - log_n_mean) * (log_rs[i] - log_rs_mean)
        den += (log_n[i] - log_n_mean) ** 2
    if den <= 0.0:
        return float("nan")
    slope = num / den
    return float(slope)


@_maybe_njit
def _sample_entropy_kernel(x: np.ndarray) -> float:
    """Sample entropy with m=2, r=0.2*std. Ported from src/features.py."""
    n = x.shape[0]
    if n < 32:
        return float("nan")
    s = np.std(x)
    if s <= 0.0:
        return float("nan")
    r = 0.2 * s
    m = 2

    def _count(m_arg: int) -> int:
        cnt = 0
        for i in range(n - m_arg):
            for j in range(i + 1, n - m_arg):
                # Chebyshev distance
                d = 0.0
                for k in range(m_arg):
                    diff = abs(x[i + k] - x[j + k])
                    if diff > d:
                        d = diff
                if d <= r:
                    cnt += 1
        return cnt

    a = _count(m + 1)
    b = _count(m)
    if a == 0 or b == 0:
        return float("nan")
    return float(-math.log(a / b))


def hurst_kernel(x: np.ndarray) -> float:
    """Public Hurst kernel (handles fallback when numba unavailable)."""
    return float(_hurst_rs_kernel(np.asarray(x, dtype=np.float64)))


def sample_entropy_kernel(x: np.ndarray) -> float:
    """Public sample-entropy kernel."""
    return float(_sample_entropy_kernel(np.asarray(x, dtype=np.float64)))


def make_hurst(name: str, source_col: str = "log_close", window: int = 128, lag: int = 1) -> BoundedBatchFeature:
    """Convenience: rolling Hurst over W bars."""
    spec = PastCovariateSpec(name=name, kind=FeatureKind.NUMBA, lag=lag, window=window)
    return BoundedBatchFeature(spec, hurst_kernel, source_col)


def make_sample_entropy(name: str, source_col: str = "return", window: int = 128, lag: int = 1) -> BoundedBatchFeature:
    """Convenience: rolling sample entropy over W bars."""
    spec = PastCovariateSpec(name=name, kind=FeatureKind.NUMBA, lag=lag, window=window)
    return BoundedBatchFeature(spec, sample_entropy_kernel, source_col)


__all__ = [
    "BoundedBatchFeature",
    "hurst_kernel",
    "sample_entropy_kernel",
    "make_hurst",
    "make_sample_entropy",
    "NUMBA_AVAILABLE",
]
