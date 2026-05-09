"""Shared fixtures for integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_minute_parquet(tmp_path: Path) -> Path:
    """Synthetic 1m parquet for fast integration tests.

    Random walk in log-space with small jitter on OHLC.
    """
    rng = np.random.default_rng(42)
    n = 20000
    sigma = 0.0008
    log_ret = rng.normal(0.0, sigma, size=n)
    log_ret[0] = 0.0
    log_close = 10.0 + np.cumsum(log_ret)
    close = np.exp(log_close)
    bar_range = np.abs(rng.normal(0.0, sigma * 1.2, size=n))
    high = close * (1.0 + bar_range)
    low = close * (1.0 - bar_range)
    open_ = np.r_[close[0], close[:-1]]
    volume = np.abs(rng.normal(100.0, 20.0, size=n))

    base_open_time_ms = 1_700_000_000_000
    open_time = np.arange(n) * 60_000 + base_open_time_ms
    close_time = open_time + 59_999

    df = pl.DataFrame({
        "open_time": open_time, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "close_time": close_time,
        "quote_volume": volume * close, "trades": np.full(n, 50, dtype=np.int64),
        "taker_buy_base": volume * 0.5,
        "taker_buy_quote": volume * close * 0.5,
        "segment_id": np.zeros(n, dtype=np.int64),
    })
    out = tmp_path / "synthetic_1m.parquet"
    df.write_parquet(out)
    return out
