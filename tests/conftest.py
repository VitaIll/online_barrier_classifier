"""Pytest fixtures for the barrier classifier test suite.

Tests run on a tiny synthetic dataset where possible so they execute in
under 30 seconds without requiring the full Binance download. Tests that
require the real dataset are gated on `RAW_PARQUET` existing and skipped
otherwise (so CI / fast-mode loops can still pass).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import utils  # noqa: E402


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def raw_parquet_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw_data" / "klines_1m.parquet"


@pytest.fixture(scope="session")
def has_raw_data(raw_parquet_path: Path) -> bool:
    return raw_parquet_path.exists()


def requires_raw_data(fn):
    """Decorator skipping a test when the raw parquet is missing."""
    return pytest.mark.skipif(
        not (REPO_ROOT / "data" / "raw_data" / "klines_1m.parquet").exists(),
        reason="raw_data/klines_1m.parquet not present; run notebook 01 first.",
    )(fn)


# -----------------------------------------------------------------------------
# Synthetic OHLCV
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 5 days of synthetic 1m OHLCV that obeys the validity invariants.

    Used by tests that need a realistic frame but not the actual Binance data.
    Returns a DataFrame indexed by UTC bar-complete timestamp with the same
    columns as the cleansed parquet from notebook 01.
    """
    rng = np.random.default_rng(seed=12345)
    n = 60 * 24 * 5  # 5 days of 1m bars

    # Geometric brownian motion with mild drift and stochastic volatility
    log_vol = np.cumsum(rng.normal(0.0, 0.005, size=n)) * 0.0  # held flat for stability
    sigma = 0.0008 * np.exp(0.0 * log_vol)
    log_returns = rng.normal(loc=0.0, scale=sigma, size=n)
    log_returns[0] = 0.0
    log_close = 10.0 + np.cumsum(log_returns)
    close = np.exp(log_close)

    # OHLC consistent with close path
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    bar_range = np.abs(rng.normal(0.0, sigma * 1.5, size=n))
    high = np.maximum(open_, close) * (1.0 + bar_range)
    low = np.minimum(open_, close) * (1.0 - bar_range)

    volume = rng.lognormal(mean=2.0, sigma=0.5, size=n)
    quote_volume = volume * close
    num_trades = rng.integers(low=10, high=500, size=n).astype(np.int64)
    taker_buy_base = volume * rng.uniform(0.3, 0.7, size=n)
    taker_buy_quote = taker_buy_base * close

    ts = pd.date_range(
        start="2024-01-01 00:01", periods=n, freq="1min", tz="UTC"
    )
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "num_trades": num_trades,
            "taker_buy_base": taker_buy_base,
            "taker_buy_quote": taker_buy_quote,
        },
        index=ts,
    )
    df.index.name = "ts"
    return df


@pytest.fixture(scope="session")
def synthetic_boundaries(synthetic_ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute base series + boundary sampling on the synthetic dataset."""
    df = utils.compute_base_series(synthetic_ohlcv.copy())
    boundaries = df.iloc[:: utils.M].copy().reset_index()
    boundaries["k"] = np.arange(len(boundaries), dtype=int)
    return df, boundaries


@pytest.fixture(scope="session")
def synthetic_labels(synthetic_boundaries) -> pd.DataFrame:
    """Boundaries with labels constructed by `construct_labels`."""
    df_minute, df_boundaries = synthetic_boundaries
    df_labelled = utils.construct_labels(
        df_boundaries.copy(),
        df_minute,
        utils.M,
        utils.ETA,
        utils.C,
    )
    return df_labelled


# -----------------------------------------------------------------------------
# Tiny m_k vectors for weight tests
# -----------------------------------------------------------------------------

@pytest.fixture
def m_k_vector() -> np.ndarray:
    """A handcrafted m_k array spanning deep losses, near-misses, and wins."""
    return np.array(
        [
            -0.020,  # deep loss
            -0.010,
            -0.005,
            -0.001,
            0.0000,
            0.0010,
            0.0023,  # = c (boundary of net break-even)
            0.0024,  # near-miss negative (just below phi)
            0.0025,  # = phi (positive boundary)
            0.0030,  # marginal positive
            0.0080,
            0.0250,  # deep winner
        ],
        dtype=float,
    )
