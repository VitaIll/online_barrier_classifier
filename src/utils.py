"""
Shared utility functions used across notebooks and transformers.

This module contains only functions that are:
1. Used in multiple places, and
2. Correctness-critical or performance-sensitive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# NUMERICAL UTILITIES
# =============================================================================

def safe_divide(
    numerator: float,
    denominator: float,
    neutral: float = 0.0,
    eps: float = 1e-10,
) -> tuple[float, bool]:
    """
    Perform safe division with explicit handling of near-zero denominators.

    Parameters
    ----------
    numerator : float
        The numerator value.
    denominator : float
        The denominator value.
    neutral : float, default 0.0
        Value to return when division is undefined.
    eps : float, default 1e-10
        Threshold below which denominator is considered zero.

    Returns
    -------
    tuple[float, bool]
        (result, flag) where flag=True indicates division was undefined.
    """
    if abs(denominator) < eps:
        return neutral, True
    return numerator / denominator, False


def parkinson_variance(high: float, low: float) -> float:
    """
    Compute single-bar Parkinson variance estimator.

    sigma2_P = (1 / (4 * ln(2))) * (ln(H/L))^2

    Parameters
    ----------
    high : float
        High price of the bar.
    low : float
        Low price of the bar.

    Returns
    -------
    float
        Estimated variance for the bar period.
    """
    if low <= 0 or high <= 0 or high < low:
        return 0.0
    log_range = np.log(high / low)
    return (log_range ** 2) / (4 * np.log(2))


def garman_klass_variance(
    open_: float,
    high: float,
    low: float,
    close: float,
) -> float:
    """
    Compute single-bar Garman-Klass variance estimator.

    sigma2_GK = 0.5 * (ln(H/L))^2 - (2 ln(2) - 1) * (ln(C/O))^2

    Parameters
    ----------
    open_ : float
        Open price of the bar.
    high : float
        High price of the bar.
    low : float
        Low price of the bar.
    close : float
        Close price of the bar.

    Returns
    -------
    float
        Estimated variance for the bar period.
    """
    if low <= 0 or high <= 0 or open_ <= 0 or close <= 0:
        return 0.0
    if high < low:
        return 0.0

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    return 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def config_hash(config: dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 hash of a configuration dictionary.

    Used for cache invalidation: if config hash changes, dependent
    artifacts should be regenerated.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    str
        Hexadecimal hash string (first 16 characters).
    """
    config_str = yaml.dump(config, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# =============================================================================
# TIMESTAMP UTILITIES
# =============================================================================

def infer_unix_timestamp_unit(ts: int) -> str:
    """
    Infer whether a Unix timestamp is in milliseconds or microseconds.

    Binance spot public data uses microseconds from 2025-01-01 onward.
    This helper uses a magnitude heuristic that is stable for modern dates.

    Parameters
    ----------
    ts : int
        Unix timestamp.

    Returns
    -------
    str
        Either "ms" or "us".
    """
    # 2026-01-01 in milliseconds is ~1.77e12; in microseconds is ~1.77e15.
    return "us" if ts >= 10**14 else "ms"


def to_ms_timestamp(ts: int, unit: str | None = None) -> int:
    """
    Normalize a Unix timestamp to integer milliseconds.

    Parameters
    ----------
    ts : int
        Unix timestamp in either ms or us.
    unit : {"ms", "us"}, optional
        Explicit unit. If None, inferred via infer_unix_timestamp_unit.

    Returns
    -------
    int
        Unix timestamp in milliseconds.
    """
    unit = unit or infer_unix_timestamp_unit(ts)
    if unit == "ms":
        return int(ts)
    if unit == "us":
        return int(ts // 1_000)
    raise ValueError(f"Unsupported unit: {unit}")


def ts_to_datetime(ts: int, unit: str | None = None) -> pd.Timestamp:
    """
    Convert a Unix timestamp (ms or us) to a UTC pandas Timestamp.

    Parameters
    ----------
    ts : int
        Unix timestamp.
    unit : {"ms", "us"}, optional
        Explicit unit. If None, inferred via infer_unix_timestamp_unit.

    Returns
    -------
    pd.Timestamp
        UTC-localized timestamp.
    """
    unit = unit or infer_unix_timestamp_unit(ts)
    return pd.Timestamp(ts, unit=unit, tz="UTC")


def datetime_to_ts(dt: pd.Timestamp, unit: str = "ms") -> int:
    """
    Convert a pandas Timestamp to a Unix timestamp in the requested unit.

    Parameters
    ----------
    dt : pd.Timestamp
        Pandas timestamp (any timezone or naive).
    unit : {"ms", "us"}, default "ms"
        Target unit.

    Returns
    -------
    int
        Unix timestamp in requested unit.
    """
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    dt = dt.tz_convert("UTC")

    # dt.value is nanoseconds since epoch.
    ns = int(dt.value)
    if unit == "ms":
        return ns // 1_000_000
    if unit == "us":
        return ns // 1_000
    raise ValueError(f"Unsupported unit: {unit}")
