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


# =============================================================================
# LABEL CONSTRUCTION (matches notebooks/feature_build.ipynb cell 5 exactly)
# =============================================================================
#
# Project contract: y_k = 1[ ln(H_{k+1} / C_k) >= alpha ] where H_{k+1} is the
# high of the NEXT decision bar (not the current one), C_k is current bar's
# close, and alpha is calibrated as the 90th-percentile of training-window
# excursions. Labels are NaN when the next bar belongs to a different
# `segment_id` (1-min gap separator) — those bars cannot be labeled because
# the lookahead crosses a gap.
#
# This is the ONE canonical implementation. Notebooks should call these
# helpers; do not re-implement inline.

LOG_EXCURSION_COL = "log_excursion"
NEXT_HIGH_COL = "next_high"
NEXT_SEGMENT_COL = "next_segment_id"


def compute_log_excursion(
    bars_df: pd.DataFrame,
    *,
    high_col: str = "high",
    close_col: str = "close",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Add ``next_high``, ``next_segment_id``, and ``log_excursion`` columns.

    `log_excursion[k] = ln(high[k+1] / close[k])` if the next bar is in the
    same segment, else NaN. The two helper columns are kept on the returned
    frame because callers (e.g. label construction, calibration) need them.

    Returns a NEW DataFrame (does not mutate input).
    """
    out = bars_df.copy()
    out[NEXT_HIGH_COL] = out[high_col].shift(-1)
    out[NEXT_SEGMENT_COL] = out[segment_col].shift(-1)
    out[LOG_EXCURSION_COL] = np.where(
        out[segment_col] == out[NEXT_SEGMENT_COL],
        np.log(out[NEXT_HIGH_COL] / out[close_col]),
        np.nan,
    )
    return out


def calibrate_alpha(
    bars_df: pd.DataFrame,
    train_fraction: float,
    quantile: float = 0.9,
    *,
    excursion_col: str = LOG_EXCURSION_COL,
) -> float:
    """Calibrate the barrier ``alpha`` from the training window's excursions.

    Matches the notebook contract: take the first ``int(N * train_fraction) - 1``
    rows (the ``-1`` keeps a one-bar gap so the last calibration bar's
    excursion does not look into the test window), drop NaN excursions, take
    the requested quantile.

    Raises ``RuntimeError`` if no valid excursions exist in the calibration
    subset (typical cause: too-small dataframe, bad ``train_fraction``, or all
    bars at gap boundaries).
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0,1], got {quantile}")

    train_end_idx = int(len(bars_df) * train_fraction)
    calibration_df = bars_df.iloc[: max(train_end_idx - 1, 0)]
    train_excursions = calibration_df[excursion_col].dropna()
    if train_excursions.empty:
        raise RuntimeError(
            "No valid excursions in calibration subset. Check segment gaps, "
            "date range, and train_fraction."
        )
    return float(train_excursions.quantile(quantile))


def construct_labels(
    bars_df: pd.DataFrame,
    alpha: float,
    *,
    excursion_col: str = LOG_EXCURSION_COL,
    label_col: str = "label",
) -> pd.DataFrame:
    """Materialize integer labels ``y_k = 1[log_excursion_k >= alpha]``.

    NaN ``log_excursion`` (gap-crossing bars) yields NaN label — the caller
    decides whether to drop or impute. Notebook cell 7 drops NaN labels and
    casts the rest to int; this function preserves NaN so tests can verify
    causality at the boundary.

    Returns a NEW DataFrame (does not mutate input). Adds ``label`` column.
    """
    out = bars_df.copy()
    out[label_col] = np.where(
        out[excursion_col].notna(),
        (out[excursion_col] >= alpha).astype(int),
        np.nan,
    )
    return out


# =============================================================================
# CHRONOLOGICAL SPLIT (matches notebooks/offline_train.ipynb cell 3 exactly)
# =============================================================================
#
# Project contract: NO embargo, NO walk-forward CV. The split is a single
# chronological cut at `train_fraction * N`, then the last `val_fraction` of
# the train window is carved off as validation.
#
# val_size formula (from the notebook):
#   split_idx = int(N * train_fraction)
#   val_size  = int(split_idx * val_fraction)   # NOT int(N * tf * vf)
#
# The two truncations can differ by 1 from a single-step computation; the
# round-trip test in test_label_split_utils.py pins this exact behaviour.


def chronological_split_indices(
    n: int,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> tuple[int, int]:
    """Return ``(train_end, val_end)`` row indices for a chronological split.

    Slicing ``df.iloc[:train_end]`` gives train; ``df.iloc[train_end:val_end]``
    gives val; ``df.iloc[val_end:]`` gives test.

    ``val_fraction`` is interpreted as the fraction of the **train window**
    (not of the whole dataset) that becomes validation, matching the
    notebook semantics. With ``train_fraction=0.6`` and ``val_fraction=0.2``,
    the resulting splits are ~48% / ~12% / 40% of N.

    Edge case: if the train window has fewer than 2 rows, no validation set
    is carved (``train_end == val_end``).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0,1), got {val_fraction}")

    split_idx = int(n * train_fraction)
    val_size = int(split_idx * val_fraction)
    val_size = max(val_size, 1) if split_idx >= 2 else 0
    if val_size > 0 and val_size < split_idx:
        train_end = split_idx - val_size
        val_end = split_idx
    else:
        train_end = split_idx
        val_end = split_idx
    return train_end, val_end


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` chronologically into ``(train, val, test)`` DataFrames.

    See ``chronological_split_indices`` for the index math. The returned
    DataFrames are independent copies (callers may mutate freely).

    The relative order ``train.index < val.index < test.index`` holds when
    the input has its natural row order — which is the notebook contract
    (no shuffle).
    """
    train_end, val_end = chronological_split_indices(
        len(df), train_fraction=train_fraction, val_fraction=val_fraction
    )
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test
