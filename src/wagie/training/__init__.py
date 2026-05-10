"""wagie.training — the SINGLE training surface.

Online training is a no-op concept here: the ARF online learner updates
streamingly from the LabelBuffer inside the Engine. Offline CatBoost
training is out-of-band: point ``wagie.model.catboost_path`` at a
pre-trained ``.cbm``.

This module is the entry for the small bits of pre-engine prep we DO need:

    compute_labels(parquet_path, m_minutes, alpha=…)
        → polars DataFrame with one-sided excursion label per decision bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from wagie.offline.label import compute_one_sided_excursion_label


def _aggregate_to_decision_bars(df: pl.DataFrame, m_minutes: int) -> pl.DataFrame:
    """Aggregate 1m kline rows to M-min decision bars."""
    sort_key = "open_time" if "open_time" in df.columns else df.columns[0]
    df = df.sort(sort_key)
    if "segment_id" not in df.columns:
        df = df.with_columns(segment_id=pl.lit(0, dtype=pl.Int64))
    df = df.with_columns(_dec=(pl.col(sort_key) // (m_minutes * 60_000)))
    bars = (
        df.group_by("_dec", maintain_order=True)
        .agg(
            open_time=pl.col(sort_key).first(),
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            volume=pl.col("volume").sum() if "volume" in df.columns else pl.lit(0.0),
            segment_id=pl.col("segment_id").first(),
        )
        .drop("_dec")
    )
    return bars


def compute_labels(
    parquet_path: str | Path,
    *,
    m_minutes: int = 20,
    alpha: Optional[float] = None,
    train_fraction: float = 0.6,
    quantile: float = 0.9,
) -> pl.DataFrame:
    """One-sided excursion label per decision bar.

    Returns a DataFrame with at least: open_time, open, high, low, close,
    log_excursion, label. When ``alpha`` is None it is calibrated from the
    train slice as the ``quantile``-th quantile of log_excursion.
    """
    raw = pl.read_parquet(str(parquet_path))
    bars = _aggregate_to_decision_bars(raw, m_minutes=m_minutes)
    out, _alpha = compute_one_sided_excursion_label(
        bars, alpha=alpha, train_fraction=train_fraction, quantile=quantile,
    )
    return out


__all__ = ["compute_labels"]
