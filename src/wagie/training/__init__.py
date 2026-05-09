"""wagie.training — the SINGLE training surface.

Online training is a no-op concept here: ARF + ACI learn streamingly from
the LabelBuffer inside the Engine. Offline CatBoost training is out-of-band:
point `wagie.model.catboost_path` at a pre-trained `.cbm`.

This module is the entry for the small bits of pre-engine prep we DO need:

    compute_labels(parquet_path, m_minutes, alpha=…)
        → polars DataFrame with one-sided excursion label per decision bar.

    warm_online_quantiles(parquet_path, m_minutes, alphas, train_frac, n_regimes)
        → dict[alpha → dict[regime_id → q_init]] for warm-starting Mondrian-ACI.

The ExperimentProtocol calls these when `spec.training.warm_calibrator_quantiles`
is True. They are ALSO directly importable for ad-hoc use.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import polars as pl

from wagie.offline.label import compute_one_sided_excursion_label
from wagie.offline.warmup import (
    finite_sample_quantile,
    lac_score,
    warm_q_init_by_regime as _warm_per_alpha,
)


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
    log_excursion, label. When `alpha` is None it is calibrated from the
    train slice as the `quantile`-th quantile of log_excursion.
    """
    raw = pl.read_parquet(str(parquet_path))
    bars = _aggregate_to_decision_bars(raw, m_minutes=m_minutes)
    out, _alpha = compute_one_sided_excursion_label(
        bars, alpha=alpha, train_fraction=train_fraction, quantile=quantile,
    )
    return out


def warm_online_quantiles(
    parquet_path: str | Path,
    *,
    m_minutes: int = 20,
    alphas: Iterable[float] = (0.05, 0.10, 0.20),
    train_frac: float = 0.6,
    n_regimes: int = 3,
    label_alpha: Optional[float] = None,
    min_per_regime: int = 50,
) -> dict[float, dict[int, float]]:
    """Compute LAC-score quantiles per (alpha, regime) on a train slice.

    Honest warm-start: we don't have a trained classifier here, so we use
    a uniform p=0.5 baseline. The returned dict drops directly into
    `cfg.model.aci.q_init_by_regime` — the streaming ACI updates from there.
    """
    df = compute_labels(
        parquet_path, m_minutes=m_minutes, alpha=label_alpha,
    )
    n = len(df)
    if n == 0:
        return {float(a): {} for a in alphas}

    cut = max(1, int(n * float(train_frac)))
    train = df.head(cut).filter(pl.col("label").is_not_null())
    if len(train) == 0:
        return {float(a): {} for a in alphas}

    # Bar-range proxy regime: log(high/low), binned into n_regimes by quantile.
    if "high" in train.columns and "low" in train.columns and n_regimes > 1:
        rng = (train["high"] / train["low"]).log().to_numpy()
        edges = np.quantile(rng, np.linspace(0.0, 1.0, n_regimes + 1)[1:-1])
        regime = np.digitize(rng, edges)
    else:
        regime = np.zeros(len(train), dtype=int)

    y_cal = train["label"].to_numpy().astype(int)
    p_cal = np.full(len(train), 0.5, dtype=float)

    out: dict[float, dict[int, float]] = {}
    for a in alphas:
        out[float(a)] = _warm_per_alpha(
            p_cal=p_cal, y_cal=y_cal, regime_cal=regime, alpha=float(a),
            min_per_regime=min_per_regime, fallback=0.5,
        )
    return out


__all__ = [
    "compute_labels", "warm_online_quantiles",
    "lac_score", "finite_sample_quantile",
]
