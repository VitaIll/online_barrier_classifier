"""One-sided excursion label generator (D2).

For each decision bar k:
    log_excursion[k] = log(bar[k+1].high / bar[k].close) if same segment else NaN
    label[k]         = int(log_excursion[k] >= alpha)

This is the ONE intentional forward-look in the entire system (the label by
definition refers to the future). Pure polars; mlfinpy would give us the
more general triple-barrier path, but D2 says ONE-SIDED EXCURSION.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import polars as pl


def compute_one_sided_excursion(
    bars: pl.DataFrame,
    *,
    high_col: str = "high",
    close_col: str = "close",
    segment_col: str = "segment_id",
) -> pl.DataFrame:
    """Add `next_high`, `next_segment_id`, `log_excursion` columns to a 20m bars DF.

    Per D6: matches src/utils.py::compute_log_excursion semantics.
    """
    return bars.with_columns(
        next_high=pl.col(high_col).shift(-1),
        next_segment_id=pl.col(segment_col).shift(-1),
    ).with_columns(
        log_excursion=pl.when(pl.col(segment_col) == pl.col("next_segment_id"))
        .then((pl.col("next_high") / pl.col(close_col)).log())
        .otherwise(None),
    )


def calibrate_alpha(
    bars: pl.DataFrame,
    *,
    train_fraction: float = 0.6,
    quantile: float = 0.9,
) -> float:
    """Fit alpha (TP barrier) as the q-th quantile of log_excursion on the train slice.

    Train is the first `train_fraction` of rows, dropping nulls.
    """
    if "log_excursion" not in bars.columns:
        raise ValueError("compute_one_sided_excursion must be called first")
    n_train = int(len(bars) * train_fraction)
    train = bars.slice(0, n_train).filter(pl.col("log_excursion").is_not_null())
    excs = train["log_excursion"].to_numpy()
    if len(excs) == 0:
        raise RuntimeError("No valid excursions on the train split")
    return float(np.quantile(excs, quantile))


def compute_one_sided_excursion_label(
    bars: pl.DataFrame,
    *,
    alpha: Optional[float] = None,
    train_fraction: float = 0.6,
    quantile: float = 0.9,
    high_col: str = "high",
    close_col: str = "close",
    segment_col: str = "segment_id",
) -> tuple[pl.DataFrame, float]:
    """End-to-end: add log_excursion, calibrate alpha (if not given), add label.

    Returns (bars_with_label, alpha).
    """
    bars = compute_one_sided_excursion(
        bars, high_col=high_col, close_col=close_col, segment_col=segment_col
    )
    if alpha is None:
        alpha = calibrate_alpha(bars, train_fraction=train_fraction, quantile=quantile)
    bars = bars.with_columns(
        label=pl.when(pl.col("log_excursion").is_not_null())
        .then((pl.col("log_excursion") >= alpha).cast(pl.Int32))
        .otherwise(None),
    )
    return bars, alpha


__all__ = [
    "compute_one_sided_excursion",
    "calibrate_alpha",
    "compute_one_sided_excursion_label",
]
