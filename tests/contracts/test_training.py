"""wagie.training — pre-engine prep helpers.

Coverage targets:
    - wagie.training.compute_labels
"""

from __future__ import annotations

import polars as pl

from wagie.training import compute_labels


def test_compute_labels_returns_dataframe_with_label_column(synthetic_minute_parquet):
    df = compute_labels(synthetic_minute_parquet)
    assert isinstance(df, pl.DataFrame)
    assert "label" in df.columns


def test_compute_labels_alpha_passthrough(synthetic_minute_parquet):
    """Explicit alpha bypasses the train-quantile calibration step."""
    df = compute_labels(synthetic_minute_parquet, alpha=0.001)
    assert "label" in df.columns
    # All labels are 0 or 1 (or null near segment boundaries).
    labels = df["label"].drop_nulls().to_list()
    assert set(labels).issubset({0, 1})
