"""Tests for the H-101 label + split utilities in src/utils.py.

These pin THIS project's contract (NOT the sibling's):
- 20-min decision bars; label uses NEXT bar's high (single-bar lookahead).
- alpha calibrated as 90th-pct of training-window log-excursions, with the
  notebook's `int(N * train_fraction) - 1` slice (the `-1` matters and is
  pinned by test_calibrate_alpha_matches_notebook_slice).
- Chronological split: train_fraction over N, then val_fraction over the
  train window (NOT over N). NO embargo, NO walk-forward CV.

Round-trip against the persisted `bars_20m_features.parquet` (when present)
is in `test_split_round_trip_matches_persisted_dataset`; the test skips
cleanly if the parquet isn't on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    LOG_EXCURSION_COL,
    NEXT_HIGH_COL,
    NEXT_SEGMENT_COL,
    calibrate_alpha,
    chronological_split,
    chronological_split_indices,
    compute_log_excursion,
    construct_labels,
)


REPO = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def two_segment_bars():
    """7-bar synthetic with a segment break between bars 3 and 4.

    bars: idx 0..6
    segments: [0,0,0,0, 1,1,1]   (gap after bar 3)
    closes:  [100, 101, 100, 102, 103, 104, 103]
    highs:   [102, 103, 101, 105, 106, 107, 104]

    Hand-computed log_excursion (next_high / close):
      idx 0: ln(103/100)
      idx 1: ln(101/101) = 0
      idx 2: ln(105/100)
      idx 3: NaN (next bar in different segment)
      idx 4: ln(107/103)
      idx 5: ln(104/104) = 0
      idx 6: NaN (no next bar)
    """
    return pd.DataFrame(
        {
            "close":      [100.0, 101.0, 100.0, 102.0, 103.0, 104.0, 103.0],
            "high":       [102.0, 103.0, 101.0, 105.0, 106.0, 107.0, 104.0],
            "segment_id": [0,     0,     0,     0,     1,     1,     1    ],
        }
    )


# -----------------------------------------------------------------------------
# compute_log_excursion
# -----------------------------------------------------------------------------

def test_log_excursion_pointwise(two_segment_bars):
    out = compute_log_excursion(two_segment_bars)
    expected = [
        np.log(103.0 / 100.0),
        np.log(101.0 / 101.0),
        np.log(105.0 / 100.0),
        np.nan,
        np.log(107.0 / 103.0),
        np.log(104.0 / 104.0),
        np.nan,
    ]
    actual = out[LOG_EXCURSION_COL].to_numpy()
    np.testing.assert_allclose(actual[:3], expected[:3], rtol=1e-12, atol=0)
    assert np.isnan(actual[3])
    np.testing.assert_allclose(actual[4:6], expected[4:6], rtol=1e-12, atol=0)
    assert np.isnan(actual[6])


def test_log_excursion_does_not_mutate_input(two_segment_bars):
    snapshot = two_segment_bars.copy()
    compute_log_excursion(two_segment_bars)
    pd.testing.assert_frame_equal(two_segment_bars, snapshot)


def test_log_excursion_adds_helper_columns(two_segment_bars):
    out = compute_log_excursion(two_segment_bars)
    assert NEXT_HIGH_COL in out.columns
    assert NEXT_SEGMENT_COL in out.columns
    assert LOG_EXCURSION_COL in out.columns


def test_log_excursion_nan_at_segment_boundary_uses_next_segment_check(two_segment_bars):
    """Bar 3 (last of segment 0) must have NaN excursion despite next bar existing."""
    out = compute_log_excursion(two_segment_bars)
    assert np.isnan(out[LOG_EXCURSION_COL].iloc[3])
    # Sanity: the helper segment column shows the gap.
    assert out[NEXT_SEGMENT_COL].iloc[3] == 1.0
    assert out["segment_id"].iloc[3] == 0


def test_log_excursion_nan_at_series_end(two_segment_bars):
    out = compute_log_excursion(two_segment_bars)
    assert np.isnan(out[LOG_EXCURSION_COL].iloc[-1])


# -----------------------------------------------------------------------------
# calibrate_alpha
# -----------------------------------------------------------------------------

def test_calibrate_alpha_matches_notebook_slice():
    """The `-1` in `iloc[: max(train_end - 1, 0)]` matters: it excludes the
    last train-window bar so its excursion does not look into the test window.
    """
    n = 100
    rng = np.random.default_rng(0)
    bars = pd.DataFrame(
        {
            "close": 100 + rng.normal(size=n).cumsum() * 0.01,
            "high": 100.5 + rng.normal(size=n).cumsum() * 0.01,
            "segment_id": np.zeros(n, dtype=int),
        }
    )
    bars = compute_log_excursion(bars)

    alpha = calibrate_alpha(bars, train_fraction=0.6, quantile=0.9)
    # Expected: quantile of excursions at iloc[: int(100*0.6) - 1] = iloc[:59]
    expected_excursions = bars[LOG_EXCURSION_COL].iloc[:59].dropna()
    expected_alpha = float(expected_excursions.quantile(0.9))
    assert alpha == pytest.approx(expected_alpha, rel=1e-12)


def test_calibrate_alpha_rejects_invalid_train_fraction(two_segment_bars):
    bars = compute_log_excursion(two_segment_bars)
    with pytest.raises(ValueError, match="train_fraction"):
        calibrate_alpha(bars, train_fraction=0.0)
    with pytest.raises(ValueError, match="train_fraction"):
        calibrate_alpha(bars, train_fraction=1.0)


def test_calibrate_alpha_rejects_invalid_quantile(two_segment_bars):
    bars = compute_log_excursion(two_segment_bars)
    with pytest.raises(ValueError, match="quantile"):
        calibrate_alpha(bars, train_fraction=0.6, quantile=-0.1)
    with pytest.raises(ValueError, match="quantile"):
        calibrate_alpha(bars, train_fraction=0.6, quantile=1.5)


def test_calibrate_alpha_raises_when_no_valid_excursions():
    bars = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "segment_id": [0, 1, 2],  # every bar is its own segment → all excursions NaN
        }
    )
    bars = compute_log_excursion(bars)
    with pytest.raises(RuntimeError, match="No valid excursions"):
        calibrate_alpha(bars, train_fraction=0.6)


# -----------------------------------------------------------------------------
# construct_labels
# -----------------------------------------------------------------------------

def test_construct_labels_pointwise(two_segment_bars):
    bars = compute_log_excursion(two_segment_bars)
    alpha = np.log(103.0 / 100.0)  # bar 0 sits exactly on the boundary
    labelled = construct_labels(bars, alpha)

    # Hand-verified: ln(107/103)=0.0381 > alpha=0.0296 → label[4]=1
    expected = [
        1,        # ln(103/100) >= alpha
        0,        # ln(1) = 0 < alpha
        1,        # ln(105/100) > alpha
        np.nan,   # gap (segment 0 → segment 1)
        1,        # ln(107/103) > alpha
        0,        # ln(1) = 0 < alpha
        np.nan,   # series end (no next bar)
    ]
    actual = labelled["label"].to_numpy()
    for i, e in enumerate(expected):
        if pd.isna(e):
            assert pd.isna(actual[i]), f"bar {i}: expected NaN, got {actual[i]}"
        else:
            assert actual[i] == e, f"bar {i}: expected {e}, got {actual[i]}"


def test_construct_labels_does_not_mutate_input(two_segment_bars):
    bars = compute_log_excursion(two_segment_bars)
    snapshot = bars.copy()
    construct_labels(bars, alpha=0.001)
    pd.testing.assert_frame_equal(bars, snapshot)


# -----------------------------------------------------------------------------
# chronological_split_indices
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n, train_frac, val_frac, expected_train_end, expected_val_end",
    [
        # Match notebook canonical config:
        # n=10000, tf=0.6, vf=0.2 → split_idx=6000, val_size=int(6000*0.2)=1200
        # train: [0, 4800), val: [4800, 6000), test: [6000, 10000)
        (10000, 0.6, 0.2, 4800, 6000),
        # Asymmetric example — exposes the two-step int() truncation
        # (not the same as int(n * tf * vf)).
        (10001, 0.6, 0.2, 4800, 6000),  # int(10001*0.6)=6000, int(6000*0.2)=1200
        # Edge: val_fraction=0 → notebook clamps val_size = max(0,1) = 1, so
        # one row still goes to val (this is the notebook's quirk; pinning it).
        (1000, 0.7, 0.0, 699, 700),
        # Edge: tiny dataset → split_idx=3, val_size=int(3*0.5)=1, train=[:2], val=[2:3]
        (10, 0.3, 0.5, 2, 3),
    ],
)
def test_chronological_split_indices_canonical(
    n, train_frac, val_frac, expected_train_end, expected_val_end
):
    train_end, val_end = chronological_split_indices(
        n, train_fraction=train_frac, val_fraction=val_frac
    )
    assert train_end == expected_train_end, f"train_end mismatch for n={n}"
    assert val_end == expected_val_end, f"val_end mismatch for n={n}"


def test_chronological_split_indices_rejects_invalid_n():
    with pytest.raises(ValueError, match="n must be >= 0"):
        chronological_split_indices(-1)


def test_chronological_split_indices_rejects_invalid_train_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        chronological_split_indices(100, train_fraction=0.0)
    with pytest.raises(ValueError, match="train_fraction"):
        chronological_split_indices(100, train_fraction=1.0)


def test_chronological_split_indices_rejects_invalid_val_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        chronological_split_indices(100, val_fraction=-0.1)
    with pytest.raises(ValueError, match="val_fraction"):
        chronological_split_indices(100, val_fraction=1.0)


# -----------------------------------------------------------------------------
# chronological_split
# -----------------------------------------------------------------------------

def test_chronological_split_partition_disjoint_and_ordered():
    n = 5000
    df = pd.DataFrame({"k": np.arange(n), "y": np.zeros(n)})
    train, val, test = chronological_split(df, train_fraction=0.6, val_fraction=0.2)

    # Sizes match the index helper.
    train_end, val_end = chronological_split_indices(n, 0.6, 0.2)
    assert len(train) == train_end
    assert len(val) == val_end - train_end
    assert len(test) == n - val_end

    # Partition: every row appears exactly once.
    assert len(train) + len(val) + len(test) == n
    seen = pd.concat([train["k"], val["k"], test["k"]])
    assert seen.is_unique

    # Strictly chronological (no overlap).
    if len(val):
        assert train["k"].max() < val["k"].min()
        assert val["k"].max() < test["k"].min()
    else:
        assert train["k"].max() < test["k"].min()


def test_chronological_split_returns_independent_copies():
    df = pd.DataFrame({"k": np.arange(100)})
    train, _, _ = chronological_split(df, 0.6, 0.2)
    train["k"] = -1
    assert df["k"].iloc[0] == 0, "input df was mutated by split"


# -----------------------------------------------------------------------------
# Round-trip against persisted dataset
# -----------------------------------------------------------------------------

def _has_persisted_dataset() -> bool:
    return (REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet").exists()


@pytest.mark.skipif(
    not _has_persisted_dataset(),
    reason="bars_20m_features.parquet not on disk (data/ is gitignored)",
)
def test_split_round_trip_matches_persisted_dataset():
    """The new utility's split sizes must match what offline_train.ipynb cell 2
    produces on the actual persisted dataset.
    """
    df = pd.read_parquet(
        REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet",
        columns=["open_time", "label"],
    )
    train, val, test = chronological_split(df, train_fraction=0.6, val_fraction=0.2)

    # Notebook cell 2 (verbatim re-implementation):
    nb_split_idx = int(len(df) * 0.6)
    nb_train_full = df.iloc[:nb_split_idx]
    nb_test = df.iloc[nb_split_idx:]
    nb_val_size = int(len(nb_train_full) * 0.2)
    nb_val_size = max(nb_val_size, 1) if len(nb_train_full) >= 2 else 0
    if 0 < nb_val_size < len(nb_train_full):
        nb_train = nb_train_full.iloc[:-nb_val_size]
        nb_val = nb_train_full.iloc[-nb_val_size:]
    else:
        nb_train = nb_train_full
        nb_val = nb_train_full.iloc[0:0]

    assert len(train) == len(nb_train)
    assert len(val) == len(nb_val)
    assert len(test) == len(nb_test)
    # Boundaries (open_time) match too:
    assert train["open_time"].iloc[-1] == nb_train["open_time"].iloc[-1]
    assert test["open_time"].iloc[0] == nb_test["open_time"].iloc[0]
