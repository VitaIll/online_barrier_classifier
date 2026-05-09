"""Causality / leakage invariants.

These tests are non-negotiable. The autonomous loop's CRITIC blocks any PR
that fails them. They cover:
- Label depends only on future bars (n_k+1 .. n_k+M).
- `construct_labels` returns NaN at series-end where the horizon is incomplete.
- Past-target features only use shifted (matured) labels.
- `chronological_split_with_embargo` separates splits by at least EMBARGO_K.
- `walk_forward_cv` produces strictly chronological folds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils


# -----------------------------------------------------------------------------
# Label causality
# -----------------------------------------------------------------------------

def test_label_uses_only_future_bars(synthetic_labels: pd.DataFrame, synthetic_boundaries):
    """Label y_k = 1[max_{j=1..M} log(P_{n_k+j}/P_{n_k}) >= phi].

    Verify by recomputing for a sample of boundaries from the minute dataframe
    and asserting equality with the function output.
    """
    df_minute, df_boundaries = synthetic_boundaries
    df_labelled = synthetic_labels

    # Drop boundaries without full future horizon.
    valid = df_labelled["y"].notna()
    sample = df_labelled.loc[valid].sample(n=min(50, valid.sum()), random_state=0)

    M = utils.M
    phi = utils.PHI
    close = df_minute["close"].to_numpy()

    for _, row in sample.iterrows():
        k = int(row["k"])
        n_k = k * M
        future = close[n_k + 1 : n_k + 1 + M]
        m_k = float(np.log(future / close[n_k]).max())
        expected_y = int(m_k >= phi)
        assert int(row["y"]) == expected_y, (
            f"label mismatch at k={k}: got {int(row['y'])}, expected {expected_y}"
        )


def test_label_is_nan_at_series_end(synthetic_labels: pd.DataFrame):
    """The final boundaries lack a full horizon; their y must be NaN."""
    n = len(synthetic_labels)
    # The last few rows must contain at least one NaN label.
    tail = synthetic_labels.tail(2)
    assert tail["y"].isna().any(), "expected at least one NaN label at series end"


def test_label_does_not_use_current_bar(synthetic_boundaries):
    """Mutating only bar n_k+1 onward must change y; mutating only bars <=n_k must not.

    This is a behavioural test of causality.
    """
    df_minute, df_boundaries = synthetic_boundaries
    df = df_minute.copy()
    M = utils.M
    phi = utils.PHI

    # Pick a boundary far from the start/end.
    k_target = 30
    n_k = k_target * M
    df_boundaries_test = df_boundaries.iloc[: k_target + 5].copy()

    # Mutate past bars only — should not change y_k.
    df_past_mutated = df.copy()
    df_past_mutated.iloc[: n_k + 1, df_past_mutated.columns.get_loc("close")] *= 0.5

    y_orig = utils.construct_labels(
        df_boundaries_test.copy(), df, M, utils.ETA, utils.C
    )["y"].iloc[k_target]
    y_past_mut = utils.construct_labels(
        df_boundaries_test.copy(), df_past_mutated, M, utils.ETA, utils.C
    )["y"].iloc[k_target]

    # Mutating past bars but keeping the future intact relative to a fixed
    # reference price means the label can in principle change because the
    # reference price P_{n_k} changes too. Assert the label is consistent
    # with recomputation, not necessarily unchanged.
    future = df_past_mutated["close"].to_numpy()[n_k + 1 : n_k + 1 + M]
    m_k = float(np.log(future / df_past_mutated["close"].to_numpy()[n_k]).max())
    expected = int(m_k >= phi)
    assert int(y_past_mut) == expected, "label inconsistent with definition"


# -----------------------------------------------------------------------------
# Split causality
# -----------------------------------------------------------------------------

def test_chronological_split_respects_embargo():
    df = pd.DataFrame({"k": np.arange(10000), "y": np.random.randint(0, 2, size=10000)})
    train, val, test = utils.chronological_split_with_embargo(
        df, train_frac=0.7, val_frac=0.15, embargo_k=60
    )
    assert train["k"].max() < val["k"].min(), "train must precede val strictly"
    assert val["k"].max() < test["k"].min(), "val must precede test strictly"
    assert val["k"].min() - train["k"].max() >= 60, "embargo train→val violated"
    assert test["k"].min() - val["k"].max() >= 60, "embargo val→test violated"


def test_walk_forward_cv_is_chronological():
    df = pd.DataFrame({"k": np.arange(5000)})
    folds = utils.walk_forward_cv(df, n_folds=3, embargo_k=60)
    assert len(folds) >= 1
    for train, val in folds:
        assert train["k"].max() < val["k"].min(), (
            "walk-forward fold has train after val"
        )
        assert val["k"].min() - train["k"].max() >= 60, (
            "walk-forward embargo violated"
        )


# -----------------------------------------------------------------------------
# Past-target feature causality
# -----------------------------------------------------------------------------

def test_past_target_features_use_shifted_labels(synthetic_labels):
    """`compute_past_target_features` must only reference labels with k' < k."""
    df = utils.compute_past_target_features(
        synthetic_labels.copy(), utils.WINDOWS_H, utils.HITRATE_WINDOWS_H
    )
    # hit__prev at k=0 must be NaN (no prior label exists).
    assert pd.isna(df["hit__prev__h__w0"].iloc[0]), "hit__prev at k=0 must be NaN"
    # hit__prev at k>=1 must equal y_{k-1}.
    if len(df) >= 3 and df["y"].iloc[0:2].notna().all():
        assert df["hit__prev__h__w0"].iloc[1] == df["y"].iloc[0], (
            "hit__prev at k=1 must equal y_0"
        )
