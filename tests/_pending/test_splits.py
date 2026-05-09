"""Property-style tests on chronological_split_with_embargo and walk_forward_cv.

We use simple parametric sweeps rather than the full hypothesis library here
to keep round-0 dependencies minimal. Round 1+ may add hypothesis-based
property tests as a separate hypothesis (BACKLOG H-004).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils


@pytest.mark.parametrize("n", [1000, 5000, 25000])
@pytest.mark.parametrize("train_frac,val_frac", [(0.6, 0.2), (0.7, 0.15), (0.8, 0.1)])
@pytest.mark.parametrize("embargo_k", [0, 30, 60, 200])
def test_split_invariants_under_param_sweep(n, train_frac, val_frac, embargo_k):
    # Skip combinations where embargoes consume the val or test split entirely.
    # Two embargoes are subtracted from the post-train tail; require each
    # downstream split to have >= 50 rows of headroom.
    val_size = int(val_frac * n) - embargo_k
    test_size = int((1 - train_frac - val_frac) * n) - embargo_k
    if val_size < 50 or test_size < 50:
        pytest.skip(
            f"infeasible: n={n}, embargo={embargo_k}, val_size={val_size}, test_size={test_size}"
        )

    df = pd.DataFrame({"k": np.arange(n)})
    train, val, test = utils.chronological_split_with_embargo(
        df, train_frac=train_frac, val_frac=val_frac, embargo_k=embargo_k
    )

    # Non-empty
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    # Strictly chronological
    assert train["k"].max() < val["k"].min()
    assert val["k"].max() < test["k"].min()

    # Embargo respected (>= because the function may produce a slightly larger gap
    # depending on how it slices)
    assert val["k"].min() - train["k"].max() >= embargo_k
    assert test["k"].min() - val["k"].max() >= embargo_k

    # No row appears in two splits
    train_set = set(train["k"])
    val_set = set(val["k"])
    test_set = set(test["k"])
    assert not (train_set & val_set)
    assert not (val_set & test_set)
    assert not (train_set & test_set)


@pytest.mark.parametrize("n_folds", [1, 2, 3, 5])
def test_walk_forward_folds_grow_monotonically(n_folds):
    df = pd.DataFrame({"k": np.arange(20000)})
    folds = utils.walk_forward_cv(df, n_folds=n_folds, embargo_k=60)
    assert len(folds) >= 1, "must produce at least one fold"

    prev_train_max = -1
    for train, val in folds:
        assert train["k"].max() > prev_train_max, "train should grow each fold"
        assert train["k"].max() < val["k"].min()
        assert val["k"].min() - train["k"].max() >= 60
        prev_train_max = train["k"].max()
