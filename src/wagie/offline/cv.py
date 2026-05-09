"""Cross-validation utilities — leverages skfolio.CombinatorialPurgedCV.

Per D4: embargo = 5 decision periods. We pass `embargo_size` as integer
observations (5 for our 20-min bars).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

try:
    from skfolio.model_selection import CombinatorialPurgedCV  # type: ignore
    SKFOLIO_AVAILABLE = True
except ImportError:
    SKFOLIO_AVAILABLE = False
    CombinatorialPurgedCV = None  # type: ignore


def make_purged_cv(
    n_folds: int = 10,
    n_test_folds: int = 2,
    purged_size: int = 1,    # label horizon = 1 decision bar
    embargo_size: int = 5,   # D4: 5 decision periods
):
    """Construct a skfolio CombinatorialPurgedCV with the project defaults."""
    if not SKFOLIO_AVAILABLE:
        raise ImportError(
            "skfolio is required for purged CV; pip install 'skfolio>=0.5'"
        )
    return CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=purged_size,
        embargo_size=embargo_size,
    )


def chronological_split(n: int, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[slice, slice, slice]:
    """Single-cut chronological split (no embargo) for streaming partitions;
    for HPO use make_purged_cv."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n),
    )


__all__ = ["make_purged_cv", "chronological_split", "SKFOLIO_AVAILABLE"]
