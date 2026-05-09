"""Classification ranking metrics — ROC, PR.

DE-EMPHASIZED: Calibration is the leading metric in this repo (the online
layer's value shows in Brier/ECE, not ranking). These are kept for
diagnostic completeness only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(-p)
    y_sorted = y[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_sorted) + 1, dtype=float)
    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return 0.0
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.trapz(precision, recall))


__all__ = ["roc_auc", "pr_auc"]
