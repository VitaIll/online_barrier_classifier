"""Classification ranking metrics — ROC, PR.

DE-EMPHASIZED: Calibration is the leading metric in this repo (the online
layer's value shows in Brier/ECE, not ranking). These are kept for
diagnostic completeness only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    """ROC AUC via Mann–Whitney U on tied ranks.

    Equivalent to `sklearn.metrics.roc_auc_score` for binary 0/1 labels.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Tied-rank Mann-Whitney U: rank in ascending order; positives with
    # higher predicted scores must accumulate higher ranks.
    order = np.argsort(p, kind="mergesort")  # ascending
    p_sorted = p[order]
    y_sorted = y[order]
    # Average rank for ties
    ranks = np.empty_like(p_sorted, dtype=float)
    i = 0
    n = len(p_sorted)
    while i < n:
        j = i
        while j + 1 < n and p_sorted[j + 1] == p_sorted[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed
        ranks[i:j + 1] = avg
        i = j + 1
    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    """Precision–Recall AUC (average precision).

    Computed as the step-function integral of precision over recall —
    matches `sklearn.metrics.average_precision_score` for binary 0/1.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")  # descending
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    # Average precision: sum of precision at each positive sample,
    # weighted by recall increment from the previous step.
    prev_recall = 0.0
    ap = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            ap += float(precision[i]) * (float(recall[i]) - prev_recall)
            prev_recall = float(recall[i])
    return ap


__all__ = ["roc_auc", "pr_auc"]
