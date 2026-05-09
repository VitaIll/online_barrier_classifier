"""Plot helpers for visual-first round reporting.

Ports four helpers from sibling barrier_classifier/src/utils.py into a
separate `plotting` module (cleaner than the sibling's all-in-utils
arrangement). Each function:

- accepts an optional ``ax``/``fig`` so callers can compose multi-panel
  figures
- returns the axes/figure (chainable)
- imports matplotlib lazily (module-level import would force matplotlib
  on every src.utils consumer, which we don't want)

Two intentional deviations from sibling:

1. ``plot_feature_importance`` takes ``importances`` as a numpy array,
   NOT a model object. The sibling's ``model``-typed signature couples
   plotting to a single estimator class; passing the array works for
   ``CatBoostEnsemble``, single CatBoost, sklearn estimators with
   ``feature_importances_``, and anything else.
2. ``plot_calibration_by_regime`` accepts ``n_regimes`` and explicit
   ``labels`` (matching ``src.utils.calibration_by_regime``), not just
   the hard-coded 3-regime split.

Sample-weight plotters (``plot_weight_profiles``, ``plot_weight_distributions``
in sibling) are NOT ported here: this project has no sample weights until
H-102 lands. They'll be added in that round so they can be tested against
real weight data, not stub inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import (
    DEFAULT_REGIME_LABELS,
    expected_calibration_error,
)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    n_bins: int = 10,
    ax: Any = None,
    label: str = "Model",
    color: str | None = None,
    plot_perfect: bool = True,
    show_ece: bool = True,
):
    """Reliability diagram with optional ECE annotation.

    Bin centers (mean predicted within bin) on x; bin accuracy (mean y_true
    within bin) on y. Empty bins are skipped. The diagonal y=x marks
    perfect calibration.
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
        else:
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        if mask.sum() == 0:
            continue
        xs.append(float(y_pred_proba[mask].mean()))
        ys.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    if plot_perfect:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, label="Perfect")
    if xs:
        ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=1.5)
    ece = expected_calibration_error(y_true, y_pred_proba, n_bins=n_bins)
    if show_ece:
        ax.set_title(f"Calibration curve (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_calibration_by_regime(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    regime_signal: np.ndarray,
    *,
    n_bins: int = 10,
    n_regimes: int = 3,
    labels: tuple[str, ...] | None = None,
    fig: Any = None,
):
    """Side-by-side reliability diagrams, one per regime quantile bucket.

    ``regime_signal`` is bucketed via ``pd.qcut`` (matches
    ``src.utils.calibration_by_regime``). Default labels are
    ``("low","med","high")`` for ``n_regimes=3``.

    Returns the figure containing one Axes per regime.
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    regime_signal = np.asarray(regime_signal)

    if labels is None:
        labels = DEFAULT_REGIME_LABELS if n_regimes == 3 else tuple(
            f"q{i}" for i in range(n_regimes)
        )
    if len(labels) != n_regimes:
        raise ValueError(f"labels length {len(labels)} != n_regimes {n_regimes}")

    buckets = pd.qcut(regime_signal, n_regimes, labels=list(labels))
    if fig is None:
        fig, axes = plt.subplots(
            1, n_regimes, figsize=(5 * n_regimes, 4.5), sharex=True, sharey=True,
        )
    else:
        axes = fig.subplots(1, n_regimes, sharex=True, sharey=True)
    if n_regimes == 1:
        axes = [axes]

    for ax, regime in zip(axes, labels):
        mask = np.asarray(buckets == regime)
        if mask.sum() == 0:
            ax.set_title(f"{regime} (n=0)")
            ax.axis("off")
            continue
        plot_calibration_curve(
            y_true[mask], y_pred_proba[mask], n_bins=n_bins, ax=ax,
            label=f"{regime}",
        )
        ax.set_title(
            f"{regime} (n={int(mask.sum()):,}; "
            f"base={float(y_true[mask].mean()):.3f})"
        )
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str] | tuple[str, ...] | np.ndarray,
    *,
    top_n: int = 30,
    ax: Any = None,
    title: str | None = None,
):
    """Horizontal barh of the top-N most important features.

    ``importances`` must be a 1-D array aligned with ``feature_names``.
    """
    import matplotlib.pyplot as plt

    importances = np.asarray(importances, dtype=float).ravel()
    feature_names = list(feature_names)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"importances ({len(importances)}) and feature_names "
            f"({len(feature_names)}) must align"
        )

    top_n = min(top_n, len(importances))
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.28 * top_n + 1.5)))

    order = np.argsort(importances)[::-1][:top_n]
    feats = [feature_names[i] for i in order]
    vals = importances[order]

    ax.barh(list(reversed(feats)), list(reversed(vals)), color="#3b7dd8", edgecolor="black")
    ax.set_title(title or f"Top {top_n} feature importances")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")
    return ax


def plot_threshold_curves(
    threshold_df: pd.DataFrame,
    *,
    ax: Any = None,
    title: str | None = None,
):
    """Plot trade_rate / precision / recall versus threshold.

    ``threshold_df`` is the output of ``src.utils.threshold_analysis``.
    """
    import matplotlib.pyplot as plt

    required = {"threshold", "trade_rate", "precision", "recall"}
    missing = required - set(threshold_df.columns)
    if missing:
        raise ValueError(f"threshold_df missing columns: {missing}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.plot(threshold_df["threshold"], threshold_df["trade_rate"],
            label="Trade rate", color="#3b7dd8", linewidth=1.5)
    ax.plot(threshold_df["threshold"], threshold_df["precision"],
            label="Precision (hit rate)", color="#56b870", linewidth=1.5)
    ax.plot(threshold_df["threshold"], threshold_df["recall"],
            label="Recall", color="#e6a23c", linewidth=1.5)
    ax.set_title(title or "Threshold sweep")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax
