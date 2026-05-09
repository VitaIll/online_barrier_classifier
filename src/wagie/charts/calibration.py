"""Calibration charts — reliability diagram + Brier breakdown.

Reliability is THE leading chart: bin predicted probability vs observed
outcome rate. A well-calibrated stream tracks the diagonal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .theme import PALETTE, apply_theme, figsize


def reliability_diagram(
    y_true: Sequence[int],
    p_pred: Sequence[float],
    out_path: Path,
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_theme(plt)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)

    fig, ax = plt.subplots(figsize=figsize("single"))
    if len(y) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    p_means, y_means, ns = [], [], []
    for i in range(n_bins):
        m = (p >= qs[i]) & (p < qs[i + 1])
        if m.sum() > 0:
            p_means.append(p[m].mean())
            y_means.append(y[m].mean())
            ns.append(m.sum())

    ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], label="perfect")
    if p_means:
        sizes = [25 + 2 * n for n in ns]
        ax.scatter(p_means, y_means, s=sizes, color=PALETTE["primary"],
                   alpha=0.7, edgecolor="white", label="empirical")
        ax.plot(p_means, y_means, "-", color=PALETTE["primary"], alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def probability_histogram(
    p_pred: Sequence[float],
    out_path: Path,
    *,
    n_bins: int = 50,
    title: str = "Predicted probability distribution",
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_theme(plt)
    p = np.asarray(p_pred, dtype=float)
    fig, ax = plt.subplots(figsize=figsize("single"))
    if len(p) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        ax.hist(p, bins=n_bins, color=PALETTE["primary"], alpha=0.8,
                edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, 1)
    ax.set_xlabel("p_predicted")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = ["reliability_diagram", "probability_histogram"]
