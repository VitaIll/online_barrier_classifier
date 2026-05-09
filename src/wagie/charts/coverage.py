"""Coverage charts — ACI quantile drift over time, per-α coverage gap."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .theme import PALETTE, apply_theme, figsize


def coverage_gap_bars(
    coverage_per_alpha: Sequence[Mapping],
    out_path: Path,
    *,
    title: str = "Empirical vs target coverage by α",
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_theme(plt)
    fig, ax = plt.subplots(figsize=figsize("single"))
    if not coverage_per_alpha:
        ax.text(0.5, 0.5, "no coverage data", ha="center", va="center")
    else:
        alphas = [c["alpha"] for c in coverage_per_alpha]
        emp = [c["empirical"] for c in coverage_per_alpha]
        tgt = [c["target"] for c in coverage_per_alpha]
        x = np.arange(len(alphas))
        w = 0.4
        ax.bar(x - w/2, tgt, width=w, color=PALETTE["muted"], alpha=0.7,
               label="target (1-α)")
        ax.bar(x + w/2, emp, width=w, color=PALETTE["primary"], alpha=0.9,
               label="empirical")
        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a:.2f}" for a in alphas])
        ax.legend()
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def quantile_drift(
    q_history: Sequence[float],
    out_path: Path,
    *,
    alpha: float = 0.10,
    title: str = "ACI quantile drift",
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_theme(plt)
    fig, ax = plt.subplots(figsize=figsize("wide"))
    q = np.asarray(q_history, dtype=float)
    if len(q) == 0:
        ax.text(0.5, 0.5, "no quantile history", ha="center", va="center")
    else:
        ax.plot(np.arange(1, len(q) + 1), q, color=PALETTE["primary"],
                linewidth=1.2, label=f"q (α={alpha:.2f})")
        ax.set_ylim(0, 1)
    ax.set_xlabel("step")
    ax.set_ylabel("q")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = ["coverage_gap_bars", "quantile_drift"]
