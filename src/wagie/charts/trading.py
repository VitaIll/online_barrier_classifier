"""Trading charts — equity curve, drawdown, PnL distribution."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .theme import PALETTE, apply_theme, figsize


def equity_curve(
    pnl_log: Sequence[float],
    out_path: Path,
    *,
    title: str = "Equity curve (log return)",
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_theme(plt)
    fig, ax = plt.subplots(figsize=figsize("wide"))
    p = np.asarray(pnl_log, dtype=float)
    if len(p) == 0:
        ax.text(0.5, 0.5, "no fills", ha="center", va="center")
    else:
        eq = np.cumsum(p)
        ax.plot(np.arange(1, len(eq) + 1), eq,
                color=PALETTE["primary"], linewidth=1.4)
        ax.axhline(0.0, color=PALETTE["muted"], linewidth=0.6)
        if (eq < 0).any():
            ax.fill_between(np.arange(1, len(eq) + 1), eq, 0,
                            where=eq < 0, alpha=0.15, color=PALETTE["warning"])
    ax.set_xlabel("trade index")
    ax.set_ylabel("cumulative log return")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def drawdown_chart(
    pnl_log: Sequence[float],
    out_path: Path,
    *,
    title: str = "Drawdown (log return)",
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_theme(plt)
    fig, ax = plt.subplots(figsize=figsize("wide"))
    p = np.asarray(pnl_log, dtype=float)
    if len(p) == 0:
        ax.text(0.5, 0.5, "no fills", ha="center", va="center")
    else:
        eq = np.cumsum(p)
        cummax = np.maximum.accumulate(eq)
        dd = eq - cummax
        ax.fill_between(np.arange(1, len(dd) + 1), dd, 0,
                        color=PALETTE["warning"], alpha=0.4)
        ax.plot(np.arange(1, len(dd) + 1), dd,
                color=PALETTE["warning"], linewidth=1.0)
    ax.set_xlabel("trade index")
    ax.set_ylabel("drawdown (log)")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def pnl_distribution(
    pnl_log: Sequence[float],
    out_path: Path,
    *,
    n_bins: int = 50,
    title: str = "Per-trade PnL distribution (log)",
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_theme(plt)
    fig, ax = plt.subplots(figsize=figsize("single"))
    p = np.asarray(pnl_log, dtype=float)
    if len(p) == 0:
        ax.text(0.5, 0.5, "no fills", ha="center", va="center")
    else:
        ax.hist(p, bins=n_bins, color=PALETTE["primary"],
                alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(0.0, color=PALETTE["muted"], linewidth=0.6)
        ax.axvline(float(np.mean(p)), color=PALETTE["accent"],
                   linewidth=1.2, label=f"mean={float(np.mean(p)):+.4f}")
        ax.legend()
    ax.set_xlabel("log return per trade")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = ["equity_curve", "drawdown_chart", "pnl_distribution"]
