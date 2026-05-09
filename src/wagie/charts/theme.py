"""Consistent matplotlib theme across the entire chart battery."""

from __future__ import annotations

from typing import Optional


PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "warning": "#d62728",
    "muted": "#7f7f7f",
    "background": "#ffffff",
    "grid": "#e6e6e6",
}


def apply_theme(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": PALETTE["background"],
        "axes.facecolor": PALETTE["background"],
        "axes.edgecolor": PALETTE["muted"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def figsize(kind: str = "single") -> tuple[float, float]:
    return {
        "single": (6.0, 4.0),
        "wide": (9.0, 4.0),
        "tall": (6.0, 6.0),
        "panel": (10.0, 7.0),
    }.get(kind, (6.0, 4.0))
