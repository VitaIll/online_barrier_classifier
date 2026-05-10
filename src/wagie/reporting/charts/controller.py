"""Plotly-first controller charts (tau trajectory, entry-rate overlay, sigma_ve).

The controller block in :class:`~wagie.metrics.MetricsReport.controller` carries
per-bar trajectories for the gating controller (entry-rate EWMA controller
with tail-bound τ_floor, sigma_ve gate, and pause spans). These charts make
the controller state legible at a glance: τ(t) over time, the realised vs
target entry-rate overlay, and the σ_ve distribution against σ_max.

Each function mirrors the convention established in
:mod:`wagie.reporting.charts.trading` — Plotly path on success, matplotlib
PNG fallback on any failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from . import (
    ChartArtifact,
    PLOTLY_PALETTE,
    plotly_layout_defaults,
    plotly_to_div,
    render_with_fallback,
)


def tau_trajectory(
    controller_data: dict,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-controller-tau",
    title: str = "τ trajectory",
) -> ChartArtifact:
    """Line plot of τ(t) with τ_floor reference and pause spans shaded.

    ``controller_data`` keys consumed:
        * ``tau_traj``   — list of τ values per bar
        * ``pause_spans`` — list of (start, end) bar-index pairs
        * ``tau_floor``   — optional float, drawn as a horizontal reference
    """
    out_dir = Path(out_dir)
    tau = list((controller_data or {}).get("tau_traj") or [])
    pause_spans = list((controller_data or {}).get("pause_spans") or [])
    tau_floor: Optional[float] = (controller_data or {}).get("tau_floor")

    def _plotly() -> str:
        if not tau:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        y = np.asarray(tau, dtype=float)
        x = np.arange(1, len(y) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=PLOTLY_PALETTE["primary"], width=1.4),
            name="τ(t)",
            hovertemplate="bar %{x}<br>τ %{y:.4f}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "bar index"}
        layout["yaxis"]["title"] = {"text": "τ"}
        shapes: list[dict] = []
        # Pause spans as light vertical bands.
        for span in pause_spans:
            try:
                s, e = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError):
                continue
            shapes.append({
                "type": "rect", "xref": "x", "yref": "paper",
                "x0": s + 1, "x1": e + 1, "y0": 0, "y1": 1,
                "fillcolor": _rgba(PLOTLY_PALETTE["warning"], 0.12),
                "line": {"width": 0}, "layer": "below",
            })
        # τ_floor reference as horizontal dashed line.
        if isinstance(tau_floor, (int, float)):
            shapes.append({
                "type": "line", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1, "y0": float(tau_floor), "y1": float(tau_floor),
                "line": {"color": PLOTLY_PALETTE["muted"],
                         "dash": "dash", "width": 1.0},
            })
        if shapes:
            layout["shapes"] = shapes
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Optional[Path]:
        if not tau:
            return None
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        from wagie.charts.theme import PALETTE, apply_theme, figsize as _fs

        apply_theme(plt)
        fig, ax = plt.subplots(figsize=_fs("wide"))
        y = np.asarray(tau, dtype=float)
        x = np.arange(1, len(y) + 1)
        # Pause spans drawn first so the τ line sits on top.
        for span in pause_spans:
            try:
                s, e = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError):
                continue
            ax.axvspan(s + 1, e + 1, color=PALETTE["warning"], alpha=0.12)
        ax.plot(x, y, color=PALETTE["primary"], linewidth=1.4, label="τ(t)")
        if isinstance(tau_floor, (int, float)):
            ax.axhline(float(tau_floor), color=PALETTE["muted"],
                       linestyle="--", linewidth=1.0, label="τ_floor")
            ax.legend()
        ax.set_xlabel("bar index")
        ax.set_ylabel("τ")
        ax.set_title(title)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "tau_trajectory.png",
        div_id=div_id,
        caption=title,
    )


def entry_rate_overlay(
    controller_data: dict,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-controller-rate",
    title: str = "Entry-rate EWMA — realised vs target",
) -> ChartArtifact:
    """Overlay r̂_ewma vs r*_ewma over the bar axis."""
    out_dir = Path(out_dir)
    r_hat = list((controller_data or {}).get("r_hat_traj") or [])
    r_star = list((controller_data or {}).get("r_star_traj") or [])

    def _plotly() -> str:
        if not r_hat and not r_star:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        fig = go.Figure()
        if r_star:
            ys = np.asarray(r_star, dtype=float)
            xs = np.arange(1, len(ys) + 1)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=PLOTLY_PALETTE["muted"],
                          dash="dash", width=1.2),
                name="r*_ewma (target)",
                hovertemplate="bar %{x}<br>r* %{y:.4f}<extra></extra>",
            ))
        if r_hat:
            yh = np.asarray(r_hat, dtype=float)
            xh = np.arange(1, len(yh) + 1)
            fig.add_trace(go.Scatter(
                x=xh, y=yh, mode="lines",
                line=dict(color=PLOTLY_PALETTE["primary"], width=1.4),
                name="r̂_ewma (realised)",
                hovertemplate="bar %{x}<br>r̂ %{y:.4f}<extra></extra>",
            ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "bar index"}
        layout["yaxis"]["title"] = {"text": "entry rate (EWMA)"}
        layout["showlegend"] = True
        layout["legend"] = {"x": 0.02, "y": 0.98}
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Optional[Path]:
        if not r_hat and not r_star:
            return None
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        from wagie.charts.theme import PALETTE, apply_theme, figsize as _fs

        apply_theme(plt)
        fig, ax = plt.subplots(figsize=_fs("wide"))
        if r_star:
            ys = np.asarray(r_star, dtype=float)
            ax.plot(np.arange(1, len(ys) + 1), ys,
                    color=PALETTE["muted"], linestyle="--",
                    linewidth=1.2, label="r*_ewma (target)")
        if r_hat:
            yh = np.asarray(r_hat, dtype=float)
            ax.plot(np.arange(1, len(yh) + 1), yh,
                    color=PALETTE["primary"], linewidth=1.4,
                    label="r̂_ewma (realised)")
        ax.set_xlabel("bar index")
        ax.set_ylabel("entry rate (EWMA)")
        ax.set_title(title)
        ax.legend()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "entry_rate_overlay.png",
        div_id=div_id,
        caption=title,
    )


def sigma_ve_distribution(
    controller_data: dict,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-controller-sigma-ve",
    n_bins: int = 50,
    title: str = "σ_ve distribution",
) -> ChartArtifact:
    """Histogram of σ_ve with σ_max as a vertical reference line."""
    out_dir = Path(out_dir)
    sigma_ve = list((controller_data or {}).get("sigma_ve_dist") or [])
    sigma_max: Optional[float] = (controller_data or {}).get("sigma_max")

    def _plotly() -> str:
        if not sigma_ve:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        s = np.asarray(sigma_ve, dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return ""

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=s, nbinsx=int(n_bins),
            marker=dict(
                color=PLOTLY_PALETTE["primary"],
                line=dict(color="white", width=0.5),
            ),
            opacity=0.85, name="σ_ve",
            hovertemplate="bin %{x}<br>count %{y}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "σ_ve"}
        layout["yaxis"]["title"] = {"text": "count"}
        layout["bargap"] = 0.02
        if isinstance(sigma_max, (int, float)):
            layout["shapes"] = [{
                "type": "line", "xref": "x",
                "x0": float(sigma_max), "x1": float(sigma_max),
                "yref": "paper", "y0": 0, "y1": 1,
                "line": {"color": PLOTLY_PALETTE["warning"], "width": 1.4},
            }]
            layout["annotations"] = [{
                "x": float(sigma_max), "y": 1.0, "xref": "x", "yref": "paper",
                "text": f"σ_max={float(sigma_max):.4f}",
                "showarrow": False, "yanchor": "bottom",
                "font": {"color": PLOTLY_PALETTE["warning"], "size": 10},
            }]
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Optional[Path]:
        import numpy as np

        if not sigma_ve:
            return None
        s = np.asarray(sigma_ve, dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return None

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from wagie.charts.theme import PALETTE, apply_theme, figsize as _fs

        apply_theme(plt)
        fig, ax = plt.subplots(figsize=_fs("single"))
        ax.hist(s, bins=int(n_bins), color=PALETTE["primary"],
                alpha=0.85, edgecolor="white", linewidth=0.5)
        if isinstance(sigma_max, (int, float)):
            ax.axvline(float(sigma_max), color=PALETTE["warning"],
                       linewidth=1.4, label=f"σ_max={float(sigma_max):.4f}")
            ax.legend()
        ax.set_xlabel("σ_ve")
        ax.set_ylabel("count")
        ax.set_title(title)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "sigma_ve_distribution.png",
        div_id=div_id,
        caption=title,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` string to an ``rgba(r,g,b,a)`` Plotly color."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


__all__ = [
    "tau_trajectory",
    "entry_rate_overlay",
    "sigma_ve_distribution",
]
