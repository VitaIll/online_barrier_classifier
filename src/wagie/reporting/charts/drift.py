"""Plotly-first drift charts (rolling-Brier offline + online).

Backed by :class:`~wagie.metrics.MetricsReport.drift`. Each chart renders
a rolling-window Brier trace with the single-shot baseline as a horizontal
reference and a 1.15× warning band — that 15% inflation is the abort
threshold per the production-readiness spec, so the band makes drift
crossings visually obvious.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

from . import (
    ChartArtifact,
    PLOTLY_PALETTE,
    plotly_layout_defaults,
    plotly_to_div,
    render_with_fallback,
)


def _abort_threshold(baseline: Optional[float]) -> Optional[float]:
    if baseline is None or not isinstance(baseline, (int, float)):
        return None
    if not math.isfinite(float(baseline)):
        return None
    return float(baseline) * 1.15


def rolling_brier_offline(
    drift_data: dict,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-drift-brier-offline",
    title: str = "Rolling Brier — offline",
) -> ChartArtifact:
    """Rolling Brier trace for the offline model."""
    return _rolling_brier_chart(
        drift_data=drift_data,
        series_key="brier_offline_rolling",
        series_label="offline rolling Brier",
        out_dir=out_dir,
        use_plotly=use_plotly,
        div_id=div_id,
        title=title,
        png_name="rolling_brier_offline.png",
    )


def rolling_brier_online(
    drift_data: dict,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-drift-brier-online",
    title: str = "Rolling Brier — online",
) -> ChartArtifact:
    """Rolling Brier trace for the online (calibrated) model."""
    return _rolling_brier_chart(
        drift_data=drift_data,
        series_key="brier_online_rolling",
        series_label="online rolling Brier",
        out_dir=out_dir,
        use_plotly=use_plotly,
        div_id=div_id,
        title=title,
        png_name="rolling_brier_online.png",
    )


# ---------------------------------------------------------------------------
# Shared rolling-Brier implementation
# ---------------------------------------------------------------------------

def _rolling_brier_chart(
    *,
    drift_data: dict,
    series_key: str,
    series_label: str,
    out_dir: Path,
    use_plotly: bool,
    div_id: str,
    title: str,
    png_name: str,
) -> ChartArtifact:
    out_dir = Path(out_dir)
    series = list((drift_data or {}).get(series_key) or [])
    baseline = (drift_data or {}).get("baseline_brier")
    threshold = _abort_threshold(baseline)
    window = int((drift_data or {}).get("rolling_window") or 0)

    def _plotly() -> str:
        if not series:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        y = np.asarray(series, dtype=float)
        # Filter out leading NaNs but preserve x alignment for the line.
        x = np.arange(1, len(y) + 1)
        mask = np.isfinite(y)
        if not mask.any():
            return ""

        fig = go.Figure()
        # Warning band [baseline, 1.15*baseline]: drawn first so the line sits on top.
        if (baseline is not None and threshold is not None
                and isinstance(baseline, (int, float))):
            band_x = list(x) + list(x[::-1])
            band_top = [float(threshold)] * len(x)
            band_bot = [float(baseline)] * len(x)
            band_y = band_top + band_bot[::-1]
            fig.add_trace(go.Scatter(
                x=band_x, y=band_y, fill="toself",
                fillcolor=_rgba(PLOTLY_PALETTE["warning"], 0.10),
                line=dict(width=0), hoverinfo="skip",
                showlegend=False, name="abort band",
            ))
        # Rolling Brier line itself.
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="lines",
            line=dict(color=PLOTLY_PALETTE["primary"], width=1.4),
            name=series_label,
            hovertemplate="bar %{x}<br>Brier %{y:.4f}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {
            "text": f"bar index (window={window})" if window
            else "bar index",
        }
        layout["yaxis"]["title"] = {"text": "Brier"}
        shapes: list[dict] = []
        # Baseline reference (solid grey).
        if baseline is not None and isinstance(baseline, (int, float)):
            shapes.append({
                "type": "line", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1,
                "y0": float(baseline), "y1": float(baseline),
                "line": {"color": PLOTLY_PALETTE["muted"], "width": 1.0},
            })
        # Abort threshold reference (dashed warning).
        if threshold is not None:
            shapes.append({
                "type": "line", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1,
                "y0": float(threshold), "y1": float(threshold),
                "line": {"color": PLOTLY_PALETTE["warning"],
                         "dash": "dash", "width": 1.2},
            })
        if shapes:
            layout["shapes"] = shapes
        annotations: list[dict] = []
        if baseline is not None and isinstance(baseline, (int, float)):
            annotations.append({
                "xref": "paper", "yref": "y", "x": 0.01,
                "y": float(baseline), "xanchor": "left",
                "text": f"baseline={float(baseline):.4f}",
                "showarrow": False, "yanchor": "bottom",
                "font": {"color": PLOTLY_PALETTE["muted"], "size": 10},
            })
        if threshold is not None:
            annotations.append({
                "xref": "paper", "yref": "y", "x": 0.01,
                "y": float(threshold), "xanchor": "left",
                "text": f"abort=1.15·baseline={float(threshold):.4f}",
                "showarrow": False, "yanchor": "bottom",
                "font": {"color": PLOTLY_PALETTE["warning"], "size": 10},
            })
        if annotations:
            layout["annotations"] = annotations
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Optional[Path]:
        import numpy as np

        if not series:
            return None
        y = np.asarray(series, dtype=float)
        mask = np.isfinite(y)
        if not mask.any():
            return None

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from wagie.charts.theme import PALETTE, apply_theme, figsize as _fs

        apply_theme(plt)
        fig, ax = plt.subplots(figsize=_fs("wide"))
        x = np.arange(1, len(y) + 1)
        if (baseline is not None and threshold is not None
                and isinstance(baseline, (int, float))):
            ax.axhspan(float(baseline), float(threshold),
                       color=PALETTE["warning"], alpha=0.10)
            ax.axhline(float(baseline), color=PALETTE["muted"],
                       linewidth=1.0,
                       label=f"baseline={float(baseline):.4f}")
            ax.axhline(float(threshold), color=PALETTE["warning"],
                       linestyle="--", linewidth=1.2,
                       label=f"abort=1.15·baseline={float(threshold):.4f}")
        elif baseline is not None and isinstance(baseline, (int, float)):
            ax.axhline(float(baseline), color=PALETTE["muted"],
                       linewidth=1.0,
                       label=f"baseline={float(baseline):.4f}")
        ax.plot(x[mask], y[mask],
                color=PALETTE["primary"], linewidth=1.4,
                label=series_label)
        if window:
            ax.set_xlabel(f"bar index (window={window})")
        else:
            ax.set_xlabel("bar index")
        ax.set_ylabel("Brier")
        ax.set_title(title)
        ax.legend()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / png_name,
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


__all__ = ["rolling_brier_offline", "rolling_brier_online"]
