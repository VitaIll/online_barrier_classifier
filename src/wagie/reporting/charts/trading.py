"""Plotly-first trading charts (equity, drawdown, PnL distribution).

Each function returns a :class:`ChartArtifact`. The Plotly path produces an
inline ``<div>`` HTML fragment (theme + colors aligned with the matplotlib
sibling in :mod:`wagie.charts.trading`). On any Plotly failure we fall back
to the existing matplotlib PNG implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import (
    ChartArtifact,
    PLOTLY_PALETTE,
    plotly_layout_defaults,
    plotly_to_div,
    render_with_fallback,
)


def equity_curve(
    pnl_log: Sequence[float],
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-equity",
    title: str = "Equity curve (log return)",
) -> ChartArtifact:
    """Cumulative log-return equity curve."""
    out_dir = Path(out_dir)

    def _plotly() -> str:
        if len(pnl_log) == 0:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        p = np.asarray(pnl_log, dtype=float)
        eq = np.cumsum(p)
        x = np.arange(1, len(eq) + 1)

        fig = go.Figure()
        # Negative-region warning fill (drawn below the main line).
        if (eq < 0).any():
            neg = np.where(eq < 0, eq, 0.0)
            fig.add_trace(go.Scatter(
                x=x, y=neg, mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor=_rgba(PLOTLY_PALETTE["warning"], 0.15),
                hoverinfo="skip", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=x, y=eq, mode="lines",
            line=dict(color=PLOTLY_PALETTE["primary"], width=1.6),
            name="cum log return",
            hovertemplate="trade %{x}<br>cum log return %{y:.4f}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "trade index"}
        layout["yaxis"]["title"] = {"text": "cumulative log return"}
        layout["shapes"] = [{
            "type": "line", "xref": "paper", "x0": 0, "x1": 1,
            "yref": "y", "y0": 0, "y1": 0,
            "line": {"color": PLOTLY_PALETTE["muted"], "width": 0.6},
        }]
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Path:
        from wagie.charts.trading import equity_curve as _eq
        return _eq(pnl_log, out_path, title=title)

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "equity_curve.png",
        div_id=div_id,
        caption=title,
    )


def drawdown_chart(
    pnl_log: Sequence[float],
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-drawdown",
    title: str = "Drawdown (log return)",
) -> ChartArtifact:
    """Drawdown over the cumulative log-return path."""
    out_dir = Path(out_dir)

    def _plotly() -> str:
        if len(pnl_log) == 0:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        p = np.asarray(pnl_log, dtype=float)
        eq = np.cumsum(p)
        cummax = np.maximum.accumulate(eq)
        dd = eq - cummax
        x = np.arange(1, len(dd) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=dd, mode="lines",
            line=dict(color=PLOTLY_PALETTE["warning"], width=1.0),
            fill="tozeroy",
            fillcolor=_rgba(PLOTLY_PALETTE["warning"], 0.4),
            name="drawdown",
            hovertemplate="trade %{x}<br>drawdown %{y:.4f}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "trade index"}
        layout["yaxis"]["title"] = {"text": "drawdown (log)"}
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Path:
        from wagie.charts.trading import drawdown_chart as _dd
        return _dd(pnl_log, out_path, title=title)

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "drawdown.png",
        div_id=div_id,
        caption=title,
    )


def pnl_distribution(
    pnl_log: Sequence[float],
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-pnl-dist",
    n_bins: int = 50,
    title: str = "Per-trade PnL distribution (log)",
) -> ChartArtifact:
    """Histogram of per-trade log returns with mean line."""
    out_dir = Path(out_dir)

    def _plotly() -> str:
        if len(pnl_log) == 0:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        p = np.asarray(pnl_log, dtype=float)
        mean = float(np.mean(p))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=p, nbinsx=int(n_bins),
            marker=dict(
                color=PLOTLY_PALETTE["primary"],
                line=dict(color="white", width=0.5),
            ),
            opacity=0.85, name="pnl",
            hovertemplate="bin %{x}<br>count %{y}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "log return per trade"}
        layout["yaxis"]["title"] = {"text": "count"}
        layout["bargap"] = 0.02
        layout["shapes"] = [
            {  # zero line
                "type": "line", "xref": "x", "x0": 0, "x1": 0,
                "yref": "paper", "y0": 0, "y1": 1,
                "line": {"color": PLOTLY_PALETTE["muted"], "width": 0.6},
            },
            {  # mean line
                "type": "line", "xref": "x", "x0": mean, "x1": mean,
                "yref": "paper", "y0": 0, "y1": 1,
                "line": {"color": PLOTLY_PALETTE["accent"], "width": 1.4},
            },
        ]
        layout["annotations"] = [{
            "x": mean, "y": 1.0, "xref": "x", "yref": "paper",
            "text": f"mean={mean:+.4f}",
            "showarrow": False, "yanchor": "bottom",
            "font": {"color": PLOTLY_PALETTE["accent"], "size": 10},
        }]
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Path:
        from wagie.charts.trading import pnl_distribution as _pd
        return _pd(pnl_log, out_path, n_bins=n_bins, title=title)

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "pnl_distribution.png",
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


__all__ = ["equity_curve", "drawdown_chart", "pnl_distribution"]
