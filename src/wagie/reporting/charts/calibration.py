"""Plotly-first calibration charts (reliability + probability histogram).

Mirrors :mod:`wagie.charts.calibration` for the matplotlib fallback. The
reliability binning uses quantile breakpoints to match the matplotlib
implementation EXACTLY so the two paths are visually interchangeable.
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


def reliability_diagram(
    y_true: Sequence[int],
    p_pred: Sequence[float],
    *,
    out_dir: Path,
    n_bins: int = 10,
    use_plotly: bool = True,
    div_id: str = "fig-reliability",
    title: str = "Reliability diagram",
) -> ChartArtifact:
    """Quantile-binned reliability diagram with the y=x reference line."""
    out_dir = Path(out_dir)

    def _plotly() -> str:
        import numpy as np
        import plotly.graph_objects as go

        y = np.asarray(y_true, dtype=int)
        p = np.asarray(p_pred, dtype=float)
        if len(y) == 0:
            return ""

        # Match matplotlib version: quantile bin breakpoints, with the
        # outer edges pushed to +/- inf so all values are captured.
        qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        qs[0] = -np.inf
        qs[-1] = np.inf
        p_means: list[float] = []
        y_means: list[float] = []
        ns: list[int] = []
        for i in range(n_bins):
            m = (p >= qs[i]) & (p < qs[i + 1])
            if m.sum() > 0:
                p_means.append(float(p[m].mean()))
                y_means.append(float(y[m].mean()))
                ns.append(int(m.sum()))

        fig = go.Figure()
        # Diagonal reference (y = x).
        fig.add_trace(go.Scatter(
            x=[0.0, 1.0], y=[0.0, 1.0], mode="lines",
            line=dict(color=PLOTLY_PALETTE["muted"], dash="dash", width=1.0),
            name="perfect", hoverinfo="skip",
        ))
        if p_means:
            # Connector line to mirror the matplotlib alpha=0.4 trail.
            fig.add_trace(go.Scatter(
                x=p_means, y=y_means, mode="lines",
                line=dict(color=PLOTLY_PALETTE["primary"], width=1.0),
                opacity=0.4, hoverinfo="skip", showlegend=False,
            ))
            # Marker sizes mirror sqrt-ish growth from the matplotlib path
            # (s = 25 + 2*n in the matplotlib scatter); Plotly uses pixel
            # diameter, so we approximate visually.
            sizes = [max(8.0, 8.0 + 0.6 * (n ** 0.5) * 2) for n in ns]
            fig.add_trace(go.Scatter(
                x=p_means, y=y_means, mode="markers",
                marker=dict(
                    color=PLOTLY_PALETTE["primary"],
                    size=sizes,
                    line=dict(color="white", width=1.0),
                    opacity=0.75,
                ),
                name="empirical",
                customdata=ns,
                hovertemplate=(
                    "p_pred %{x:.3f}<br>"
                    "p_obs %{y:.3f}<br>"
                    "n %{customdata}<extra></extra>"
                ),
            ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "predicted probability"}
        layout["xaxis"]["range"] = [0.0, 1.0]
        layout["yaxis"]["title"] = {"text": "observed positive rate"}
        layout["yaxis"]["range"] = [0.0, 1.0]
        layout["showlegend"] = True
        layout["legend"] = {"x": 0.02, "y": 0.98}
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Path:
        from wagie.charts.calibration import reliability_diagram as _rd
        return _rd(y_true, p_pred, out_path, n_bins=n_bins, title=title)

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "reliability.png",
        div_id=div_id,
        caption=title,
    )


def probability_histogram(
    p_pred: Sequence[float],
    *,
    out_dir: Path,
    n_bins: int = 50,
    use_plotly: bool = True,
    div_id: str = "fig-phist",
    title: str = "Predicted probability distribution",
) -> ChartArtifact:
    """Histogram of predicted probabilities, x-range [0, 1]."""
    out_dir = Path(out_dir)

    def _plotly() -> str:
        if len(p_pred) == 0:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        p = np.asarray(p_pred, dtype=float)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=p, nbinsx=int(n_bins),
            xbins=dict(start=0.0, end=1.0, size=1.0 / max(int(n_bins), 1)),
            marker=dict(
                color=PLOTLY_PALETTE["primary"],
                line=dict(color="white", width=0.5),
            ),
            opacity=0.85, name="p",
            hovertemplate="bin %{x}<br>count %{y}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "p_predicted"}
        layout["xaxis"]["range"] = [0.0, 1.0]
        layout["yaxis"]["title"] = {"text": "count"}
        layout["bargap"] = 0.02
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Path:
        from wagie.charts.calibration import probability_histogram as _ph
        return _ph(p_pred, out_path, n_bins=n_bins, title=title)

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "probability_histogram.png",
        div_id=div_id,
        caption=title,
    )


__all__ = ["reliability_diagram", "probability_histogram"]
