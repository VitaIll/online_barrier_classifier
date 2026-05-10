"""Plotly-first cross-validation charts (per-fold Sharpe).

Mirrors the matplotlib path embedded in
``wagie.experiments.protocol.ExperimentProtocol._render_cv_charts``;
that body is replicated here as the matplotlib fallback so this module
can be used outside the ExperimentProtocol context too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import (
    ChartArtifact,
    PLOTLY_PALETTE,
    plotly_layout_defaults,
    plotly_to_div,
    render_with_fallback,
)


def per_fold_sharpe(
    per_fold: list[dict],
    pbo: float,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-cv-sharpe",
    title: Optional[str] = None,
) -> ChartArtifact:
    """Bar chart of per-fold annualized Sharpe with PBO in the title."""
    out_dir = Path(out_dir)
    final_title = title or f"per-fold Sharpe (PBO={float(pbo):.2f})"

    def _plotly() -> str:
        if not per_fold:
            return ""
        import plotly.graph_objects as go

        xs = [r["fold"] for r in per_fold]
        sharpes = [float(r.get("sharpe", 0.0)) for r in per_fold]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=xs, y=sharpes,
            marker=dict(color=PLOTLY_PALETTE["primary"], opacity=0.85),
            name="sharpe",
            hovertemplate="fold %{x}<br>sharpe %{y:.3f}<extra></extra>",
        ))
        layout = plotly_layout_defaults(final_title)
        layout["xaxis"]["title"] = {"text": "fold"}
        layout["yaxis"]["title"] = {"text": "Sharpe (annualized)"}
        layout["shapes"] = [{
            "type": "line", "xref": "paper", "x0": 0, "x1": 1,
            "yref": "y", "y0": 0, "y1": 0,
            "line": {"color": PLOTLY_PALETTE["muted"], "width": 0.6},
        }]
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Path:
        return _render_per_fold_sharpe_mpl(
            per_fold, pbo=pbo, out_path=out_path, title=final_title,
        )

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "per_fold_sharpe.png",
        div_id=div_id,
        caption=final_title,
    )


def _render_per_fold_sharpe_mpl(
    per_fold: list[dict],
    *,
    pbo: float,
    out_path: Path,
    title: str,
) -> Path:
    """Replicates the body of ExperimentProtocol._render_cv_charts."""
    import matplotlib.pyplot as plt

    from wagie.charts.theme import PALETTE, apply_theme, figsize

    apply_theme(plt)
    fig, ax = plt.subplots(figsize=figsize("wide"))
    if per_fold:
        xs = [r["fold"] for r in per_fold]
        sharpes = [float(r.get("sharpe", 0.0)) for r in per_fold]
        ax.bar(xs, sharpes, color=PALETTE["primary"], alpha=0.85)
        ax.axhline(0, color=PALETTE["muted"], linewidth=0.6)
        ax.set_xlabel("fold")
        ax.set_ylabel("Sharpe (annualized)")
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, "no folds", ha="center", va="center")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = ["per_fold_sharpe"]
