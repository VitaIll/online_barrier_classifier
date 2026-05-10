"""Per-regime calibration breakdown — grouped Brier vs ECE bars.

Each input row is ``{"regime_id": int, "n": int, "brier": float, "ece": float}``.
Plotly path renders side-by-side grouped bars; matplotlib fallback is a
small hand-rolled chart in the local ``_render_regime_brier_mpl`` (no
sibling exists in :mod:`wagie.charts`).
"""

from __future__ import annotations

from pathlib import Path

from . import (
    ChartArtifact,
    PLOTLY_PALETTE,
    plotly_layout_defaults,
    plotly_to_div,
    render_with_fallback,
)


def regime_brier_bars(
    per_regime: list[dict],
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-regime-brier",
    title: str = "Brier per regime",
) -> ChartArtifact:
    """Grouped bar chart: Brier and ECE per regime."""
    out_dir = Path(out_dir)

    def _plotly() -> str:
        if not per_regime:
            return ""
        import plotly.graph_objects as go

        regimes = [str(r.get("regime_id", "?")) for r in per_regime]
        brier = [float(r.get("brier", 0.0)) for r in per_regime]
        ece = [float(r.get("ece", 0.0)) for r in per_regime]
        ns = [int(r.get("n", 0)) for r in per_regime]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=regimes, y=brier, name="Brier",
            marker=dict(color=PLOTLY_PALETTE["primary"], opacity=0.85),
            customdata=ns,
            hovertemplate=(
                "regime %{x}<br>Brier %{y:.4f}"
                "<br>n %{customdata}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            x=regimes, y=ece, name="ECE",
            marker=dict(color=PLOTLY_PALETTE["secondary"], opacity=0.85),
            customdata=ns,
            hovertemplate=(
                "regime %{x}<br>ECE %{y:.4f}"
                "<br>n %{customdata}<extra></extra>"
            ),
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "regime_id"}
        layout["yaxis"]["title"] = {"text": "score"}
        layout["barmode"] = "group"
        layout["bargap"] = 0.18
        layout["bargroupgap"] = 0.06
        layout["showlegend"] = True
        layout["legend"] = {"x": 0.02, "y": 0.98}
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Path:
        return _render_regime_brier_mpl(per_regime, out_path=out_path, title=title)

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "regime_brier.png",
        div_id=div_id,
        caption=title,
    )


def _render_regime_brier_mpl(
    per_regime: list[dict],
    *,
    out_path: Path,
    title: str,
) -> Path:
    """Side-by-side matplotlib bars for Brier and ECE per regime."""
    import matplotlib.pyplot as plt
    import numpy as np

    from wagie.charts.theme import PALETTE, apply_theme, figsize

    apply_theme(plt)
    fig, ax = plt.subplots(figsize=figsize("wide"))
    if not per_regime:
        ax.text(0.5, 0.5, "no regimes", ha="center", va="center")
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    regimes = [str(r.get("regime_id", "?")) for r in per_regime]
    brier = np.array([float(r.get("brier", 0.0)) for r in per_regime])
    ece = np.array([float(r.get("ece", 0.0)) for r in per_regime])

    idx = np.arange(len(regimes), dtype=float)
    width = 0.4
    ax.bar(idx - width / 2, brier, width=width, color=PALETTE["primary"],
           alpha=0.85, label="Brier")
    ax.bar(idx + width / 2, ece, width=width, color=PALETTE["secondary"],
           alpha=0.85, label="ECE")
    ax.set_xticks(idx)
    ax.set_xticklabels(regimes)
    ax.set_xlabel("regime_id")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = ["regime_brier_bars"]
