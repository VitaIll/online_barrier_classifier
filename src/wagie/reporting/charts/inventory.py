"""Plotly-first inventory charts (size over time, hold-age distribution).

Backed by :class:`~wagie.metrics.MetricsReport.inventory` which carries
per-bar inventory size and the per-bar maximum hold-age trajectory. The
hold-age histogram can optionally stratify by exit reason when caller-supplied
``exit_reasons`` aligns with ``hold_ages`` 1:1.
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


def inventory_size_over_time(
    inventory_data: dict,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-inventory-size",
    title: str = "Inventory size over time",
) -> ChartArtifact:
    """Step plot of open-position count per bar."""
    out_dir = Path(out_dir)
    sizes = list((inventory_data or {}).get("size_traj") or [])

    def _plotly() -> str:
        if not sizes:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        y = np.asarray(sizes, dtype=float)
        x = np.arange(1, len(y) + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=PLOTLY_PALETTE["primary"],
                      width=1.4, shape="hv"),
            name="inventory size",
            hovertemplate="bar %{x}<br>size %{y}<extra></extra>",
        ))
        layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "bar index"}
        layout["yaxis"]["title"] = {"text": "open positions"}
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Optional[Path]:
        if not sizes:
            return None
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        from wagie.charts.theme import PALETTE, apply_theme, figsize as _fs

        apply_theme(plt)
        fig, ax = plt.subplots(figsize=_fs("wide"))
        y = np.asarray(sizes, dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.step(x, y, where="post",
                color=PALETTE["primary"], linewidth=1.4,
                label="inventory size")
        ax.set_xlabel("bar index")
        ax.set_ylabel("open positions")
        ax.set_title(title)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "inventory_size.png",
        div_id=div_id,
        caption=title,
    )


def hold_age_distribution(
    inventory_data: dict,
    *,
    out_dir: Path,
    use_plotly: bool = True,
    div_id: str = "fig-inventory-hold-age",
    n_bins: int = 30,
    title: str = "Hold-age distribution",
) -> ChartArtifact:
    """Histogram of per-bar max hold age, optionally stratified by exit reason.

    ``inventory_data`` keys consumed:
        * ``hold_age_max_traj`` — list of integers (per-bar max age)
        * ``hold_ages``         — optional list of per-fill hold ages
        * ``exit_reasons``      — optional list aligned 1:1 with ``hold_ages``;
                                  enables stratified bars when present
    """
    out_dir = Path(out_dir)
    per_fill_ages = list((inventory_data or {}).get("hold_ages") or [])
    exit_reasons = list((inventory_data or {}).get("exit_reasons") or [])
    fallback = list((inventory_data or {}).get("hold_age_max_traj") or [])
    # Prefer per-fill ages when available — that's the "real" distribution.
    use_per_fill = bool(per_fill_ages)
    ages = per_fill_ages if use_per_fill else fallback

    def _plotly() -> str:
        if not ages:
            return ""
        import numpy as np
        import plotly.graph_objects as go

        a = np.asarray(ages, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return ""

        fig = go.Figure()
        # Stratified by exit reason when we have aligned reasons.
        stratified = (
            use_per_fill
            and len(exit_reasons) == len(per_fill_ages)
            and len(set(exit_reasons)) > 1
        )
        if stratified:
            palette_cycle = [
                PLOTLY_PALETTE["primary"],
                PLOTLY_PALETTE["secondary"],
                PLOTLY_PALETTE["accent"],
                PLOTLY_PALETTE["warning"],
                PLOTLY_PALETTE["muted"],
            ]
            seen: list[str] = []
            for reason in exit_reasons:
                key = str(reason)
                if key not in seen:
                    seen.append(key)
            for i, reason in enumerate(seen):
                ys = [
                    float(per_fill_ages[k])
                    for k in range(len(per_fill_ages))
                    if str(exit_reasons[k]) == reason
                    and per_fill_ages[k] == per_fill_ages[k]  # NaN-safe
                ]
                if not ys:
                    continue
                fig.add_trace(go.Histogram(
                    x=ys, nbinsx=int(n_bins),
                    marker=dict(
                        color=palette_cycle[i % len(palette_cycle)],
                        line=dict(color="white", width=0.5),
                    ),
                    opacity=0.65, name=reason,
                ))
            layout = plotly_layout_defaults(title)
            layout["barmode"] = "overlay"
            layout["showlegend"] = True
            layout["legend"] = {"x": 0.98, "y": 0.98, "xanchor": "right"}
        else:
            fig.add_trace(go.Histogram(
                x=a, nbinsx=int(n_bins),
                marker=dict(
                    color=PLOTLY_PALETTE["primary"],
                    line=dict(color="white", width=0.5),
                ),
                opacity=0.85, name="hold age",
                hovertemplate="bin %{x}<br>count %{y}<extra></extra>",
            ))
            layout = plotly_layout_defaults(title)
        layout["xaxis"]["title"] = {"text": "hold age (bars)"}
        layout["yaxis"]["title"] = {"text": "count"}
        layout["bargap"] = 0.02
        fig.update_layout(**layout)
        return plotly_to_div(fig, div_id=div_id)

    def _matplotlib(out_path: Path) -> Optional[Path]:
        import numpy as np

        if not ages:
            return None
        a = np.asarray(ages, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from wagie.charts.theme import PALETTE, apply_theme, figsize as _fs

        apply_theme(plt)
        fig, ax = plt.subplots(figsize=_fs("single"))

        stratified = (
            use_per_fill
            and len(exit_reasons) == len(per_fill_ages)
            and len(set(exit_reasons)) > 1
        )
        if stratified:
            palette_cycle = [
                PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
                PALETTE["warning"], PALETTE["muted"],
            ]
            seen: list[str] = []
            for reason in exit_reasons:
                key = str(reason)
                if key not in seen:
                    seen.append(key)
            for i, reason in enumerate(seen):
                ys = [
                    float(per_fill_ages[k])
                    for k in range(len(per_fill_ages))
                    if str(exit_reasons[k]) == reason
                    and per_fill_ages[k] == per_fill_ages[k]
                ]
                if not ys:
                    continue
                ax.hist(ys, bins=int(n_bins),
                        color=palette_cycle[i % len(palette_cycle)],
                        alpha=0.65, edgecolor="white", linewidth=0.5,
                        label=reason)
            ax.legend()
        else:
            ax.hist(a, bins=int(n_bins),
                    color=PALETTE["primary"], alpha=0.85,
                    edgecolor="white", linewidth=0.5)
        ax.set_xlabel("hold age (bars)")
        ax.set_ylabel("count")
        ax.set_title(title)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    return render_with_fallback(
        plotly_fn=_plotly,
        matplotlib_fn=_matplotlib,
        use_plotly=use_plotly,
        png_path=out_dir / "hold_age_distribution.png",
        div_id=div_id,
        caption=title,
    )


__all__ = [
    "inventory_size_over_time",
    "hold_age_distribution",
]
