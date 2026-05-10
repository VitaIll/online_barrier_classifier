"""Controller section emitter — order = 15 (between calibration and trading).

The gating controller is the chokepoint between p_online and order flow:
the τ trajectory shows how the entry threshold adapts to the realised vs
target entry rate (r̂_ewma vs r*_ewma) under a tail-bound τ_floor, and the
σ_ve histogram against σ_max shows how often the volatility gate trips.

Pulls ``metrics['controller']`` (populated by ``MetricsBattery`` from the
engine's per-bar trajectories). When that block is absent (legacy strategies
without the controller), the section emits an informative empty card.
"""

from __future__ import annotations

from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class ControllerSection(SectionEmitter):
    name = "controller"
    title = "Controller"
    order = 15

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}
        controller = m.get("controller")
        if not controller:
            return self.empty_record(
                message="Controller block not present (legacy strategy)",
            )

        files: list[str] = []
        uses_plotly = False
        out_dir = ctx.section_figs_dir(self.name)

        # ------------------------------------------------------------------
        # KPI cards summarising the controller state.
        # ------------------------------------------------------------------
        tau_traj = list(controller.get("tau_traj") or [])
        r_hat_traj = list(controller.get("r_hat_traj") or [])
        r_star_traj = list(controller.get("r_star_traj") or [])
        pause_spans = list(controller.get("pause_spans") or [])
        sigma_max = controller.get("sigma_max")

        final_tau = tau_traj[-1] if tau_traj else None
        mean_r_hat = (
            sum(r_hat_traj) / len(r_hat_traj) if r_hat_traj else None
        )
        mean_r_star = (
            sum(r_star_traj) / len(r_star_traj) if r_star_traj else None
        )
        total_paused = self._total_paused_bars(pause_spans)

        cards = [
            self._kpi_card("final τ", self.fmt_float(final_tau, ".4f")),
            self._kpi_card("mean r̂_ewma",
                           self.fmt_float(mean_r_hat, ".4f")),
            self._kpi_card("mean r*_ewma",
                           self.fmt_float(mean_r_star, ".4f")),
            self._kpi_card("paused bars", self.fmt_int(total_paused)),
            self._kpi_card("σ_max", self.fmt_float(sigma_max, ".4f")),
        ]
        kpi_html = f'<div class="kpi-row">{"".join(cards)}</div>'

        # ------------------------------------------------------------------
        # Charts — three stacked figures.
        # ------------------------------------------------------------------
        chart_html_parts: list[str] = []
        try:
            from wagie.reporting.charts import controller as pctrl  # type: ignore

            specs = (
                ("tau_trajectory", "ctrl-tau", "τ(t) with τ_floor"),
                ("entry_rate_overlay", "ctrl-rate",
                 "r̂_ewma vs r*_ewma"),
                ("sigma_ve_distribution", "ctrl-sigma",
                 "σ_ve distribution vs σ_max"),
            )
            for fn_name, div_id, title in specs:
                fn = getattr(pctrl, fn_name, None)
                if fn is None:
                    continue
                art = fn(
                    controller,
                    out_dir=out_dir,
                    use_plotly=ctx.use_plotly,
                    div_id=div_id,
                    title=title,
                )
                if art.kind == "png" and art.png_path is not None:
                    art.rel_png = ctx.rel(art.png_path)
                    files.append(art.rel_png)
                uses_plotly = uses_plotly or (art.kind == "plotly")
                chart_html_parts.append(art.to_html())
        except ImportError:
            chart_html_parts.append(
                '<div class="empty"><em>controller charts unavailable</em></div>'
            )

        charts_html = "".join(chart_html_parts)

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'The entry-rate EWMA controller adjusts τ to track the target '
            'entry rate under a tail-bound τ_floor; the σ_ve gate suppresses '
            'entries when realised one-step volatility blows past σ_max. '
            'Pause spans appear shaded on the τ chart.'
            '</p>'
        )

        fragment = intro + kpi_html + charts_html

        return SectionRecord(
            name=self.name,
            title=self.title,
            order=self.order,
            html_fragment=fragment,
            files=files,
            metadata={"uses_plotly": uses_plotly},
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kpi_card(label: str, value: str) -> str:
        return (
            f'<div class="kpi"><div class="k">{label}</div>'
            f'<div class="v">{value}</div></div>'
        )

    @staticmethod
    def _total_paused_bars(pause_spans: list) -> int:
        total = 0
        for span in pause_spans:
            try:
                s, e = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError):
                continue
            # Inclusive span: e - s + 1 bars.
            if e >= s:
                total += (e - s + 1)
        return total


__all__ = ["ControllerSection"]
