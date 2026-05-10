"""Calibration section emitter — the LEAD section per repo policy.

Order = 10, immediately after Overview. The online layer's value shows
in Brier and the calibration curve, NOT in ROC/PR (which are diagnostic
only and live in the operational section). This section therefore leads
with the reliability diagram, not a wall of numbers.

Per-regime breakdown is rendered when ``metrics['calibration_by_regime']``
is non-empty: a small HTML table plus the grouped Brier-vs-ECE chart in
the third tab of a CSS-only ``.tabset``.
"""

from __future__ import annotations

from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class CalibrationEmitter(SectionEmitter):
    name = "calibration"
    title = "Calibration"
    order = 10

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}
        files: list[str] = []
        uses_plotly = False
        out_dir = ctx.section_figs_dir(self.name)

        # Predictions for charts.
        y_true, p_pred = self._extract_predictions(ctx)

        # ---------------- KPI cards ---------------------------------------
        cards = [
            self._kpi_card("Brier", self.fmt_float(m.get("brier")),
                           m.get("brier_ci")),
            self._kpi_card("ECE", self.fmt_float(m.get("ece")),
                           m.get("ece_ci")),
            self._kpi_card("n_predictions", self.fmt_int(len(y_true))),
        ]
        kpi_html = f'<div class="kpi-row">{"".join(cards)}</div>'

        # ---------------- Reliability tab (lead chart) --------------------
        rel_html = ""
        try:
            from wagie.reporting.charts import calibration as pcal  # type: ignore

            art = pcal.reliability_diagram(
                y_true,
                p_pred,
                out_dir=out_dir,
                use_plotly=ctx.use_plotly,
                div_id="cal-reliability",
                title="Reliability diagram",
            )
            if art.kind == "png" and art.png_path is not None:
                art.rel_png = ctx.rel(art.png_path)
                files.append(art.rel_png)
            uses_plotly = uses_plotly or (art.kind == "plotly")
            rel_html = art.to_html()
        except ImportError:
            rel_html = (
                '<div class="empty"><em>calibration charts unavailable</em></div>'
            )

        # ---------------- Histogram tab -----------------------------------
        hist_html = ""
        try:
            from wagie.reporting.charts import calibration as pcal  # type: ignore

            art = pcal.probability_histogram(
                p_pred,
                out_dir=out_dir,
                use_plotly=ctx.use_plotly,
                div_id="cal-phist",
                title="Predicted probability distribution",
            )
            if art.kind == "png" and art.png_path is not None:
                art.rel_png = ctx.rel(art.png_path)
                files.append(art.rel_png)
            uses_plotly = uses_plotly or (art.kind == "plotly")
            hist_html = art.to_html()
        except ImportError:
            hist_html = (
                '<div class="empty"><em>calibration charts unavailable</em></div>'
            )

        # ---------------- Regime breakdown tab ----------------------------
        per_regime = m.get("calibration_by_regime") or []
        regime_chart_html = ""
        regime_table_html = ""
        if per_regime:
            try:
                from wagie.reporting.charts import calibration_regime as pcr  # type: ignore

                art = pcr.regime_brier_bars(
                    list(per_regime),
                    out_dir=out_dir,
                    use_plotly=ctx.use_plotly,
                    div_id="cal-regbrier",
                    title="Brier per regime",
                )
                if art.kind == "png" and art.png_path is not None:
                    art.rel_png = ctx.rel(art.png_path)
                    files.append(art.rel_png)
                uses_plotly = uses_plotly or (art.kind == "plotly")
                regime_chart_html = art.to_html()
            except ImportError:
                regime_chart_html = (
                    '<div class="empty">'
                    '<em>regime calibration chart unavailable</em></div>'
                )

            regime_table_html = self._render_regime_table(per_regime)

        regime_body = (
            (regime_chart_html + regime_table_html)
            if per_regime
            else '<div class="empty"><em>no per-regime predictions</em></div>'
        )

        # ---------------- Build tabset ------------------------------------
        # First tab open by default (only first <details open>).
        tabset_html = (
            '<div class="tabset">'
            '<details name="cal-tabs" open><summary>Reliability</summary>'
            f'<div class="tab-body">{rel_html}</div></details>'
            '<details name="cal-tabs"><summary>Histogram</summary>'
            f'<div class="tab-body">{hist_html}</div></details>'
            '<details name="cal-tabs"><summary>Regime breakdown</summary>'
            f'<div class="tab-body">{regime_body}</div></details>'
            '</div>'
        )

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Calibration is the primary truth. Brier (proper score) and ECE '
            '(empirical reliability) are reported with bootstrap CIs. '
            'ROC/PR live in the operational section as diagnostics only.'
            '</p>'
        )
        fragment = intro + kpi_html + tabset_html

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
    def _kpi_card(label: str, value: str, ci: Any = None) -> str:
        ci_html = ""
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lo = float(ci[0])
                hi = float(ci[1])
                ci_html = (
                    f'<div class="ci">95% CI [{lo:.4f}, {hi:.4f}]</div>'
                )
            except (TypeError, ValueError):
                ci_html = ""
        return (
            f'<div class="kpi"><div class="k">{label}</div>'
            f'<div class="v">{value}</div>{ci_html}</div>'
        )

    def _render_regime_table(self, per_regime: list[dict]) -> str:
        rows: list[str] = []
        for row in per_regime:
            rid = self.escape(row.get("regime_id", "?"))
            n = self.fmt_int(row.get("n", 0))
            br = self.fmt_float(row.get("brier"))
            ec = self.fmt_float(row.get("ece"))
            rows.append(
                f'<tr><td>{rid}</td><td class="num">{n}</td>'
                f'<td class="num">{br}</td><td class="num">{ec}</td></tr>'
            )
        body = "".join(rows) if rows else (
            '<tr><td colspan="4"><em>no regimes</em></td></tr>'
        )
        return (
            '<table><thead><tr>'
            '<th>regime</th><th class="num">n</th>'
            '<th class="num">Brier</th><th class="num">ECE</th>'
            f'</tr></thead><tbody>{body}</tbody></table>'
        )

    @staticmethod
    def _extract_predictions(ctx: EmitterContext) -> tuple[list[int], list[float]]:
        eng = ctx.engine_result
        if eng is None:
            return [], []
        y = list(getattr(eng, "label_history", []) or [])
        p = list(getattr(eng, "p_online_history", []) or [])
        n = min(len(y), len(p))
        return [int(yy) for yy in y[:n]], [float(pp) for pp in p[:n]]


__all__ = ["CalibrationEmitter"]
