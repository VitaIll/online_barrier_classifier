"""Overview section emitter — top-of-report KPI cards + lead chart pair.

This is the first section the reader sees. It MUST lead with a chart pair
(equity curve + reliability diagram) per repo policy ("lead with charts"),
with KPI cards above. Calibration is the lead metric; ROC/PR are NOT
shown here (operational section handles those, labelled diagnostic).

For BLOCKED runs, the blocked_reasons list is rendered inside the section
as a ``<ul>`` so the reader sees the why before the data below it.
"""

from __future__ import annotations

from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class OverviewEmitter(SectionEmitter):
    name = "overview"
    title = "Overview"
    order = 0

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}
        files: list[str] = []
        uses_plotly = False

        # ------------------------------------------------------------------
        # KPI cards — calibration leads, then trading.
        # ------------------------------------------------------------------
        kpi_html = self._render_kpis(m)

        # ------------------------------------------------------------------
        # Blocked banner inside the section (mirrors the page banner up top
        # but keeps a tight summary in the section flow too).
        # ------------------------------------------------------------------
        blocked_html = ""
        accepted = bool(m.get("accepted", True))
        reasons = m.get("blocked_reasons") or ctx.run_meta.blocked_reasons or []
        if not accepted and reasons:
            items = "".join(
                f"<li>{self.escape(r)}</li>" for r in reasons
            )
            blocked_html = (
                '<div class="banner blocked" role="alert">'
                '<strong>BLOCKED</strong> — accept-gate failed.'
                f'<ul>{items}</ul></div>'
            )

        # ------------------------------------------------------------------
        # Charts: equity curve (left) and reliability diagram (right).
        # We use a 2-column responsive grid via inline style so we don't
        # need to extend the CSS.
        # ------------------------------------------------------------------
        out_dir = ctx.section_figs_dir(self.name)

        eq_html = ""
        rel_html = ""

        # Equity — uses fills from engine_result.ledger.fills.
        pnl_log = self._extract_pnl(ctx)
        try:
            from wagie.reporting.charts import trading as ptrade  # type: ignore

            art = ptrade.equity_curve(
                pnl_log,
                out_dir=out_dir,
                use_plotly=ctx.use_plotly,
                div_id="ov-equity",
                title="Equity (cum log return)",
            )
            if art.kind == "png" and art.png_path is not None:
                art.rel_png = ctx.rel(art.png_path)
                files.append(art.rel_png)
            uses_plotly = uses_plotly or (art.kind == "plotly")
            eq_html = art.to_html()
        except ImportError:
            eq_html = '<div class="empty"><em>trading charts unavailable</em></div>'

        # Reliability — uses streaming p_online_history + label_history.
        y_true, p_pred = self._extract_predictions(ctx)
        try:
            from wagie.reporting.charts import calibration as pcal  # type: ignore

            art = pcal.reliability_diagram(
                y_true,
                p_pred,
                out_dir=out_dir,
                use_plotly=ctx.use_plotly,
                div_id="ov-reliability",
                title="Reliability (calibration leads)",
            )
            if art.kind == "png" and art.png_path is not None:
                art.rel_png = ctx.rel(art.png_path)
                files.append(art.rel_png)
            uses_plotly = uses_plotly or (art.kind == "plotly")
            rel_html = art.to_html()
        except ImportError:
            rel_html = '<div class="empty"><em>calibration charts unavailable</em></div>'

        grid_html = (
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;'
            'margin-top:6px">'
            f'<div>{eq_html}</div><div>{rel_html}</div></div>'
        )

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Calibration is the primary truth (Brier / ECE). Trading metrics '
            'sit alongside it; ranking metrics are diagnostic-only and live '
            'in the operational section.</p>'
        )

        fragment = blocked_html + intro + kpi_html + grid_html

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

    def _render_kpis(self, m: dict[str, Any]) -> str:
        trading = m.get("trading", {}) or {}

        cards: list[str] = []
        # Calibration LEADS.
        cards.append(self._kpi_card(
            "Brier", self.fmt_float(m.get("brier")), m.get("brier_ci"),
        ))
        cards.append(self._kpi_card(
            "ECE", self.fmt_float(m.get("ece")), m.get("ece_ci"),
        ))
        cards.append(self._kpi_card(
            "Sharpe",
            self.fmt_float(trading.get("sharpe"), "+.3f"),
            m.get("sharpe_ci"),
        ))
        cards.append(self._kpi_card(
            "n_trades", self.fmt_int(trading.get("n_trades", 0)),
        ))
        cards.append(self._kpi_card(
            "n_decisions", self.fmt_int(m.get("n_decisions", 0)),
        ))
        accepted = bool(m.get("accepted", True))
        accept_state = "ACCEPTED" if accepted else "BLOCKED"
        klass = "good" if accepted else "bad"
        cards.append(
            f'<div class="kpi {klass}"><div class="k">accept</div>'
            f'<div class="v">{accept_state}</div></div>'
        )
        return f'<div class="kpi-row">{"".join(cards)}</div>'

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

    @staticmethod
    def _extract_pnl(ctx: EmitterContext) -> list[float]:
        eng = ctx.engine_result
        if eng is None:
            return []
        ledger = getattr(eng, "ledger", None)
        if ledger is None:
            return []
        fills = getattr(ledger, "fills", None) or []
        out: list[float] = []
        for f in fills:
            try:
                out.append(float(getattr(f, "pnl_log_net", 0.0)))
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _extract_predictions(ctx: EmitterContext) -> tuple[list[int], list[float]]:
        eng = ctx.engine_result
        if eng is None:
            return [], []
        y = list(getattr(eng, "label_history", []) or [])
        p = list(getattr(eng, "p_online_history", []) or [])
        n = min(len(y), len(p))
        return [int(yy) for yy in y[:n]], [float(pp) for pp in p[:n]]


__all__ = ["OverviewEmitter"]
