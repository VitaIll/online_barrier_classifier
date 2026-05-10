"""Trading section emitter — order = 20 (after calibration).

Trading is the bottom-line, but per repo policy calibration leads. This
section therefore comes second. KPI cards lead with Sharpe (with CI),
Sortino, max-DD, profit factor, hit rate, total log return, n_trades,
and DSR (when populated). The TP/SL/timeout breakdown is rendered as a
small table below the cards, then three separate ``<figure>``s with
captions: equity curve, drawdown, PnL distribution.
"""

from __future__ import annotations

from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class TradingEmitter(SectionEmitter):
    name = "trading"
    title = "Trading"
    order = 20

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}
        trading = m.get("trading", {}) or {}
        files: list[str] = []
        uses_plotly = False
        out_dir = ctx.section_figs_dir(self.name)

        # ------------------------------------------------------------------
        # KPI cards.
        # ------------------------------------------------------------------
        cards = [
            self._kpi_card("Sharpe",
                           self.fmt_float(trading.get("sharpe"), "+.3f"),
                           m.get("sharpe_ci")),
            self._kpi_card("Sortino",
                           self.fmt_float(trading.get("sortino"), "+.3f")),
            self._kpi_card("max-DD",
                           self.fmt_float(trading.get("max_drawdown_log"),
                                          "+.4f")),
            self._kpi_card("profit factor",
                           self.fmt_float(trading.get("profit_factor"), ".3f")),
            self._kpi_card("hit rate",
                           self.fmt_float(trading.get("hit_rate"), ".3f")),
            self._kpi_card("total log return",
                           self.fmt_float(trading.get("total_log_return"),
                                          "+.4f")),
            self._kpi_card("n_trades",
                           self.fmt_int(trading.get("n_trades", 0))),
        ]
        dsr = m.get("dsr")
        if dsr is not None:
            cards.append(self._kpi_card("DSR (p)", self.fmt_float(dsr, ".4f")))
        kpi_html = f'<div class="kpi-row">{"".join(cards)}</div>'

        # ------------------------------------------------------------------
        # Outcome breakdown table (TP / SL / timeout).
        # ------------------------------------------------------------------
        outcome_html = self._render_outcome_table(trading)

        # ------------------------------------------------------------------
        # Charts: each as its own <figure> stacked.
        # ------------------------------------------------------------------
        pnl_log = self._extract_pnl(ctx)
        chart_html_parts: list[str] = []
        try:
            from wagie.reporting.charts import trading as ptrade  # type: ignore

            specs = (
                ("equity_curve", "tr-equity",
                 "Cumulative log return — equity curve"),
                ("drawdown_chart", "tr-dd",
                 "Drawdown (log) over the equity path"),
                ("pnl_distribution", "tr-pnl",
                 "Per-trade log-return distribution"),
            )
            for fn_name, div_id, title in specs:
                fn = getattr(ptrade, fn_name, None)
                if fn is None:
                    continue
                art = fn(
                    pnl_log,
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
                '<div class="empty"><em>trading charts unavailable</em></div>'
            )

        if not pnl_log and not chart_html_parts:
            chart_html_parts.append(
                '<div class="empty"><em>no fills — no trading charts</em></div>'
            )

        charts_html = "".join(chart_html_parts)

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Trading metrics are reported with bootstrap CIs where available. '
            'Calibration above is the primary truth; this section is the '
            'bottom-line outcome of decisions taken.'
            '</p>'
        )

        fragment = intro + kpi_html + outcome_html + charts_html

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

    def _render_outcome_table(self, trading: dict) -> str:
        n_tp = self.fmt_int(trading.get("n_tp", 0))
        n_sl = self.fmt_int(trading.get("n_sl", 0))
        n_timeout = self.fmt_int(trading.get("n_timeout", 0))
        return (
            '<table><thead><tr>'
            '<th>outcome</th><th class="num">count</th>'
            '</tr></thead><tbody>'
            f'<tr><td>take-profit</td><td class="num">{n_tp}</td></tr>'
            f'<tr><td>stop-loss</td><td class="num">{n_sl}</td></tr>'
            f'<tr><td>timeout</td><td class="num">{n_timeout}</td></tr>'
            '</tbody></table>'
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


__all__ = ["TradingEmitter"]
