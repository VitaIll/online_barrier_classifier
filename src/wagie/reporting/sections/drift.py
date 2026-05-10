"""Drift section emitter — order = 35 (between operational and coverage).

Renders rolling-Brier traces for the offline and online predictions against
the single-shot baseline Brier and a 1.15× warning band — that 15% inflation
is the abort threshold per the production-readiness spec, so any rolling
crossing of the band is a tradeable signal of model decay.

When ``metrics['drift']`` is absent (the calibration trace was not
captured), the section emits an informative empty card.
"""

from __future__ import annotations

import math
from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class DriftSection(SectionEmitter):
    name = "drift"
    title = "Drift"
    order = 35

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}
        drift = m.get("drift")
        if not drift:
            return self.empty_record(
                message="Drift block not present (no calibration trace captured)",
            )

        files: list[str] = []
        uses_plotly = False
        out_dir = ctx.section_figs_dir(self.name)

        # ------------------------------------------------------------------
        # KPI cards.
        # ------------------------------------------------------------------
        baseline = drift.get("baseline_brier")
        threshold = (
            float(baseline) * 1.15
            if isinstance(baseline, (int, float))
            and math.isfinite(float(baseline))
            else None
        )
        offline_series = list(drift.get("brier_offline_rolling") or [])
        online_series = list(drift.get("brier_online_rolling") or [])

        max_offline = self._finite_max(offline_series)
        max_online = self._finite_max(online_series)
        n_crossings_offline = self._count_crossings(offline_series, threshold)
        n_crossings_online = self._count_crossings(online_series, threshold)

        cards = [
            self._kpi_card("baseline Brier",
                           self.fmt_float(baseline, ".4f")),
            self._kpi_card("abort threshold",
                           self.fmt_float(threshold, ".4f")),
            self._kpi_card("max rolling Brier (online)",
                           self.fmt_float(max_online, ".4f")),
            self._kpi_card("max rolling Brier (offline)",
                           self.fmt_float(max_offline, ".4f")),
            self._kpi_card("abort crossings (online)",
                           self.fmt_int(n_crossings_online)),
            self._kpi_card("abort crossings (offline)",
                           self.fmt_int(n_crossings_offline)),
        ]
        kpi_html = f'<div class="kpi-row">{"".join(cards)}</div>'

        # ------------------------------------------------------------------
        # Two stacked rolling-Brier charts.
        # ------------------------------------------------------------------
        chart_html_parts: list[str] = []
        try:
            from wagie.reporting.charts import drift as pdrift  # type: ignore

            specs = (
                ("rolling_brier_offline", "drift-brier-offline",
                 "Rolling Brier — offline"),
                ("rolling_brier_online", "drift-brier-online",
                 "Rolling Brier — online"),
            )
            for fn_name, div_id, title in specs:
                fn = getattr(pdrift, fn_name, None)
                if fn is None:
                    continue
                art = fn(
                    drift,
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
                '<div class="empty"><em>drift charts unavailable</em></div>'
            )

        charts_html = "".join(chart_html_parts)

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Rolling Brier vs the single-shot baseline. The shaded warning '
            'band marks the 15%-inflation abort threshold from the '
            'production-readiness spec — any sustained crossing is a '
            'tradeable drift signal that warrants recalibration.'
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
    def _finite_max(series: list) -> Any:
        """Maximum of finite values in ``series``; None if no finite values."""
        best: float | None = None
        for v in series:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(vf):
                continue
            if best is None or vf > best:
                best = vf
        return best

    @staticmethod
    def _count_crossings(series: list, threshold: Any) -> int:
        """Count the number of UP-crossings of ``threshold`` in ``series``.

        Each crossing is one transition from "<= threshold" (or NaN/missing)
        to "> threshold". Two consecutive over-threshold values count once.
        """
        if threshold is None or not isinstance(threshold, (int, float)):
            return 0
        thr = float(threshold)
        if not math.isfinite(thr):
            return 0
        count = 0
        prev_above = False
        for v in series:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                # treat as below; resets the streak
                prev_above = False
                continue
            if not math.isfinite(vf):
                prev_above = False
                continue
            above = vf > thr
            if above and not prev_above:
                count += 1
            prev_above = above
        return count


__all__ = ["DriftSection"]
