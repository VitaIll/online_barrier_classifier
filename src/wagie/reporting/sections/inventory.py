"""Inventory section emitter — order = 25 (between trading and operational).

Renders the open-position trajectory and the hold-age distribution. When a
ledger of fills is available on the engine result, hold ages are derived
per-fill (entry_ts → exit_ts in bar units) and stratified by exit reason
(TP / SL / TIMEOUT / MANUAL / OPERATOR). When the engine result is absent
(report rebuild), the hold-age histogram falls back to the per-bar
``hold_age_max_traj`` carried in the metrics block.
"""

from __future__ import annotations

from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class InventorySection(SectionEmitter):
    name = "inventory"
    title = "Inventory"
    order = 25

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}
        inventory = m.get("inventory")
        if not inventory:
            return self.empty_record(
                message="Inventory block not present (legacy strategy)",
            )

        files: list[str] = []
        uses_plotly = False
        out_dir = ctx.section_figs_dir(self.name)

        # ------------------------------------------------------------------
        # Enrich the inventory dict with per-fill hold ages + exit reasons
        # when the engine result is in scope. This drives the stratified
        # histogram in :mod:`wagie.reporting.charts.inventory`.
        # ------------------------------------------------------------------
        per_fill_ages, exit_reasons = self._extract_fill_hold_ages(ctx)
        chart_data = dict(inventory)
        if per_fill_ages:
            chart_data["hold_ages"] = per_fill_ages
            chart_data["exit_reasons"] = exit_reasons

        # ------------------------------------------------------------------
        # KPI cards.
        # ------------------------------------------------------------------
        size_traj = list(inventory.get("size_traj") or [])
        hold_age_max_traj = list(inventory.get("hold_age_max_traj") or [])

        max_inv = max(size_traj) if size_traj else None
        mean_inv = (sum(size_traj) / len(size_traj)) if size_traj else None
        # Mean hold age — prefer per-fill ages over per-bar maxima.
        if per_fill_ages:
            mean_hold = sum(per_fill_ages) / len(per_fill_ages)
        elif hold_age_max_traj:
            mean_hold = (
                sum(hold_age_max_traj) / len(hold_age_max_traj)
            )
        else:
            mean_hold = None
        batched_exit = self._count_batched_exits(ctx)

        cards = [
            self._kpi_card("max inventory", self.fmt_int(max_inv) if max_inv is not None else "n/a"),
            self._kpi_card("mean inventory", self.fmt_float(mean_inv, ".2f")),
            self._kpi_card("mean hold age", self.fmt_float(mean_hold, ".2f")),
            self._kpi_card("batched exits", self.fmt_int(batched_exit)),
        ]
        kpi_html = f'<div class="kpi-row">{"".join(cards)}</div>'

        # ------------------------------------------------------------------
        # Charts — two stacked figures.
        # ------------------------------------------------------------------
        chart_html_parts: list[str] = []
        try:
            from wagie.reporting.charts import inventory as pinv  # type: ignore

            specs = (
                ("inventory_size_over_time", "inv-size",
                 "Inventory size over time"),
                ("hold_age_distribution", "inv-hold-age",
                 "Hold-age distribution"),
            )
            for fn_name, div_id, title in specs:
                fn = getattr(pinv, fn_name, None)
                if fn is None:
                    continue
                art = fn(
                    chart_data,
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
                '<div class="empty"><em>inventory charts unavailable</em></div>'
            )

        charts_html = "".join(chart_html_parts)

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Open-position size and hold-age distributions. The hold-age '
            'histogram is stratified by exit reason when per-fill data is '
            'available; "batched exits" counts simultaneous closes — a '
            'signal of correlated risk hitting timeout together.'
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
    def _extract_fill_hold_ages(
        ctx: EmitterContext,
    ) -> tuple[list[float], list[str]]:
        """Per-fill (hold_age_in_ts_units, exit_reason_string).

        Returns two lists in 1:1 alignment. Empty when the engine result
        is absent (typical for ``rebuild`` mode where we operate from the
        rebuild bundle only).
        """
        eng = ctx.engine_result
        if eng is None:
            return [], []
        ledger = getattr(eng, "ledger", None)
        if ledger is None:
            return [], []
        fills = getattr(ledger, "fills", None) or []
        ages: list[float] = []
        reasons: list[str] = []
        for f in fills:
            entry_ts = getattr(f, "entry_ts", None)
            exit_ts = getattr(f, "exit_ts", None)
            if entry_ts is None or exit_ts is None:
                continue
            try:
                age = float(int(exit_ts) - int(entry_ts))
            except (TypeError, ValueError):
                continue
            if age < 0:
                continue
            reason = getattr(f, "reason", None)
            try:
                reason_str = (
                    reason.value if hasattr(reason, "value")
                    else str(reason) if reason is not None else "unknown"
                )
            except (TypeError, AttributeError):
                reason_str = "unknown"
            ages.append(age)
            reasons.append(reason_str)
        return ages, reasons

    @staticmethod
    def _count_batched_exits(ctx: EmitterContext) -> int:
        """Count batched (simultaneous) exits — fills sharing an ``exit_ts``.

        A batched exit is a sign of correlated barrier hits (typical when
        many timeouts trip in the same bar). We sum *extra* fills per
        same-bar bucket so a bucket of 3 contributes 2.
        """
        eng = ctx.engine_result
        if eng is None:
            return 0
        ledger = getattr(eng, "ledger", None)
        if ledger is None:
            return 0
        fills = getattr(ledger, "fills", None) or []
        from collections import Counter
        ts_keys: list[int] = []
        for f in fills:
            exit_ts = getattr(f, "exit_ts", None)
            if exit_ts is None:
                continue
            try:
                ts_keys.append(int(exit_ts))
            except (TypeError, ValueError):
                continue
        c = Counter(ts_keys)
        return int(sum(max(0, n - 1) for n in c.values()))


__all__ = ["InventorySection"]
