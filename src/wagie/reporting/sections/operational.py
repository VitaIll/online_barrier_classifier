"""Operational & Diagnostic section emitter — order = 30.

This is where ROC-AUC and PR-AUC live. Per repo policy ("calibration
first, ranking diagnostic"), they are EXPLICITLY labelled diagnostic
only — calibration is the primary truth. Counters for n_decisions,
n_filled, n_actions_approved, n_actions_rejected sit at the top as
KPI cards. No charts are emitted from this section.
"""

from __future__ import annotations

from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class OperationalEmitter(SectionEmitter):
    name = "operational"
    title = "Operational & Diagnostic"
    order = 30

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}

        cards = [
            self._kpi_card("n_decisions", self.fmt_int(m.get("n_decisions", 0))),
            self._kpi_card("n_filled", self.fmt_int(m.get("n_filled", 0))),
            self._kpi_card("approved",
                           self.fmt_int(m.get("n_actions_approved", 0))),
            self._kpi_card("rejected",
                           self.fmt_int(m.get("n_actions_rejected", 0))),
        ]
        kpi_html = f'<div class="kpi-row">{"".join(cards)}</div>'

        # ------------------------------------------------------------------
        # Diagnostic ranking metrics — DELIBERATELY de-emphasised. Small
        # text, explicit "(diagnostic only ...)" caveat per repo policy.
        # ------------------------------------------------------------------
        roc_str = self._fmt_with_ci(
            self.fmt_float(m.get("roc_auc"), ".4f"),
            m.get("roc_ci_delong"),
        )
        pr_str = self._fmt_with_ci(
            self.fmt_float(m.get("pr_auc"), ".4f"),
            m.get("pr_ci"),
        )
        diag_html = (
            '<p style="font-size:0.86em;color:var(--muted);margin-top:6px">'
            '<strong>Diagnostic ranking metrics</strong> '
            '(diagnostic only — calibration is the primary truth):<br>'
            f'ROC-AUC = <code>{roc_str}</code>'
            f' &middot; PR-AUC = <code>{pr_str}</code>'
            '</p>'
        )

        # State / pipeline-hash readout — useful for replay diagnostics.
        state_hash = m.get("pipeline_state_hash") or ""
        if isinstance(state_hash, (bytes, bytearray)):
            try:
                state_hash = state_hash.hex()
            except (TypeError, ValueError):
                state_hash = ""
        state_hash_str = self.escape(str(state_hash)[:16] if state_hash else "n/a")

        provenance_html = (
            '<p style="font-size:0.86em;color:var(--muted);margin-top:6px">'
            f'pipeline state hash <code>{state_hash_str}</code>'
            '</p>'
        )

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Operational counters and diagnostic ranking metrics. The '
            'calibration section above carries the primary truth — these '
            'numbers are for sanity-checking and pipeline diagnostics.'
            '</p>'
        )
        fragment = intro + kpi_html + diag_html + provenance_html

        return SectionRecord(
            name=self.name,
            title=self.title,
            order=self.order,
            html_fragment=fragment,
            files=[],
            metadata={"uses_plotly": False},
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
    def _fmt_with_ci(value_str: str, ci: Any) -> str:
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lo = float(ci[0])
                hi = float(ci[1])
                return f"{value_str} [{lo:.4f}, {hi:.4f}]"
            except (TypeError, ValueError):
                return value_str
        return value_str


__all__ = ["OperationalEmitter"]
