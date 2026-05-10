"""Coverage & Drift section emitter — order = 40.

Placeholder section for the conformal coverage / ADWIN drift wiring that
lands in a later round. Today this section either:

  - renders ``metrics['conformal']`` and/or ``metrics['drift']`` as a
    small ``<pre>`` JSON dump when present (for early-bird visibility); or
  - displays an explanatory empty card when neither is populated.

Per the spec for this section, the placeholder is intentional and is
documented as such in the rendered HTML so a reader doesn't think the
report is broken.
"""

from __future__ import annotations

import json
from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class CoverageEmitter(SectionEmitter):
    name = "coverage"
    title = "Coverage & Drift"
    order = 40

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        m = ctx.metrics or {}
        conformal = m.get("conformal")
        drift = m.get("drift")

        parts: list[str] = []
        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Conformal coverage layer + ADWIN drift markers. The online ARF '
            'acts as the streaming conformal layer over the offline model, '
            'providing conditional coverage and tail-failure capture.'
            '</p>'
        )
        parts.append(intro)

        any_data = False

        if conformal:
            any_data = True
            parts.append('<h3 style="margin-top:14px">Conformal coverage</h3>')
            parts.append(self._json_pre(conformal))

        if drift:
            any_data = True
            parts.append('<h3 style="margin-top:14px">Drift markers</h3>')
            parts.append(self._json_pre(drift))

        if not any_data:
            parts.append(
                '<div class="empty">'
                '<p><em>Conformal coverage and ADWIN drift markers are wired '
                'in a later round; this section is the placeholder.</em></p>'
                '</div>'
            )

        fragment = "".join(parts)

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

    def _json_pre(self, obj: Any) -> str:
        try:
            text = json.dumps(obj, indent=2, default=str)
        except (TypeError, ValueError):
            text = repr(obj)
        return f"<pre>{self.escape(text)}</pre>"


__all__ = ["CoverageEmitter"]
