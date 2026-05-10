"""Spec & Provenance section emitter — order = 90 (last).

Renders the run spec (YAML pre-block in a collapsed ``<details>``) and a
short provenance table covering pipeline state hash, git short SHA,
generation timestamp, mode, and accept-state. Placed last because
provenance is a sanity-check / replay aid, not a headline metric.
"""

from __future__ import annotations

import json
from typing import Any

from ..manifest import SectionRecord
from . import EmitterContext, SectionEmitter, register


@register
class SpecEmitter(SectionEmitter):
    name = "spec"
    title = "Spec & Provenance"
    order = 90

    def emit(self, ctx: EmitterContext) -> SectionRecord:
        spec_dict = ctx.spec_dict or {}
        m = ctx.metrics or {}
        rm = ctx.run_meta

        spec_text = self._dump_yaml(spec_dict)
        spec_block = (
            '<details><summary>show spec</summary>'
            f'<pre>{self.escape(spec_text)}</pre>'
            '</details>'
        )

        # Provenance table.
        state_hash = m.get("pipeline_state_hash") or rm.state_hash or ""
        if isinstance(state_hash, (bytes, bytearray)):
            try:
                state_hash = state_hash.hex()
            except (TypeError, ValueError):
                state_hash = ""
        state_hash = str(state_hash)
        state_hash_short = state_hash[:16] if state_hash else "n/a"

        accepted = bool(m.get("accepted", rm.accepted))
        accept_str = "ACCEPTED" if accepted else "BLOCKED"

        # generated_at_utc — best effort: pull from manifest? The renderer
        # writes it after we run, so the most reliable handle is to stamp
        # "now" here too; the manifest header on the page is the canonical
        # one. We use the run_meta as a proxy and fall back to the report's
        # own timestamp when this runs (it'll align within a few ms).
        from ..manifest import ReportManifest
        generated = ReportManifest.now_utc()

        rows = [
            ("state_hash", state_hash_short),
            ("git_sha", rm.git_sha[:8] if rm.git_sha else "n/a"),
            ("generated_at_utc", generated),
            ("mode", rm.mode or "n/a"),
            ("spec_hash", rm.spec_hash[:12] if rm.spec_hash else "n/a"),
            ("run_id", rm.run_id or "n/a"),
            ("accepted", accept_str),
        ]
        body = "".join(
            f'<tr><td>{self.escape(k)}</td>'
            f'<td><code>{self.escape(v)}</code></td></tr>'
            for k, v in rows
        )
        prov_table = (
            '<table><thead><tr><th>field</th><th>value</th></tr></thead>'
            f'<tbody>{body}</tbody></table>'
        )

        intro = (
            '<p style="margin:0 0 6px 0;color:var(--muted);font-size:0.92em">'
            'Full run spec (collapsed) and provenance pointers — replay '
            'inputs for the deterministic re-run path.'
            '</p>'
        )
        fragment = intro + spec_block + prov_table

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
    def _dump_yaml(obj: Any) -> str:
        """Best-effort YAML dump; fall back to indented JSON when PyYAML
        is unavailable or the object isn't safely dumpable."""
        try:
            import yaml  # type: ignore
            return yaml.safe_dump(
                obj, sort_keys=False, default_flow_style=False,
            )
        except Exception:
            try:
                return json.dumps(obj, indent=2, default=str)
            except (TypeError, ValueError):
                return repr(obj)


__all__ = ["SpecEmitter"]
