"""wagie.reporting — the SINGLE unified report.

There is exactly ONE report on disk: ``artifacts/report/index.html``.
Every ``wagie experiment run`` overwrites it (archiving the previous
state to ``artifacts/report/_archive/`` first; only the last 10 are
kept). Side experiments use ``--experiment <name>`` and write to
``artifacts/experiments/<name>/`` without touching the canonical report.

Public API:
    from wagie.reporting import ReportRenderer, RunMeta
    ReportRenderer(report_root=Path("artifacts/report")).render(
        engine_result=..., metrics=..., spec_dict=...,
        run_meta=RunMeta(run_id=..., spec_name=..., spec_hash=...),
        use_plotly=True,
    )

Per-section rebuild (no engine re-run, reads ``state/rebuild_bundle.pkl``)::

    ReportRenderer().rebuild(sections=["calibration", "trading"])

To extend the report, write a new ``SectionEmitter`` subclass under
``wagie.reporting.sections.<name>`` and add it to the auto-load list in
``wagie.reporting.sections.__init__._autoload_builtin_sections``.
"""

from __future__ import annotations

from .archive import (
    ARCHIVE_DIRNAME,
    DEFAULT_KEEP,
    archive_current,
    list_archives,
    prune_archive,
    wipe_payload,
)
from .manifest import ReportManifest, RunMeta, SectionRecord
from .renderer import (
    DEFAULT_REPORT_ROOT,
    REBUILD_BUNDLE_NAME,
    RebuildBundle,
    ReportRenderer,
)
from .sections import (
    EmitterContext,
    SectionEmitter,
    all_emitters,
    get_emitter,
    register,
    register_emitter,
)


__all__ = [
    # Renderer
    "ReportRenderer", "RebuildBundle",
    "DEFAULT_REPORT_ROOT", "REBUILD_BUNDLE_NAME",
    # Manifest
    "ReportManifest", "RunMeta", "SectionRecord",
    # Sections
    "SectionEmitter", "EmitterContext",
    "register", "register_emitter", "all_emitters", "get_emitter",
    # Archive
    "archive_current", "list_archives", "prune_archive", "wipe_payload",
    "ARCHIVE_DIRNAME", "DEFAULT_KEEP",
]
