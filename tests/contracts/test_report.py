"""Smoke tests for the public ``wagie.reporting`` API surface.

The legacy markdown ``Report`` class has been removed. The unified HTML
report is now produced by ``wagie.reporting.ReportRenderer``. Detailed
behavioural tests live in:

  - ``tests/contracts/test_manifest.py``       — manifest schema
  - ``tests/contracts/test_archive.py``        — archive rotation
  - ``tests/contracts/test_renderer.py``       — renderer behaviour
  - ``tests/contracts/test_section_registry.py`` — section emitter registry

This file just pins the public surface so an accidental rename or import
removal trips a clear contract failure.
"""

from __future__ import annotations

import pytest


def test_legacy_markdown_report_is_gone() -> None:
    """The old ``from wagie.reporting import Report`` import must fail.

    Anyone still depending on the markdown class needs to migrate to
    ``ReportRenderer`` — fail loudly so they don't think there's a
    silent fallback.
    """
    with pytest.raises(ImportError):
        from wagie.reporting import Report  # noqa: F401


def test_public_api_exposes_renderer_and_run_meta() -> None:
    """The new public surface: ReportRenderer + RunMeta + the manifest."""
    from wagie.reporting import (  # noqa: F401
        ARCHIVE_DIRNAME,
        DEFAULT_KEEP,
        DEFAULT_REPORT_ROOT,
        REBUILD_BUNDLE_NAME,
        EmitterContext,
        RebuildBundle,
        ReportManifest,
        ReportRenderer,
        RunMeta,
        SectionEmitter,
        SectionRecord,
        all_emitters,
        archive_current,
        get_emitter,
        list_archives,
        prune_archive,
        register,
        register_emitter,
        wipe_payload,
    )


def test_report_renderer_construction_smoke() -> None:
    """Default construction works and exposes the documented attrs."""
    from wagie.reporting import DEFAULT_KEEP, DEFAULT_REPORT_ROOT, ReportRenderer

    rr = ReportRenderer()
    assert rr.report_root == DEFAULT_REPORT_ROOT
    assert rr.archive_keep == DEFAULT_KEEP
    assert rr.enable_archive is True


def test_run_meta_required_fields_present() -> None:
    from wagie.reporting import RunMeta

    rm = RunMeta(run_id="x", spec_name="x", spec_hash="x")
    assert rm.run_id == "x"
    assert rm.mode == "backtest"
    assert rm.accepted is True
    assert rm.use_plotly is True
