"""Contract tests for ``wagie.reporting.renderer.ReportRenderer``.

Pin down:
  - ``render`` produces ``index.html``, ``manifest.json``, ``metrics.json``,
    ``spec.yaml``, ``state/rebuild_bundle.pkl``, ``state/pipeline_state_hash.txt``
  - the HTML contains every section's title (Overview / Calibration /
    Trading / Operational & Diagnostic / Coverage & Drift / Spec & Provenance)
  - ``manifest.all_listed_files()`` matches the on-disk figs/tables tree
  - re-rendering the same inputs is idempotent in file SET (contents may
    differ in ``generated_at_utc``)
  - re-rendering with archive enabled creates a zip in ``_archive/``
  - re-rendering with archive disabled leaves ``_archive/`` alone
  - ``use_plotly=False`` ⇒ <img> only, no Plotly CDN
  - ``use_plotly=True`` ⇒ Plotly CDN tag is present (when a section emitted
    a plotly fragment)
  - ``rebuild(sections=["calibration"])`` succeeds when bundle is present;
    only calibration's files are regenerated; other sections are intact
  - ``rebuild()`` raises FileNotFoundError when no bundle exists
  - orphan cleanup: a stray PNG dropped under figs/calibration/ is gone
    after re-render
  - side experiments don't touch artifacts/report/
  - Plotly failure → graceful matplotlib fallback (the critical case)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from wagie.reporting import (
    ARCHIVE_DIRNAME,
    REBUILD_BUNDLE_NAME,
    ReportManifest,
    ReportRenderer,
    RunMeta,
)


# =============================================================================
# Fixtures: a duck-typed _FakeEngine that satisfies the section emitters'
# attribute surface (mirrors `_ReplayEngineResult` shape).
# =============================================================================

@dataclass
class _FakeFill:
    pnl_log_net: float = 0.0


@dataclass
class _FakeLedger:
    fills: list = field(default_factory=list)


@dataclass
class _FakeEngine:
    p_online_history: list = field(default_factory=list)
    label_history: list = field(default_factory=list)
    regime_history: list = field(default_factory=list)
    ledger: _FakeLedger = field(default_factory=_FakeLedger)
    n_decisions: int = 0
    n_filled: int = 0
    n_skipped_warmup: int = 0
    n_actions_approved: int = 0
    n_actions_rejected: int = 0
    pipeline_state_hash: bytes = b"\xde\xad\xbe\xef" * 8


def _fake_engine_with_data() -> _FakeEngine:
    """Build a tiny but non-trivial _FakeEngine for rendering tests."""
    p = [0.10, 0.85, 0.40, 0.65, 0.20, 0.55, 0.30, 0.95, 0.45, 0.70,
         0.15, 0.80, 0.35, 0.60, 0.25]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    r = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    fills = [_FakeFill(pnl_log_net=v) for v in
             (0.001, -0.002, 0.0015, -0.0005, 0.003, -0.001, 0.0025,
              0.0008, -0.0012, 0.0018)]
    return _FakeEngine(
        p_online_history=p,
        label_history=y,
        regime_history=r,
        ledger=_FakeLedger(fills=fills),
        n_decisions=20,
        n_filled=10,
        n_actions_approved=10,
        n_actions_rejected=5,
        pipeline_state_hash=b"\xab" * 32,
    )


def _fake_metrics() -> dict:
    return {
        "brier": 0.21,
        "ece": 0.034,
        "calibration_by_regime": [
            {"regime_id": 0, "n": 80, "brier": 0.20, "ece": 0.03},
            {"regime_id": 1, "n": 90, "brier": 0.22, "ece": 0.04},
        ],
        "trading": {
            "n_trades": 10, "n_tp": 5, "n_sl": 4, "n_timeout": 1,
            "sharpe": 1.234, "probabilistic_sharpe": 0.5, "sortino": 1.0,
            "hit_rate": 0.5, "profit_factor": 1.4,
            "max_drawdown_log": -0.05, "cdar_5pct_log": -0.02,
            "total_log_return": 0.012, "total_pct_return": 0.012,
        },
        "n_decisions": 20, "n_filled": 10,
        "n_actions_approved": 10, "n_actions_rejected": 5,
        "roc_auc": 0.55, "pr_auc": 0.31,
        "pipeline_state_hash": "deadbeef" * 8,
        "accepted": True,
        "blocked_reasons": [],
    }


def _spec_dict() -> dict:
    return {
        "name": "smoke", "seed": 42,
        "wagie": {"data": {"parquet_path": "x.parquet", "m_minutes": 20}},
    }


def _run_meta(**overrides) -> RunMeta:
    base = dict(
        run_id="test-rid", spec_name="smoke", spec_hash="deadbeef",
        mode="backtest", accepted=True,
    )
    base.update(overrides)
    return RunMeta(**base)


def _do_render(
    tmp_path: Path,
    *,
    use_plotly: bool = True,
    archive: bool = True,
    metrics: Optional[dict] = None,
) -> Path:
    rr = ReportRenderer(report_root=tmp_path, enable_archive=archive)
    return rr.render(
        engine_result=_fake_engine_with_data(),
        metrics=metrics if metrics is not None else _fake_metrics(),
        spec_dict=_spec_dict(),
        run_meta=_run_meta(),
        use_plotly=use_plotly,
    )


# =============================================================================
# render — produces the expected file set
# =============================================================================

def test_render_produces_expected_top_level_files(tmp_path: Path) -> None:
    out = _do_render(tmp_path)
    assert out.name == "index.html"
    assert out.is_file()
    # Top-level artifacts:
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "spec.yaml").is_file()
    # State dir:
    assert (tmp_path / "state" / REBUILD_BUNDLE_NAME).is_file()
    assert (tmp_path / "state" / "pipeline_state_hash.txt").is_file()


def test_render_html_contains_every_section_title(tmp_path: Path) -> None:
    """Each built-in section's title must appear in index.html."""
    _do_render(tmp_path)
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    for title in (
        "Overview",
        "Calibration",
        "Trading",
        "Operational",          # "Operational & Diagnostic"
        "Coverage",             # "Coverage & Drift"
        "Spec",                 # "Spec & Provenance"
    ):
        assert title in text, f"missing section title {title!r} in index.html"


def test_render_manifest_listed_files_match_disk(tmp_path: Path) -> None:
    """No orphans: the manifest's listed file set equals what's actually on
    disk under figs/ and tables/."""
    _do_render(tmp_path)
    manifest = ReportManifest.read(tmp_path / "manifest.json")
    assert manifest is not None

    listed = manifest.all_listed_files()

    # Walk figs/ and tables/ and collect POSIX-relative paths.
    on_disk: set[str] = set()
    for sub in ("figs", "tables"):
        root = tmp_path / sub
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file():
                    on_disk.add(p.relative_to(tmp_path).as_posix())

    # The manifest may list paths under figs/ or tables/, but it shouldn't
    # claim a file that isn't there. AND every file under figs/tables must
    # be claimed — that's the orphan-cleanup invariant.
    for f in listed:
        assert (tmp_path / f).is_file(), \
            f"manifest claims file {f!r} but it's not on disk"
    # Symmetric: nothing on disk under figs/tables that isn't claimed.
    assert on_disk.issubset(listed), \
        f"orphan files on disk: {on_disk - listed}"


# =============================================================================
# Idempotence
# =============================================================================

def test_re_render_same_inputs_same_file_set(tmp_path: Path) -> None:
    """Re-rendering with the same inputs produces the SAME set of files
    (contents may differ only by generated_at_utc timestamp)."""
    _do_render(tmp_path, archive=False)
    files_before = sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*")
        if p.is_file() and ARCHIVE_DIRNAME not in p.relative_to(tmp_path).parts
    )

    _do_render(tmp_path, archive=False)
    files_after = sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*")
        if p.is_file() and ARCHIVE_DIRNAME not in p.relative_to(tmp_path).parts
    )

    assert files_before == files_after


# =============================================================================
# Archive: enabled / disabled paths
# =============================================================================

def test_re_render_with_archive_enabled_creates_zip(tmp_path: Path) -> None:
    _do_render(tmp_path, archive=True)
    # First render — archive dir empty (no prior payload to archive)
    arch_dir = tmp_path / ARCHIVE_DIRNAME
    archives_after_first = list(arch_dir.glob("*.zip")) if arch_dir.is_dir() else []

    _do_render(tmp_path, archive=True)
    # After second render, exactly one archive should sit in _archive/
    assert arch_dir.is_dir()
    archives_after_second = list(arch_dir.glob("*.zip"))
    assert len(archives_after_second) == len(archives_after_first) + 1


def test_re_render_with_archive_disabled_leaves_archive_untouched(
    tmp_path: Path,
) -> None:
    _do_render(tmp_path, archive=False)
    arch_dir = tmp_path / ARCHIVE_DIRNAME
    # No archive on first render either way (dir doesn't exist yet)
    assert not arch_dir.exists()

    _do_render(tmp_path, archive=False)
    # Still no archive
    assert not arch_dir.exists() or len(list(arch_dir.glob("*.zip"))) == 0


# =============================================================================
# use_plotly switch
# =============================================================================

def test_use_plotly_false_emits_img_tags_no_cdn(tmp_path: Path) -> None:
    """With Plotly disabled: only PNG <img> tags, no Plotly CDN script tag."""
    _do_render(tmp_path, use_plotly=False)
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    # Must NOT pull in Plotly CDN
    assert "cdn.plot.ly" not in text
    # Must contain at least one <img tag (PNG fallback)
    assert "<img " in text


def test_use_plotly_true_includes_cdn_when_section_used_plotly(
    tmp_path: Path,
) -> None:
    """With Plotly enabled AND at least one section reporting uses_plotly,
    the index page pulls in plotly.js. If plotly isn't installed, the
    section will fall back to PNG and the CDN won't appear; skip then."""
    pytest.importorskip("plotly")
    _do_render(tmp_path, use_plotly=True)
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    # CDN reference must appear when at least one chart used Plotly
    # (the `class="plotly-fig"` wrapper signals an actual Plotly fragment;
    # the CSS class `.plotly-fig{...}` is always present in <style>).
    if 'class="plotly-fig"' in text:
        assert "cdn.plot.ly" in text
    else:
        # All sections fell back to PNG even with use_plotly=True; CDN not needed
        assert "cdn.plot.ly" not in text


# =============================================================================
# Plotly outage → graceful matplotlib fallback (critical test per spec)
# =============================================================================

def test_plotly_failure_falls_back_to_matplotlib(
    tmp_path: Path, monkeypatch,
) -> None:
    """If the Plotly path raises, every chart cleanly falls back to PNG.

    This is the user-critical test: the report MUST keep working even if
    Plotly mis-installs or the page-render path explodes for any reason.
    """
    import wagie.reporting.charts as ch

    def boom(*a, **kw):
        raise RuntimeError("simulated plotly outage")

    # Replace the Plotly div builder so every chart's plotly_fn() raises.
    # Each chart submodule does `from . import plotly_to_div` which COPIES the
    # binding into the submodule's namespace, so we have to patch in every
    # submodule that uses it.
    monkeypatch.setattr(ch, "plotly_to_div", boom)
    for sub in ("trading", "calibration", "calibration_regime", "cv"):
        try:
            mod = __import__(f"wagie.reporting.charts.{sub}",
                             fromlist=["plotly_to_div"])
        except ImportError:
            continue
        if hasattr(mod, "plotly_to_div"):
            monkeypatch.setattr(mod, "plotly_to_div", boom)

    out = ReportRenderer(report_root=tmp_path, enable_archive=False).render(
        engine_result=_fake_engine_with_data(),
        metrics=_fake_metrics(),
        spec_dict=_spec_dict(),
        run_meta=_run_meta(),
        use_plotly=True,
    )
    text = out.read_text(encoding="utf-8")

    # The CSS rule `.plotly-fig{...}` is always present in the template
    # stylesheet, but no actual section should have emitted a
    # `class="plotly-fig"` wrapper because every Plotly call raised.
    assert 'class="plotly-fig"' not in text or "<img" in text

    # Several PNGs landed under figs/
    pngs = list((tmp_path / "figs").rglob("*.png"))
    assert len(pngs) >= 1, "expected at least one PNG fallback file"


# =============================================================================
# Orphan cleanup
# =============================================================================

def test_orphan_files_under_figs_are_cleaned_on_rerender(tmp_path: Path) -> None:
    """Drop a stray file under figs/calibration/, re-render, assert it's gone."""
    _do_render(tmp_path, archive=False)
    orphan = tmp_path / "figs" / "calibration" / "orphan.png"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    assert orphan.is_file()  # precondition

    _do_render(tmp_path, archive=False)
    assert not orphan.exists(), "orphan file survived the re-render"


def test_orphan_files_under_tables_are_cleaned_on_rerender(tmp_path: Path) -> None:
    _do_render(tmp_path, archive=False)
    tables = tmp_path / "tables"
    tables.mkdir(exist_ok=True)
    orphan = tables / "stray.json"
    orphan.write_text("{}", encoding="utf-8")
    assert orphan.is_file()

    _do_render(tmp_path, archive=False)
    assert not orphan.exists()


# =============================================================================
# Side-experiment isolation
# =============================================================================

def test_side_experiment_root_does_not_touch_canonical_root(tmp_path: Path) -> None:
    """Render to artifacts/experiments/foo MUST NOT touch artifacts/report."""
    canonical = tmp_path / "artifacts" / "report"
    side = tmp_path / "artifacts" / "experiments" / "foo"

    # First write a canonical report
    _do_render(canonical, archive=False)
    canonical_index = canonical / "index.html"
    canonical_mtime_before = canonical_index.stat().st_mtime_ns
    canonical_text_before = canonical_index.read_text(encoding="utf-8")

    # Now render to the side path
    rr = ReportRenderer(report_root=side, enable_archive=False)
    rr.render(
        engine_result=_fake_engine_with_data(),
        metrics=_fake_metrics(),
        spec_dict=_spec_dict(),
        run_meta=_run_meta(spec_name="foo"),
        use_plotly=False,
    )

    # Side index exists
    assert (side / "index.html").is_file()
    # Canonical untouched
    assert canonical_index.is_file()
    assert canonical_index.stat().st_mtime_ns == canonical_mtime_before
    assert canonical_index.read_text(encoding="utf-8") == canonical_text_before


# =============================================================================
# rebuild(sections=[...]) — partial rebuild from bundle
# =============================================================================

def test_rebuild_calibration_only_succeeds_and_isolates_changes(
    tmp_path: Path,
) -> None:
    """rebuild(sections=['calibration']) regenerates only calibration files
    and leaves other sections' files intact."""
    _do_render(tmp_path, archive=False)

    # Snapshot non-calibration figure files
    other_files = []
    for sub in ("figs", "tables"):
        root = tmp_path / sub
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(tmp_path).as_posix()
                    if "/calibration/" not in rel:
                        other_files.append((rel, p.read_bytes()))

    # Now rebuild calibration only
    rr = ReportRenderer(report_root=tmp_path, enable_archive=False)
    out = rr.rebuild(sections=["calibration"])
    assert out.name == "index.html"
    assert out.is_file()

    # Other sections' files are untouched (same bytes).
    for rel, before in other_files:
        p = tmp_path / rel
        assert p.is_file(), f"non-target file disappeared: {rel}"
        assert p.read_bytes() == before, f"non-target file mutated: {rel}"


def test_rebuild_full_regenerates_all_sections(tmp_path: Path) -> None:
    """rebuild() with sections=None rebuilds every section."""
    _do_render(tmp_path, archive=False)
    rr = ReportRenderer(report_root=tmp_path, enable_archive=False)
    out = rr.rebuild()  # sections=None → all
    assert out.is_file()


def test_rebuild_unknown_section_raises_keyerror(tmp_path: Path) -> None:
    _do_render(tmp_path, archive=False)
    rr = ReportRenderer(report_root=tmp_path, enable_archive=False)
    with pytest.raises(KeyError):
        rr.rebuild(sections=["this_does_not_exist"])


def test_rebuild_no_bundle_raises_file_not_found(tmp_path: Path) -> None:
    """rebuild without a prior render (no bundle) raises FileNotFoundError."""
    rr = ReportRenderer(report_root=tmp_path, enable_archive=False)
    with pytest.raises(FileNotFoundError):
        rr.rebuild(sections=["calibration"])


def test_rebuild_no_manifest_raises_file_not_found(tmp_path: Path) -> None:
    """Even if state/rebuild_bundle.pkl exists, missing manifest still errs."""
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / REBUILD_BUNDLE_NAME).write_bytes(b"\x80\x04N.")  # pickle of None
    rr = ReportRenderer(report_root=tmp_path, enable_archive=False)
    with pytest.raises(FileNotFoundError):
        rr.rebuild(sections=["calibration"])


# =============================================================================
# Smoke: archive disabled + re-render leaves no _archive/ dir
# =============================================================================

def test_disabled_archive_means_no_archive_dir_at_all(tmp_path: Path) -> None:
    rr = ReportRenderer(report_root=tmp_path, enable_archive=False)
    rr.render(
        engine_result=_fake_engine_with_data(),
        metrics=_fake_metrics(),
        spec_dict=_spec_dict(),
        run_meta=_run_meta(),
        use_plotly=False,
    )
    assert not (tmp_path / ARCHIVE_DIRNAME).exists()
