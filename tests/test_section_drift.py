"""Contract tests for ``DriftSection`` + the drift charts."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from wagie.reporting.charts import ChartArtifact
from wagie.reporting.charts import drift as pdrift
from wagie.reporting.manifest import RunMeta
from wagie.reporting.sections import (
    EmitterContext,
    SectionEmitter,
    get_registry,
)
from wagie.reporting.sections.drift import DriftSection


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _full_drift_block() -> dict:
    nan = float("nan")
    return {
        "brier_offline_rolling": (
            [nan, nan, nan, nan, nan]
            + [0.20, 0.22, 0.24, 0.30, 0.28, 0.19, 0.18, 0.21]
        ),
        "brier_online_rolling": (
            [nan, nan, nan, nan, nan]
            + [0.18, 0.19, 0.19, 0.21, 0.20, 0.18, 0.17, 0.19]
        ),
        "baseline_brier": 0.20,
        "rolling_window": 5,
    }


def _make_ctx(tmp_path: Path, *, metrics: dict, use_plotly: bool = True) -> EmitterContext:
    return EmitterContext(
        report_root=tmp_path,
        figs_dir=tmp_path / "figs",
        tables_dir=tmp_path / "tables",
        metrics=metrics, spec_dict={},
        run_meta=RunMeta(run_id="r", spec_name="s", spec_hash="h"),
        use_plotly=use_plotly,
    )


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------

def test_drift_section_is_registered() -> None:
    reg = get_registry()
    assert "drift" in reg
    assert reg["drift"] is DriftSection
    assert issubclass(reg["drift"], SectionEmitter)


def test_drift_section_class_attributes() -> None:
    assert DriftSection.name == "drift"
    assert DriftSection.title == "Drift"
    assert DriftSection.order == 35


# ---------------------------------------------------------------------------
# emit() contract — populated and empty
# ---------------------------------------------------------------------------

def test_emit_full_block_returns_valid_record(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={"drift": _full_drift_block()})
    rec = DriftSection().emit(ctx)
    assert rec.name == "drift"
    assert rec.title == "Drift"
    assert rec.order == 35
    assert rec.html_fragment.strip() != ""
    assert "baseline Brier" in rec.html_fragment
    assert "abort threshold" in rec.html_fragment
    assert "max rolling Brier" in rec.html_fragment
    assert "abort crossings" in rec.html_fragment


def test_emit_missing_drift_returns_empty_state(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={"drift": None})
    rec = DriftSection().emit(ctx)
    assert "Drift block not present" in rec.html_fragment
    assert rec.files == []


def test_emit_no_drift_key_returns_empty_state(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={})
    rec = DriftSection().emit(ctx)
    assert "Drift block not present" in rec.html_fragment


def test_emit_matplotlib_path_writes_pngs(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path, metrics={"drift": _full_drift_block()},
        use_plotly=False,
    )
    rec = DriftSection().emit(ctx)
    assert any(p.endswith(".png") for p in rec.files)
    for rel in rec.files:
        assert (tmp_path / rel).is_file()
        assert (tmp_path / rel).stat().st_size > 1024


def test_emit_plotly_path_inlines_html(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path, metrics={"drift": _full_drift_block()}, use_plotly=True,
    )
    rec = DriftSection().emit(ctx)
    assert "<div" in rec.html_fragment
    assert rec.metadata.get("uses_plotly") is True


# ---------------------------------------------------------------------------
# Numeric summary helpers behave as documented
# ---------------------------------------------------------------------------

def test_finite_max_skips_nans() -> None:
    nan = float("nan")
    assert DriftSection._finite_max([nan, nan, 0.18, 0.20, 0.30, nan]) == 0.30
    assert DriftSection._finite_max([]) is None
    assert DriftSection._finite_max([nan, nan]) is None


def test_count_crossings_counts_up_transitions_only() -> None:
    nan = float("nan")
    # threshold 0.25; series goes ABOVE twice (once at 0.30, again at 0.40)
    series = [nan, 0.10, 0.20, 0.30, 0.30, 0.10, 0.20, 0.40, 0.05]
    assert DriftSection._count_crossings(series, 0.25) == 2


def test_count_crossings_with_no_threshold_returns_zero() -> None:
    assert DriftSection._count_crossings([0.10, 0.50, 0.30], None) == 0


# ---------------------------------------------------------------------------
# Chart-level contract
# ---------------------------------------------------------------------------

def test_charts_handle_empty_data_gracefully(tmp_path: Path) -> None:
    empty = {
        "brier_offline_rolling": [], "brier_online_rolling": [],
        "baseline_brier": None, "rolling_window": 0,
    }
    for fn in (pdrift.rolling_brier_offline, pdrift.rolling_brier_online):
        art = fn(empty, out_dir=tmp_path, use_plotly=True, div_id="x", title="t")
        assert isinstance(art, ChartArtifact)
        assert art.is_empty()
        art2 = fn(empty, out_dir=tmp_path, use_plotly=False, div_id="x", title="t")
        assert isinstance(art2, ChartArtifact)
        assert art2.is_empty()


def test_charts_handle_empty_dict(tmp_path: Path) -> None:
    for fn in (pdrift.rolling_brier_offline, pdrift.rolling_brier_online):
        art = fn({}, out_dir=tmp_path, use_plotly=True, div_id="x", title="t")
        assert art.is_empty()


def test_charts_handle_all_nan_series(tmp_path: Path) -> None:
    data = {
        "brier_offline_rolling": [float("nan"), float("nan")],
        "brier_online_rolling": [float("nan")],
        "baseline_brier": 0.2, "rolling_window": 5,
    }
    art = pdrift.rolling_brier_offline(
        data, out_dir=tmp_path, use_plotly=True, div_id="x", title="t",
    )
    assert art.is_empty()
    art2 = pdrift.rolling_brier_online(
        data, out_dir=tmp_path, use_plotly=False, div_id="x", title="t",
    )
    assert art2.is_empty()


def test_chart_plotly_path_contains_baseline_and_threshold_annotations(tmp_path: Path) -> None:
    data = _full_drift_block()
    art = pdrift.rolling_brier_offline(
        data, out_dir=tmp_path, use_plotly=True, div_id="x", title="t",
    )
    assert art.kind == "plotly"
    # Both baseline and the 1.15× threshold should be reflected in the inline html.
    assert "baseline" in art.html.lower()
    assert "abort" in art.html.lower() or "1.15" in art.html


def test_chart_matplotlib_path_writes_png(tmp_path: Path) -> None:
    data = _full_drift_block()
    art = pdrift.rolling_brier_online(
        data, out_dir=tmp_path, use_plotly=False, div_id="x", title="t",
    )
    assert art.kind == "png"
    assert art.png_path is not None
    assert art.png_path.is_file()
    assert art.png_path.stat().st_size > 1024


def test_chart_handles_missing_baseline_gracefully(tmp_path: Path) -> None:
    data = {
        "brier_online_rolling": [0.18, 0.20, 0.22, 0.19],
        "baseline_brier": None, "rolling_window": 4,
    }
    art = pdrift.rolling_brier_online(
        data, out_dir=tmp_path, use_plotly=True, div_id="x", title="t",
    )
    # Series is non-empty, so should still produce a valid plot.
    assert art.kind == "plotly"
