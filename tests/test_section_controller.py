"""Contract tests for ``ControllerSection`` + the controller charts.

Pin down:
  - ``@register_emitter`` registers the ControllerSection class on import.
  - ``emit()`` returns a valid SectionRecord with non-empty content when the
    full controller block is present, and an informative empty-state when
    the block is absent or None.
  - All three controller chart functions handle empty arrays gracefully
    (no exceptions; returns ChartArtifact of kind=='empty').
  - Both Plotly and matplotlib paths produce valid output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Force a non-interactive backend before any test imports matplotlib.
matplotlib.use("Agg", force=True)

from wagie.reporting.charts import controller as pctrl
from wagie.reporting.charts import ChartArtifact
from wagie.reporting.manifest import RunMeta
from wagie.reporting.sections import (
    EmitterContext,
    SectionEmitter,
    get_registry,
)
from wagie.reporting.sections.controller import ControllerSection


# ---------------------------------------------------------------------------
# Fixtures (plain helpers — kept module-local to avoid conftest changes)
# ---------------------------------------------------------------------------

def _full_controller_block() -> dict:
    return {
        "tau_traj": [0.50, 0.55, 0.60, 0.70, 0.65, 0.60, 0.55, 0.52],
        "r_hat_traj": [0.10, 0.12, 0.20, 0.18, 0.16, 0.14, 0.12, 0.11],
        "r_star_traj": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
        "pause_spans": [(2, 3)],
        "sigma_ve_dist": [0.010, 0.020, 0.015, 0.025, 0.018, 0.022, 0.014],
        "sigma_max": 0.030,
        "tau_floor": 0.40,
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

def test_controller_section_is_registered() -> None:
    """Importing the module must register the section in the global registry."""
    reg = get_registry()
    assert "controller" in reg
    assert reg["controller"] is ControllerSection
    assert issubclass(reg["controller"], SectionEmitter)


def test_controller_section_class_attributes() -> None:
    assert ControllerSection.name == "controller"
    assert ControllerSection.title == "Controller"
    assert ControllerSection.order == 15


# ---------------------------------------------------------------------------
# emit() contract — populated and empty
# ---------------------------------------------------------------------------

def test_emit_full_block_returns_valid_record(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={"controller": _full_controller_block()})
    rec = ControllerSection().emit(ctx)
    assert rec.name == "controller"
    assert rec.title == "Controller"
    assert rec.order == 15
    assert rec.html_fragment.strip() != ""
    # KPI cards mention the documented summary metrics.
    assert "final τ" in rec.html_fragment
    assert "mean r̂_ewma" in rec.html_fragment
    assert "σ_max" in rec.html_fragment
    assert "paused bars" in rec.html_fragment
    # Should mention plotly use in the metadata.
    assert isinstance(rec.metadata, dict)


def test_emit_missing_controller_block_returns_empty_state(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={"controller": None})
    rec = ControllerSection().emit(ctx)
    assert rec.name == "controller"
    assert rec.html_fragment.strip() != ""
    # Empty-state message must be human-informative, not silent.
    assert "Controller block not present" in rec.html_fragment
    assert "legacy" in rec.html_fragment.lower()
    assert rec.files == []


def test_emit_no_controller_key_returns_empty_state(tmp_path: Path) -> None:
    """A metrics dict without the key at all is treated like a missing block."""
    ctx = _make_ctx(tmp_path, metrics={})
    rec = ControllerSection().emit(ctx)
    assert "Controller block not present" in rec.html_fragment


def test_emit_matplotlib_path_writes_pngs(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path, metrics={"controller": _full_controller_block()},
        use_plotly=False,
    )
    rec = ControllerSection().emit(ctx)
    # We expect at least one PNG artifact produced via the matplotlib fallback.
    assert any(p.endswith(".png") for p in rec.files)
    for rel in rec.files:
        # Each listed file must exist on disk under the report root.
        assert (tmp_path / rel).is_file()
        assert (tmp_path / rel).stat().st_size > 1024


def test_emit_plotly_path_inlines_html(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path, metrics={"controller": _full_controller_block()},
        use_plotly=True,
    )
    rec = ControllerSection().emit(ctx)
    # Plotly path inlines a <div>...plotly.js fragment without writing PNGs.
    assert "plotly" in rec.html_fragment.lower() or "<div" in rec.html_fragment
    assert rec.metadata.get("uses_plotly") is True


# ---------------------------------------------------------------------------
# Chart-level contract
# ---------------------------------------------------------------------------

def test_charts_handle_empty_data_gracefully(tmp_path: Path) -> None:
    empty: dict = {
        "tau_traj": [], "r_hat_traj": [], "r_star_traj": [],
        "pause_spans": [], "sigma_ve_dist": [], "sigma_max": None,
    }
    for fn in (
        pctrl.tau_trajectory,
        pctrl.entry_rate_overlay,
        pctrl.sigma_ve_distribution,
    ):
        art = fn(empty, out_dir=tmp_path, use_plotly=True, div_id="x", title="t")
        assert isinstance(art, ChartArtifact)
        assert art.is_empty()
        # matplotlib path should also be empty (returns kind="empty").
        art2 = fn(empty, out_dir=tmp_path, use_plotly=False, div_id="x", title="t")
        assert isinstance(art2, ChartArtifact)
        assert art2.is_empty()


def test_charts_handle_none_data_dict_gracefully(tmp_path: Path) -> None:
    """Passing ``{}`` (or even ``None`` defended by the empty-key fallback)
    must not raise."""
    for fn in (
        pctrl.tau_trajectory,
        pctrl.entry_rate_overlay,
        pctrl.sigma_ve_distribution,
    ):
        art = fn({}, out_dir=tmp_path, use_plotly=True, div_id="x", title="t")
        assert art.is_empty()


def test_chart_plotly_and_matplotlib_paths_both_produce_output(tmp_path: Path) -> None:
    data = _full_controller_block()
    # Plotly path → inline html, no PNG.
    art_plotly = pctrl.tau_trajectory(
        data, out_dir=tmp_path, use_plotly=True,
        div_id="x", title="t",
    )
    assert art_plotly.kind == "plotly"
    assert "<div" in art_plotly.html
    # Matplotlib path → PNG file on disk.
    art_mpl = pctrl.tau_trajectory(
        data, out_dir=tmp_path, use_plotly=False,
        div_id="x", title="t",
    )
    assert art_mpl.kind == "png"
    assert art_mpl.png_path is not None
    assert art_mpl.png_path.is_file()
    assert art_mpl.png_path.stat().st_size > 1024


def test_sigma_ve_chart_handles_all_nonfinite(tmp_path: Path) -> None:
    """A σ_ve_dist with only NaN/inf values must produce kind='empty'."""
    data = {
        "sigma_ve_dist": [float("nan"), float("inf"), float("-inf")],
        "sigma_max": 0.05,
    }
    art = pctrl.sigma_ve_distribution(
        data, out_dir=tmp_path, use_plotly=True, div_id="x", title="t",
    )
    assert art.is_empty()
    art2 = pctrl.sigma_ve_distribution(
        data, out_dir=tmp_path, use_plotly=False, div_id="x", title="t",
    )
    assert art2.is_empty()
