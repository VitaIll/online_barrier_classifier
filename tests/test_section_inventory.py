"""Contract tests for ``InventorySection`` + the inventory charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from wagie.reporting.charts import ChartArtifact
from wagie.reporting.charts import inventory as pinv
from wagie.reporting.manifest import RunMeta
from wagie.reporting.sections import (
    EmitterContext,
    SectionEmitter,
    get_registry,
)
from wagie.reporting.sections.inventory import InventorySection


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _full_inventory_block() -> dict:
    return {
        "size_traj": [0, 1, 2, 2, 3, 2, 1, 0],
        "hold_age_max_traj": [0, 1, 2, 3, 5, 7, 4, 2],
    }


def _make_ctx(
    tmp_path: Path,
    *,
    metrics: dict,
    use_plotly: bool = True,
    engine_result=None,
) -> EmitterContext:
    return EmitterContext(
        report_root=tmp_path,
        figs_dir=tmp_path / "figs",
        tables_dir=tmp_path / "tables",
        metrics=metrics, spec_dict={},
        run_meta=RunMeta(run_id="r", spec_name="s", spec_hash="h"),
        engine_result=engine_result,
        use_plotly=use_plotly,
    )


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------

def test_inventory_section_is_registered() -> None:
    reg = get_registry()
    assert "inventory" in reg
    assert reg["inventory"] is InventorySection
    assert issubclass(reg["inventory"], SectionEmitter)


def test_inventory_section_class_attributes() -> None:
    assert InventorySection.name == "inventory"
    assert InventorySection.title == "Inventory"
    assert InventorySection.order == 25


# ---------------------------------------------------------------------------
# emit() contract — populated and empty
# ---------------------------------------------------------------------------

def test_emit_full_block_returns_valid_record(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={"inventory": _full_inventory_block()})
    rec = InventorySection().emit(ctx)
    assert rec.name == "inventory"
    assert rec.title == "Inventory"
    assert rec.order == 25
    assert rec.html_fragment.strip() != ""
    assert "max inventory" in rec.html_fragment
    assert "mean inventory" in rec.html_fragment
    assert "mean hold age" in rec.html_fragment
    assert "batched exits" in rec.html_fragment


def test_emit_missing_inventory_returns_empty_state(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={"inventory": None})
    rec = InventorySection().emit(ctx)
    assert "Inventory block not present" in rec.html_fragment
    assert "legacy" in rec.html_fragment.lower()
    assert rec.files == []


def test_emit_no_inventory_key_returns_empty_state(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, metrics={})
    rec = InventorySection().emit(ctx)
    assert "Inventory block not present" in rec.html_fragment


def test_emit_matplotlib_path_writes_pngs(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path, metrics={"inventory": _full_inventory_block()},
        use_plotly=False,
    )
    rec = InventorySection().emit(ctx)
    assert any(p.endswith(".png") for p in rec.files)
    for rel in rec.files:
        assert (tmp_path / rel).is_file()
        assert (tmp_path / rel).stat().st_size > 1024


def test_emit_plotly_path_inlines_html(tmp_path: Path) -> None:
    ctx = _make_ctx(
        tmp_path, metrics={"inventory": _full_inventory_block()},
        use_plotly=True,
    )
    rec = InventorySection().emit(ctx)
    assert "<div" in rec.html_fragment
    assert rec.metadata.get("uses_plotly") is True


# ---------------------------------------------------------------------------
# Stratified hold-age distribution from engine ledger fills
# ---------------------------------------------------------------------------

def test_emit_with_engine_fills_uses_per_fill_ages(tmp_path: Path) -> None:
    """When the engine result is in scope, per-fill ages + reasons feed the
    stratified hold-age histogram."""
    from types import SimpleNamespace

    fills = [
        SimpleNamespace(entry_ts=0, exit_ts=10,
                        reason=SimpleNamespace(value="tp")),
        SimpleNamespace(entry_ts=5, exit_ts=10,
                        reason=SimpleNamespace(value="sl")),
        SimpleNamespace(entry_ts=8, exit_ts=20,
                        reason=SimpleNamespace(value="timeout")),
        # batched exit: shares exit_ts=10 with the first two
        SimpleNamespace(entry_ts=4, exit_ts=10,
                        reason=SimpleNamespace(value="manual")),
    ]
    ledger = SimpleNamespace(fills=fills)
    eng = SimpleNamespace(ledger=ledger)
    ctx = _make_ctx(
        tmp_path, metrics={"inventory": _full_inventory_block()},
        use_plotly=True, engine_result=eng,
    )
    rec = InventorySection().emit(ctx)
    # batched_exits = 2 (3 fills sharing exit_ts=10 → 2 extra)
    assert "batched exits" in rec.html_fragment
    # KPI mean hold age changes when per-fill ages are derived (10, 5, 12, 6 → mean 8.25)
    assert "mean hold age" in rec.html_fragment


# ---------------------------------------------------------------------------
# Chart-level contract
# ---------------------------------------------------------------------------

def test_charts_handle_empty_data_gracefully(tmp_path: Path) -> None:
    empty = {"size_traj": [], "hold_age_max_traj": []}
    for fn in (pinv.inventory_size_over_time, pinv.hold_age_distribution):
        art = fn(empty, out_dir=tmp_path, use_plotly=True, div_id="x", title="t")
        assert isinstance(art, ChartArtifact)
        assert art.is_empty()
        art2 = fn(empty, out_dir=tmp_path, use_plotly=False, div_id="x", title="t")
        assert isinstance(art2, ChartArtifact)
        assert art2.is_empty()


def test_charts_handle_empty_dict(tmp_path: Path) -> None:
    for fn in (pinv.inventory_size_over_time, pinv.hold_age_distribution):
        art = fn({}, out_dir=tmp_path, use_plotly=True, div_id="x", title="t")
        assert art.is_empty()


def test_inventory_size_chart_both_paths(tmp_path: Path) -> None:
    data = _full_inventory_block()
    art_plotly = pinv.inventory_size_over_time(
        data, out_dir=tmp_path, use_plotly=True, div_id="x", title="t",
    )
    assert art_plotly.kind == "plotly"
    assert "<div" in art_plotly.html
    art_mpl = pinv.inventory_size_over_time(
        data, out_dir=tmp_path, use_plotly=False, div_id="x", title="t",
    )
    assert art_mpl.kind == "png"
    assert art_mpl.png_path is not None
    assert art_mpl.png_path.is_file()


def test_hold_age_chart_stratified_path(tmp_path: Path) -> None:
    """When ``hold_ages`` + ``exit_reasons`` are aligned, the stratified
    histogram produces output for both paths."""
    data = {
        "size_traj": [], "hold_age_max_traj": [],
        "hold_ages": [10.0, 25.0, 30.0, 18.0, 12.0, 22.0],
        "exit_reasons": ["tp", "sl", "timeout", "tp", "manual", "tp"],
    }
    art_plotly = pinv.hold_age_distribution(
        data, out_dir=tmp_path, use_plotly=True, div_id="x", title="t",
    )
    assert art_plotly.kind == "plotly"
    art_mpl = pinv.hold_age_distribution(
        data, out_dir=tmp_path, use_plotly=False, div_id="x", title="t",
    )
    assert art_mpl.kind == "png"
    assert art_mpl.png_path is not None and art_mpl.png_path.is_file()


def test_hold_age_chart_falls_back_to_max_traj(tmp_path: Path) -> None:
    """No per-fill data → falls back to the ``hold_age_max_traj`` series."""
    data = {"size_traj": [], "hold_age_max_traj": [1, 3, 5, 7, 4, 2, 6, 8]}
    art = pinv.hold_age_distribution(
        data, out_dir=tmp_path, use_plotly=True, div_id="x", title="t",
    )
    assert art.kind == "plotly"
