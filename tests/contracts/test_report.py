"""Contract tests for ``wagie.reporting.Report``.

Pins down the markdown report's section ordering, image-ref portability,
empty-metrics fallback, spec-block YAML round-trip, and run_id surfacing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from wagie.reporting import Report


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _make_chart_pngs(charts_dir: Path) -> dict[str, Path]:
    """Create a minimal valid PNG (8-byte signature) for each documented key."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    png_sig = b"\x89PNG\r\n\x1a\n"
    paths: dict[str, Path] = {}
    for key in ("equity_curve", "drawdown", "pnl_distribution",
                "reliability", "p_histogram"):
        p = charts_dir / f"{key}.png"
        p.write_bytes(png_sig)
        paths[key] = p
    return paths


def _full_metrics() -> dict:
    return {
        "brier": 0.21,
        "ece": 0.034,
        "calibration_by_regime": [
            {"regime_id": 0, "n": 80, "brier": 0.20, "ece": 0.03},
            {"regime_id": 1, "n": 90, "brier": 0.22, "ece": 0.04},
        ],
        "trading": {
            "n_trades": 12, "n_tp": 5, "n_sl": 4, "n_timeout": 3,
            "sharpe": 1.234, "probabilistic_sharpe": 0.5, "sortino": 1.0,
            "hit_rate": 0.42, "profit_factor": 1.1,
            "max_drawdown_log": -0.05, "cdar_5pct_log": -0.02,
            "total_log_return": 0.012, "total_pct_return": 0.012,
        },
        "n_decisions": 30, "n_filled": 12,
        "n_actions_approved": 12, "n_actions_rejected": 5,
        "roc_auc": 0.55, "pr_auc": 0.31,
        "pipeline_state_hash": "deadbeef" * 8,
    }


def _spec_dict() -> dict:
    return {
        "name": "smoke", "seed": 42,
        "wagie": {"data": {"parquet_path": "x.parquet", "m_minutes": 20}},
    }


# -----------------------------------------------------------------
# Section ordering & content
# -----------------------------------------------------------------

def test_render_produces_four_ordered_sections(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    chart_paths = _make_chart_pngs(tmp_path / "charts")
    Report().render(
        spec_dict=_spec_dict(),
        metrics=_full_metrics(),
        chart_paths=chart_paths,
        out_path=out,
    )
    body = out.read_text(encoding="utf-8")
    # Find each header's index and assert ordering
    headers = [
        "## 1. Calibration",
        "## 2. Trading",
        "## 3. Operational",
        "## 4. Spec",
    ]
    indexes = [body.index(h) for h in headers]
    assert indexes == sorted(indexes), f"sections out of order: {indexes}"
    # Assert there is no longer a Conformal coverage section.
    assert "Conformal coverage" not in body


def test_render_writes_calibration_metrics(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    Report().render(
        spec_dict=_spec_dict(), metrics=_full_metrics(),
        chart_paths={}, out_path=out,
    )
    body = out.read_text()
    assert "Brier" in body
    assert "ECE" in body
    assert "0.21" in body  # Brier value


def test_render_writes_per_regime_calibration_table(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    Report().render(
        spec_dict=_spec_dict(), metrics=_full_metrics(),
        chart_paths={}, out_path=out,
    )
    body = out.read_text()
    assert "regime" in body.lower()
    assert "Brier" in body
    # Both regimes present
    assert "| 0 |" in body
    assert "| 1 |" in body


# -----------------------------------------------------------------
# Image refs are relative paths anchored at the report file
# -----------------------------------------------------------------

def test_image_refs_are_relative_when_under_report_parent(tmp_path: Path) -> None:
    """Charts inside the run dir produce relative refs (no absolute paths)."""
    run_dir = tmp_path / "runs" / "20260101_smoke_aaaa"
    out = run_dir / "report.md"
    charts = _make_chart_pngs(run_dir / "charts")

    Report().render(
        spec_dict=_spec_dict(), metrics=_full_metrics(),
        chart_paths=charts, out_path=out, run_id="rid-001",
    )
    body = out.read_text(encoding="utf-8")

    # Every image ref is a markdown image; must be relative
    # The report writes things like ![alt](charts/equity_curve.png).
    assert "![Equity curve](charts/equity_curve.png)" in body
    assert "![Reliability diagram](charts/reliability.png)" in body
    assert "![PnL distribution](charts/pnl_distribution.png)" in body

    # No drive-letter / absolute-windows-path leakage (no "C:/" or "C:\")
    assert "C:/" not in body
    assert "C:\\" not in body


def test_image_refs_fall_back_to_posix_when_outside_report_parent(tmp_path: Path) -> None:
    """If chart lives outside report parent, fall back to as_posix() path
    (no exception). Documents the existing graceful-fallback behavior."""
    out_dir = tmp_path / "report_dir"
    out_dir.mkdir()
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    chart_paths = _make_chart_pngs(other_dir)
    out = out_dir / "report.md"

    Report().render(
        spec_dict=_spec_dict(), metrics=_full_metrics(),
        chart_paths=chart_paths, out_path=out,
    )
    body = out.read_text(encoding="utf-8")
    # Still produced an image markdown ref for equity_curve
    assert "![Equity curve](" in body


# -----------------------------------------------------------------
# Empty metrics fallback
# -----------------------------------------------------------------

def test_completely_empty_metrics_does_not_crash(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    rep = Report()
    p = rep.render(spec_dict={}, metrics={}, chart_paths={}, out_path=out)
    assert p == out
    assert out.is_file()
    # All four sections still rendered
    body = out.read_text()
    for h in ("## 1. Calibration", "## 2. Trading",
              "## 3. Operational", "## 4. Spec"):
        assert h in body


# -----------------------------------------------------------------
# Spec block is a valid YAML round-trip
# -----------------------------------------------------------------

def test_spec_block_round_trips_as_yaml(tmp_path: Path) -> None:
    spec = _spec_dict()
    out = tmp_path / "report.md"
    Report().render(
        spec_dict=spec, metrics=_full_metrics(),
        chart_paths={}, out_path=out,
    )
    body = out.read_text(encoding="utf-8")
    # Extract the fenced YAML block
    fence_open = body.index("```yaml")
    fence_close = body.index("```", fence_open + len("```yaml"))
    yaml_text = body[fence_open + len("```yaml"): fence_close].strip()
    parsed = yaml.safe_load(yaml_text)
    assert parsed["name"] == "smoke"
    assert parsed["seed"] == 42
    assert parsed["wagie"]["data"]["parquet_path"] == "x.parquet"


# -----------------------------------------------------------------
# Run id surfacing
# -----------------------------------------------------------------

def test_run_id_appears_in_output_when_passed(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    rid = "20260101-120000_smoke_abc12345"
    Report().render(
        spec_dict=_spec_dict(), metrics=_full_metrics(),
        chart_paths={}, out_path=out, run_id=rid,
    )
    body = out.read_text(encoding="utf-8")
    assert rid in body
    assert f"`{rid}`" in body


def test_run_id_placeholder_when_omitted(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    Report().render(
        spec_dict=_spec_dict(), metrics=_full_metrics(),
        chart_paths={}, out_path=out,
    )
    body = out.read_text(encoding="utf-8")
    # The dash em-dash is the documented placeholder
    assert "**run_id:**" in body


def test_state_hash_only_shown_when_provenance_enabled(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    metrics = _full_metrics()
    Report(show_provenance=False).render(
        spec_dict=_spec_dict(), metrics=metrics,
        chart_paths={}, out_path=out,
    )
    body = out.read_text(encoding="utf-8")
    assert "state_hash" not in body

    # And inversely
    out2 = tmp_path / "report2.md"
    Report(show_provenance=True).render(
        spec_dict=_spec_dict(), metrics=metrics,
        chart_paths={}, out_path=out2,
    )
    assert "state_hash" in out2.read_text(encoding="utf-8")


def test_custom_title_appears_as_h1(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    Report(title="my experiment").render(
        spec_dict=_spec_dict(), metrics=_full_metrics(),
        chart_paths={}, out_path=out,
    )
    body = out.read_text(encoding="utf-8")
    assert body.splitlines()[0] == "# my experiment"


def test_render_creates_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "report.md"
    Report().render(
        spec_dict=_spec_dict(), metrics={},
        chart_paths={}, out_path=out,
    )
    assert out.is_file()
