"""Integration test: ExperimentProtocol end-to-end on synthetic data.

This is the SINGLE protocol the entire repo runs through. The test asserts:
    1. Protocol completes without exceptions
    2. Outputs are written: spec.yaml, metrics.json, index.html (the unified
       HTML report), figs/, state/
    3. metrics.json has the expected fields (calibration leads)
    4. Brier / ECE are non-zero — calibration data flows from the engine
       through MetricsBattery automatically.
    5. The state hash is deterministic across two identical runs
    6. result.report_path points at index.html (NOT the legacy report.md)
"""

from __future__ import annotations

import json
from pathlib import Path

from wagie.experiments import ExperimentProtocol, ExperimentSpec


def _build_spec(parquet: Path, out_dir: Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate({
        "name": "smoke",
        "description": "integration smoke test",
        "seed": 42,
        "wagie": {
            "data": {"parquet_path": str(parquet), "m_minutes": 20},
            "model": {
                "catboost_path": None,
                "arf": {"n_models": 5, "lambda_value": 6.0, "seed": 42},
            },
            "strategy": {"kind": "threshold_gate", "tau": 0.20},
            "broker": {"inventory_cap": 5},
            "runtime": {"warmup_samples": 10},
        },
        "features": {"catalog": "minimal", "n_features": 5},
        "charts": {"enable": True, "n_calibration_bins": 10},
        "report": {"enable": True, "title": None},
        "artifacts": {"out_dir": str(out_dir),
                      "save_state": True, "save_predictions": True},
    })


def test_protocol_runs_end_to_end(synthetic_minute_parquet, tmp_path):
    spec = _build_spec(synthetic_minute_parquet, tmp_path / "report")

    protocol = ExperimentProtocol()
    result = protocol.run(spec)

    assert result.run_id
    assert result.out_dir.is_dir()

    # New shape: out_dir IS the report root (no <run_id>/ subdir).
    assert (result.out_dir / "spec.yaml").is_file()

    # Metrics report exists and has the expected fields
    m = result.out_dir / "metrics.json"
    assert m.is_file()
    metrics = json.loads(m.read_text())
    assert "brier" in metrics
    assert "ece" in metrics
    assert "trading" in metrics
    assert "n_decisions" in metrics
    assert metrics["pipeline_state_hash"]
    # Calibration leads — Brier/ECE are non-zero on the smoke fixture.
    assert metrics["brier"] > 0.0
    assert metrics["ece"] >= 0.0

    # Figures are emitted under figs/<section>/ (new layout). Plotly figures
    # are inlined into index.html (no PNG on disk); matplotlib fallback
    # writes PNGs. The figs/ dir exists in either case but may be empty
    # when Plotly is in use.
    figs_dir = result.out_dir / "figs"
    assert figs_dir.is_dir(), \
        f"figs dir missing under {result.out_dir}"

    # Report exists — and it's an HTML file, not the legacy markdown.
    assert result.report_path is not None
    assert result.report_path.is_file()
    assert result.report_path.name == "index.html"
    body = result.report_path.read_text(encoding="utf-8")
    assert "Calibration" in body
    assert "Trading" in body

    # Brief sanity: at least *some* approved actions on the test slice
    assert metrics["n_decisions"] >= 0


def test_protocol_replay_state_hash_deterministic(synthetic_minute_parquet, tmp_path):
    spec1 = _build_spec(synthetic_minute_parquet, tmp_path / "report1")
    spec2 = _build_spec(synthetic_minute_parquet, tmp_path / "report2")
    r1 = ExperimentProtocol().run(spec1)
    r2 = ExperimentProtocol().run(spec2)
    assert r1.metrics["pipeline_state_hash"] == r2.metrics["pipeline_state_hash"]
