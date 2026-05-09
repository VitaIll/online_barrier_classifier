"""Integration tests for the four ExperimentProtocol modes:

    1. CV mode    — spec.cv.enabled=True dispatches to wagie.cv
    2. Training   — warm_calibrator_quantiles populates aci.q_init_by_regime
    3. Charts off — no charts/ dir is produced
    4. Report off — no report.md
    5. run_id     — timestamp_<name>_<hash> regex
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from wagie.experiments import ExperimentProtocol, ExperimentSpec


RUN_ID_RE = re.compile(r"^\d{8}-\d{6}_[A-Za-z0-9_\-]+_[0-9a-f]{8}$")


def _base_spec_dict(parquet: Path, out_dir: Path, name: str = "modes") -> dict:
    return {
        "name": name,
        "description": "protocol-mode test",
        "seed": 42,
        "wagie": {
            "data": {"parquet_path": str(parquet), "m_minutes": 20},
            "model": {
                "catboost_path": None,
                "arf": {"n_models": 5, "lambda_value": 6.0, "seed": 42},
                "aci": {"alphas": [0.05, 0.10, 0.20], "gamma": 0.01,
                        "n_regimes": 3, "q_init": 0.5},
            },
            "strategy": {"kind": "pure_conformal", "alpha": 0.10},
            "broker": {"inventory_cap": 5},
            "runtime": {"warmup_samples": 10},
        },
        "features": {"catalog": "minimal", "n_features": 5},
        "charts": {"enable": True, "n_calibration_bins": 10},
        "report": {"enable": True, "title": None},
        "artifacts": {"out_dir": str(out_dir),
                      "save_state": True, "save_predictions": True},
    }


# --------------------------- run_id format ---------------------------------

def test_run_id_format_matches_regex(synthetic_minute_parquet, tmp_path):
    """run_id == <timestamp>_<name>_<8-hex-hash>."""
    spec = ExperimentSpec.model_validate(
        _base_spec_dict(synthetic_minute_parquet, tmp_path / "runs",
                        name="runidtest"),
    )
    result = ExperimentProtocol().run(spec)
    assert RUN_ID_RE.match(result.run_id), \
        f"run_id={result.run_id!r} does not match {RUN_ID_RE.pattern}"
    # Anchored sub-checks: name embedded, hash matches spec.hash() prefix.
    assert "_runidtest_" in result.run_id
    assert result.run_id.endswith(spec.hash()[:8])


# ------------------------------ CV mode ------------------------------------

def test_protocol_cv_mode_emits_per_fold_and_pbo(
    synthetic_minute_parquet, tmp_path,
):
    """spec.cv.enabled=True ⇒ metrics.json has mode=cv, per_fold list,
    pbo value, and a per_fold_sharpe.png chart."""
    raw = _base_spec_dict(synthetic_minute_parquet, tmp_path / "runs", name="cvmode")
    # NOTE: skfolio.CombinatorialPurgedCV requires n_test_folds >= 2.
    # The repo's CVSpec accepts n_test_folds=1 but the underlying library
    # rejects it. We use n_test_folds=2 to exercise the protocol path.
    raw["cv"] = {
        "enabled": True, "n_folds": 4, "n_test_folds": 2,
        "embargo_size": 1, "purged_size": 1,
    }
    spec = ExperimentSpec.model_validate(raw)
    result = ExperimentProtocol().run(spec)

    metrics_path = result.out_dir / "metrics.json"
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text())

    assert metrics["mode"] == "cv"
    assert "per_fold" in metrics and isinstance(metrics["per_fold"], list)
    assert "pbo" in metrics
    pbo = float(metrics["pbo"])
    assert 0.0 <= pbo <= 1.0, f"PBO out of [0,1]: {pbo}"

    # Charts: per-fold sharpe png
    charts_dir = result.out_dir / "charts"
    assert charts_dir.is_dir()
    sharpe_png = charts_dir / "01_per_fold_sharpe.png"
    assert sharpe_png.is_file(), \
        f"missing per-fold sharpe chart: {list(charts_dir.glob('*'))}"
    assert sharpe_png.stat().st_size > 100


# --------------------------- Training warm-up -------------------------------

def test_training_warm_calibrator_populates_q_init_by_regime(
    synthetic_minute_parquet, tmp_path,
):
    """warm_calibrator_quantiles=True ⇒ spec.wagie.model.aci.q_init_by_regime
    is populated by the protocol before the engine runs."""
    raw = _base_spec_dict(synthetic_minute_parquet, tmp_path / "runs", name="warm")
    raw["training"] = {"warm_calibrator_quantiles": True, "warm_train_frac": 0.5}
    spec = ExperimentSpec.model_validate(raw)

    assert spec.wagie.model.aci.q_init_by_regime is None  # pre-warm

    ExperimentProtocol().run(spec)

    q = spec.wagie.model.aci.q_init_by_regime
    assert q is not None, "warm_calibrator did not populate q_init_by_regime"
    assert isinstance(q, dict)
    # Expect one entry per alpha.
    expected_alphas = set(float(a) for a in spec.wagie.model.aci.alphas)
    assert set(float(k) for k in q.keys()) == expected_alphas
    # Each alpha maps to a regime → quantile dict.
    for alpha, per_regime in q.items():
        assert isinstance(per_regime, dict), \
            f"alpha={alpha} should map to regime dict, got {type(per_regime)}"


# ----------------------------- Charts disabled -----------------------------

def test_charts_disabled_no_charts_dir(synthetic_minute_parquet, tmp_path):
    """charts.enable=False ⇒ no charts/ dir produced."""
    raw = _base_spec_dict(synthetic_minute_parquet, tmp_path / "runs",
                          name="nochartz")
    raw["charts"]["enable"] = False
    spec = ExperimentSpec.model_validate(raw)
    result = ExperimentProtocol().run(spec)

    charts_dir = result.out_dir / "charts"
    assert not charts_dir.exists(), \
        f"charts dir should not exist when disabled, got {list(charts_dir.iterdir()) if charts_dir.exists() else None}"
    assert result.chart_paths == {} or not result.chart_paths


# ----------------------------- Report disabled -----------------------------

def test_report_disabled_no_report_md(synthetic_minute_parquet, tmp_path):
    """report.enable=False ⇒ no report.md produced."""
    raw = _base_spec_dict(synthetic_minute_parquet, tmp_path / "runs",
                          name="noreport")
    raw["report"]["enable"] = False
    spec = ExperimentSpec.model_validate(raw)
    result = ExperimentProtocol().run(spec)

    report_md = result.out_dir / "report.md"
    assert not report_md.exists(), "report.md should not exist when disabled"
    assert result.report_path is None
