"""Contract tests for `wagie.experiments.spec.ExperimentSpec`.

The ExperimentSpec is the canonical YAML schema for a wagie experiment. These
tests pin down the public surface:
  - YAML round-trip (`from_yaml` / `to_yaml`)
  - `extra='forbid'` rejects typos at every nesting level
  - `.hash()` is deterministic and changes on a single-bit edit
  - sub-spec defaults remain stable (CVSpec, TrainingSpec, ArtifactsSpec, ...)
  - `FeaturesSpec.catalog` is restricted to the documented Literal set
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from wagie.experiments.spec import (
    ArtifactsSpec,
    ChartsSpec,
    CVSpec,
    ExperimentSpec,
    FeaturesSpec,
    ReportSpec,
    TrainingSpec,
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _minimal_spec_dict(parquet_path: str = "data/foo.parquet") -> dict:
    """Smallest possible payload that still validates."""
    return {
        "name": "smoke",
        "wagie": {
            "data": {"parquet_path": parquet_path, "m_minutes": 20},
        },
    }


# -----------------------------------------------------------------
# from_yaml / to_yaml round-trip
# -----------------------------------------------------------------

def test_from_yaml_round_trip(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_minimal_spec_dict()))

    spec = ExperimentSpec.from_yaml(spec_path)
    assert spec.name == "smoke"
    assert spec.wagie.data.parquet_path == "data/foo.parquet"
    assert spec.wagie.data.m_minutes == 20


def test_to_yaml_is_parseable_and_round_trips(tmp_path: Path) -> None:
    spec = ExperimentSpec.model_validate(_minimal_spec_dict())
    yaml_text = spec.to_yaml()

    # parseable as YAML
    parsed = yaml.safe_load(yaml_text)
    assert isinstance(parsed, dict)
    assert parsed["name"] == "smoke"

    # round-trip via temp file
    p = tmp_path / "rt.yaml"
    p.write_text(yaml_text)
    rebuilt = ExperimentSpec.from_yaml(p)
    assert rebuilt.hash() == spec.hash()
    assert rebuilt.model_dump() == spec.model_dump()


def test_from_yaml_missing_wagie_block_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "x"}))
    with pytest.raises(ValueError, match="missing required `wagie:` block"):
        ExperimentSpec.from_yaml(bad)


def test_from_yaml_empty_file_raises(tmp_path: Path) -> None:
    """Empty YAML still triggers the missing-wagie error rather than silently passing."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError, match="missing required `wagie:` block"):
        ExperimentSpec.from_yaml(p)


# -----------------------------------------------------------------
# extra='forbid' at every level
# -----------------------------------------------------------------

def test_top_level_typo_is_rejected() -> None:
    bad = _minimal_spec_dict()
    bad["seeed"] = 42  # typo of 'seed'
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(bad)


def test_features_spec_typo_rejected() -> None:
    with pytest.raises(ValidationError):
        FeaturesSpec(catalogue="default")  # type: ignore[call-arg]


def test_charts_spec_typo_rejected() -> None:
    with pytest.raises(ValidationError):
        ChartsSpec(enabled=True)  # field is `enable`, not `enabled`  # type: ignore[call-arg]


def test_report_spec_typo_rejected() -> None:
    with pytest.raises(ValidationError):
        ReportSpec(titel="oops")  # type: ignore[call-arg]


def test_artifacts_spec_typo_rejected() -> None:
    with pytest.raises(ValidationError):
        ArtifactsSpec(out="artifacts/runs")  # type: ignore[call-arg]


def test_cv_spec_typo_rejected() -> None:
    with pytest.raises(ValidationError):
        CVSpec(n_fold=10)  # type: ignore[call-arg]


def test_training_spec_typo_rejected() -> None:
    """TrainingSpec uses extra='forbid' — unknown fields raise."""
    with pytest.raises(ValidationError):
        TrainingSpec(warm_calibrator=False)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        TrainingSpec(warm_calibrator_quantiles=True)  # type: ignore[call-arg]


def test_nested_typo_in_features_through_top_spec() -> None:
    payload = _minimal_spec_dict()
    payload["features"] = {"catalogue": "default"}  # typo of 'catalog'
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_nested_typo_in_charts_through_top_spec() -> None:
    payload = _minimal_spec_dict()
    payload["charts"] = {"enabled": True}
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


# -----------------------------------------------------------------
# Hash determinism + sensitivity
# -----------------------------------------------------------------

def test_hash_is_deterministic() -> None:
    s1 = ExperimentSpec.model_validate(_minimal_spec_dict())
    s2 = ExperimentSpec.model_validate(_minimal_spec_dict())
    assert s1.hash() == s2.hash()
    # 16 hex chars per spec.py
    assert len(s1.hash()) == 16
    int(s1.hash(), 16)  # is valid hex


def test_hash_changes_on_single_bit_edit() -> None:
    base = _minimal_spec_dict()
    s1 = ExperimentSpec.model_validate(base)

    # Single-character / single-bit-ish change — name -> "smokf"
    diff = dict(base)
    diff["name"] = "smokf"
    s2 = ExperimentSpec.model_validate(diff)
    assert s1.hash() != s2.hash()

    # Same payload but different seed (also a single-bit class diff)
    diff2 = dict(base)
    diff2["seed"] = 43
    s3 = ExperimentSpec.model_validate(diff2)
    assert s1.hash() != s3.hash()


# -----------------------------------------------------------------
# Sub-spec defaults
# -----------------------------------------------------------------

def test_cv_spec_defaults() -> None:
    cv = CVSpec()
    assert cv.enabled is True
    assert cv.n_folds == 10
    assert cv.n_test_folds == 2
    assert cv.purged_size == 1
    assert cv.embargo_size == 5


def test_training_spec_defaults() -> None:
    """TrainingSpec is intentionally near-empty after the Mondrian-ACI
    removal (the streaming ARF needs no warm-up)."""
    t = TrainingSpec()
    # Stable empty shape — no fields to assert beyond construction.
    assert isinstance(t, TrainingSpec)


def test_artifacts_spec_defaults() -> None:
    a = ArtifactsSpec()
    assert a.out_dir == "artifacts/report"
    assert a.save_state is True
    assert a.save_predictions is True
    assert a.archive_keep == 10
    assert a.enable_archive is True


def test_charts_spec_defaults() -> None:
    c = ChartsSpec()
    assert c.enable is True
    assert c.n_calibration_bins == 10


def test_report_spec_defaults() -> None:
    r = ReportSpec()
    assert r.enable is True
    assert r.title is None
    assert r.use_plotly is True


def test_features_spec_defaults() -> None:
    f = FeaturesSpec()
    assert f.catalog == "default"
    assert f.n_features is None
    assert f.regime_feature == "parkinson_var_rolling_mean_24"
    assert f.regime_edges == (1e-6, 1e-5)
    assert f.regime_labels == ("low", "med", "high")


def test_experiment_spec_defaults_present_when_unspecified() -> None:
    spec = ExperimentSpec.model_validate(_minimal_spec_dict())
    # description and seed take defaults
    assert spec.description == ""
    assert spec.seed == 42
    # cv is optional; default None
    assert spec.cv is None
    # default factories produce live sub-models, not None
    assert isinstance(spec.features, FeaturesSpec)
    assert isinstance(spec.training, TrainingSpec)
    assert isinstance(spec.charts, ChartsSpec)
    assert isinstance(spec.report, ReportSpec)
    assert isinstance(spec.artifacts, ArtifactsSpec)


# -----------------------------------------------------------------
# Catalog Literal
# -----------------------------------------------------------------

@pytest.mark.parametrize("ok", ["default", "minimal", "full"])
def test_features_catalog_accepts_documented_values(ok: str) -> None:
    f = FeaturesSpec(catalog=ok)
    assert f.catalog == ok


@pytest.mark.parametrize("bad", ["DEFAULT", "small", "extended", "", "minimal "])
def test_features_catalog_rejects_other_values(bad: str) -> None:
    with pytest.raises(ValidationError):
        FeaturesSpec(catalog=bad)  # type: ignore[arg-type]
