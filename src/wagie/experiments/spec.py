"""ExperimentSpec — the canonical YAML schema for an experiment.

A spec is the declarative description of one experiment run. The protocol
consumes it; nothing else in the package writes to it. To extend, add a field
to a sub-model — never bypass the spec.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from wagie.config import WagieConfig


class FeaturesSpec(BaseModel):
    """How features are computed at run time."""

    catalog: Literal["default", "minimal", "full"] = "default"
    n_features: Optional[int] = None
    regime_feature: str = "parkinson_var_rolling_mean_24"
    regime_edges: tuple[float, ...] = (1e-6, 1e-5)
    regime_labels: tuple[str, ...] = ("low", "med", "high")

    model_config = ConfigDict(extra="forbid")


class ChartsSpec(BaseModel):
    enable: bool = True
    n_calibration_bins: int = 10

    model_config = ConfigDict(extra="forbid")


class ReportSpec(BaseModel):
    enable: bool = True
    title: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class ArtifactsSpec(BaseModel):
    """Where to write outputs. The protocol treats this as the only sink."""

    out_dir: str = "artifacts/runs"
    save_state: bool = True
    save_predictions: bool = True

    model_config = ConfigDict(extra="forbid")


class CVSpec(BaseModel):
    """Cross-validation knobs. When present, the protocol runs CV instead of
    a single backtest. CV folds use `wagie.cv.cross_validation`."""

    enabled: bool = True
    n_folds: int = 10
    n_test_folds: int = 2
    purged_size: int = 1
    embargo_size: int = 5

    model_config = ConfigDict(extra="forbid")


class TrainingSpec(BaseModel):
    """Training-side knobs the protocol applies before the engine runs.

    Currently: warm the streaming Mondrian-ACI quantiles from a held-out
    train slice. Offline CatBoost training is out-of-band — point
    `wagie.model.catboost_path` at a pre-trained .cbm.
    """

    warm_calibrator_quantiles: bool = False
    warm_train_frac: float = 0.6

    model_config = ConfigDict(extra="forbid")


class ExperimentSpec(BaseModel):
    """The single source of truth for one experiment.

    The protocol calls:
        wagie experiment run my_spec.yaml

    and produces: artifacts/runs/<run_id>/{config.yaml, metrics.json,
    charts/, report.md, state/}.
    """

    name: str
    description: str = ""
    seed: int = 42

    # The wagie engine config (data, model, strategy, broker, runtime, cv).
    wagie: WagieConfig

    features: FeaturesSpec = Field(default_factory=FeaturesSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    cv: Optional[CVSpec] = None
    charts: ChartsSpec = Field(default_factory=ChartsSpec)
    report: ReportSpec = Field(default_factory=ReportSpec)
    artifacts: ArtifactsSpec = Field(default_factory=ArtifactsSpec)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentSpec":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "wagie" not in raw:
            raise ValueError(f"spec {path}: missing required `wagie:` block")
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"),
                              sort_keys=False, default_flow_style=False)

    def hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


__all__ = ["ExperimentSpec", "FeaturesSpec", "ChartsSpec",
           "ReportSpec", "ArtifactsSpec", "CVSpec", "TrainingSpec"]
