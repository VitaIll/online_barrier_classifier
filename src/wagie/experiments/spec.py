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
    """Toggles for the unified HTML report.

    The protocol always renders into the SINGLE canonical report dir
    (``artifacts.out_dir``, default ``artifacts/report``). When
    ``enable=False`` no report is rendered (rare — used for smoke tests
    where you only want metrics on disk). ``title`` overrides the default
    page title; ``use_plotly`` toggles the interactive Plotly layer.
    """

    enable: bool = True
    title: Optional[str] = None
    use_plotly: bool = True

    model_config = ConfigDict(extra="forbid")


class ArtifactsSpec(BaseModel):
    """Where to write outputs.

    Layout (canonical, single report)::

        <out_dir>/
        ├── index.html         # the SINGLE report
        ├── manifest.json
        ├── figs/, tables/
        ├── spec.yaml, metrics.json
        ├── state/             # pipeline_state_hash + rebuild_bundle.pkl
        └── _archive/          # last `archive_keep` snapshots (zip)

    Each ``wagie experiment run`` overwrites the live tree (after
    archiving the previous state into ``_archive/``). Side experiments
    via ``wagie experiment run --experiment NAME`` write to
    ``artifacts/experiments/<NAME>/`` and never archive.
    """

    out_dir: str = "artifacts/report"
    save_state: bool = True
    save_predictions: bool = True
    archive_keep: int = 10
    enable_archive: bool = True

    model_config = ConfigDict(extra="forbid")


class CVSpec(BaseModel):
    """Cross-validation knobs. When present, the protocol runs CV instead of
    a single backtest. CV folds use ``wagie.cv.cross_validation``."""

    enabled: bool = True
    n_folds: int = 10
    n_test_folds: int = 2
    purged_size: int = 1
    embargo_size: int = 5

    model_config = ConfigDict(extra="forbid")


class TrainingSpec(BaseModel):
    """Training-side knobs the protocol applies before the engine runs.

    Offline CatBoost training is out-of-band — point
    ``wagie.model.catboost_path`` at a pre-trained .cbm. This sub-spec is
    intentionally near-empty after the Mondrian-ACI removal: the streaming
    ARF needs no warm-up.
    """

    # Reserved for future hooks (e.g. selected_features warm-up). Kept as a
    # stable empty shape so existing YAMLs remain valid.

    model_config = ConfigDict(extra="forbid")


class ExperimentSpec(BaseModel):
    """The single source of truth for one experiment.

    The protocol calls:
        wagie experiment run my_spec.yaml

    and produces ``<artifacts.out_dir>/{index.html, manifest.json,
    spec.yaml, metrics.json, figs/, tables/, state/, _archive/}`` —
    the single canonical report. The previous run is zipped into
    ``_archive/`` (last 10 by default) before being overwritten.

    Pre-registration block (round-040 additive): ``hypothesis_id``,
    ``predicted_effect_min``, ``min_n_trades``, ``min_effect_vs_seed_band``
    are evaluated by the protocol AFTER metrics compute. If any gate
    fails, the protocol still renders the report but with
    ``RunMeta.accepted=False`` and the failure reasons surfaced in the
    report's blocked banner; ``ExperimentResult.accepted`` is False.
    ``n_trials_for_dsr`` controls the deflated Sharpe ratio
    multiple-testing trials count.
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

    # ---------------- Pre-registration / accept-gate (round-040) ----------
    hypothesis_id: str = "unspecified"
    predicted_effect_min: Optional[float] = None
    min_n_trades: int = 0
    min_effect_vs_seed_band: float = 0.0
    n_trials_for_dsr: int = 1

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentSpec":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "wagie" not in raw:
            raise ValueError(f"spec {path}: missing required `wagie:` block")
        # Quietly drop any legacy `aci` block from old YAMLs.
        wagie = raw.get("wagie", {})
        model = wagie.get("model") if isinstance(wagie, dict) else None
        if isinstance(model, dict):
            model.pop("aci", None)
        # Drop legacy strategy.alpha / .layer if present.
        strategy = wagie.get("strategy") if isinstance(wagie, dict) else None
        if isinstance(strategy, dict):
            strategy.pop("alpha", None)
            strategy.pop("layer", None)
        # Drop legacy training.warm_calibrator_quantiles / warm_train_frac.
        training = raw.get("training")
        if isinstance(training, dict):
            training.pop("warm_calibrator_quantiles", None)
            training.pop("warm_train_frac", None)
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"),
                              sort_keys=False, default_flow_style=False)

    def hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


__all__ = ["ExperimentSpec", "FeaturesSpec", "ChartsSpec",
           "ReportSpec", "ArtifactsSpec", "CVSpec", "TrainingSpec"]
