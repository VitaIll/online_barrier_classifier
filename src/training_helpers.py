"""Training-stage helpers used by 03_model_training.ipynb (and future CLI runs).

This module exists so that the autonomous loop's training step is reproducible
without notebook surgery. Provides:

- `param_provenance()`: deterministic resolution of which parameter set wins
  given USE_PINNED_PARAMS, RUN_HPO, file-loaded params, and a fallback.
  Replaces the silent override the notebook had pre-round-003.
- `log_training_run()`: a thin wrapper around `src.mlflow_utils.round_run`
  that logs the standard tag set, the best params, the per-split metric block,
  the calibration-by-regime block, and any plot artifacts.
- `metric_block_from_predictions()`: builds the canonical metric dict from
  `(y_true, y_pred_proba)` in one call so the notebook doesn't repeat itself.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Param provenance
# ----------------------------------------------------------------------------

def param_provenance(
    *,
    use_pinned: bool,
    run_hpo: bool,
    pinned_params: Optional[Mapping[str, Any]] = None,
    hpo_best_params: Optional[Mapping[str, Any]] = None,
    file_loaded_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Resolve the priority chain `pinned > hpo > file > fallback`.

    The pre-round-003 notebook silently overrode HPO+file output. This function
    makes the choice explicit and the result auditable. Returns a dict with
    `source` ∈ {"pinned", "hpo", "file", "fallback"} and `params`.
    """
    if use_pinned and pinned_params:
        return {"source": "pinned", "params": dict(pinned_params)}
    if run_hpo and hpo_best_params:
        return {"source": "hpo", "params": dict(hpo_best_params)}
    if file_loaded_params:
        return {"source": "file", "params": dict(file_loaded_params)}
    return {"source": "fallback", "params": {}}


# ----------------------------------------------------------------------------
# MLflow logging wrapper
# ----------------------------------------------------------------------------

def _scalarize(value: Any) -> Any:
    """Coerce values into a form MLflow can accept as a tag (string) or metric."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return repr(value)
        return value
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return str(value)


def log_training_run(
    *,
    round_id: str,
    hypothesis_id: str,
    fast_mode: bool,
    provenance: Mapping[str, Any],
    fixed_params: Mapping[str, Any],
    metrics_validation: Optional[Mapping[str, float]] = None,
    metrics_test: Optional[Mapping[str, float]] = None,
    calibration_by_regime: Optional[Mapping[str, Any]] = None,
    artifact_paths: Optional[list[Path | str]] = None,
    extra_tags: Optional[Mapping[str, Any]] = None,
    data_start: Optional[str] = None,
    data_end: Optional[str] = None,
):
    """Open an MLflow run, log the standard tag/param/metric/artifact set, return run.

    No-ops gracefully (returns a null context) if `mlflow` is not importable so
    the notebook still runs. Round-3 only adds optional logging; round-4+ may
    promote MLflow to a hard requirement once the loop is exercised end-to-end.
    """
    try:
        import mlflow  # noqa: F401

        from . import mlflow_utils  # local import to keep this module cheap
    except ImportError:
        # Hard fall-through: training continues without MLflow logging.
        return nullcontext()

    return _LoggedRun(
        round_id=round_id,
        hypothesis_id=hypothesis_id,
        fast_mode=fast_mode,
        provenance=provenance,
        fixed_params=fixed_params,
        metrics_validation=metrics_validation,
        metrics_test=metrics_test,
        calibration_by_regime=calibration_by_regime,
        artifact_paths=artifact_paths,
        extra_tags=extra_tags,
        data_start=data_start,
        data_end=data_end,
    )


class _LoggedRun:
    """Context manager: opens MLflow run, logs once, closes."""

    def __init__(self, **kw: Any) -> None:
        self.kw = kw
        self._cm = None
        self._run = None

    def __enter__(self):
        from . import mlflow_utils
        kw = self.kw

        # Build extra tags with provenance source so it's queryable in MLflow.
        prov = kw.get("provenance") or {}
        tags = {"param_source": _scalarize(prov.get("source", "unknown"))}
        if kw.get("extra_tags"):
            for k, v in kw["extra_tags"].items():
                tags[str(k)] = _scalarize(v)

        self._cm = mlflow_utils.round_run(
            round_id=str(kw.get("round_id")),
            hypothesis_id=str(kw.get("hypothesis_id")),
            fast_mode=bool(kw.get("fast_mode", False)),
            data_start=kw.get("data_start"),
            data_end=kw.get("data_end"),
            extra_tags=tags,
        )
        self._run = self._cm.__enter__()

        import mlflow

        # Params: fixed + chosen.
        for k, v in (kw.get("fixed_params") or {}).items():
            mlflow.log_param(f"fixed.{k}", _scalarize(v))
        for k, v in (prov.get("params") or {}).items():
            mlflow.log_param(f"chosen.{k}", _scalarize(v))

        # Metrics: validation + test.
        if kw.get("metrics_validation"):
            mlflow_utils.log_metrics_dict(kw["metrics_validation"], prefix="val.")
        if kw.get("metrics_test"):
            mlflow_utils.log_metrics_dict(kw["metrics_test"], prefix="test.")

        # Calibration by regime: log scalars in a flat namespace.
        cal = kw.get("calibration_by_regime") or {}
        for split_name, regimes in cal.items():
            if not isinstance(regimes, Mapping):
                continue
            for regime, stats in regimes.items():
                if not isinstance(stats, Mapping):
                    continue
                for stat_name, val in stats.items():
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(v):
                        mlflow.log_metric(
                            f"{split_name}.regime.{regime}.{stat_name}", v
                        )

        # Artifacts.
        for p in kw.get("artifact_paths") or []:
            mlflow_utils.log_artifact_safely(p)

        return self._run

    def __exit__(self, exc_type, exc, tb):
        if self._cm is not None:
            return self._cm.__exit__(exc_type, exc, tb)
        return False


# ----------------------------------------------------------------------------
# Predictions -> metric block
# ----------------------------------------------------------------------------

def metric_block_from_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    base_rate: Optional[float] = None,
) -> dict[str, float]:
    """Standard metric block used by the notebook + log_training_run."""
    from . import utils as bc_utils

    metrics = bc_utils.compute_all_metrics(np.asarray(y_true), np.asarray(y_pred_proba))
    n = int(len(y_true))
    metrics["n_samples"] = n
    metrics["base_rate"] = float(base_rate) if base_rate is not None else float(np.mean(y_true))
    return metrics
