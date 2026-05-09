"""MLflow tracking helpers for the barrier classifier loop.

The autonomous loop logs every training run to a local MLflow store
(`experiments/mlruns/`). Each run is tagged with the round number,
hypothesis ID, branch, git SHA, FAST_MODE flag, and data-slice timestamps
so the LEDGER can reference runs unambiguously.

Tracking URI default: `sqlite:///<repo_root>/experiments/mlruns/mlflow.db`.
Override via env var `BARRIER_MLFLOW_URI`.

This is a thin wrapper that doesn't add abstraction over MLflow itself —
it just standardises the tag set and the experiment-naming convention.
Round 1 wires this into 03_model_training.ipynb and scripts/run_round.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'experiments' / 'mlruns' / 'mlflow.db').as_posix()}"


def get_tracking_uri() -> str:
    return os.environ.get("BARRIER_MLFLOW_URI", DEFAULT_TRACKING_URI)


def _git_sha(short: bool = True) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@contextmanager
def round_run(
    *,
    round_id: str,
    hypothesis_id: str,
    fast_mode: bool,
    data_start: str | None = None,
    data_end: str | None = None,
    extra_tags: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open an MLflow run with the loop's standard tag set.

    Usage
    -----
    >>> with round_run(round_id="001", hypothesis_id="H-010",
    ...                fast_mode=True) as run:
    ...     mlflow.log_param("learning_rate", 0.01)
    ...     mlflow.log_metric("brier", 0.075)

    The experiment is named `barrier_round_{round_id}` automatically; this
    keeps the MLflow UI organized chronologically.

    Parameters
    ----------
    round_id : str
        Three-digit round identifier (e.g. "001").
    hypothesis_id : str
        Hypothesis from the BACKLOG (e.g. "H-010").
    fast_mode : bool
        Whether the run used the FAST_MODE data slice.
    data_start, data_end : str, optional
        ISO-formatted timestamps describing the data slice used.
    extra_tags : Mapping[str, Any], optional
        Additional tags to attach to the run.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow not installed. Add `mlflow>=2.10` to requirements.txt."
        ) from exc

    mlflow.set_tracking_uri(get_tracking_uri())
    experiment_name = f"barrier_round_{round_id}"
    mlflow.set_experiment(experiment_name)

    tags: dict[str, Any] = {
        "round_id": round_id,
        "hypothesis_id": hypothesis_id,
        "fast_mode": str(bool(fast_mode)).lower(),
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
    }
    if data_start:
        tags["data_start"] = data_start
    if data_end:
        tags["data_end"] = data_end
    if extra_tags:
        tags.update({str(k): v for k, v in extra_tags.items()})

    with mlflow.start_run(run_name=hypothesis_id, tags=tags) as run:
        yield run


def log_metrics_dict(metrics: Mapping[str, float], *, prefix: str = "") -> None:
    """Log a flat metrics dict, with optional prefix and finite-only filter."""
    import math

    import mlflow

    for name, value in metrics.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = f"{prefix}{name}" if prefix else name
        mlflow.log_metric(key, value)


def log_artifact_safely(local_path: str | Path) -> None:
    """Log an artifact only if it exists; useful for plot helpers that may
    not write a file in single-class-test slices."""
    import mlflow

    p = Path(local_path)
    if p.exists():
        mlflow.log_artifact(str(p))
