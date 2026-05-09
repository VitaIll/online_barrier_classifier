"""ExperimentResult — what the protocol returns after a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from wagie.engine import EngineResult


@dataclass
class ExperimentResult:
    """Everything an experiment produces. The CLI prints a one-line summary
    of this; tests read fields off it directly."""

    run_id: str
    out_dir: Path
    spec_path: Path
    engine_result: EngineResult
    metrics: dict
    chart_paths: dict[str, Path] = field(default_factory=dict)
    report_path: Optional[Path] = None
    spec_hash: str = ""

    @property
    def headline(self) -> str:
        t = self.metrics.get("trading", {}) or {}
        return (
            f"run_id={self.run_id} | n_trades={t.get('n_trades', 0)} "
            f"sharpe={t.get('sharpe', 0.0):+.3f} "
            f"brier={self.metrics.get('brier', 0.0):.5f} "
            f"ece={self.metrics.get('ece', 0.0):.5f}"
        )


__all__ = ["ExperimentResult"]
