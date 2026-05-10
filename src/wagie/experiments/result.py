"""ExperimentResult — what the protocol returns after a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from wagie.engine import EngineResult


@dataclass
class ExperimentResult:
    """Everything an experiment produces. The CLI prints a one-line summary
    of this; tests read fields off it directly.

    Accept-gate fields (round-040 additive):
      - ``accepted`` is False when the protocol's ``_check_accept_gates``
        rejected the run (sample-size, predicted-effect-floor, or seed-band
        gates). ``blocked_reasons`` lists the failed gates.
      - ``blocked_path`` points to ``BLOCKED.md`` when gates fail (set by
        the protocol). ``report_path`` is None in that case.
    """

    run_id: str
    out_dir: Path
    spec_path: Path
    engine_result: EngineResult
    metrics: dict
    chart_paths: dict[str, Path] = field(default_factory=dict)
    report_path: Optional[Path] = None
    spec_hash: str = ""
    accepted: bool = True
    blocked_reasons: list[str] = field(default_factory=list)
    blocked_path: Optional[Path] = None

    @property
    def headline(self) -> str:
        t = self.metrics.get("trading", {}) or {}
        prefix = "" if self.accepted else "[BLOCKED] "
        return (
            f"{prefix}run_id={self.run_id} | n_trades={t.get('n_trades', 0)} "
            f"sharpe={t.get('sharpe', 0.0):+.3f} "
            f"brier={self.metrics.get('brier', 0.0):.5f} "
            f"ece={self.metrics.get('ece', 0.0):.5f}"
        )


__all__ = ["ExperimentResult"]
