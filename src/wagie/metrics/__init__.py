"""wagie.metrics — the SINGLE way to measure.

All metrics flow through `MetricsBattery.compute(result)` which returns a
`MetricsReport`. The battery leads with calibration (Brier, ECE), then trading,
then conformal coverage. ROC/PR are diagnostic-only.

Public API:
    from wagie.metrics import MetricsBattery, MetricsReport
    report = MetricsBattery().compute(engine_result)
    report.to_dict()
    report.to_json()
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from wagie.engine import EngineResult

from .calibration import (
    Brier, brier_score, expected_calibration_error,
    ReliabilityBin, reliability_bins,
)
from .classification import roc_auc, pr_auc
from .coverage import CoverageStats, empirical_coverage
from .trading import TradingMetrics, compute_trading_metrics, deflated_sharpe


@dataclass
class MetricsReport:
    """The unified output of a measurement pass over an EngineResult.

    Calibration leads (Brier, ECE). Trading is the bottom line. Coverage shows
    the conformal layer is doing its job. Classification ranking is diagnostic.
    """

    # Calibration (lead)
    brier: float = 0.0
    ece: float = 0.0
    reliability: list[dict] = field(default_factory=list)

    # Trading
    trading: dict = field(default_factory=dict)

    # Coverage (per α)
    coverage: list[dict] = field(default_factory=list)

    # Classification (diagnostic)
    roc_auc: float = 0.5
    pr_auc: float = 0.0

    # Provenance
    n_decisions: int = 0
    n_filled: int = 0
    n_actions_approved: int = 0
    n_actions_rejected: int = 0
    pipeline_state_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class MetricsBattery:
    """The ONE measurement protocol. Configure once, run on any EngineResult."""

    m_minutes: int = 20
    n_calibration_bins: int = 10

    def compute(
        self,
        result: EngineResult,
        *,
        y_true: Optional[Sequence[int]] = None,
        p_pred: Optional[Sequence[float]] = None,
        in_set_by_alpha: Optional[dict[float, Sequence[int]]] = None,
    ) -> MetricsReport:
        """Compute every metric in one pass.

        Args:
            result: EngineResult from the engine event loop.
            y_true: Optional matured labels for calibration metrics.
            p_pred: Optional predicted probabilities aligned with y_true.
            in_set_by_alpha: Optional in_set indicator per α level for coverage.
        """
        rep = MetricsReport(
            n_decisions=int(result.n_decisions),
            n_filled=int(result.n_filled),
            n_actions_approved=int(getattr(result, "n_actions_approved", 0)),
            n_actions_rejected=int(getattr(result, "n_actions_rejected", 0)),
            pipeline_state_hash=result.pipeline_state_hash.hex()[:16],
        )

        # Trading from fills
        tm = compute_trading_metrics(result.ledger.fills, m_minutes=self.m_minutes)
        rep.trading = tm.to_dict()

        # Calibration (only if labels provided)
        if y_true is not None and p_pred is not None and len(y_true) > 0:
            rep.brier = brier_score(y_true, p_pred)
            rep.ece = expected_calibration_error(y_true, p_pred,
                                                  n_bins=self.n_calibration_bins)
            rep.reliability = [
                {"p_mean": b.p_mean, "y_mean": b.y_mean, "n": b.n}
                for b in reliability_bins(y_true, p_pred, n_bins=self.n_calibration_bins)
            ]
            rep.roc_auc = roc_auc(y_true, p_pred)
            rep.pr_auc = pr_auc(y_true, p_pred)

        # Coverage per α
        if in_set_by_alpha and y_true is not None:
            rep.coverage = [
                {"alpha": cs.alpha, "empirical": cs.empirical_coverage,
                 "target": cs.target_coverage, "gap": cs.gap, "n": cs.n}
                for cs in (empirical_coverage(in_set, y_true, alpha)
                           for alpha, in_set in in_set_by_alpha.items())
            ]

        return rep


__all__ = [
    "MetricsBattery", "MetricsReport",
    # Re-exports for backward compatibility
    "TradingMetrics", "compute_trading_metrics", "deflated_sharpe",
    "Brier", "brier_score", "expected_calibration_error",
    "ReliabilityBin", "reliability_bins",
    "CoverageStats", "empirical_coverage",
    "roc_auc", "pr_auc",
]
