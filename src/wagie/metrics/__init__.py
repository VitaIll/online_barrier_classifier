"""wagie.metrics — the SINGLE way to measure.

All metrics flow through ``MetricsBattery.compute(result)`` which returns a
``MetricsReport``. The battery LEADS with calibration (Brier, ECE, per-regime
ECE), then trading. ROC/PR are diagnostic-only.

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
from .trading import TradingMetrics, compute_trading_metrics, deflated_sharpe


@dataclass
class MetricsReport:
    """The unified output of a measurement pass over an EngineResult.

    Calibration leads (Brier, ECE, per-regime). Trading is the bottom line.
    Classification ranking is diagnostic.
    """

    # Calibration (lead)
    brier: float = 0.0
    ece: float = 0.0
    reliability: list[dict] = field(default_factory=list)
    # Per-regime stratification of Brier/ECE: list of {regime_id, brier, ece, n}
    calibration_by_regime: list[dict] = field(default_factory=list)

    # Trading
    trading: dict = field(default_factory=dict)

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
        regime_ids: Optional[Sequence[int]] = None,
    ) -> MetricsReport:
        """Compute every metric in one pass.

        Args:
            result: EngineResult from the engine event loop. If it carries
                ``p_online_history`` / ``label_history`` from streaming
                calibration capture, those are used by default.
            y_true: Optional matured labels for calibration metrics. When
                None, falls back to ``result.label_history``.
            p_pred: Optional predicted probabilities aligned with y_true.
                When None, falls back to ``result.p_online_history``.
            regime_ids: Optional regime ids parallel to y_true / p_pred.
                Drives per-regime calibration stratification. When None,
                falls back to ``result.regime_history`` if available.
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

        # Streaming calibration capture from the engine — opt-in fallback when
        # the caller didn't supply explicit y_true / p_pred.
        if y_true is None and hasattr(result, "label_history"):
            y_true = result.label_history
        if p_pred is None and hasattr(result, "p_online_history"):
            p_pred = result.p_online_history
        if regime_ids is None and hasattr(result, "regime_history"):
            regime_ids = result.regime_history

        # Calibration (only if labels provided AND non-empty)
        if (
            y_true is not None and p_pred is not None
            and len(y_true) > 0 and len(p_pred) > 0
        ):
            rep.brier = brier_score(y_true, p_pred)
            rep.ece = expected_calibration_error(y_true, p_pred,
                                                  n_bins=self.n_calibration_bins)
            rep.reliability = [
                {"p_mean": b.p_mean, "y_mean": b.y_mean, "n": b.n}
                for b in reliability_bins(y_true, p_pred, n_bins=self.n_calibration_bins)
            ]
            rep.roc_auc = roc_auc(y_true, p_pred)
            rep.pr_auc = pr_auc(y_true, p_pred)

            if regime_ids is not None and len(regime_ids) == len(y_true):
                rep.calibration_by_regime = self._per_regime_calibration(
                    y_true, p_pred, regime_ids,
                )

        return rep

    def _per_regime_calibration(
        self,
        y_true: Sequence[int],
        p_pred: Sequence[float],
        regime_ids: Sequence[int],
    ) -> list[dict]:
        """Brier + ECE stratified by regime_id. Skips regimes with n<2."""
        buckets: dict[int, tuple[list[int], list[float]]] = {}
        for y, p, r in zip(y_true, p_pred, regime_ids):
            try:
                rid = int(r)
            except (TypeError, ValueError):
                continue
            ys, ps = buckets.setdefault(rid, ([], []))
            ys.append(int(y))
            ps.append(float(p))
        out: list[dict] = []
        for rid in sorted(buckets):
            ys, ps = buckets[rid]
            if len(ys) < 2:
                continue
            out.append({
                "regime_id": rid,
                "n": len(ys),
                "brier": brier_score(ys, ps),
                "ece": expected_calibration_error(ys, ps,
                                                   n_bins=self.n_calibration_bins),
            })
        return out


__all__ = [
    "MetricsBattery", "MetricsReport",
    # Re-exports for backward compatibility
    "TradingMetrics", "compute_trading_metrics", "deflated_sharpe",
    "Brier", "brier_score", "expected_calibration_error",
    "ReliabilityBin", "reliability_bins",
    "roc_auc", "pr_auc",
]
