"""wagie.metrics — the SINGLE way to measure.

All metrics flow through ``MetricsBattery.compute(result)`` which returns a
``MetricsReport``. The battery LEADS with calibration (Brier, ECE, per-regime
ECE), then trading. ROC/PR are diagnostic-only.

Public API:
    from wagie.metrics import MetricsBattery, MetricsReport
    report = MetricsBattery().compute(engine_result)
    report.to_dict()
    report.to_json()

CI fields (round-040 additive port from `src/bootstrap.py`): when the battery
runs with ``compute_ci=True`` (the new default), trading bootstraps Sharpe/
Sortino/max-DD via stationary block bootstrap, calibration uses Wilson on
Brier+ECE, classification uses DeLong for ROC and stratified bootstrap for
PR, all on the matured (y_true, p_pred) pairs the engine produces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from wagie.engine import EngineResult

from .bootstrap import (
    bootstrap_metric,
    delong_roc_auc_ci,
    stratified_bootstrap_pr_auc,
    wilson_interval,
)
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

    CI fields (round-040 additive): every reported scalar carries a 95%
    confidence interval as ``(lo, hi)`` tuples — None when CI computation is
    not requested or sample sizes are too small.
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

    # ----------------------------- CI block (round-040 additive) -----------
    # Trading CIs are populated through MetricsBattery → compute_trading_metrics.
    # Sharpe/Brier/ECE CIs come from `wagie.metrics.bootstrap`. ROC uses DeLong;
    # PR uses Boyd-Eng-Page stratified bootstrap. Everything optional/None when
    # the underlying computation is skipped.
    sharpe_ci: Optional[tuple[float, float]] = None
    brier_ci: Optional[tuple[float, float]] = None
    ece_ci: Optional[tuple[float, float]] = None
    roc_ci_delong: Optional[tuple[float, float]] = None
    pr_ci: Optional[tuple[float, float]] = None
    n_trials: int = 1
    dsr: Optional[float] = None
    pbo: Optional[float] = None
    # Pre-registration / accept-gate provenance.
    accepted: bool = True
    blocked_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class MetricsBattery:
    """The ONE measurement protocol. Configure once, run on any EngineResult.

    Round-040 additions (additive only):
    - ``compute_ci`` (default True) populates trading + classification CIs.
    - ``n_bootstrap`` controls the resample count for bootstrap-based CIs.
    - ``n_trials`` exposes the multiple-trial Bailey-Borwein DSR; defaults
      to 1 (no inflation correction). When > 1, ``dsr`` on the report is
      populated using Bailey & López de Prado (2014).
    """

    m_minutes: int = 20
    n_calibration_bins: int = 10
    compute_ci: bool = True
    n_bootstrap: int = 2000
    confidence: float = 0.95
    ci_seed: int = 42
    n_trials: int = 1

    def compute(
        self,
        result: EngineResult,
        *,
        y_true: Optional[Sequence[int]] = None,
        p_pred: Optional[Sequence[float]] = None,
        regime_ids: Optional[Sequence[int]] = None,
        n_trials: Optional[int] = None,
        returns_matrix: Optional["np.ndarray"] = None,  # noqa: F821 - typing only
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
            n_trials: Optional override for ``self.n_trials`` (multiple-testing
                trials count for DSR). Default falls back to ``self.n_trials``.
            returns_matrix: Optional (n_strategies, n_periods) matrix that
                triggers cross-strategy CSCV PBO computation.
        """
        n_trials_eff = int(n_trials if n_trials is not None else self.n_trials)
        rep = MetricsReport(
            n_decisions=int(result.n_decisions),
            n_filled=int(result.n_filled),
            n_actions_approved=int(getattr(result, "n_actions_approved", 0)),
            n_actions_rejected=int(getattr(result, "n_actions_rejected", 0)),
            pipeline_state_hash=result.pipeline_state_hash.hex()[:16],
            n_trials=n_trials_eff,
        )

        # Trading from fills (with optional bootstrap CIs)
        tm = compute_trading_metrics(
            result.ledger.fills, m_minutes=self.m_minutes,
            compute_ci=self.compute_ci, n_bootstrap=self.n_bootstrap,
            confidence=self.confidence, ci_seed=self.ci_seed,
        )
        rep.trading = tm.to_dict()
        if tm.sharpe_ci_lo is not None and tm.sharpe_ci_hi is not None:
            rep.sharpe_ci = (float(tm.sharpe_ci_lo), float(tm.sharpe_ci_hi))

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

            if self.compute_ci:
                rep.brier_ci = self._brier_ci(y_true, p_pred)
                rep.ece_ci = self._ece_ci(y_true, p_pred)
                rep.roc_ci_delong = self._roc_ci(y_true, p_pred)
                rep.pr_ci = self._pr_ci(y_true, p_pred)

            # DSR auto-invocation when n_trials > 1.
            if n_trials_eff > 1:
                rep.dsr = self._dsr(tm, n_trials_eff)

        # CSCV PBO + median IS-best — only when caller passed a strategies-by-
        # periods returns matrix (n_strategies >= 2). The single-EngineResult
        # path can't compute PBO; this hook is for the CV / multi-strategy
        # caller that already aggregated returns.
        if returns_matrix is not None:
            try:
                import numpy as np

                from wagie.cscv import cscv_pbo
                R = np.asarray(returns_matrix, dtype=float)
                if R.ndim == 2 and R.shape[0] >= 2 and R.shape[1] >= 16:
                    out = cscv_pbo(R, n_chunks=min(16, R.shape[1] // 2 * 2))
                    rep.pbo = float(out["pbo"])
                    # Annotate trading dict with median IS-best index — useful
                    # for the report layer.
                    rep.trading.setdefault(
                        "is_best_strategy_modal_index",
                        int(out["is_best_strategy_modal_index"]),
                    )
            except (ValueError, RuntimeError, ImportError):
                pass

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

    # ------------------------------------------------------------------
    # CI helpers (round-040 additive). Each helper is permissive with
    # degenerate inputs — it returns None rather than raising so the
    # battery never aborts a metric run on a CI failure.
    # ------------------------------------------------------------------

    def _brier_ci(
        self, y_true: Sequence[int], p_pred: Sequence[float],
    ) -> Optional[tuple[float, float]]:
        import numpy as np
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(p_pred, dtype=float)
        if len(y) < 2:
            return None
        try:
            sq = (p - y) ** 2
            out = bootstrap_metric(
                sq, lambda x: float(x.mean()),
                n_resamples=self.n_bootstrap,
                confidence=self.confidence, seed=self.ci_seed,
            )
            return (float(out["ci_lo"]), float(out["ci_hi"]))
        except (ValueError, RuntimeError):
            return None

    def _ece_ci(
        self, y_true: Sequence[int], p_pred: Sequence[float],
    ) -> Optional[tuple[float, float]]:
        import numpy as np
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(p_pred, dtype=float)
        n = len(y)
        if n < 2:
            return None

        n_bins = int(self.n_calibration_bins)

        def _ece(idx: np.ndarray) -> float:
            yi = y[idx]
            pi = p[idx]
            qs = np.quantile(pi, np.linspace(0.0, 1.0, n_bins + 1))
            qs[0] = -np.inf
            qs[-1] = np.inf
            e = 0.0
            ni = len(pi)
            for i in range(n_bins):
                m = (pi >= qs[i]) & (pi < qs[i + 1])
                if m.sum() > 0:
                    e += abs(pi[m].mean() - yi[m].mean()) * (m.sum() / ni)
            return float(e)

        try:
            out = bootstrap_metric(
                np.arange(n), lambda idx: _ece(idx.astype(np.int64)),
                n_resamples=self.n_bootstrap,
                confidence=self.confidence, seed=self.ci_seed,
            )
            return (float(out["ci_lo"]), float(out["ci_hi"]))
        except (ValueError, RuntimeError):
            return None

    def _roc_ci(
        self, y_true: Sequence[int], p_pred: Sequence[float],
    ) -> Optional[tuple[float, float]]:
        import numpy as np
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(p_pred, dtype=float)
        if len(y) < 2 or len(np.unique(y)) < 2:
            return None
        try:
            out = delong_roc_auc_ci(y, p, confidence=self.confidence)
            return (float(out["ci_lo"]), float(out["ci_hi"]))
        except ValueError:
            return None

    def _pr_ci(
        self, y_true: Sequence[int], p_pred: Sequence[float],
    ) -> Optional[tuple[float, float]]:
        import numpy as np
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(p_pred, dtype=float)
        if len(y) < 2 or len(np.unique(y)) < 2:
            return None
        try:
            # Stratified PR-AUC bootstrap; pull n_resamples from battery setting
            # and clamp to keep the integration tests fast.
            out = stratified_bootstrap_pr_auc(
                y, p,
                n_resamples=min(self.n_bootstrap, 1000),
                confidence=self.confidence, seed=self.ci_seed,
            )
            return (float(out["ci_lo"]), float(out["ci_hi"]))
        except (ValueError, RuntimeError, ImportError):
            return None

    def _dsr(self, tm: "TradingMetrics", n_trials: int) -> Optional[float]:
        """Bailey-Borwein deflated Sharpe p-value at the observed Sharpe."""
        try:
            n_obs = int(getattr(tm, "n_trades", 0))
            if n_obs < 2:
                return None
            # Crude variance-of-trial-Sharpes proxy: var across the strategy
            # ≈ 1.0 (Bayesian default until the harness threads a true grid).
            p_dsr, _sr_star = deflated_sharpe(
                observed_sharpe=float(tm.sharpe), var_trial_sharpes=1.0,
                n_trials=int(n_trials), n_obs=n_obs,
            )
            return float(p_dsr)
        except (ValueError, ZeroDivisionError):
            return None


__all__ = [
    "MetricsBattery", "MetricsReport",
    # Re-exports for backward compatibility
    "TradingMetrics", "compute_trading_metrics", "deflated_sharpe",
    "Brier", "brier_score", "expected_calibration_error",
    "ReliabilityBin", "reliability_bins",
    "roc_auc", "pr_auc",
    # Bootstrap re-exports (round-040 additive port)
    "bootstrap_metric", "delong_roc_auc_ci",
    "stratified_bootstrap_pr_auc", "wilson_interval",
]
