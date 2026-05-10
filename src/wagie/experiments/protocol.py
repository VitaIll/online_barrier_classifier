"""ExperimentProtocol — the SINGLE protocol for running experiments.

There is exactly one way to run an experiment in this repo:

    from wagie.experiments import ExperimentSpec, ExperimentProtocol
    spec = ExperimentSpec.from_yaml("experiments/baseline.yaml")
    result = ExperimentProtocol().run(spec)

Or via CLI:

    wagie experiment run experiments/baseline.yaml

Stages (one tick — backtest):
    1. Build pipeline + engine via wagie.run.run()
    2. Run engine event loop → EngineResult
    3. MetricsBattery.compute → MetricsReport
    4. Evaluate accept-gates → RunMeta(accepted=, blocked_reasons=)
    5. ReportRenderer.render → the SINGLE canonical report on disk

When ``spec.cv`` is set, the protocol delegates to ``wagie.cv.cross_validation``
and feeds the per-fold result + aggregated metrics dict into the renderer.

Side-experiment escape hatch: pass ``experiment_name="..."`` to
``ExperimentProtocol.run`` (or ``--experiment NAME`` to the CLI) and the
output is redirected to ``artifacts/experiments/<name>/`` with archiving
disabled (it's a one-off, no rotation).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from wagie.engine import EngineResult
from wagie.metrics import MetricsBattery
from wagie.reporting import ReportRenderer, RunMeta
from wagie.run import run as run_backtest

from .result import ExperimentResult
from .spec import ExperimentSpec


logger = logging.getLogger(__name__)


def _make_run_id(spec: ExperimentSpec) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{ts}_{spec.name}_{spec.hash()[:8]}"


# =============================================================================
# Accept-gate (round-040 additive)
# =============================================================================

def _check_accept_gates(metrics: dict, spec: ExperimentSpec) -> list[str]:
    """Evaluate the spec's pre-registration gates against ``metrics``.

    Returns a list of human-readable failure reasons. Empty list = PASS.
    Gates evaluated:
      1. Sample size: ``trading.n_trades`` >= ``spec.min_n_trades``.
      2. Predicted-effect floor: ``trading.sharpe`` >=
         ``spec.predicted_effect_min`` (when set). For backtest mode
         additionally requires the lower CI bound to clear zero when a
         CI was computed.
      3. Seed-band gap: when the metrics carry a ``sharpe_seed_band``
         (CV mode), the median fold Sharpe must clear the band's lo by
         at least ``spec.min_effect_vs_seed_band``.
    """
    reasons: list[str] = []
    trading = metrics.get("trading") or {}
    n_trades = int(trading.get("n_trades", 0) or 0)
    if n_trades < int(spec.min_n_trades):
        reasons.append(
            f"min_n_trades: n_trades={n_trades} < required={spec.min_n_trades}"
        )

    sharpe = float(trading.get("sharpe", 0.0) or 0.0)
    if spec.predicted_effect_min is not None:
        if sharpe < float(spec.predicted_effect_min):
            reasons.append(
                f"predicted_effect_min: sharpe={sharpe:+.4f} < "
                f"required={spec.predicted_effect_min:+.4f}"
            )
        # Also enforce CI lower bound > 0 when a Sharpe CI is present.
        sh_ci = metrics.get("sharpe_ci")
        if sh_ci and isinstance(sh_ci, (list, tuple)) and len(sh_ci) == 2:
            if float(sh_ci[0]) <= 0.0:
                reasons.append(
                    f"sharpe_ci_lo<=0: ci_lo={float(sh_ci[0]):+.4f} not > 0 "
                    "(CI contains zero — point estimate is not "
                    "statistically distinguishable from null)"
                )

    band = metrics.get("sharpe_seed_band")
    if band and isinstance(band, dict):
        med = float(band.get("med", 0.0))
        lo = float(band.get("lo", 0.0))
        gap = med - lo
        if gap < float(spec.min_effect_vs_seed_band):
            reasons.append(
                f"min_effect_vs_seed_band: med-lo gap={gap:+.4f} < "
                f"required={spec.min_effect_vs_seed_band:+.4f} "
                f"(seed-band median {med:+.4f}, lo {lo:+.4f})"
            )

    return reasons


@dataclass
class ExperimentProtocol:
    """The ONE protocol. All experiments flow through here.

    ``metrics_battery`` is injectable for tests and bespoke users; the
    default is the canonical shipped battery. Reporting is owned by
    :class:`wagie.reporting.ReportRenderer`; the protocol composes the
    inputs but does not own any HTML/markdown writers itself.
    """

    metrics_battery: MetricsBattery = None

    def __post_init__(self):
        if self.metrics_battery is None:
            self.metrics_battery = MetricsBattery()

    # -----------------------------------------------------------------
    # Public entry point — dispatches on spec
    # -----------------------------------------------------------------

    def run(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
        experiment_name: Optional[str] = None,
    ) -> ExperimentResult:
        """Run the experiment described by ``spec``.

        Parameters
        ----------
        spec
            Validated spec.
        spec_path
            Original YAML path (echoed back on the result).
        experiment_name
            When set, redirect output to
            ``artifacts/experiments/<experiment_name>/`` and disable
            archiving for this render. Used by the CLI's
            ``--experiment NAME`` flag for one-off side runs that must
            not touch the canonical report.
        """
        # Side-experiment escape hatch — clone of the spec with the
        # artifacts dir overridden so we never touch the canonical report.
        if experiment_name:
            new_spec = spec.model_copy(deep=True)
            new_spec.artifacts.out_dir = str(
                Path("artifacts/experiments") / experiment_name
            )
            new_spec.artifacts.enable_archive = False
            spec = new_spec

        if spec.cv is not None and spec.cv.enabled:
            return self._run_cv(spec, spec_path=spec_path)
        return self._run_backtest(spec, spec_path=spec_path)

    # -----------------------------------------------------------------
    # Backtest mode
    # -----------------------------------------------------------------

    def _run_backtest(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
    ) -> ExperimentResult:
        run_id = _make_run_id(spec)
        report_root = Path(spec.artifacts.out_dir)
        logger.info(
            f"experiment[backtest] run_id={run_id} report_root={report_root}"
        )

        feature_builder, base_bar, regime = _build_features(spec)
        self.metrics_battery.m_minutes = spec.wagie.data.m_minutes
        # Plumb pre-registration n_trials → DSR auto-invocation.
        self.metrics_battery.n_trials = int(spec.n_trials_for_dsr)
        engine_result: EngineResult = run_backtest(
            spec.wagie,
            feature_builder=feature_builder,
            base_bar=base_bar,
            regime_feature=regime,
        )

        metrics_report = self.metrics_battery.compute(engine_result)
        metrics_dict = metrics_report.to_dict()

        # Accept-gate: evaluate pre-registration gates BEFORE rendering so
        # the renderer can surface BLOCKED status in the banner.
        blocked_reasons = _check_accept_gates(metrics_dict, spec)
        accepted = not blocked_reasons
        if not accepted:
            metrics_dict["accepted"] = False
            metrics_dict["blocked_reasons"] = list(blocked_reasons)
            logger.warning(
                f"experiment[backtest] BLOCKED run_id={run_id} "
                f"reasons={blocked_reasons}"
            )

        report_path: Optional[Path] = None
        if spec.report.enable:
            run_meta = RunMeta(
                run_id=run_id,
                spec_name=spec.name,
                spec_hash=spec.hash(),
                mode="backtest",
                accepted=accepted,
                blocked_reasons=list(blocked_reasons),
                state_hash=engine_result.pipeline_state_hash.hex()
                if engine_result.pipeline_state_hash else "",
                use_plotly=bool(spec.report.use_plotly),
            )
            renderer = ReportRenderer(
                report_root=report_root,
                archive_keep=int(spec.artifacts.archive_keep),
                enable_archive=bool(spec.artifacts.enable_archive),
            )
            report_path = renderer.render(
                engine_result=engine_result,
                metrics=metrics_dict,
                spec_dict=spec.model_dump(mode="json"),
                run_meta=run_meta,
                use_plotly=bool(spec.report.use_plotly),
                title=spec.report.title,
            )

        result = ExperimentResult(
            run_id=run_id, out_dir=report_root,
            spec_path=Path(spec_path) if spec_path else report_root / "spec.yaml",
            engine_result=engine_result, metrics=metrics_dict,
            chart_paths={}, report_path=report_path,
            spec_hash=spec.hash(),
            accepted=accepted, blocked_reasons=blocked_reasons,
            blocked_path=None,
        )
        logger.info(result.headline)
        return result

    # -----------------------------------------------------------------
    # CV mode
    # -----------------------------------------------------------------

    def _run_cv(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
    ) -> ExperimentResult:
        from wagie.cv import cross_validation

        run_id = _make_run_id(spec)
        report_root = Path(spec.artifacts.out_dir)
        logger.info(f"experiment[cv] run_id={run_id} report_root={report_root}")

        cv = spec.cv
        cv_result = cross_validation(
            spec.wagie,
            n_folds=cv.n_folds,
            n_test_folds=cv.n_test_folds,
            purged_size=cv.purged_size,
            embargo_size=cv.embargo_size,
        )

        metrics_dict = _cv_to_metrics_dict(
            cv_result, m_minutes=spec.wagie.data.m_minutes,
        )

        # Accept-gate (round-040 additive) — same semantics as backtest mode.
        blocked_reasons = _check_accept_gates(metrics_dict, spec)
        accepted = not blocked_reasons
        if not accepted:
            metrics_dict["accepted"] = False
            metrics_dict["blocked_reasons"] = list(blocked_reasons)
            logger.warning(
                f"experiment[cv] BLOCKED run_id={run_id} "
                f"reasons={blocked_reasons}"
            )

        report_path: Optional[Path] = None
        if spec.report.enable:
            run_meta = RunMeta(
                run_id=run_id,
                spec_name=spec.name,
                spec_hash=spec.hash(),
                mode="cv",
                accepted=accepted,
                blocked_reasons=list(blocked_reasons),
                state_hash="",
                use_plotly=bool(spec.report.use_plotly),
            )
            renderer = ReportRenderer(
                report_root=report_root,
                archive_keep=int(spec.artifacts.archive_keep),
                enable_archive=bool(spec.artifacts.enable_archive),
            )
            # CV mode passes ``cv_result`` directly to the renderer; the
            # section emitters know how to read it. No EngineResult stub.
            report_path = renderer.render(
                engine_result=None,
                metrics=metrics_dict,
                spec_dict=spec.model_dump(mode="json"),
                run_meta=run_meta,
                cv_result=cv_result,
                use_plotly=bool(spec.report.use_plotly),
                title=spec.report.title,
            )

        return ExperimentResult(
            run_id=run_id, out_dir=report_root,
            spec_path=Path(spec_path) if spec_path else report_root / "spec.yaml",
            engine_result=None, metrics=metrics_dict,
            chart_paths={}, report_path=report_path,
            spec_hash=spec.hash(),
            accepted=accepted, blocked_reasons=blocked_reasons,
            blocked_path=None,
        )


def _cv_to_metrics_dict(cv_result, *, m_minutes: int) -> dict:
    """Aggregate CV per-fold rows into the same shape as MetricsReport.

    Round-040 fix: previously hard-coded ``brier=0.0, ece=0.0`` even when
    folds contained real fills — that was the "CV silently writes zeros"
    audit finding. Now we:
      - Compute the cross-fold Sharpe **band** (5/50/95-pct of per-fold
        Sharpe) as a seed-band proxy.
      - Stationary-block bootstrap a single Sharpe CI on the concatenated
        per-fold PnL series (real number, not a mean of point estimates).
      - Mark Brier/ECE as ``None`` so downstream code cannot mistake the
        CV mode for one that actually produced calibration metrics. The
        engine in this repo doesn't capture (y_true, p_pred) pairs at the
        fill-level, so any CV calibration number would be a fabrication.
    """
    import numpy as np

    from wagie.metrics.bootstrap import bootstrap_sharpe

    folds = cv_result.per_fold or []
    if not folds:
        agg_sharpe = 0.0
        agg_n_trades = 0
        sharpe_seed_band = None
        sharpe_ci = None
    else:
        n_folds = max(len(folds), 1)
        agg_sharpe = sum(r.get("sharpe", 0.0) for r in folds) / n_folds
        agg_n_trades = sum(r.get("n_trades", 0) for r in folds)
        per_fold_sharpe = np.array([r.get("sharpe", 0.0) for r in folds],
                                   dtype=float)
        if len(per_fold_sharpe) >= 1:
            sharpe_seed_band = {
                "lo": float(np.quantile(per_fold_sharpe, 0.05)),
                "med": float(np.quantile(per_fold_sharpe, 0.50)),
                "hi": float(np.quantile(per_fold_sharpe, 0.95)),
                "n_folds": int(len(per_fold_sharpe)),
            }
        else:
            sharpe_seed_band = None

        # Stationary-block bootstrap on the concatenated per-fold-PnL proxy
        # (each per-fold row may carry total_log_return; treat per-fold totals
        # as a 1D series for a coarse CI). Folds < 2 → no CI.
        per_fold_returns = np.array(
            [r.get("total_log_return", 0.0) for r in folds], dtype=float,
        )
        if len(per_fold_returns) >= 2 and per_fold_returns.std() > 0:
            try:
                out = bootstrap_sharpe(per_fold_returns,
                                       n_resamples=2000, seed=42)
                sharpe_ci = (float(out["ci_lo"]), float(out["ci_hi"]))
            except (ValueError, RuntimeError):
                sharpe_ci = None
        else:
            sharpe_ci = None

    return {
        "mode": "cv",
        "n_folds": cv_result.n_folds,
        "pbo": cv_result.pbo,
        "per_fold": folds,
        "trading": {
            "n_trades": agg_n_trades,
            "sharpe": agg_sharpe,
            # Honest defaults — CV mode does NOT compute these per-trade fields.
            "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_log_return": 0.0, "total_pct_return": 0.0,
            "probabilistic_sharpe": 0.0, "sortino": 0.0,
            "max_drawdown_log": 0.0, "cdar_5pct_log": 0.0,
            "sharpe_ci_lo": (sharpe_ci[0] if sharpe_ci else None),
            "sharpe_ci_hi": (sharpe_ci[1] if sharpe_ci else None),
            "sortino_ci_lo": None, "sortino_ci_hi": None,
            "max_dd_ci_lo": None, "max_dd_ci_hi": None,
            "block_length": None, "n_bootstrap": None,
        },
        # Calibration: NOT computed in CV mode (engine doesn't capture
        # per-decision (y_true, p_pred)). Use None to signal absence rather
        # than fake zeros.
        "brier": None, "ece": None,
        "reliability": [], "calibration_by_regime": [],
        "roc_auc": None, "pr_auc": None,
        "n_decisions": agg_n_trades, "n_filled": 0,
        "n_actions_approved": 0, "n_actions_rejected": 0,
        "pipeline_state_hash": "",
        # CI / pre-registration block.
        "sharpe_ci": list(sharpe_ci) if sharpe_ci else None,
        "sharpe_seed_band": sharpe_seed_band,
        "brier_ci": None, "ece_ci": None,
        "roc_ci_delong": None, "pr_ci": None,
        "n_trials": 1, "dsr": None,
        "accepted": True, "blocked_reasons": [],
    }


def _build_features(spec: ExperimentSpec):
    """Resolve the features sub-spec to live (base_bar, builder, regime)."""
    from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
    from wagie.features.catalog import default_streaming_features

    base_bar = BaseBarFeatures()
    cat = default_streaming_features()
    if spec.features.n_features is not None and spec.features.n_features > 0:
        cat = cat[: int(spec.features.n_features)]
    elif spec.features.catalog == "minimal":
        cat = cat[:5]
    builder = FeatureBuilder(cat)

    cuts = RegimeCuts(
        feature=spec.features.regime_feature,
        edges=spec.features.regime_edges,
        labels=spec.features.regime_labels,
    )
    return builder, base_bar, RegimeFeature(cuts)


__all__ = ["ExperimentProtocol"]
