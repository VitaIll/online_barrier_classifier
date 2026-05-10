"""ExperimentProtocol — the SINGLE protocol for running experiments.

There is exactly one way to run an experiment in this repo:

    from wagie.experiments import ExperimentSpec, ExperimentProtocol
    spec = ExperimentSpec.from_yaml("experiments/baseline.yaml")
    result = ExperimentProtocol().run(spec)

Or via CLI:

    wagie experiment run experiments/baseline.yaml

Stages (one tick — backtest):
    1. Resolve out_dir, write spec snapshot
    2. Build pipeline + engine via wagie.run.run()
    3. Run engine event loop → EngineResult
    4. MetricsBattery.compute → MetricsReport
    5. ChartBattery.render_all → png paths
    6. Report.render → report.md
    7. Persist metrics.json + state hash

When ``spec.cv`` is set, the protocol delegates to ``wagie.cv.cross_validation``
and aggregates per-fold metrics into the same MetricsReport / Report shape.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from wagie.charts import ChartBattery
from wagie.engine import EngineResult
from wagie.metrics import MetricsBattery
from wagie.reporting import Report
from wagie.run import run as run_backtest

from .result import ExperimentResult
from .spec import ExperimentSpec


logger = logging.getLogger(__name__)


def _make_run_id(spec: ExperimentSpec) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{ts}_{spec.name}_{spec.hash()[:8]}"


@dataclass
class ExperimentProtocol:
    """The ONE protocol. All experiments flow through here.

    ``metrics_battery``, ``chart_battery``, ``report`` are injectable for
    tests and bespoke users; defaults are the canonical shipped batteries.
    """

    metrics_battery: MetricsBattery = None
    chart_battery: ChartBattery = None
    report: Report = None

    def __post_init__(self):
        if self.metrics_battery is None:
            self.metrics_battery = MetricsBattery()
        if self.chart_battery is None:
            self.chart_battery = ChartBattery()
        if self.report is None:
            self.report = Report()

    # -----------------------------------------------------------------
    # Public entry point — dispatches on spec
    # -----------------------------------------------------------------

    def run(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
    ) -> ExperimentResult:
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
        out_dir = Path(spec.artifacts.out_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
        logger.info(f"experiment[backtest] run_id={run_id} out_dir={out_dir}")

        feature_builder, base_bar, regime = _build_features(spec)
        self.metrics_battery.m_minutes = spec.wagie.data.m_minutes
        engine_result: EngineResult = run_backtest(
            spec.wagie,
            feature_builder=feature_builder,
            base_bar=base_bar,
            regime_feature=regime,
        )

        metrics_report = self.metrics_battery.compute(engine_result)
        metrics_dict = metrics_report.to_dict()
        (out_dir / "metrics.json").write_text(metrics_report.to_json(), encoding="utf-8")

        chart_paths: dict[str, Path] = {}
        if spec.charts.enable:
            self.chart_battery.n_calibration_bins = spec.charts.n_calibration_bins
            chart_paths = self.chart_battery.render_all(
                engine_result, out_dir / "charts",
            )

        report_path: Optional[Path] = None
        if spec.report.enable:
            self.report.title = spec.report.title or f"experiment: {spec.name}"
            report_path = self.report.render(
                spec_dict=spec.model_dump(mode="json"),
                metrics=metrics_dict,
                chart_paths=chart_paths,
                out_path=out_dir / "report.md",
                run_id=run_id,
            )

        if spec.artifacts.save_state:
            state_dir = out_dir / "state"
            state_dir.mkdir(exist_ok=True)
            (state_dir / "pipeline_state_hash.txt").write_text(
                engine_result.pipeline_state_hash.hex(), encoding="utf-8",
            )

        result = ExperimentResult(
            run_id=run_id, out_dir=out_dir,
            spec_path=Path(spec_path) if spec_path else out_dir / "spec.yaml",
            engine_result=engine_result, metrics=metrics_dict,
            chart_paths=chart_paths, report_path=report_path,
            spec_hash=spec.hash(),
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
        out_dir = Path(spec.artifacts.out_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
        logger.info(f"experiment[cv] run_id={run_id} out_dir={out_dir}")

        cv = spec.cv
        cv_result = cross_validation(
            spec.wagie,
            n_folds=cv.n_folds,
            n_test_folds=cv.n_test_folds,
            purged_size=cv.purged_size,
            embargo_size=cv.embargo_size,
        )

        metrics_dict = _cv_to_metrics_dict(cv_result, m_minutes=spec.wagie.data.m_minutes)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics_dict, indent=2, default=str), encoding="utf-8",
        )

        # Charts: CV gets its own minimal panel (per-fold sharpe bars + PBO marker)
        chart_paths: dict[str, Path] = {}
        if spec.charts.enable:
            chart_paths = self._render_cv_charts(cv_result, out_dir / "charts")

        report_path: Optional[Path] = None
        if spec.report.enable:
            self.report.title = spec.report.title or f"cv: {spec.name}"
            report_path = self.report.render(
                spec_dict=spec.model_dump(mode="json"),
                metrics=metrics_dict, chart_paths=chart_paths,
                out_path=out_dir / "report.md", run_id=run_id,
            )

        # CV doesn't have a single EngineResult; build a stub so ExperimentResult
        # remains uniform.
        from wagie.engine import EngineResult as _ER
        from wagie.io.brokers import BrokerLedger
        from wagie.core.portfolio import Portfolio
        stub_engine = _ER(
            ledger=BrokerLedger(fills=[], n_open_at_finalize=0, config={}),
            n_decisions=int(sum(r.get("n_trades", 0) for r in cv_result.per_fold)),
            n_filled=0, n_skipped_warmup=0,
            n_actions_approved=0, n_actions_rejected=0,
            pipeline_state_hash=b"\x00" * 32,
            final_portfolio=Portfolio(),
        )
        return ExperimentResult(
            run_id=run_id, out_dir=out_dir,
            spec_path=Path(spec_path) if spec_path else out_dir / "spec.yaml",
            engine_result=stub_engine, metrics=metrics_dict,
            chart_paths=chart_paths, report_path=report_path,
            spec_hash=spec.hash(),
        )

    def _render_cv_charts(self, cv_result, out_dir: Path) -> dict[str, Path]:
        import matplotlib.pyplot as plt
        from wagie.charts.theme import PALETTE, apply_theme, figsize

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}

        # Per-fold Sharpe bar chart
        apply_theme(plt)
        fig, ax = plt.subplots(figsize=figsize("wide"))
        if cv_result.per_fold:
            xs = [r["fold"] for r in cv_result.per_fold]
            sharpes = [r.get("sharpe", 0.0) for r in cv_result.per_fold]
            ax.bar(xs, sharpes, color=PALETTE["primary"], alpha=0.85)
            ax.axhline(0, color=PALETTE["muted"], linewidth=0.6)
            ax.set_xlabel("fold")
            ax.set_ylabel("Sharpe (annualized)")
            ax.set_title(f"per-fold Sharpe (PBO={cv_result.pbo:.2f})")
        else:
            ax.text(0.5, 0.5, "no folds", ha="center", va="center")
        path = out_dir / "01_per_fold_sharpe.png"
        fig.savefig(path)
        plt.close(fig)
        out["per_fold_sharpe"] = path
        return out


def _cv_to_metrics_dict(cv_result, *, m_minutes: int) -> dict:
    """Aggregate CV per-fold rows into the same shape as MetricsReport."""
    folds = cv_result.per_fold or []
    if not folds:
        agg_sharpe = 0.0
        agg_n_trades = 0
    else:
        n = max(len(folds), 1)
        agg_sharpe = sum(r.get("sharpe", 0.0) for r in folds) / n
        agg_n_trades = sum(r.get("n_trades", 0) for r in folds)
    return {
        "mode": "cv",
        "n_folds": cv_result.n_folds,
        "pbo": cv_result.pbo,
        "per_fold": folds,
        "trading": {
            "n_trades": agg_n_trades,
            "sharpe": agg_sharpe,
            "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_log_return": 0.0, "total_pct_return": 0.0,
            "probabilistic_sharpe": 0.0, "sortino": 0.0,
            "max_drawdown_log": 0.0, "cdar_5pct_log": 0.0,
        },
        "brier": 0.0, "ece": 0.0,
        "reliability": [], "calibration_by_regime": [],
        "roc_auc": 0.5, "pr_auc": 0.0,
        "n_decisions": agg_n_trades, "n_filled": 0,
        "n_actions_approved": 0, "n_actions_rejected": 0,
        "pipeline_state_hash": "",
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
