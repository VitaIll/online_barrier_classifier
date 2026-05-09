"""ExperimentProtocol — the SINGLE protocol for running experiments.

There is exactly one way to run an experiment in this repo:

    from wagie.experiments import ExperimentSpec, ExperimentProtocol
    spec = ExperimentSpec.from_yaml("experiments/baseline.yaml")
    result = ExperimentProtocol().run(spec)

Or via CLI:

    python -m wagie experiment run experiments/baseline.yaml

The protocol orchestrates: data load → engine run → measure → chart → report
→ persist. It is the only object in the package that knows about all four
sub-batteries.
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

    Stages:
        1. Resolve out_dir, write spec snapshot
        2. Build pipeline + engine via wagie.run.run()
        3. Run engine event loop → EngineResult
        4. MetricsBattery.compute → MetricsReport (calibration + trading + coverage)
        5. ChartBattery.render_all → png paths
        6. Report.render → report.md
        7. Persist metrics.json + state hash
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

    def run(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
    ) -> ExperimentResult:
        run_id = _make_run_id(spec)
        out_dir = Path(spec.artifacts.out_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: snapshot the spec
        (out_dir / "spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
        logger.info(f"experiment run_id={run_id} out_dir={out_dir}")

        # Stage 2-3: build features (from FeaturesSpec) + run the engine
        feature_builder, base_bar, regime = _build_features(spec)
        self.metrics_battery.m_minutes = spec.wagie.data.m_minutes
        engine_result: EngineResult = run_backtest(
            spec.wagie,
            feature_builder=feature_builder,
            base_bar=base_bar,
            regime_feature=regime,
        )

        # Stage 4: measure
        metrics_report = self.metrics_battery.compute(engine_result)
        metrics_dict = metrics_report.to_dict()
        (out_dir / "metrics.json").write_text(metrics_report.to_json(), encoding="utf-8")

        # Stage 5: chart battery
        chart_paths: dict[str, Path] = {}
        if spec.charts.enable:
            self.chart_battery.n_calibration_bins = spec.charts.n_calibration_bins
            chart_paths = self.chart_battery.render_all(
                engine_result, out_dir / "charts",
            )

        # Stage 6: report
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

        # Stage 7: state hash + brief json
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
