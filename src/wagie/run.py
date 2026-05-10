"""Engine builder + entry point used by ExperimentProtocol and wagie.cv.

Wires the canonical pipeline (BaseBarFeatures | FeatureBuilder | Regime?
| CatBoost? | ARF | LabelBuffer | Strategy) and runs the engine.
End-users go through `wagie experiment run <spec.yaml>`; this module is the
implementation detail underneath.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from wagie.config import WagieConfig
from wagie.core.numeric import LogReturn
from wagie.core.time import Duration
from wagie.engine import Engine, EngineResult
from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeFeature
from wagie.io.brokers import SimBroker
from wagie.io.clock import TestClock
from wagie.io.sources import ParquetReplaySource
from wagie.pipeline import (
    FrozenCatBoostPredictor,
    LabelBuffer,
    OnlineARFCorrector,
    Pipeline,
)
from wagie.risk import RiskEngine
from wagie.strategy import build_strategy


logger = logging.getLogger(__name__)


def build_pipeline(
    cfg: WagieConfig,
    feature_builder: Optional[FeatureBuilder] = None,
    base_bar: Optional[BaseBarFeatures] = None,
    regime_feature: Optional[RegimeFeature] = None,
) -> tuple[Pipeline, LabelBuffer]:
    """Assemble the canonical sealed Pipeline from a nested WagieConfig."""
    selected_features = None
    if cfg.model.selected_features_path:
        sel = json.loads(Path(cfg.model.selected_features_path).read_text())
        selected_features = sel.get("features") if isinstance(sel, dict) else sel

    stages: list = []
    if base_bar is not None:
        stages.append(base_bar)
    if feature_builder is not None:
        stages.append(feature_builder)
    if regime_feature is not None:
        stages.append(regime_feature)

    if cfg.model.catboost_path:
        stages.append(FrozenCatBoostPredictor(
            model_path=cfg.model.catboost_path,
            feature_columns=None,
            ensemble_n=cfg.model.catboost_ensemble_n,
        ))

    stages.append(OnlineARFCorrector(
        n_models=cfg.model.arf.n_models,
        max_features=cfg.model.arf.max_features,
        lambda_value=cfg.model.arf.lambda_value,
        seed=cfg.model.arf.seed,
        selected_features=selected_features,
    ))

    label_buf = LabelBuffer(alpha_label=cfg.broker.label_alpha)
    stages.append(label_buf)

    strat_kwargs: dict = {}
    if cfg.strategy.tau is not None:
        strat_kwargs["tau"] = cfg.strategy.tau
    if cfg.strategy.k is not None:
        strat_kwargs["k"] = cfg.strategy.k
    strat_kwargs.update(cfg.strategy.extra)
    strat_kwargs.setdefault("take_profit", LogReturn(cfg.broker.take_profit_log))
    strat_kwargs.setdefault("stop_loss", LogReturn(cfg.broker.stop_loss_log))
    strat_kwargs.setdefault("expiry", Duration.from_minutes(cfg.broker.expiry_minutes))
    stages.append(build_strategy(cfg.strategy.kind, **strat_kwargs))

    return Pipeline(stages), label_buf


def run(
    cfg: WagieConfig,
    feature_builder: Optional[FeatureBuilder] = None,
    base_bar: Optional[BaseBarFeatures] = None,
    regime_feature: Optional[RegimeFeature] = None,
    risk_engine: Optional[RiskEngine] = None,
) -> EngineResult:
    """Backtest run. ParquetReplaySource + SimBroker + TestClock + RiskEngine."""
    source = ParquetReplaySource(
        parquet_path=cfg.data.parquet_path,
        m_minutes=cfg.data.m_minutes,
        start_ts_ms=cfg.data.start_ts_ms,
        end_ts_ms=cfg.data.end_ts_ms,
    )
    broker = SimBroker(
        m_minutes=cfg.data.m_minutes,
        cost=cfg.broker.cost_bps * 1e-4,
        execution_latency_minutes=cfg.broker.execution_latency_minutes,
        inventory_cap=cfg.broker.inventory_cap,
    )
    pipeline, label_buf = build_pipeline(cfg, feature_builder, base_bar, regime_feature)

    engine = Engine(
        source=source, pipeline=pipeline, broker=broker, clock=TestClock(0),
        label_buffer=label_buf,
        risk_engine=risk_engine or RiskEngine.default(),
        warmup_samples=cfg.runtime.warmup_samples,
        capture_audit=cfg.runtime.capture_audit,
    )
    logger.info(f"wagie.run starting (config hash {cfg.hash()})")
    return engine.run()


__all__ = ["run", "build_pipeline"]
