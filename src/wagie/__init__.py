"""wagie — streaming-native ML harness; prod ≡ backtest by construction.

Single-protocol experimentation lives at:
    wagie.experiments.ExperimentProtocol  (run from CLI: wagie experiment run)

Single batteries:
    wagie.metrics.MetricsBattery
    wagie.charts.ChartBattery
    wagie.reporting.ReportRenderer  (single canonical HTML report)


Public vs internal surface
--------------------------

The package surface is tiered into ``wagie.public`` (everything in
:data:`__all__`) and ``wagie.internal`` (anything else exported from a
submodule). Both are importable, but only the public set is part of the
backwards-compatibility contract:

    wagie.public  : stable, semver-guarded; ``__all__`` enumerates it.
                    Treat additions as minor-version bumps, removals as major.
    wagie.internal: implementation detail. May change in any release. If you
                    need something here, file an issue first so it can be
                    promoted properly.

Inspect the live tiering with:

    wagie info               # public surface count
    wagie info --internal    # plus the internal symbol list

The internal modules used by the engine and protocol (e.g.
``wagie.run.build_pipeline``, ``wagie.engine.Engine`` private hooks,
``wagie.persistence.checkpoint_periodic``) are intentionally NOT on
``__all__`` to keep ``import wagie`` lightweight and the public contract
narrow.

Logging:
    Set ``WAGIE_LOG_FORMAT=json`` to switch the CLI's logger to one-line
    JSON records (see :mod:`wagie.observability`). The default text
    formatter remains for interactive use.

Error policy:
    Stage failures inside the pipeline used to be silently swallowed. They
    are now wrapped in :class:`wagie.core.errors.StageError`, propagated to
    the engine, and counted on :attr:`wagie.engine.EngineResult.n_stage_errors`
    so failures are visible in ``metrics.json``.
"""

__version__ = "0.3.0"

from wagie.charts import ChartBattery
from wagie.console import TradingConsole
from wagie.core import (
    DEFAULT_INSTRUMENT,
    Action,
    ActionKind,
    Bps,
    DecisionBar,
    Duration,
    Event,
    ExitReason,
    InstrumentId,
    LogReturn,
    MinuteBar,
    Observation,
    Portfolio,
    Position,
    Price,
    Probability,
    Quantity,
    Side,
    Stage,
    StageKind,
    Symbol,
    Timestamp,
    TimeWindow,
    Venue,
)
from wagie.engine import Engine, EngineResult
from wagie.experiments import (
    ExperimentProtocol,
    ExperimentResult,
    ExperimentSpec,
)
from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
from wagie.io.brokers import BinanceBroker, SimBroker
from wagie.io.clock import Clock, LiveClock, TestClock
from wagie.io.sources import BinanceLiveSource, ParquetReplaySource
from wagie.metrics import MetricsBattery, MetricsReport
from wagie.pipeline import (
    FrozenCatBoostPredictor,
    LabelBuffer,
    OnlineARFCorrector,
    Pipeline,
)
from wagie.reporting import ReportRenderer, RunMeta
from wagie.risk import (
    KillSwitchPolicy,
    MaxDrawdownPolicy,
    MaxLossPerPositionPolicy,
    MaxOrderRatePolicy,
    MaxPositionsPolicy,
    RiskEngine,
    RiskPolicy,
)
from wagie.strategy import (
    EvCalibratedSize,
    Strategy,
    StrategyBase,
    StrategyContext,
    ThresholdGate,
)


__all__ = [
    "__version__",
    # Core types
    "Symbol", "Venue", "InstrumentId", "DEFAULT_INSTRUMENT",
    "Timestamp", "Duration", "TimeWindow",
    "Price", "Quantity", "LogReturn", "Probability", "Bps",
    "Event", "MinuteBar", "DecisionBar",
    "Side", "ExitReason", "ActionKind", "Action",
    "Position", "Portfolio",
    "Observation",
    "Stage", "StageKind",
    # Pipeline
    "Pipeline",
    "FrozenCatBoostPredictor", "OnlineARFCorrector",
    "LabelBuffer",
    # Strategy
    "Strategy", "StrategyContext", "StrategyBase",
    "ThresholdGate", "EvCalibratedSize",
    # Risk
    "RiskEngine", "RiskPolicy",
    "MaxPositionsPolicy", "MaxDrawdownPolicy", "MaxLossPerPositionPolicy",
    "MaxOrderRatePolicy", "KillSwitchPolicy",
    # Features
    "BaseBarFeatures", "FeatureBuilder", "RegimeCuts", "RegimeFeature",
    # IO
    "Clock", "TestClock", "LiveClock",
    "ParquetReplaySource", "BinanceLiveSource",
    "SimBroker", "BinanceBroker",
    # Engine + console
    "Engine", "EngineResult",
    "TradingConsole",
    # Single batteries
    "MetricsBattery", "MetricsReport",
    "ChartBattery",
    "ReportRenderer", "RunMeta",
    # Single experiment protocol
    "ExperimentProtocol", "ExperimentSpec", "ExperimentResult",
]
