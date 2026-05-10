"""wagie — streaming-native ML harness; prod ≡ backtest by construction.

Single-protocol experimentation lives at:
    wagie.experiments.ExperimentProtocol  (run from CLI: wagie experiment run)

Single batteries:
    wagie.metrics.MetricsBattery
    wagie.charts.ChartBattery
    wagie.reporting.Report
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
from wagie.reporting import Report
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
    "Report",
    # Single experiment protocol
    "ExperimentProtocol", "ExperimentSpec", "ExperimentResult",
]
