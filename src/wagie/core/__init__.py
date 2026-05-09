"""wagie.core — the foundation primitives.

Eleven conceptual layers of small, sealed, mathematically-coherent types.
Every other module in wagie consumes types from here. New extensions extend
these primitives.

Layers:
    1. identity     — Symbol, Venue, InstrumentId, *Id (typed identifiers)
    2. numeric      — Price, Quantity, LogReturn, Probability, Bps (typed numerics)
    3. time         — Timestamp, Duration, TimeWindow (affine time arithmetic)
    4. event        — Event base + concrete events (algebraic data type)
    5. state        — CausalState, Stateful Protocol (state machines)
    6. observation  — Observation (typed pipeline frame)
    7. action       — Side, ExitReason, ActionKind, Action
    8. probabilistic — Quantile, PredictionInterval, CoverageGuarantee
    9. validation   — Split, CrossValSplitter Protocol
    10. pipeline    — StageKind, Stage Protocol
    11. stream      — Stream, Source, Sink Protocols
"""

# Layer 1: identity
from wagie.core.identity import (
    DEFAULT_INSTRUMENT, InstrumentId, OrderId, PositionId, Symbol, TraderId, Venue,
)

# Layer 2: numeric
from wagie.core.numeric import (
    Bps, LogReturn, Price, Probability, Quantity,
)

# Layer 3: time
from wagie.core.time import (
    Duration, Timestamp, TimeWindow,
)

# Layer 4: event
from wagie.core.event import (
    BarrierTouched, DecisionBar, DriftDetected, Event,
    MinuteBar, OrderFilled, OrderSubmitted,
    PositionClosed, PositionOpened, PositionScaledIn, PositionScaledOut,
    PositionStopModified, RiskPolicyChanged, RiskRejected,
)

# Layer 5: state
from wagie.core.state import (
    CausalState, Stateful,
)

# Layer 6: observation
from wagie.core.observation import (
    FeatureMap, Observation,
)

# Layer 7: action — Action sum type
from wagie.core.action import (
    Action, ActionKind, ExitReason, Side,
)

# Layer 7b: position + portfolio (G8)
from wagie.core.portfolio import Portfolio
from wagie.core.position import Position

# Layer 8: probabilistic
from wagie.core.probabilistic import (
    CoverageGuarantee, PredictionInterval, Quantile,
)

# Layer 9: validation
from wagie.core.validation import (
    CrossValSplitter, Split,
)

# Layer 10: pipeline
from wagie.core.pipeline import (
    Stage, StageKind,
)

# Layer 11: stream
from wagie.core.stream import (
    Sink, Source, Stream,
)


__all__ = [
    # 1. identity
    "Symbol", "Venue", "InstrumentId", "OrderId", "PositionId", "TraderId",
    "DEFAULT_INSTRUMENT",
    # 2. numeric
    "Price", "Quantity", "LogReturn", "Probability", "Bps",
    # 3. time
    "Timestamp", "Duration", "TimeWindow",
    # 4. event
    "Event", "MinuteBar", "DecisionBar", "OrderSubmitted", "OrderFilled",
    "BarrierTouched", "DriftDetected",
    "PositionOpened", "PositionScaledIn", "PositionScaledOut",
    "PositionStopModified", "PositionClosed",
    "RiskRejected", "RiskPolicyChanged",
    # 5. state
    "CausalState", "Stateful",
    # 6. observation
    "Observation", "FeatureMap",
    # 7. action
    "Side", "ExitReason", "ActionKind", "Action",
    # 7b. position + portfolio
    "Position", "Portfolio",
    # 8. probabilistic
    "Quantile", "PredictionInterval", "CoverageGuarantee",
    # 9. validation
    "Split", "CrossValSplitter",
    # 10. pipeline
    "StageKind", "Stage",
    # 11. stream
    "Stream", "Source", "Sink",
]
