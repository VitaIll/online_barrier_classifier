"""Pipeline primitives — Stage Protocol + StageKind enum.

Stage is the atomic unit of a Pipeline. Concrete stages (CatBoostPredictor,
OnlineARFCorrector, MondrianACICalibrator, strategies) implement the Stage
Protocol.

The sealed `wagie.Pipeline` (in wagie.pipeline.sealed) validates that stage
ordering monotonically increases by StageKind.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Protocol, runtime_checkable

from wagie.core.observation import Observation


class StageKind(IntEnum):
    """Pipeline stage kinds, in canonical execution order.

    The sealed Pipeline enforces this ordering at construction:
        AGGREGATOR < FEATURE < PREDICTOR < CORRECTOR
        < CALIBRATOR < LABEL_BUFFER < STRATEGY
    """

    AGGREGATOR = 0      # MinuteBar -> DecisionBar (lives in DataSource, not Pipeline)
    FEATURE = 1         # adds features to Observation
    PREDICTOR = 2       # adds p_offline (frozen offline model)
    CORRECTOR = 3       # adds p_online (online learner)
    CALIBRATOR = 4      # adds q_lo, in_set
    LABEL_BUFFER = 5    # delayed-label staging
    STRATEGY = 6        # adds decision
    DRIFT_DETECTOR = 7  # emits DriftDetected events (sidechannel)


@runtime_checkable
class Stage(Protocol):
    """The contract every Pipeline stage satisfies.

    transform: pure function on Observation; called per bar.
    update:    consumes (matured features, label) — only the LABEL_BUFFER
               and downstream learning stages override.
    """

    name: str
    kind: StageKind

    def transform(self, obs: Observation) -> Observation: ...
    def update(self, obs: Observation, label: Optional[int] = None) -> None: ...

    # Stateful contract (from wagie.core.state.Stateful):
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def state_hash(self) -> bytes: ...
    def reset(self) -> None: ...


__all__ = ["StageKind", "Stage"]
