"""Sealed Pipeline — validates stage ordering at construction.

Replaces the river.compose.Pipeline-as-everything design with a typed
container that:
    1. Walks an explicit list of Stage objects (each implements wagie.core.Stage)
    2. Validates monotone StageKind ordering at __init__
    3. Operates on Observation (NOT dict)
    4. Exposes state_dict / load_state_dict / state_hash uniformly
    5. Uses pattern-matched dispatch based on StageKind for predict-vs-learn
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from wagie.core.observation import Observation
from wagie.core.pipeline import Stage, StageKind
from wagie.strategy import Strategy, StrategyBase, StrategyContext


@dataclass
class Pipeline:
    """Sealed, validated pipeline of Stages operating on Observation.

    Construction validates:
        - all stages have a `kind: StageKind`
        - kinds are monotonically non-decreasing
        - last stage has kind == STRATEGY
        - no duplicate names
    """

    stages: tuple[Stage, ...]

    def __init__(self, stages: Iterable[Stage]):
        self.stages = tuple(stages)
        self._validate()

    def _validate(self) -> None:
        if not self.stages:
            raise ValueError("Pipeline requires at least one stage")
        seen_kinds = []
        seen_names = set()
        for s in self.stages:
            kind = getattr(s, "kind", None)
            if not isinstance(kind, StageKind):
                raise TypeError(f"stage {s!r} has no StageKind `kind` attribute")
            if seen_kinds and kind.value < seen_kinds[-1].value:
                raise ValueError(
                    f"stage ordering violates monotone kind: "
                    f"{seen_kinds[-1].name} (kind={seen_kinds[-1].value}) "
                    f"followed by {kind.name} (kind={kind.value})"
                )
            seen_kinds.append(kind)
            name = getattr(s, "name", None)
            if name is None:
                name = type(s).__name__
            if name in seen_names:
                raise ValueError(f"duplicate stage name: {name!r}")
            seen_names.add(name)
        if self.stages[-1].kind != StageKind.STRATEGY:
            raise ValueError(
                f"Pipeline must end with a STRATEGY stage; got {self.stages[-1].kind}"
            )

    # -- transform / learn ---------------------------------------------------

    def transform_one(self, obs: Observation, ctx: Optional[StrategyContext] = None) -> Observation:
        """Walk the stack. Each non-strategy stage transforms; the strategy
        consumes (obs, ctx) and writes Sequence[Action] into obs."""
        if ctx is None:
            ctx = StrategyContext()
        for stage in self.stages:
            if stage.kind == StageKind.STRATEGY:
                if hasattr(stage, "decide"):
                    actions = stage.decide(obs, ctx)
                    obs = obs.with_actions(actions)
                else:
                    obs = stage.transform(obs)
            else:
                obs = stage.transform(obs)
        return obs

    def learn_one(self, obs: Observation, label: Optional[int] = None) -> None:
        """Drive learning on stages that consume labels (calibrator, online)."""
        for stage in self.stages:
            try:
                stage.update(obs, label=label)
            except TypeError:
                # Stage's update() doesn't accept label; call with no args
                try:
                    stage.update(obs)
                except Exception:
                    pass
            except Exception:
                pass

    # -- Stateful --------------------------------------------------------------

    def state_dict(self) -> dict:
        return {s.name: s.state_dict() for s in self.stages}

    def load_state_dict(self, state: dict) -> None:
        for s in self.stages:
            if s.name in state:
                s.load_state_dict(state[s.name])

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"Pipeline|")
        for s in self.stages:
            h.update(s.name.encode())
            h.update(b"=")
            try:
                h.update(s.state_hash())
            except Exception:
                h.update(repr(s).encode())
            h.update(b"|")
        return h.digest()

    def reset(self) -> None:
        for s in self.stages:
            try:
                s.reset()
            except Exception:
                pass

    @property
    def n_stages(self) -> int:
        return len(self.stages)


__all__ = ["Pipeline"]
