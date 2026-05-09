"""Feature contracts.

FeatureSpec carries lag (Nixtla pattern) + window + forecasting_safe (darts).
Feature is the unified Protocol; FeatureBuilder is a Stage that walks features.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.core.time import Duration


class FeatureKind(Enum):
    STREAMING = "streaming"
    POLARS = "polars"
    NUMBA = "numba"


@dataclass(frozen=True)
class FeatureSpec:
    """Static metadata about a feature. Bound at config time.
    `lag` and `window` together prevent forward-looking features by construction."""

    name: str
    kind: FeatureKind
    lag: int = 1
    window: int = 1
    forecasting_safe: bool = True
    symbol: str = "default"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lag < 1 and self.forecasting_safe:
            raise ValueError(
                f"FeatureSpec {self.name!r}: forecasting_safe=True forbids lag<1; got {self.lag}"
            )
        if self.window < 1:
            raise ValueError(f"FeatureSpec {self.name!r}: window>=1; got {self.window}")

    @property
    def update_samples(self) -> int:
        return self.lag + self.window


@dataclass(frozen=True)
class PastCovariateSpec(FeatureSpec):
    """Lag >= 1 enforced."""
    def __post_init__(self):
        super().__post_init__()
        if self.lag < 1:
            raise ValueError(f"PastCovariateSpec {self.name!r}: lag>=1; got {self.lag}")


@dataclass(frozen=True)
class FutureCovariateSpec(FeatureSpec):
    """Calendar features. lag=0 allowed."""
    lag: int = 0
    forecasting_safe: bool = False
    def __post_init__(self):
        if self.window < 1:
            raise ValueError(f"FutureCovariateSpec {self.name!r}: window>=1; got {self.window}")


@runtime_checkable
class Feature(Protocol):
    """Unified feature contract. Operates on the Observation.

    Each Feature reads bar/feature values from obs.to_dict() and emits its
    own value via with_features().
    """
    spec: FeatureSpec
    def update_one(self, obs_dict: dict) -> dict: ...
    def reset_state(self) -> None: ...
    def state_hash(self) -> bytes: ...


class FeatureBuilder:
    """Walks features in order; satisfies wagie.core.Stage.

    Each Feature reads and emits a flat dict; the FeatureBuilder converts
    Observation <-> dict at the boundary so each Feature stays simple.
    """

    name: str = "features"
    kind: StageKind = StageKind.FEATURE

    def __init__(self, features: Iterable[Feature]):
        self._features: list[Feature] = list(features)
        self._n_seen = 0
        seen = set()
        for f in self._features:
            if f.spec.name in seen:
                raise ValueError(f"duplicate feature name {f.spec.name!r}")
            seen.add(f.spec.name)

    @property
    def update_samples(self) -> int:
        return max((f.spec.update_samples for f in self._features), default=0)

    @property
    def feature_names(self) -> list[str]:
        return [f.spec.name for f in self._features]

    # ---- Stage Protocol ----

    def transform(self, obs: Observation) -> Observation:
        d = obs.to_dict()
        for f in self._features:
            d = f.update_one(d)
        self._n_seen += 1
        # Extract only the new feature columns (those not already in the bar dict)
        new_features = {
            f.spec.name: float(d.get(f.spec.name, float("nan")))
            for f in self._features
        }
        return obs.with_features_dict(new_features)

    def update(self, obs: Observation, label: Optional[Any] = None) -> None:
        return None

    def state_dict(self) -> dict:
        return {"n_seen": self._n_seen}

    def load_state_dict(self, state: dict) -> None:
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"FeatureBuilder|")
        for f in self._features:
            h.update(f.spec.name.encode())
            h.update(b"=")
            try:
                h.update(f.state_hash())
            except Exception:
                h.update(b"?")
            h.update(b"|")
        return h.digest()

    def reset(self) -> None:
        for f in self._features:
            try:
                f.reset_state()
            except Exception:
                pass
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen


def _hash_floats(values: Iterable[float]) -> bytes:
    h = hashlib.sha256()
    for v in values:
        h.update(repr(float(v)).encode())
        h.update(b",")
    return h.digest()


__all__ = [
    "FeatureKind",
    "FeatureSpec",
    "PastCovariateSpec",
    "FutureCovariateSpec",
    "Feature",
    "FeatureBuilder",
    "_hash_floats",
]
