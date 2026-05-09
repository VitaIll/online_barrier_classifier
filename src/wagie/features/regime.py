"""Regime feature — bins a continuous source feature into discrete regimes.

Frozen edges fitted offline; runtime emits regime_id into the Observation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind


@dataclass(frozen=True)
class RegimeCuts:
    feature: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]

    @property
    def n_regimes(self) -> int:
        return len(self.edges) + 1

    def assign(self, value: float) -> int:
        if value is None or (isinstance(value, float) and value != value):
            return 0
        return int(sum(value > e for e in self.edges))

    @classmethod
    def from_quantiles(cls, values: Sequence[float], feature: str,
                       n_regimes: int = 3, labels: Optional[Sequence[str]] = None) -> "RegimeCuts":
        v = np.asarray([float(x) for x in values if x is not None and x == x])
        if v.size == 0:
            raise ValueError("Cannot fit RegimeCuts on empty array")
        qs = np.linspace(0.0, 1.0, n_regimes + 1)[1:-1]
        edges = tuple(float(q) for q in np.quantile(v, qs))
        if labels is None:
            labels = ("low", "med", "high") if n_regimes == 3 else tuple(f"r{i}" for i in range(n_regimes))
        return cls(feature=feature, edges=edges, labels=tuple(labels))

    def to_dict(self) -> dict:
        return {"feature": self.feature, "edges": list(self.edges), "labels": list(self.labels)}

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeCuts":
        return cls(feature=d["feature"], edges=tuple(float(e) for e in d["edges"]),
                   labels=tuple(d.get("labels") or [f"r{i}" for i in range(len(d["edges"]) + 1)]))


class RegimeFeature:
    """Reads `source_col` from Observation.features; emits regime_id."""

    name: str = "regime"
    kind: StageKind = StageKind.FEATURE

    def __init__(self, cuts: RegimeCuts, source_col: Optional[str] = None):
        self.cuts = cuts
        self.source_col = source_col or cuts.feature
        self._n_seen = 0

    def transform(self, obs: Observation) -> Observation:
        v = obs.features.get(self.source_col, float("nan"))
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        rid = self.cuts.assign(v)
        self._n_seen += 1
        return obs.with_regime(rid)

    def update(self, obs: Observation, label=None) -> None:
        return None

    def state_dict(self) -> dict:
        return {"cuts": self.cuts.to_dict(), "n_seen": self._n_seen}

    def load_state_dict(self, state: dict) -> None:
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"Regime|")
        h.update(self.source_col.encode())
        for e in self.cuts.edges:
            h.update(repr(e).encode())
            h.update(b",")
        return h.digest()

    def reset(self) -> None:
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen


__all__ = ["RegimeCuts", "RegimeFeature"]
