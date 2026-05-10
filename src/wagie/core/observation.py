"""Observation — typed pipeline frame, replaces dict[str, Any].

Each Pipeline stage is a function (Observation) -> Observation that ADDS
information via with_* methods (immutable replace). The full Observation at
the strategy stage carries: bar, features, p_offline, p_online, regime, q_lo,
in_set, decision.

Mathematical structure:
    - Product type (every with_* yields a new immutable Observation)
    - Independent updates commute (with_p_offline ∘ with_regime ==
      with_regime ∘ with_p_offline)
    - The fold of with_* over an empty Observation yields the full one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Optional

from wagie.core.action import Action, ActionKind
from wagie.core.event import DecisionBar
from wagie.core.numeric import Probability
from wagie.core.time import Timestamp


class FeatureMap(Mapping[str, float]):
    """Immutable typed mapping str -> float. Supports with_added()."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, float] | None = None):
        object.__setattr__(self, "_data", dict(data) if data else {})

    def __getitem__(self, k: str) -> float:
        return self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, k) -> bool:
        return k in self._data

    def get(self, k, default=None):
        return self._data.get(k, default)

    def with_added(self, **kw: float) -> "FeatureMap":
        merged = {**self._data, **{k: float(v) for k, v in kw.items()}}
        return FeatureMap(merged)

    def with_added_dict(self, d: Mapping[str, float]) -> "FeatureMap":
        merged = {**self._data, **{k: float(v) for k, v in d.items()}}
        return FeatureMap(merged)

    def to_dict(self) -> dict:
        return dict(self._data)

    def __repr__(self) -> str:
        n = len(self._data)
        keys = list(self._data.keys())
        sample = ", ".join(keys[:3]) + ("..." if n > 3 else "")
        return f"FeatureMap(n={n}, [{sample}])"


@dataclass(frozen=True, slots=True)
class Observation:
    """The strongly-typed pipeline frame. Replaces dict[str, Any].

    Each Pipeline stage is a function (Observation) -> Observation that
    *adds* (never removes) information via with_* methods.

    `drift_signals` carries per-iteration drift detector counts surfaced by
    online stages (e.g. `OnlineARFCorrector` exposes the underlying ARF's
    ADWIN drift / warning counts since the last `transform`). The Engine
    consumes this to emit `DriftDetected` events.
    """

    bar: DecisionBar
    features: FeatureMap = field(default_factory=FeatureMap)
    p_offline: Optional[Probability] = None
    p_online: Optional[Probability] = None
    regime_id: Optional[int] = None
    q_lo: Mapping[float, float] = field(default_factory=dict)
    in_set: Mapping[float, bool] = field(default_factory=dict)
    actions: tuple = field(default_factory=tuple)   # tuple[Action, ...]
    drift_signals: Optional[Mapping[str, int]] = None

    @property
    def as_of(self) -> Timestamp:
        """When this observation is honestly available — bar's close time."""
        return self.bar.close_time

    @property
    def instrument(self):
        return self.bar.instrument

    # ---- Immutable updates (with_*) -----------------------------------------

    def with_features(self, **kw: float) -> "Observation":
        return dataclasses.replace(self, features=self.features.with_added(**kw))

    def with_features_dict(self, d: Mapping[str, float]) -> "Observation":
        return dataclasses.replace(self, features=self.features.with_added_dict(d))

    def with_p_offline(self, p: Probability) -> "Observation":
        return dataclasses.replace(self, p_offline=p)

    def with_p_online(self, p: Probability) -> "Observation":
        return dataclasses.replace(self, p_online=p)

    def with_regime(self, regime_id: int) -> "Observation":
        return dataclasses.replace(self, regime_id=int(regime_id))

    def with_calibration(self, alpha: float, q_lo: float, in_set: bool) -> "Observation":
        new_q = {**self.q_lo, float(alpha): float(q_lo)}
        new_in = {**self.in_set, float(alpha): bool(in_set)}
        return dataclasses.replace(self, q_lo=new_q, in_set=new_in)

    def with_actions(self, actions) -> "Observation":
        """Set the sequence of actions emitted by Strategy.decide."""
        return dataclasses.replace(self, actions=tuple(actions))

    def with_drift_signals(self, signals: Mapping[str, int]) -> "Observation":
        """Attach per-iteration drift signals (counts) emitted by online stages."""
        return dataclasses.replace(self, drift_signals=dict(signals))

    # ---- Convenience: flat-dict view -----------------------------------------

    def to_dict(self) -> dict:
        """Flat dict view used by river-style consumers (FeatureBuilder, ARF)."""
        bar = self.bar
        out = {
            "open_time": float(bar.ts_init.ms),
            "close_time": float(bar.close_time.ms),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "quote_volume": float(bar.quote_volume),
            "trades": float(bar.trades),
            "taker_buy_base": float(bar.taker_buy_base),
            "taker_buy_quote": float(bar.taker_buy_quote),
            "segment_id": float(bar.segment_id),
        }
        out.update(self.features.to_dict())
        if self.p_offline is not None:
            out["p_offline"] = float(self.p_offline)
        if self.p_online is not None:
            out["p_online"] = float(self.p_online)
        if self.regime_id is not None:
            out["regime_id"] = self.regime_id
        for a, q in self.q_lo.items():
            key = f"q_lo_{int(round(a*100)):02d}"
            out[key] = float(q)
        for a, s in self.in_set.items():
            key = f"in_set_{int(round(a*100)):02d}"
            out[key] = int(s)
        return out


__all__ = ["FeatureMap", "Observation"]
