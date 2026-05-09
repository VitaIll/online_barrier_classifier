"""Online ARF correction layer — Stage Protocol.

Reads p_offline + selected features from the Observation; emits p_online
via river.forest.ARFClassifier.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

from river.forest import ARFClassifier

from wagie.core.numeric import Probability
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind


class OnlineARFCorrector:
    """Adaptive Random Forest. Inputs: dict of selected_features ∪ {p_offline}.
    Output: p_online."""

    name: str = "online_arf"
    kind: StageKind = StageKind.CORRECTOR

    def __init__(
        self,
        n_models: int = 10,
        max_features: str = "sqrt",
        lambda_value: float = 6.0,
        seed: int = 42,
        selected_features: Optional[list[str]] = None,
    ):
        self.n_models = int(n_models)
        self.max_features = max_features
        self.lambda_value = float(lambda_value)
        self.seed = int(seed)
        self.selected_features = list(selected_features) if selected_features else None
        self._arf: Optional[ARFClassifier] = None
        self._n_seen = 0

    def _ensure_init(self) -> None:
        if self._arf is None:
            self._arf = ARFClassifier(
                n_models=self.n_models,
                max_features=self.max_features,
                lambda_value=self.lambda_value,
                seed=self.seed,
            )

    def _z(self, obs: Observation) -> dict:
        bar_dict = obs.to_dict()
        if self.selected_features is None:
            # Pass everything except metadata/decision/etc.
            return {k: v for k, v in bar_dict.items()
                    if not k.startswith("_") and k not in {"decision"}}
        z = {f: bar_dict.get(f, float("nan")) for f in self.selected_features}
        if obs.p_offline is not None:
            z["p_offline"] = float(obs.p_offline)
        return z

    def transform(self, obs: Observation) -> Observation:
        self._ensure_init()
        self._n_seen += 1
        z = self._z(obs)
        if any(isinstance(v, float) and math.isnan(v) for v in z.values()):
            return obs
        try:
            proba = self._arf.predict_proba_one(z)
        except Exception:
            return obs
        p1 = float(proba.get(1, 0.5)) if proba else 0.5
        return obs.with_p_online(Probability(min(max(p1, 0.0), 1.0)))

    def update(self, obs: Observation, label: Optional[int] = None) -> None:
        if label is None:
            return
        self._ensure_init()
        z = self._z(obs)
        try:
            self._arf.learn_one(z, int(label))
        except Exception:
            pass

    # ---- Stateful ----

    def state_dict(self) -> dict:
        return {"n_seen": self._n_seen, "seed": self.seed,
                "n_models": self.n_models, "lambda_value": self.lambda_value}

    def load_state_dict(self, state: dict) -> None:
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"ARF|")
        h.update(repr((self.n_models, self.lambda_value, self.seed,
                       self.max_features, self._n_seen)).encode())
        return h.digest()

    def reset(self) -> None:
        self._arf = None
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen


__all__ = ["OnlineARFCorrector"]
