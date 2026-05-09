"""Frozen CatBoost as a Pipeline Stage.

Inherits the `wagie.core.Stage` Protocol shape: name, kind, transform, update,
state_dict, load_state_dict, state_hash, reset.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional

from catboost import CatBoostClassifier

from wagie.core.numeric import Probability
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind


class FrozenCatBoostPredictor:
    """Wraps a frozen CatBoost ensemble. Adds p_offline to the Observation.

    Auto-detects the model's expected feature column order from `feature_names_`
    if `feature_columns` is None.
    """

    name: str = "catboost_predictor"
    kind: StageKind = StageKind.PREDICTOR

    def __init__(
        self,
        model_path: str,
        feature_columns: Optional[list[str]] = None,
        ensemble_n: int = 1,
    ):
        self.model_path = str(model_path)
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.ensemble_n = int(ensemble_n)
        self._models: list[CatBoostClassifier] = []
        self._loaded = False
        self._n_seen = 0

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        path = Path(self.model_path)
        if self.ensemble_n == 1:
            m = CatBoostClassifier()
            m.load_model(str(path))
            self._models = [m]
        else:
            self._models = []
            for i in range(self.ensemble_n):
                m = CatBoostClassifier()
                fp = path.parent / f"{path.stem}_seed{i}{path.suffix}"
                m.load_model(str(fp))
                self._models.append(m)
        if self.feature_columns is None and self._models:
            self.feature_columns = list(self._models[0].feature_names_)
        self._loaded = True

    # ---- Stage Protocol ----

    def transform(self, obs: Observation) -> Observation:
        self._ensure_loaded()
        self._n_seen += 1
        cols = self.feature_columns or []
        bar_dict = obs.to_dict()
        try:
            row = [float(bar_dict.get(c, float("nan"))) for c in cols]
        except (TypeError, ValueError):
            return obs
        ps = []
        for m in self._models:
            try:
                p = float(m.predict_proba([row])[0, 1])
                if not math.isnan(p):
                    ps.append(p)
            except Exception:
                pass
        if not ps:
            return obs
        p_offline = sum(ps) / len(ps)
        return obs.with_p_offline(Probability(p_offline))

    def update(self, obs: Observation, label: Optional[int] = None) -> None:
        return None  # frozen

    # ---- Stateful ----

    def state_dict(self) -> dict:
        return {"model_path": self.model_path, "ensemble_n": self.ensemble_n,
                "feature_columns": self.feature_columns, "n_seen": self._n_seen}

    def load_state_dict(self, state: dict) -> None:
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"CatBoostPredictor|")
        h.update(self.model_path.encode())
        h.update(b"|")
        h.update(",".join(self.feature_columns or []).encode())
        return h.digest()

    def reset(self) -> None:
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen


__all__ = ["FrozenCatBoostPredictor"]
