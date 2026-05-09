"""Streaming Mondrian-ACI calibrator — Stage Protocol.

Per-regime q_t with the ACI γ-step. Adds q_lo and in_set per α to Observation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Optional

from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind


def _lac_score(p_class1: float, y: int) -> float:
    return 1.0 - p_class1 if y == 1 else p_class1


class MondrianACICalibrator:
    """Per-regime Adaptive Conformal Inference on p_online.

    State: q[regime_id][α] -> float in [0, 1].
    Update: q ← q + γ · (α - 1{miscovered}).
    """

    name: str = "mondrian_aci"
    kind: StageKind = StageKind.CALIBRATOR

    def __init__(
        self,
        alphas: Iterable[float] = (0.05, 0.10, 0.20),
        gamma: float = 0.01,
        n_regimes: int = 3,
        q_init: float = 0.5,
        q_init_by_regime: Optional[dict] = None,
    ):
        self.alphas = tuple(float(a) for a in alphas)
        self.gamma = float(gamma)
        self.n_regimes = int(n_regimes)
        self.q_init = float(q_init)
        self.q_init_by_regime = dict(q_init_by_regime or {})
        self._q: dict[int, dict[float, float]] = {}
        for r in range(self.n_regimes):
            self._q[r] = {}
            for a in self.alphas:
                per_alpha = self.q_init_by_regime.get(a, {})
                self._q[r][a] = float(per_alpha.get(r, q_init))
        self._n_seen = 0

    def _regime_of(self, obs: Observation) -> int:
        r = obs.regime_id
        if r is None:
            return 0
        ri = int(r)
        if ri < 0 or ri >= self.n_regimes:
            return 0
        return ri

    def transform(self, obs: Observation) -> Observation:
        self._n_seen += 1
        if obs.p_online is None or math.isnan(float(obs.p_online)):
            for a in self.alphas:
                obs = obs.with_calibration(a, float("nan"), False)
            return obs
        p = float(obs.p_online)
        regime = self._regime_of(obs)
        for a in self.alphas:
            q = self._q[regime][a]
            in_set = p >= 1.0 - q
            obs = obs.with_calibration(a, q, in_set)
        return obs

    def update(self, obs: Observation, label: Optional[int] = None) -> None:
        if label is None or obs.p_online is None:
            return
        try:
            y_int = int(label)
        except (TypeError, ValueError):
            return
        if math.isnan(float(obs.p_online)):
            return
        regime = self._regime_of(obs)
        s = _lac_score(float(obs.p_online), y_int)
        for a in self.alphas:
            q = self._q[regime][a]
            err = 1 if s > q else 0
            # ACI on score-threshold q (not on alpha_t). Bigger q → bigger LAC
            # set → fewer miscovers, so on miscover q must GROW (not shrink).
            # Equivalent to Gibbs–Candes (2021) eq. 5 written in the q domain.
            q_next = q + self.gamma * (err - a)
            self._q[regime][a] = max(0.0, min(1.0, q_next))

    # ---- Stateful ----

    def state_dict(self) -> dict:
        return {
            "alphas": list(self.alphas),
            "gamma": self.gamma,
            "n_regimes": self.n_regimes,
            "q": {str(r): {str(a): v for a, v in self._q[r].items()}
                  for r in self._q},
            "n_seen": self._n_seen,
        }

    def load_state_dict(self, state: dict) -> None:
        if "q" in state:
            for r_str, ad in state["q"].items():
                r = int(r_str)
                if r in self._q:
                    for a_str, v in ad.items():
                        a = float(a_str)
                        if a in self._q[r]:
                            self._q[r][a] = float(v)
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"MondrianACI|")
        h.update(repr(self.alphas).encode())
        h.update(b"|")
        for r in sorted(self._q.keys()):
            for a in sorted(self._q[r].keys()):
                h.update(f"{r}/{a}={self._q[r][a]:.18g}".encode())
                h.update(b"|")
        h.update(str(self._n_seen).encode())
        return h.digest()

    def reset(self) -> None:
        for r in self._q:
            for a in self._q[r]:
                per_alpha = self.q_init_by_regime.get(a, {})
                self._q[r][a] = float(per_alpha.get(r, self.q_init))
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen


__all__ = ["MondrianACICalibrator"]
