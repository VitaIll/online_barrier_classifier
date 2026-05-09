"""LabelBuffer — first-class predict-then-learn-with-delayed-label staging.

Records (z_k, ts, ref_close) at predict time; emits (z_prev, y_prev) when
bar k+1 arrives (segment-boundary aware).
"""

from __future__ import annotations

import collections
import hashlib
import math
from dataclasses import dataclass
from typing import Optional

from wagie.core.numeric import LogReturn
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.core.time import Timestamp


@dataclass
class _Record:
    ts_close_ns: int
    obs: Observation
    ref_close: float
    segment_id: int


class LabelBuffer:
    """Stores (Observation, ref_close, segment_id) keyed by close_time.

    The Engine queries `maybe_emit(next_bar)` BEFORE running pipeline.transform_one
    on the current bar (constitutional invariant I.7). Emits (z_prev, y_prev)
    when the next bar's high crosses the one-sided excursion threshold.
    """

    name: str = "label_buffer"
    kind: StageKind = StageKind.LABEL_BUFFER

    def __init__(self, alpha_label: LogReturn | float = 0.0041113, max_buffer: int = 4):
        self.alpha_label = float(alpha_label)
        self._records: collections.deque = collections.deque(maxlen=max_buffer)
        self._n_seen = 0

    # ---- Stage Protocol ----

    def transform(self, obs: Observation) -> Observation:
        try:
            ts = int(obs.bar.close_time.ns)
            close = float(obs.bar.close)
            seg = int(obs.bar.segment_id)
        except Exception:
            return obs
        if math.isnan(close):
            return obs
        self._records.append(_Record(ts_close_ns=ts, obs=obs, ref_close=close, segment_id=seg))
        self._n_seen += 1
        return obs

    def update(self, obs: Observation, label: Optional[int] = None) -> None:
        return None

    def maybe_emit(self, next_bar) -> Optional[tuple[Observation, int]]:
        """Engine calls this when the next bar arrives. Returns (obs_prev, y_prev)
        if a record matures, else None."""
        if not self._records:
            return None
        try:
            next_seg = int(getattr(next_bar, "segment_id", 0))
            next_high = float(next_bar.high)
        except Exception:
            return None
        rec = self._records.popleft()
        if rec.segment_id != next_seg:
            return None
        if math.isnan(next_high) or rec.ref_close <= 0:
            return None
        log_excursion = math.log(next_high / rec.ref_close)
        y = int(log_excursion >= self.alpha_label)
        return (rec.obs, y)

    # ---- Stateful ----

    def state_dict(self) -> dict:
        return {"alpha_label": self.alpha_label, "n_seen": self._n_seen,
                "n_records": len(self._records)}

    def load_state_dict(self, state: dict) -> None:
        self.alpha_label = float(state.get("alpha_label", self.alpha_label))
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"LabelBuffer|")
        h.update(f"{self.alpha_label:.18g}".encode())
        for r in self._records:
            h.update(f"{r.ts_close_ns}/{r.ref_close:.18g}/{r.segment_id}".encode())
            h.update(b"|")
        return h.digest()

    def reset(self) -> None:
        self._records.clear()
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen


__all__ = ["LabelBuffer"]
