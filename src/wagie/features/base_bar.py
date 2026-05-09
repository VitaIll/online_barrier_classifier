"""Per-bar derived features — Stage Protocol form (operates on Observation).

Single Stage that takes OHLCV from Observation.bar and adds derived per-bar
quantities to Observation.features.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind


class BaseBarFeatures:
    """Stateful per-bar feature derivation. Maintains prev_close for return calc."""

    name: str = "base_bar"
    kind: StageKind = StageKind.FEATURE

    def __init__(self):
        self.prev_close: Optional[float] = None
        self.prev_segment_id: Optional[int] = None
        self._n_seen = 0

    def transform(self, obs: Observation) -> Observation:
        bar = obs.bar
        try:
            o = float(bar.open); h = float(bar.high); l = float(bar.low)
            c = float(bar.close); v = float(bar.volume); seg = int(bar.segment_id)
        except Exception:
            return obs
        feats: dict[str, float] = {}

        if o > 0: feats["log_open"] = math.log(o)
        if h > 0: feats["log_high"] = math.log(h)
        if l > 0: feats["log_low"] = math.log(l)
        if c > 0: feats["log_close"] = math.log(c)
        if v > 0: feats["log_volume"] = math.log1p(v)
        feats["log_quote_volume"] = math.log1p(float(bar.quote_volume))
        feats["log_trades"] = math.log1p(float(bar.trades))

        feats["range_hl"] = h - l
        feats["range_pct"] = (h - l) / c if c > 0 else 0.0
        feats["body"] = c - o
        feats["body_abs"] = abs(c - o)
        feats["upper_wick"] = h - max(o, c)
        feats["lower_wick"] = min(o, c) - l

        if (self.prev_close is not None and self.prev_close > 0
                and self.prev_segment_id == seg and c > 0):
            ret = math.log(c / self.prev_close)
            feats["return"] = ret
            feats["abs_return"] = abs(ret)
            feats["squared_return"] = ret * ret
            if o > 0:
                feats["open_gap_return"] = math.log(o / self.prev_close)
            else:
                feats["open_gap_return"] = 0.0
        else:
            feats["return"] = 0.0
            feats["abs_return"] = 0.0
            feats["squared_return"] = 0.0
            feats["open_gap_return"] = 0.0

        if h > 0 and l > 0 and h >= l:
            log_hl = math.log(h / l)
            feats["parkinson_var"] = (log_hl * log_hl) / (4.0 * math.log(2.0))
        else:
            feats["parkinson_var"] = 0.0

        if h > 0 and l > 0 and o > 0 and c > 0:
            log_hl = math.log(h / l)
            log_co = math.log(c / o)
            feats["garman_klass_var"] = (
                0.5 * log_hl * log_hl - (2.0 * math.log(2.0) - 1.0) * log_co * log_co
            )
        else:
            feats["garman_klass_var"] = 0.0

        feats["parkinson_vol"] = math.sqrt(max(0.0, feats["parkinson_var"]))
        feats["garman_klass_vol"] = math.sqrt(max(0.0, feats["garman_klass_var"]))

        if self.prev_close is not None and self.prev_close > 0:
            feats["true_range"] = max(h - l, abs(h - self.prev_close),
                                       abs(l - self.prev_close))
        else:
            feats["true_range"] = h - l

        if h > l:
            feats["close_position"] = (c - l) / (h - l)
            feats["open_position"] = (o - l) / (h - l)
        else:
            feats["close_position"] = 0.5
            feats["open_position"] = 0.5

        feats["oc_return"] = math.log(c / o) if (c > 0 and o > 0) else 0.0

        qv = float(bar.quote_volume)
        if v > 0 and qv > 0:
            vwap = qv / v
            feats["vwap"] = vwap
            feats["vwap_log"] = math.log(vwap) if vwap > 0 else 0.0
            feats["close_to_vwap"] = (c - vwap) / vwap if vwap > 0 else 0.0
        else:
            feats["vwap"] = c
            feats["vwap_log"] = math.log(c) if c > 0 else 0.0
            feats["close_to_vwap"] = 0.0

        tbb = float(bar.taker_buy_base)
        tbq = float(bar.taker_buy_quote)
        if v > 0:
            feats["buy_ratio"] = tbb / v
            feats["imbalance"] = (2.0 * tbb - v) / v
        else:
            feats["buy_ratio"] = 0.5
            feats["imbalance"] = 0.0
        if qv > 0:
            feats["buy_ratio_quote"] = tbq / qv
        else:
            feats["buy_ratio_quote"] = 0.5
        n_tr = float(bar.trades)
        feats["avg_trade_size"] = (v / n_tr) if n_tr > 0 else 0.0
        feats["return_x_log_volume"] = feats["return"] * feats.get("log_volume", 0.0)
        feats["return_x_imbalance"] = feats["return"] * feats["imbalance"]
        feats["range_x_log_volume"] = feats["range_hl"] * feats.get("log_volume", 0.0)

        if qv > 0 and feats["abs_return"] > 0:
            feats["illiq"] = feats["abs_return"] / qv
        else:
            feats["illiq"] = 0.0

        # Calendar features
        try:
            open_ms = bar.ts_init.ms
        except Exception:
            open_ms = 0
        if open_ms > 0:
            t_sec = open_ms // 1000
            mod = (t_sec // 60) % (24 * 60)
            angle_min = 2.0 * math.pi * mod / (24.0 * 60.0)
            feats["minute_sin"] = math.sin(angle_min)
            feats["minute_cos"] = math.cos(angle_min)
            dow = (t_sec // 86400 + 4) % 7
            angle_dow = 2.0 * math.pi * dow / 7.0
            feats["dow_sin"] = math.sin(angle_dow)
            feats["dow_cos"] = math.cos(angle_dow)
        else:
            feats["minute_sin"] = 0.0
            feats["minute_cos"] = 0.0
            feats["dow_sin"] = 0.0
            feats["dow_cos"] = 0.0

        self.prev_close = c
        self.prev_segment_id = seg
        self._n_seen += 1
        return obs.with_features_dict(feats)

    def update(self, obs: Observation, label=None) -> None:
        return None

    def state_dict(self) -> dict:
        return {"prev_close": self.prev_close, "prev_segment_id": self.prev_segment_id,
                "n_seen": self._n_seen}

    def load_state_dict(self, state: dict) -> None:
        self.prev_close = state.get("prev_close")
        self.prev_segment_id = state.get("prev_segment_id")
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"BaseBarFeatures|")
        h.update(repr(self.prev_close).encode())
        h.update(b"|")
        h.update(repr(self.prev_segment_id).encode())
        return h.digest()

    def reset(self) -> None:
        self.prev_close = None
        self.prev_segment_id = None
        self._n_seen = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen


__all__ = ["BaseBarFeatures"]
