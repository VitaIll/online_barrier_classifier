"""
Streaming feature pipeline for decision bars.

This module defines the feature extraction pipeline using River streaming
statistics. Features are computed incrementally as each decision bar arrives.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from river import stats, utils
from river.base import Transformer

if __package__ and "." in __package__:
    from ..utils import (
        safe_divide,
        parkinson_variance,
        garman_klass_variance,
    )
else:
    from utils import (
        safe_divide,
        parkinson_variance,
        garman_klass_variance,
    )


class BaseFeatureExtractor(Transformer):
    """
    Extract base features from a single decision bar.

    Computes non-rolling features that depend only on the current bar and
    optionally the previous close price.

    Parameters
    ----------
    eps : float, default 1e-10
        Numerical stability constant for division.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps
        self.prev_close: Optional[float] = None
        self.prev_segment_id: Optional[int] = None

    def learn_one(self, x: dict) -> "BaseFeatureExtractor":
        """Update state with the current bar's close price."""
        self.prev_close = x["close"]
        if "segment_id" in x and x["segment_id"] is not None:
            self.prev_segment_id = int(x["segment_id"])
        return self

    def transform_one(self, x: dict) -> dict:
        """
        Extract base features from a decision bar.

        Parameters
        ----------
        x : dict
            Decision bar dictionary.

        Returns
        -------
        dict
            Feature dictionary with base features.
        """
        o, h, l, c = x["open"], x["high"], x["low"], x["close"]
        v, q = x["volume"], x["quote_volume"]
        n_trades = x["trades"]
        taker_buy_base = x["taker_buy_base"]
        taker_buy_quote = x.get("taker_buy_quote", 0.0)
        open_time = x["open_time"]
        segment_id = x.get("segment_id")

        eps = self.eps
        features: dict[str, float] = {}
        flags: dict[str, int] = {}

        # --- Returns ---
        log_open = math.log(o) if o > 0 else 0.0
        log_high = math.log(h) if h > 0 else 0.0
        log_low = math.log(l) if l > 0 else 0.0
        log_close = math.log(c) if c > 0 else 0.0

        seg = int(segment_id) if segment_id is not None else None
        is_new_segment = (
            seg is not None
            and self.prev_segment_id is not None
            and seg != self.prev_segment_id
        )
        if is_new_segment:
            prev_close = None
            flags["flag__segment_start"] = 1
        else:
            prev_close = self.prev_close
            flags["flag__segment_start"] = 1 if self.prev_close is None else 0

        flags["flag__first_bar"] = 1 if prev_close is None else 0

        if prev_close is not None and prev_close > 0:
            ret = log_close - math.log(prev_close)
            open_gap_return = log_open - math.log(prev_close) if o > 0 else 0.0
        else:
            ret = 0.0
            open_gap_return = 0.0

        oc_return = log_close - log_open if o > 0 else 0.0

        features["log_open"] = log_open
        features["log_high"] = log_high
        features["log_low"] = log_low
        features["log_close"] = log_close
        features["return"] = ret
        features["open_gap_return"] = open_gap_return
        features["oc_return"] = oc_return
        features["abs_return"] = abs(ret)
        features["squared_return"] = ret ** 2

        # --- Range-based volatility ---
        features["range_hl"] = math.log(h / l) if h > 0 and l > 0 else 0.0
        features["parkinson_var"] = parkinson_variance(h, l)
        features["parkinson_vol"] = math.sqrt(features["parkinson_var"])
        gk_var = garman_klass_variance(o, h, l, c)
        if gk_var < 0:
            features["garman_klass_var"] = 0.0
            flags["flag__gk_negative"] = 1
        else:
            features["garman_klass_var"] = gk_var
            flags["flag__gk_negative"] = 0
        features["garman_klass_vol"] = math.sqrt(features["garman_klass_var"])

        # --- Candle geometry ---
        features["body"] = (c - o) / (o + eps)
        features["upper_wick"] = (h - max(o, c)) / (o + eps)
        features["lower_wick"] = (min(o, c) - l) / (o + eps)

        range_size = h - l
        if range_size > eps:
            features["body_fraction"] = abs(c - o) / range_size
            features["close_position"] = (c - l) / range_size
            features["open_position"] = (o - l) / range_size
            flags["flag__range_zero"] = 0
        else:
            features["body_fraction"] = 0.0
            features["close_position"] = 0.0
            features["open_position"] = 0.0
            flags["flag__range_zero"] = 1

        features["range_pct"] = range_size / (c + eps)
        if prev_close is not None and prev_close > 0:
            ref_price = prev_close
        else:
            ref_price = c
        true_range = max(h, ref_price) - min(l, ref_price)
        features["true_range"] = true_range / (ref_price + eps)

        # --- Activity ---
        features["log_volume"] = math.log(1 + v)
        features["log_quote_volume"] = math.log(1 + q)
        features["log_trades"] = math.log(1 + n_trades)
        features["trades"] = float(n_trades)

        avg_trade, flag_no_trades = safe_divide(q, n_trades, neutral=0.0, eps=1.0)
        features["avg_trade_size"] = avg_trade
        flags["flag__no_trades"] = int(flag_no_trades)

        vwap, flag_no_vol = safe_divide(q, v, neutral=0.0, eps=eps)
        features["vwap"] = vwap
        features["vwap_log"] = math.log(vwap) if vwap > 0 else 0.0
        close_to_vwap, flag_no_vwap = safe_divide(c - vwap, vwap, neutral=0.0, eps=eps)
        features["close_to_vwap"] = close_to_vwap
        flags["flag__no_vwap"] = int(flag_no_vwap or flag_no_vol)

        # --- Order flow ---
        buy_ratio, flag_no_vol = safe_divide(taker_buy_base, v, neutral=0.5, eps=eps)
        features["buy_ratio"] = buy_ratio
        flags["flag__no_volume"] = int(flag_no_vol)

        buy_ratio_quote, flag_no_quote = safe_divide(
            taker_buy_quote, q, neutral=0.5, eps=eps
        )
        features["buy_ratio_quote"] = buy_ratio_quote
        flags["flag__no_quote_volume"] = int(flag_no_quote)

        imbalance, _ = safe_divide(2 * taker_buy_base - v, v, neutral=0.0, eps=eps)
        features["imbalance"] = imbalance

        # --- Illiquidity ---
        illiq, _ = safe_divide(abs(ret), q, neutral=0.0, eps=eps)
        features["illiq"] = illiq

        # --- Seasonality ---
        dt = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
        minute_of_day = dt.hour * 60 + dt.minute
        day_of_week = dt.weekday()  # 0 = Monday

        features["minute_sin"] = math.sin(2 * math.pi * minute_of_day / 1440)
        features["minute_cos"] = math.cos(2 * math.pi * minute_of_day / 1440)
        features["dow_sin"] = math.sin(2 * math.pi * day_of_week / 7)
        features["dow_cos"] = math.cos(2 * math.pi * day_of_week / 7)

        # --- Simple interactions ---
        features["return_x_log_volume"] = ret * features["log_volume"]
        features["return_x_imbalance"] = ret * imbalance
        features["range_x_log_volume"] = features["range_hl"] * features["log_volume"]

        return {**features, **flags}


class RollingFeatureExtractor(Transformer):
    """
    Compute rolling statistics over base features.

    Maintains River rolling statistics for specified base features across
    multiple window sizes.

    Parameters
    ----------
    base_features : list of str
        Names of base features to compute rolling stats for.
    windows : list of int
        Window sizes (in decision bars).
    stats_to_compute : list of str
        Statistics: "mean", "var", "min", "max", "sum", "iqr", "ptp".
    """

    def __init__(
        self,
        base_features: Optional[list[str]] = None,
        windows: Optional[list[int]] = None,
        stats_to_compute: Optional[list[str]] = None,
    ) -> None:
        self.base_features = base_features or [
            "return",
            "abs_return",
            "squared_return",
            "range_hl",
            "range_pct",
            "parkinson_var",
            "garman_klass_var",
            "log_volume",
            "log_quote_volume",
            "imbalance",
            "buy_ratio",
            "illiq",
        ]
        self.windows = windows or [1, 2, 4, 12, 48]
        self.stats_to_compute = stats_to_compute or [
            "mean",
            "var",
            "min",
            "max",
            "iqr",
            "ptp",
        ]

        self._init_stats()

    def reset(self) -> None:
        """Reset all rolling state (use at segment boundaries)."""
        self._init_stats()

    def _init_stats(self) -> None:
        """(Re)initialize rolling statistic objects."""
        self._rolling_stats: dict[str, dict[str, object]] = {}
        for feat in self.base_features:
            self._rolling_stats[feat] = {}
            for window in self.windows:
                for stat_name in self.stats_to_compute:
                    key = f"{feat}_rolling_{stat_name}_{window}"
                    self._rolling_stats[feat][key] = self._create_stat(stat_name, window)

    def _create_stat(self, stat_name: str, window: int):
        """Create River rolling statistic object."""
        if stat_name == "mean":
            return utils.Rolling(stats.Mean(), window_size=window)
        if stat_name == "var":
            return utils.Rolling(stats.Var(ddof=1), window_size=window)
        if stat_name == "min":
            return stats.RollingMin(window_size=window)
        if stat_name == "max":
            return stats.RollingMax(window_size=window)
        if stat_name == "sum":
            return utils.Rolling(stats.Sum(), window_size=window)
        if stat_name == "iqr":
            return stats.RollingIQR(window_size=window)
        if stat_name == "ptp":
            return stats.RollingPeakToPeak(window_size=window)
        raise ValueError(f"Unknown stat: {stat_name}")

    def learn_one(self, x: dict) -> "RollingFeatureExtractor":
        """
        Update all rolling statistics with base feature values.

        Parameters
        ----------
        x : dict
            Base feature dictionary (output of BaseFeatureExtractor).
        """
        if x.get("flag__segment_start") == 1:
            self.reset()

        for feat in self.base_features:
            if feat not in x:
                continue
            value = x[feat]
            for stat_obj in self._rolling_stats[feat].values():
                stat_obj.update(value)
        return self

    def transform_one(self, x: dict) -> dict:
        """
        Get current rolling statistic values.

        Parameters
        ----------
        x : dict
            Base feature dictionary.

        Returns
        -------
        dict
            Rolling feature dictionary.
        """
        result: dict[str, float] = {}
        for feat in self.base_features:
            for key, stat_obj in self._rolling_stats[feat].items():
                result[key] = stat_obj.get()
        return result


class LagFeatureExtractor(Transformer):
    """
    Generate lagged features for selected base features.

    Parameters
    ----------
    base_features : list of str
        Names of base features to lag.
    lags : list of int
        Positive lag steps to emit (e.g., [1, 2, 3]).
    fill_value : float, default 0.0
        Value to use when a lag is not yet available.
    """

    def __init__(
        self,
        base_features: Optional[list[str]] = None,
        lags: Optional[list[int]] = None,
        fill_value: float = 0.0,
    ) -> None:
        self.base_features = base_features or [
            "return",
            "abs_return",
            "range_hl",
            "log_volume",
            "imbalance",
            "buy_ratio",
        ]
        self.lags = sorted(set(lags or [1, 2, 3]))
        self.fill_value = fill_value
        self._init_buffers()

    def reset(self) -> None:
        """Reset lag buffers (use at segment boundaries)."""
        self._init_buffers()

    def _init_buffers(self) -> None:
        max_lag = max(self.lags) if self.lags else 1
        self._buffers = {
            feat: deque(maxlen=max_lag) for feat in self.base_features
        }

    def learn_one(self, x: dict) -> "LagFeatureExtractor":
        """Update lag buffers with current feature values."""
        if x.get("flag__segment_start") == 1:
            self.reset()

        for feat in self.base_features:
            if feat in x:
                self._buffers[feat].append(x[feat])
        return self

    def transform_one(self, x: dict) -> dict:
        """Return lagged feature values based on buffered history."""
        result: dict[str, float] = {}
        for feat in self.base_features:
            buf = self._buffers.get(feat)
            if buf is None:
                continue
            for lag in self.lags:
                key = f"{feat}_lag_{lag}"
                if len(buf) >= lag:
                    result[key] = buf[-lag]
                else:
                    result[key] = self.fill_value
        return result


class ExpandingFeatureExtractor(Transformer):
    """
    Compute expanding (since-segment-start) statistics for base features.

    Parameters
    ----------
    base_features : list of str
        Names of base features to compute expanding stats for.
    stats_to_compute : list of str
        Statistics: "mean", "var", "sum", "skew", "kurtosis".
    """

    def __init__(
        self,
        base_features: Optional[list[str]] = None,
        stats_to_compute: Optional[list[str]] = None,
    ) -> None:
        self.base_features = base_features or [
            "return",
            "range_hl",
            "log_volume",
            "imbalance",
            "illiq",
        ]
        self.stats_to_compute = stats_to_compute or ["mean", "var", "skew"]
        self._init_stats()

    def reset(self) -> None:
        """Reset expanding statistics (use at segment boundaries)."""
        self._init_stats()

    def _init_stats(self) -> None:
        self._expanding_stats: dict[str, dict[str, object]] = {}
        for feat in self.base_features:
            self._expanding_stats[feat] = {}
            for stat_name in self.stats_to_compute:
                key = f"{feat}_expanding_{stat_name}"
                self._expanding_stats[feat][key] = self._create_stat(stat_name)

    def _create_stat(self, stat_name: str):
        if stat_name == "mean":
            return stats.Mean()
        if stat_name == "var":
            return stats.Var(ddof=1)
        if stat_name == "sum":
            return stats.Sum()
        if stat_name == "skew":
            return stats.Skew()
        if stat_name == "kurtosis":
            return stats.Kurtosis()
        raise ValueError(f"Unknown stat: {stat_name}")

    def learn_one(self, x: dict) -> "ExpandingFeatureExtractor":
        """Update expanding statistics with current feature values."""
        if x.get("flag__segment_start") == 1:
            self.reset()

        for feat in self.base_features:
            if feat not in x:
                continue
            value = x[feat]
            for stat_obj in self._expanding_stats[feat].values():
                stat_obj.update(value)
        return self

    def transform_one(self, x: dict) -> dict:
        """Return current expanding statistic values."""
        result: dict[str, float] = {}
        for feat in self.base_features:
            for key, stat_obj in self._expanding_stats[feat].items():
                result[key] = stat_obj.get()
        return result
