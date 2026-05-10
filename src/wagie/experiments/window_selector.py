"""Test-window selector for trading-strategy backtests.

The offline CatBoost model is valid for ~14 days from its training cutoff.
A trading experiment therefore needs to fit inside a 14-day envelope (e.g.
2 days warm-up + 12 days trading).

Given a labeled bars frame plus matching regime features, this module slides
a window over the data and ranks candidate windows on:

    * empirical event rate inside the operating band (default 5%-15%),
    * up/down-bar balance (we want a mix, not a pure trend regime),
    * Parkinson-variance stability (CV) and span (max/min ratio): we want
      *some* regime variation but not chaos.

The result is a small list of dict candidates with a composite score —
suitable for inspection in a notebook or persistence under
``artifacts/experiment_window.json``.

Pure polars + numpy. No I/O here; the caller passes already-loaded frames.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl


# ---------- helpers --------------------------------------------------------

_TS_COL_DEFAULT = "open_time"
_LABEL_COL_DEFAULT = "label"
_RETURN_COL_DEFAULT = "return"
_PARKINSON_COL_DEFAULT = "parkinson_var"


def _bracket_score(value: float, lo: float, hi: float, ideal: float) -> float:
    """0..1 score that peaks at ``ideal`` and decays linearly to 0 at the
    boundaries [lo, hi]. Outside the band returns 0.0."""
    if not np.isfinite(value):
        return 0.0
    if value < lo or value > hi:
        return 0.0
    if value == ideal:
        return 1.0
    if value < ideal:
        return float((value - lo) / (ideal - lo)) if ideal > lo else 1.0
    return float((hi - value) / (hi - ideal)) if hi > ideal else 1.0


def _composite_score(
    *,
    event_rate: float,
    up_frac: float,
    parkinson_cv: float,
    parkinson_span: float,
    event_rate_min: float,
    event_rate_max: float,
    event_rate_ideal: float = 0.10,
    span_lo: float = 1.5,
    span_hi: float = 3.0,
) -> float:
    """Combine the per-window quality signals into a single 0..1 score.

    Weights:
        * 0.45 — closeness of event rate to ``event_rate_ideal`` inside [min, max]
        * 0.25 — closeness of up-bar fraction to 0.5 (balanced regime)
        * 0.20 — Parkinson span inside the [span_lo, span_hi] sweet spot
        * 0.10 — Parkinson CV preference (mid-range, neither flat nor chaotic)
    """
    s_event = _bracket_score(event_rate, event_rate_min, event_rate_max, event_rate_ideal)
    s_balance = _bracket_score(up_frac, 0.30, 0.70, 0.50)
    s_span = _bracket_score(parkinson_span, span_lo, span_hi, (span_lo + span_hi) / 2.0)
    # CV "sweet spot": ~0.5..1.5 (variation without being explosive).
    s_cv = _bracket_score(parkinson_cv, 0.2, 2.0, 0.8)
    return float(0.45 * s_event + 0.25 * s_balance + 0.20 * s_span + 0.10 * s_cv)


def _summarise_window(
    sub: pl.DataFrame,
    feats_sub: pl.DataFrame,
    *,
    label_col: str,
    return_col: str,
    parkinson_col: str,
) -> Optional[dict]:
    """Compute summary stats for a single candidate window."""
    if sub.is_empty():
        return None
    valid = sub.filter(pl.col(label_col).is_not_null())
    n = valid.height
    if n == 0:
        return None
    event_rate = float(valid[label_col].cast(pl.Float64).mean() or 0.0)

    if return_col in feats_sub.columns:
        rets = feats_sub[return_col].drop_nulls().to_numpy()
    else:
        # Fall back to bar-over-bar log returns from close.
        closes = sub["close"].drop_nulls().to_numpy() if "close" in sub.columns else np.array([])
        rets = np.diff(np.log(closes)) if len(closes) > 1 else np.array([])

    if rets.size == 0:
        up_frac = down_frac = 0.0
    else:
        up_frac = float((rets > 0).mean())
        down_frac = float((rets < 0).mean())

    if parkinson_col in feats_sub.columns:
        pv = feats_sub[parkinson_col].drop_nulls().to_numpy()
    else:
        pv = np.array([])

    pv_pos = pv[pv > 0]
    if pv_pos.size >= 2:
        mean = float(pv_pos.mean())
        std = float(pv_pos.std(ddof=0))
        parkinson_cv = float(std / mean) if mean > 0 else 0.0
        parkinson_span = float(pv_pos.max() / pv_pos.min())
    else:
        parkinson_cv = 0.0
        parkinson_span = 1.0

    return {
        "n_bars": int(n),
        "event_rate": event_rate,
        "up_frac": up_frac,
        "down_frac": down_frac,
        "parkinson_cv": parkinson_cv,
        "parkinson_span": parkinson_span,
    }


# ---------- public API -----------------------------------------------------


def select_test_windows(
    labels_df: pl.DataFrame,
    features_df: pl.DataFrame,
    *,
    window_days: int = 14,
    slide_days: int = 1,
    event_rate_min: float = 0.05,
    event_rate_max: float = 0.15,
    n_top: int = 5,
    ts_col: str = _TS_COL_DEFAULT,
    label_col: str = _LABEL_COL_DEFAULT,
    return_col: str = _RETURN_COL_DEFAULT,
    parkinson_col: str = _PARKINSON_COL_DEFAULT,
) -> list[dict]:
    """Slide a ``window_days`` window over ``labels_df`` and rank candidates.

    Both ``labels_df`` and ``features_df`` MUST share the same timestamp
    column (``ts_col``, default ``open_time``, expected ms-since-epoch
    Int64). They are joined on the timestamp column and a per-window
    summary is computed for each candidate.

    Filters: ``event_rate`` must lie in ``[event_rate_min, event_rate_max]``.

    Returns
    -------
    list[dict]
        Up to ``n_top`` highest-scoring windows. Each dict has keys
        ``start_ts``, ``end_ts``, ``event_rate``, ``up_frac``, ``down_frac``,
        ``parkinson_cv``, ``parkinson_span``, ``score``, ``n_bars``.
    """
    if labels_df.is_empty():
        return []
    if ts_col not in labels_df.columns:
        raise ValueError(f"labels_df must contain timestamp column '{ts_col}'")
    if label_col not in labels_df.columns:
        raise ValueError(f"labels_df must contain label column '{label_col}'")
    if ts_col not in features_df.columns:
        raise ValueError(f"features_df must contain timestamp column '{ts_col}'")

    labels_df = labels_df.sort(ts_col)
    features_df = features_df.sort(ts_col)

    ts = labels_df[ts_col].to_numpy()
    if ts.size == 0:
        return []

    # ms-since-epoch arithmetic.
    day_ms = 86_400_000
    win_ms = int(window_days * day_ms)
    slide_ms = int(slide_days * day_ms)
    t0 = int(ts.min())
    t_end = int(ts.max())
    if t_end - t0 < win_ms:
        return []

    # Slim down features once.
    feats_keep = [ts_col]
    for c in (return_col, parkinson_col):
        if c in features_df.columns:
            feats_keep.append(c)
    feats_slim = features_df.select(feats_keep)

    candidates: list[dict] = []
    start = t0
    last_start = t_end - win_ms
    while start <= last_start:
        end = start + win_ms
        sub_labels = labels_df.filter(
            (pl.col(ts_col) >= start) & (pl.col(ts_col) < end)
        )
        sub_feats = feats_slim.filter(
            (pl.col(ts_col) >= start) & (pl.col(ts_col) < end)
        )
        summary = _summarise_window(
            sub_labels, sub_feats,
            label_col=label_col,
            return_col=return_col,
            parkinson_col=parkinson_col,
        )
        if summary is None:
            start += slide_ms
            continue

        if not (event_rate_min <= summary["event_rate"] <= event_rate_max):
            start += slide_ms
            continue

        score = _composite_score(
            event_rate=summary["event_rate"],
            up_frac=summary["up_frac"],
            parkinson_cv=summary["parkinson_cv"],
            parkinson_span=summary["parkinson_span"],
            event_rate_min=event_rate_min,
            event_rate_max=event_rate_max,
        )
        candidates.append({
            "start_ts": int(start),
            "end_ts": int(end),
            "event_rate": summary["event_rate"],
            "up_frac": summary["up_frac"],
            "down_frac": summary["down_frac"],
            "parkinson_cv": summary["parkinson_cv"],
            "parkinson_span": summary["parkinson_span"],
            "score": score,
            "n_bars": summary["n_bars"],
        })
        start += slide_ms

    candidates.sort(key=lambda d: d["score"], reverse=True)
    return candidates[:n_top]


__all__ = ["select_test_windows"]
