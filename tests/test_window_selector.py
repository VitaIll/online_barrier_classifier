"""Tests for ``wagie.experiments.window_selector.select_test_windows``."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from wagie.experiments.window_selector import select_test_windows


DAY_MS = 86_400_000
BAR_MS = 20 * 60 * 1000  # 20 minutes


def _synthetic_pair(
    *,
    n_days: int,
    event_rate: float,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a (labels_df, features_df) pair with a target event rate.

    Returns/parkinson are gentle gaussians; labels are i.i.d. Bernoulli with
    the requested rate.
    """
    rng = np.random.default_rng(seed)
    n = (n_days * DAY_MS) // BAR_MS
    n = int(n)
    ts = np.arange(n, dtype=np.int64) * BAR_MS
    labels = (rng.random(n) < event_rate).astype(np.int32)
    # gentle alternating returns so up_frac ~= 0.5
    rets = rng.normal(0.0, 1e-4, size=n)
    pv = np.abs(rng.normal(2e-6, 5e-7, size=n)) + 1e-9

    labels_df = pl.DataFrame({"open_time": ts, "label": labels})
    features_df = pl.DataFrame({
        "open_time": ts,
        "return": rets,
        "parkinson_var": pv,
    })
    return labels_df, features_df


def test_event_rate_below_min_is_rejected():
    """A window with event rate well below ``event_rate_min`` must not appear."""
    labels_df, features_df = _synthetic_pair(n_days=20, event_rate=0.01, seed=1)
    out = select_test_windows(
        labels_df, features_df,
        window_days=14, slide_days=1,
        event_rate_min=0.05, event_rate_max=0.15,
        n_top=5,
    )
    # Either no candidates at all, or none below the floor — same thing.
    assert all(c["event_rate"] >= 0.05 for c in out)
    # Concretely, with event_rate=0.01 we expect zero survivors.
    assert out == []


def test_event_rate_in_range_is_included():
    """A window inside the band must produce at least one ranked candidate."""
    labels_df, features_df = _synthetic_pair(n_days=20, event_rate=0.10, seed=2)
    out = select_test_windows(
        labels_df, features_df,
        window_days=14, slide_days=1,
        event_rate_min=0.05, event_rate_max=0.15,
        n_top=5,
    )
    assert len(out) > 0
    for c in out:
        assert 0.05 <= c["event_rate"] <= 0.15
        assert {"start_ts", "end_ts", "event_rate", "up_frac",
                "down_frac", "parkinson_cv", "parkinson_span", "score"} <= set(c)


def test_score_monotone_in_event_rate_proximity_to_ideal():
    """Holding all else roughly equal, the candidate whose event rate is
    closer to 0.10 must score >= a candidate further away."""
    rng = np.random.default_rng(3)
    n_days = 28
    n = (n_days * DAY_MS) // BAR_MS
    ts = np.arange(int(n), dtype=np.int64) * BAR_MS

    # First half ~6% events (near low edge), second half ~10% events (ideal).
    half = int(n) // 2
    p_first = np.full(half, 0.06)
    p_second = np.full(int(n) - half, 0.10)
    p = np.concatenate([p_first, p_second])
    labels = (rng.random(int(n)) < p).astype(np.int32)
    rets = rng.normal(0.0, 1e-4, size=int(n))
    pv = np.abs(rng.normal(2e-6, 5e-7, size=int(n))) + 1e-9

    labels_df = pl.DataFrame({"open_time": ts, "label": labels})
    features_df = pl.DataFrame({
        "open_time": ts,
        "return": rets,
        "parkinson_var": pv,
    })

    out = select_test_windows(
        labels_df, features_df,
        window_days=10, slide_days=2,
        event_rate_min=0.04, event_rate_max=0.16,
        n_top=20,
    )
    assert len(out) >= 2

    # Sort candidates by |event_rate - 0.10| ascending; their scores must be
    # in non-increasing order (closeness implies higher composite, since the
    # other components are roughly constant by construction).
    by_proximity = sorted(out, key=lambda c: abs(c["event_rate"] - 0.10))
    scores_in_proximity_order = [c["score"] for c in by_proximity]
    # Allow tiny numeric ties (fp noise) but require monotone non-increasing.
    for a, b in zip(scores_in_proximity_order, scores_in_proximity_order[1:]):
        assert a + 1e-9 >= b, (
            f"Score ordering not monotone: {a} should be >= {b}"
        )


def test_short_history_returns_empty():
    """If the history is shorter than the window, no candidates are returned."""
    labels_df, features_df = _synthetic_pair(n_days=5, event_rate=0.10, seed=4)
    out = select_test_windows(
        labels_df, features_df,
        window_days=14, slide_days=1,
        n_top=5,
    )
    assert out == []
