"""Streaming features must reset their internal state at segment boundaries.

The cleansed-data invariant: a `segment_id` advances at any 1-min gap in the
input. Streaming feature buffers (lag deques, rolling stats, warmup counts)
MUST NOT carry values from one segment into the next — silent contamination.
"""

from __future__ import annotations

import math

import pytest

from wagie.features.streaming import (
    LagFeature, RiverRollingFeature, make_lag, make_rolling_mean,
)


# ---------------------------------------------------------------------------
# RiverRollingFeature segment reset
# ---------------------------------------------------------------------------


def test_river_rolling_resets_at_segment_boundary():
    """When `segment_id` changes, the rolling stat AND the lag buffer reset.
    Warmup restarts; the first warmup-many bars in the new segment emit NaN."""
    f = make_rolling_mean("m_4", "x", window=4, lag=1)
    # Warm the feature inside segment 0.
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        out = f.update_one({"x": v, "segment_id": 0})
    # By now the rolling mean has produced a finite value.
    assert math.isfinite(out["m_4"])
    finite_in_seg0 = float(out["m_4"])

    # Cross a segment boundary: first bar in segment=1 must drop to NaN
    # (warmup restarts).
    out_first_new = f.update_one({"x": 99.0, "segment_id": 1})
    assert math.isnan(out_first_new["m_4"]), \
        "first bar in new segment must restart warmup → NaN"

    # The internal n_seen counter must have been reset to 1 (we just consumed
    # the first bar of the new segment).
    assert f._n_seen == 1
    # The lag buffer should hold ONLY the new segment's value.
    assert list(f._lag_buf) == [99.0]

    # Continue in segment 1; the rolling stat must build from scratch and
    # MUST NOT mention any value from segment 0.
    for v in (100.0, 101.0, 102.0, 103.0):
        out = f.update_one({"x": v, "segment_id": 1})
    # The mean over the most recent 4 values in segment 1 should be in the
    # neighbourhood of [99, 103]. If segment 0 had bled through, we'd see a
    # value in [3, 50].
    final = float(out["m_4"])
    assert 99.0 <= final <= 103.0, \
        f"rolling mean leaked across segments; got {final}, expected ~100"


def test_river_rolling_no_reset_within_same_segment():
    """Sanity: identical segment_id does NOT cause a reset."""
    f = make_rolling_mean("m_4", "x", window=4, lag=1)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        out = f.update_one({"x": v, "segment_id": 7})
    assert math.isfinite(out["m_4"])
    n_before = f._n_seen
    out = f.update_one({"x": 6.0, "segment_id": 7})
    assert math.isfinite(out["m_4"])
    assert f._n_seen == n_before + 1, \
        "same segment_id must not trigger a reset"


def test_river_rolling_first_seen_segment_is_adopted_silently():
    """The first observed segment_id is adopted as `_last_segment_id` without
    triggering a spurious reset."""
    f = make_rolling_mean("m_4", "x", window=4, lag=1)
    # First bar: segment_id=42. There's no prior segment, so no reset.
    f.update_one({"x": 1.0, "segment_id": 42})
    assert f._n_seen == 1
    assert f._last_segment_id == 42


def test_river_rolling_handles_missing_segment_id():
    """If `segment_id` is absent from the bar dict, the feature treats it as
    a no-op (no reset)."""
    f = make_rolling_mean("m_4", "x", window=4, lag=1)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        f.update_one({"x": v})  # no segment_id
    n_before = f._n_seen
    out = f.update_one({"x": 6.0})
    assert f._n_seen == n_before + 1
    # Still finite, no reset triggered.
    assert math.isfinite(out["m_4"])


def test_river_rolling_repeated_boundary_crossings():
    """Multiple segment crossings each trigger an independent reset; values
    never bleed across."""
    f = make_rolling_mean("m_4", "x", window=4, lag=1)
    # Seg 0: warm up
    for v in (10.0, 20.0, 30.0, 40.0, 50.0):
        f.update_one({"x": v, "segment_id": 0})
    # Seg 1: also warm up
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        out_seg1 = f.update_one({"x": v, "segment_id": 1})
    assert math.isfinite(out_seg1["m_4"])
    seg1_value = float(out_seg1["m_4"])
    # Seg 2: first bar -> NaN (warmup restart)
    out_seg2_first = f.update_one({"x": 1000.0, "segment_id": 2})
    assert math.isnan(out_seg2_first["m_4"])
    # State carries no echo of seg1 / seg0
    assert list(f._lag_buf) == [1000.0]


# ---------------------------------------------------------------------------
# Reset_state covers _last_segment_id too
# ---------------------------------------------------------------------------


def test_reset_state_clears_segment_tracking():
    f = make_rolling_mean("m_4", "x", window=4, lag=1)
    f.update_one({"x": 1.0, "segment_id": 5})
    assert f._last_segment_id == 5
    f.reset_state()
    # After reset, the feature is uninitialised again — first new bar adopts
    # whatever segment_id appears, no spurious reset.
    assert f._last_segment_id is None
    f.update_one({"x": 99.0, "segment_id": 9})
    assert f._last_segment_id == 9
    assert f._n_seen == 1
