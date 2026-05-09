"""Contract tests for LabelBuffer (predict-then-learn-with-delayed-label staging)."""

from __future__ import annotations

import math

import pytest

from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import LogReturn, Price, Quantity
from wagie.core.observation import Observation
from wagie.core.time import Duration, Timestamp
from wagie.pipeline.label_buffer import LabelBuffer


def _bar(close: float = 100.0, segment_id: int = 0, ts_ns: int = 1_000_000_000,
         high: float | None = None) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(ts_ns),
        instrument=DEFAULT_INSTRUMENT,
        open=Price(close),
        high=Price(high if high is not None else close * 1.01),
        low=Price(close * 0.99),
        close=Price(close),
        volume=Quantity(1.0),
        duration=Duration.from_minutes(20),
        segment_id=segment_id,
    )


def _obs(close: float = 100.0, segment_id: int = 0, ts_ns: int = 1_000_000_000) -> Observation:
    return Observation(bar=_bar(close=close, segment_id=segment_id, ts_ns=ts_ns))


class _NextBar:
    """Lightweight stand-in for the next bar (only .high and .segment_id read)."""

    def __init__(self, high: float, segment_id: int = 0):
        self.high = high
        self.segment_id = segment_id


# ---------------------------------------------------------------------------
# transform records and advances n_seen
# ---------------------------------------------------------------------------

def test_transform_records_and_advances_n_seen():
    lb = LabelBuffer(alpha_label=LogReturn.from_bps(40.0), max_buffer=4)
    o = _obs(close=100.0, segment_id=0, ts_ns=1_000_000_000)
    n0 = lb.n_seen
    lb.transform(o)
    assert lb.n_seen == n0 + 1
    assert len(lb._records) == 1
    rec = lb._records[0]
    assert rec.obs is o
    assert rec.ref_close == 100.0
    assert rec.segment_id == 0
    assert rec.ts_close_ns == int(o.bar.close_time.ns)


def test_transform_returns_obs_unchanged():
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    o = _obs()
    out = lb.transform(o)
    assert out is o


# ---------------------------------------------------------------------------
# maybe_emit
# ---------------------------------------------------------------------------

def test_maybe_emit_empty_buffer_returns_none():
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    assert lb.maybe_emit(_NextBar(high=110.0)) is None


def test_maybe_emit_segment_boundary_suppression():
    """If next_bar segment changes, the popped record yields None (no false label)."""
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    lb.transform(_obs(close=100.0, segment_id=0))
    assert len(lb._records) == 1
    out = lb.maybe_emit(_NextBar(high=110.0, segment_id=1))
    assert out is None
    # The record was popleft'd (consumed) — buffer empty.
    assert len(lb._records) == 0


def test_maybe_emit_nan_next_high_returns_none():
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    lb.transform(_obs(close=100.0, segment_id=0))
    out = lb.maybe_emit(_NextBar(high=float("nan"), segment_id=0))
    assert out is None


def test_maybe_emit_nonpositive_ref_close_returns_none():
    """ref_close <= 0 yields None (mathematical: log(x/0) is undefined).

    Direct ref_close manipulation since Price prevents <= 0 construction.
    """
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    lb.transform(_obs(close=100.0, segment_id=0))
    # Tamper the record's ref_close to simulate a bad upstream value.
    lb._records[0].ref_close = 0.0
    out = lb.maybe_emit(_NextBar(high=110.0, segment_id=0))
    assert out is None


def test_maybe_emit_label_one_when_excursion_meets_alpha():
    """alpha_label=0 → any positive log_excursion qualifies → y=1."""
    lb = LabelBuffer(alpha_label=0.0, max_buffer=4)
    o = _obs(close=100.0, segment_id=0)
    lb.transform(o)
    out = lb.maybe_emit(_NextBar(high=101.0, segment_id=0))
    assert out is not None
    obs_prev, y = out
    assert obs_prev is o
    assert y == 1


def test_maybe_emit_label_zero_when_alpha_too_large():
    """Very large alpha_label → no realistic excursion qualifies → y=0."""
    lb = LabelBuffer(alpha_label=10.0, max_buffer=4)  # ~0.001%-style alphas; 10.0 is huge
    o = _obs(close=100.0, segment_id=0)
    lb.transform(o)
    out = lb.maybe_emit(_NextBar(high=101.0, segment_id=0))
    assert out is not None
    _, y = out
    assert y == 0


def test_maybe_emit_returns_none_on_invalid_next_bar():
    """next_bar without .high/.segment_id returns None gracefully."""
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    lb.transform(_obs(close=100.0))

    class Bad:
        # missing .high
        segment_id = 0

    out = lb.maybe_emit(Bad())
    assert out is None


# ---------------------------------------------------------------------------
# Deque eviction
# ---------------------------------------------------------------------------

def test_max_buffer_evicts_oldest():
    lb = LabelBuffer(alpha_label=0.001, max_buffer=2)
    lb.transform(_obs(close=100.0, ts_ns=1_000_000_000))
    lb.transform(_obs(close=101.0, ts_ns=2_000_000_000))
    lb.transform(_obs(close=102.0, ts_ns=3_000_000_000))
    assert len(lb._records) == 2
    # Oldest (close=100, ts=1e9) was evicted; newest two remain.
    assert lb._records[0].ref_close == 101.0
    assert lb._records[1].ref_close == 102.0


# ---------------------------------------------------------------------------
# state_hash changes after transform
# ---------------------------------------------------------------------------

def test_state_hash_differs_after_transform():
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    h0 = lb.state_hash()
    lb.transform(_obs(close=100.0, segment_id=0, ts_ns=1_000_000_000))
    h1 = lb.state_hash()
    assert h0 != h1


def test_state_dict_round_trip():
    lb = LabelBuffer(alpha_label=0.005, max_buffer=4)
    lb.transform(_obs(close=100.0))
    sd = lb.state_dict()
    assert sd["alpha_label"] == pytest.approx(0.005)
    assert sd["n_seen"] == 1
    assert sd["n_records"] == 1

    lb2 = LabelBuffer(alpha_label=0.0, max_buffer=4)
    lb2.load_state_dict(sd)
    assert lb2.alpha_label == pytest.approx(0.005)
    assert lb2.n_seen == 1


def test_reset_clears_records_and_n_seen():
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    lb.transform(_obs(close=100.0))
    lb.transform(_obs(close=101.0))
    assert lb.n_seen == 2
    lb.reset()
    assert lb.n_seen == 0
    assert len(lb._records) == 0


def test_update_is_noop():
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    h0 = lb.state_hash()
    lb.update(_obs(), label=1)
    assert lb.state_hash() == h0


def test_transform_handles_bad_obs_gracefully():
    """If obs.bar fields are unreadable, transform returns obs unchanged (no record)."""
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)

    class Stub:
        # missing .bar attribute → AttributeError → caught by except: clause
        pass

    out = lb.transform(Stub())  # type: ignore[arg-type]
    assert isinstance(out, Stub)
    assert lb.n_seen == 0
    assert len(lb._records) == 0
