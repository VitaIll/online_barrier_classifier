"""G8 contracts: Position lifecycle + Portfolio aggregation invariants."""

from __future__ import annotations

import math

import pytest

from wagie.core import (
    DEFAULT_INSTRUMENT, Duration, ExitReason, LogReturn, Portfolio, Position,
    Price, Probability, Quantity, Side, Timestamp,
)
from wagie.core.identity import PositionId


def _mk_position(side=Side.LONG, size=1.0, entry=100.0):
    return Position.open_at(
        position_id=PositionId(1),
        instrument=DEFAULT_INSTRUMENT,
        side=side,
        size=Quantity(size),
        entry_price=Price(entry),
        ts=Timestamp(1_700_000_000_000_000_000),
        take_profit_log=LogReturn.from_bps(40),
        stop_loss_log=LogReturn.from_bps(40),
        expiry=Duration.from_minutes(20),
    )


def test_position_open_state():
    p = _mk_position()
    assert not p.closed
    assert float(p.current_size) == 1.0
    assert float(p.realized_pnl_log) == 0.0
    assert float(p.unrealized_pnl_log) == 0.0


def test_position_with_mark_updates_unrealized_pnl():
    p = _mk_position(entry=100.0)
    p2 = p.with_mark(Price(101.0), Timestamp(1_700_000_000_001_000_000))
    expected = math.log(101.0 / 100.0)
    assert abs(float(p2.unrealized_pnl_log) - expected) < 1e-12
    # Original unchanged (immutable)
    assert float(p.unrealized_pnl_log) == 0.0


def test_position_mae_mfe_track_extremes():
    p = _mk_position(entry=100.0)
    p = p.with_mark(Price(99.0), Timestamp(1))    # adverse
    p = p.with_mark(Price(101.5), Timestamp(2))   # favorable
    p = p.with_mark(Price(100.5), Timestamp(3))   # in between
    assert float(p.mae_log) <= float(p.unrealized_pnl_log)   # MAE is the worst
    assert float(p.mfe_log) >= float(p.unrealized_pnl_log)   # MFE is the best


def test_position_with_close_realizes_pnl():
    p = _mk_position(entry=100.0)
    p2 = p.with_close(
        exit_price=Price(101.0),
        ts=Timestamp(2),
        reason=ExitReason.TP,
        cost_log=LogReturn(0.0001),
    )
    assert p2.closed
    assert p2.close_reason == ExitReason.TP
    expected_gross = math.log(101.0 / 100.0)
    expected_net = expected_gross - 2.0 * 0.0001
    assert abs(float(p2.realized_pnl_log) - expected_net) < 1e-12


def test_position_scale_in_updates_avg_entry():
    p = _mk_position(entry=100.0, size=1.0)
    p2 = p.with_scale_in(size_added=Quantity(1.0), price=Price(102.0),
                          ts=Timestamp(2))
    assert float(p2.current_size) == 2.0
    assert float(p2.open_size) == 2.0
    # Volume-weighted average: (1*100 + 1*102) / 2 = 101
    assert abs(float(p2.avg_entry_price) - 101.0) < 1e-9


def test_position_scale_out_realizes_partial():
    p = _mk_position(entry=100.0, size=2.0)
    p = p.with_mark(Price(101.0), Timestamp(1))
    # scale out half (1.0 of 2.0)
    p2 = p.with_scale_out(
        size_removed=Quantity(1.0),
        exit_price=Price(101.0),
        ts=Timestamp(2),
    )
    assert float(p2.current_size) == 1.0
    assert not p2.closed
    expected_realized = 1.0 * math.log(101.0 / 100.0)
    assert abs(float(p2.realized_pnl_log) - expected_realized) < 1e-12


def test_portfolio_starts_empty():
    p = Portfolio()
    assert p.is_flat
    assert p.n_open == 0
    assert float(p.equity_log) == 0.0
    assert float(p.drawdown_log) == 0.0


def test_portfolio_with_position_adds_open():
    pos = _mk_position()
    pf = Portfolio().with_position(pos)
    assert pf.n_open == 1
    assert pf.get(PositionId(1)) is pos


def test_portfolio_equity_is_realized_plus_unrealized():
    pos = _mk_position(entry=100.0).with_mark(Price(101.0), Timestamp(1))
    pf = Portfolio().with_position(pos).with_realized(LogReturn(0.001))
    expected = 0.001 + math.log(101.0 / 100.0)
    assert abs(float(pf.equity_log) - expected) < 1e-12


def test_portfolio_drawdown_negative_after_loss():
    pos = _mk_position(entry=100.0).with_mark(Price(101.0), Timestamp(1))
    pf = Portfolio().with_position(pos)
    pf = pf.with_mark({DEFAULT_INSTRUMENT: Price(101.0)}, Timestamp(1))
    # Now mark down
    new_pos = pos.with_mark(Price(99.0), Timestamp(2))
    pf2 = pf.with_position(new_pos)
    assert float(pf2.drawdown_log) < 0.0


def test_portfolio_net_position_aggregates():
    pos1 = _mk_position(side=Side.LONG, size=1.0)
    pos2_id = PositionId(2)
    import dataclasses
    pos2 = dataclasses.replace(_mk_position(side=Side.SHORT, size=0.5),
                                position_id=pos2_id)
    pf = Portfolio().with_position(pos1).with_position(pos2)
    assert pf.net_signed(DEFAULT_INSTRUMENT) == 0.5  # 1.0 long - 0.5 short
