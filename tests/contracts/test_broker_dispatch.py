"""SimBroker.dispatch contracts — exercises every ActionKind path.

These tests construct DecisionBar + minute tuples by hand so they avoid
the parquet path. Covers HOLD, OPEN (with cap), SCALE_IN/SCALE_OUT/CLOSE/
MODIFY_STOP/CANCEL, target-id validation, and emitted events.
"""

from __future__ import annotations

from wagie.core.action import Action, ActionKind, ExitReason, Side
from wagie.core.event import (
    DecisionBar, MinuteBar, PositionClosed, PositionOpened,
    PositionScaledOut, PositionStopModified,
)
from wagie.core.identity import DEFAULT_INSTRUMENT, InstrumentId, PositionId
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.time import Duration, Timestamp
from wagie.io.brokers import SimBroker


# ---- helpers ----

INSTR: InstrumentId = DEFAULT_INSTRUMENT
TP = LogReturn.from_bps(80)   # large enough not to hit on flat tape
SL = LogReturn.from_bps(80)
EXP = Duration.from_minutes(20)


def _flat_minute(ts_ms_start: int, idx: int, price: float = 100.0) -> MinuteBar:
    open_ms = ts_ms_start + idx * 60_000
    close_ms = open_ms + 59_999
    return MinuteBar(
        ts_init=Timestamp.from_ms(open_ms),
        instrument=INSTR,
        open=Price(price), high=Price(price * 1.0001),
        low=Price(price * 0.9999), close=Price(price),
        volume=Quantity(1.0),
        ts_close=Timestamp.from_ms(close_ms),
    )


def _flat_decision_bar(ts_ms_start: int, m: int = 20, price: float = 100.0) -> DecisionBar:
    minutes = tuple(_flat_minute(ts_ms_start, i, price) for i in range(m))
    return DecisionBar(
        ts_init=Timestamp.from_ms(ts_ms_start),
        instrument=INSTR,
        open=Price(price), high=Price(price * 1.0001),
        low=Price(price * 0.9999), close=Price(price),
        volume=Quantity(float(m)),
        duration=Duration.from_minutes(m),
        underlying=minutes,
    )


def _open_action() -> Action:
    return Action.open(side=Side.LONG, size=Probability(1.0),
                       take_profit=TP, stop_loss=SL, expiry=EXP)


def _activate_pending(broker: SimBroker, ts_ms_start: int) -> DecisionBar:
    bar = _flat_decision_bar(ts_ms_start)
    broker.advance_to(bar.close_time, bar)
    return bar


# ---- HOLD ----


def test_dispatch_hold_returns_true_no_op():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    assert bro.dispatch(Action.hold(), ts=ts, instrument=INSTR) is True
    assert bro.open_orders() == 0
    assert bro.portfolio().n_open == 0


# ---- OPEN: cap respected ----


def test_open_cap_blocks_n_plus_one():
    bro = SimBroker(inventory_cap=2)
    ts = Timestamp.from_ms(1_700_000_000_000)
    assert bro.dispatch(_open_action(), ts=ts, instrument=INSTR) is True
    assert bro.dispatch(_open_action(), ts=ts, instrument=INSTR) is True
    # cap=2 reached (counts pending opens too)
    assert bro.dispatch(_open_action(), ts=ts, instrument=INSTR) is False
    assert bro.open_orders() == 2


# ---- target_position_id=None must be rejected for ref'ing kinds ----


def test_scale_in_rejects_none_target():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    bad = Action(kind=ActionKind.SCALE_IN, target_position_id=None,
                 size=Probability(0.5))
    assert bro.dispatch(bad, ts=ts, instrument=INSTR) is False


def test_scale_out_rejects_none_target():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    bad = Action(kind=ActionKind.SCALE_OUT, target_position_id=None,
                 fraction=Probability(0.5))
    assert bro.dispatch(bad, ts=ts, instrument=INSTR) is False


def test_close_rejects_none_target():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    bad = Action(kind=ActionKind.CLOSE, target_position_id=None)
    assert bro.dispatch(bad, ts=ts, instrument=INSTR) is False


def test_modify_stop_rejects_none_target():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    bad = Action(kind=ActionKind.MODIFY_STOP, target_position_id=None,
                 new_stop_loss=LogReturn.from_bps(30))
    assert bro.dispatch(bad, ts=ts, instrument=INSTR) is False


def test_cancel_rejects_none_target():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    bad = Action(kind=ActionKind.CANCEL, target_position_id=None)
    assert bro.dispatch(bad, ts=ts, instrument=INSTR) is False


# ---- SCALE_IN against missing position ----


def test_scale_in_missing_position_returns_false():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    a = Action.scale_in(PositionId(999), size=Probability(0.5))
    assert bro.dispatch(a, ts=ts, instrument=INSTR) is False


# ---- SCALE_OUT realizes the right fraction ----


def test_scale_out_realizes_fraction_emits_event():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    assert bro.dispatch(_open_action(), ts=ts0, instrument=INSTR) is True
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))
    pos = bro.portfolio().get(pid)
    initial_size = float(pos.current_size)

    ts1 = bar.close_time
    a = Action.scale_out(pid, fraction=Probability(0.4))
    assert bro.dispatch(a, ts=ts1, instrument=INSTR) is True

    pos2 = bro.portfolio().get(pid)
    # after a 0.4 scale-out we expect ~0.6 of original size remaining
    assert abs(float(pos2.current_size) - initial_size * 0.6) < 1e-9
    assert not pos2.closed
    # event emitted
    sout = [e for e in bro.events if isinstance(e, PositionScaledOut)]
    assert len(sout) == 1
    assert sout[-1].position_id == int(pid)
    assert abs(sout[-1].size_removed - initial_size * 0.4) < 1e-9


def test_scale_out_full_fraction_closes_position():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    assert bro.dispatch(_open_action(), ts=ts0, instrument=INSTR) is True
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))

    a = Action.scale_out(pid, fraction=Probability(1.0))
    assert bro.dispatch(a, ts=bar.close_time, instrument=INSTR) is True
    pos = bro.portfolio().get(pid)
    assert pos.closed is True
    # closing scale-out emits PositionClosed too
    closes = [e for e in bro.events if isinstance(e, PositionClosed)]
    assert len(closes) >= 1


# ---- CLOSE realizes pnl + emits PositionClosed ----


def test_close_realizes_and_emits():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    assert bro.dispatch(_open_action(), ts=ts0, instrument=INSTR) is True
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))

    a = Action.close(pid)
    assert bro.dispatch(a, ts=bar.close_time, instrument=INSTR) is True
    pos = bro.portfolio().get(pid)
    assert pos.closed is True
    assert pos.close_reason == ExitReason.MANUAL
    # PositionClosed event present
    closes = [e for e in bro.events if isinstance(e, PositionClosed)]
    assert any(c.position_id == int(pid) for c in closes)


def test_close_already_closed_returns_false():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    bro.dispatch(_open_action(), ts=ts0, instrument=INSTR)
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))

    assert bro.dispatch(Action.close(pid), ts=bar.close_time, instrument=INSTR) is True
    # second close on a closed position fails
    assert bro.dispatch(Action.close(pid), ts=bar.close_time, instrument=INSTR) is False


# ---- MODIFY_STOP updates SL/TP and emits ----


def test_modify_stop_updates_levels_and_emits():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    bro.dispatch(_open_action(), ts=ts0, instrument=INSTR)
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))

    new_tp = LogReturn.from_bps(123.0)
    new_sl = LogReturn.from_bps(45.0)
    a = Action.modify_stop(pid, new_take_profit=new_tp, new_stop_loss=new_sl)
    assert bro.dispatch(a, ts=bar.close_time, instrument=INSTR) is True

    pos = bro.portfolio().get(pid)
    assert abs(float(pos.take_profit_log) - float(new_tp)) < 1e-12
    assert abs(float(pos.stop_loss_log) - float(new_sl)) < 1e-12
    mods = [e for e in bro.events if isinstance(e, PositionStopModified)]
    assert len(mods) == 1
    assert mods[0].position_id == int(pid)
    assert mods[0].actor == "strategy"


# ---- CANCEL removes pending open before activation ----


def test_cancel_removes_pending_open():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    bro.dispatch(_open_action(), ts=ts0, instrument=INSTR)
    # one pending; portfolio still empty
    assert bro.portfolio().n_open == 0
    assert bro.open_orders() == 1

    pid = PositionId(1)  # first allocated id
    assert bro.dispatch(Action.cancel(pid), ts=ts0, instrument=INSTR) is True
    assert bro.open_orders() == 0
    # subsequent advance does nothing (no pending, no open)
    bar = _activate_pending(bro, base_ms)
    assert bro.portfolio().n_open == 0


def test_cancel_unknown_id_returns_false():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    assert bro.dispatch(Action.cancel(PositionId(42)), ts=ts, instrument=INSTR) is False


# ---- After dispatch chain, portfolio reflects state ----


def test_portfolio_reflects_chain_of_dispatches():
    base_ms = 1_700_000_000_000
    bro = SimBroker(inventory_cap=3)
    ts = Timestamp.from_ms(base_ms)

    # open three
    for _ in range(3):
        assert bro.dispatch(_open_action(), ts=ts, instrument=INSTR) is True
    bar = _activate_pending(bro, base_ms)
    assert bro.portfolio().n_open == 3

    pids = sorted(int(p) for p in bro.portfolio().positions.keys())
    # close first, scale_out half on second, leave third open
    assert bro.dispatch(Action.close(PositionId(pids[0])),
                        ts=bar.close_time, instrument=INSTR) is True
    assert bro.dispatch(
        Action.scale_out(PositionId(pids[1]), fraction=Probability(0.5)),
        ts=bar.close_time, instrument=INSTR) is True

    pf = bro.portfolio()
    p0 = pf.get(PositionId(pids[0]))
    p1 = pf.get(PositionId(pids[1]))
    p2 = pf.get(PositionId(pids[2]))
    assert p0.closed is True
    assert not p1.closed and float(p1.current_size) < 1.0
    assert not p2.closed and float(p2.current_size) == 1.0
    assert pf.n_open == 2


# ---- unknown ActionKind path returns False ----


def test_unknown_kind_returns_false():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)

    class _Phony:
        kind = "phony"
        target_position_id = None
        side = None
        size = None
        fraction = None
        new_take_profit = None
        new_stop_loss = None
        expiry = None
        take_profit = None
        stop_loss = None
    assert bro.dispatch(_Phony(), ts=ts, instrument=INSTR) is False


# ---- Additional edge-case coverage ---------------------------------------


def test_scale_in_pending_activates_on_advance():
    """SCALE_IN against a real position queues a pending fill that activates
    on the next advance_to (emits PositionScaledIn event).

    Use a generous expiry so the original position doesn't time out on the
    second bar.
    """
    from wagie.core.event import PositionScaledIn
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    long_open = Action.open(side=Side.LONG, size=Probability(1.0),
                            take_profit=TP, stop_loss=SL,
                            expiry=Duration.from_minutes(200))
    bro.dispatch(long_open, ts=ts0, instrument=INSTR)
    _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))
    pos = bro.portfolio().get(pid)
    initial_size = float(pos.current_size)

    # queue scale-in on the next bar
    next_base = base_ms + 20 * 60_000
    ts1 = Timestamp.from_ms(next_base)
    a = Action.scale_in(pid, size=Probability(0.5))
    assert bro.dispatch(a, ts=ts1, instrument=INSTR) is True
    _activate_pending(bro, next_base)
    pos2 = bro.portfolio().get(pid)
    assert not pos2.closed
    assert float(pos2.current_size) > initial_size
    sins = [e for e in bro.events if isinstance(e, PositionScaledIn)]
    assert len(sins) == 1


def test_scale_out_on_closed_returns_false():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    bro.dispatch(_open_action(), ts=ts0, instrument=INSTR)
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))
    # close it
    assert bro.dispatch(Action.close(pid), ts=bar.close_time, instrument=INSTR) is True
    # subsequent scale_out fails
    assert bro.dispatch(
        Action.scale_out(pid, fraction=Probability(0.5)),
        ts=bar.close_time, instrument=INSTR) is False


def test_scale_out_zero_fraction_returns_false():
    """fraction == 0 -> size_to_remove <= 0 -> return False (line 173)."""
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    bro.dispatch(_open_action(), ts=ts0, instrument=INSTR)
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))
    a = Action.scale_out(pid, fraction=Probability(0.0))
    assert bro.dispatch(a, ts=bar.close_time, instrument=INSTR) is False


def test_modify_stop_on_closed_returns_false():
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    ts0 = Timestamp.from_ms(base_ms)
    bro.dispatch(_open_action(), ts=ts0, instrument=INSTR)
    bar = _activate_pending(bro, base_ms)
    pid = next(iter(bro.portfolio().positions.keys()))
    bro.dispatch(Action.close(pid), ts=bar.close_time, instrument=INSTR)
    a = Action.modify_stop(pid, new_stop_loss=LogReturn.from_bps(10))
    assert bro.dispatch(a, ts=bar.close_time, instrument=INSTR) is False


def test_modify_stop_unknown_position_returns_false():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    a = Action.modify_stop(PositionId(999), new_stop_loss=LogReturn.from_bps(10))
    assert bro.dispatch(a, ts=ts, instrument=INSTR) is False


def test_close_unknown_position_returns_false():
    bro = SimBroker()
    ts = Timestamp.from_ms(1_700_000_000_000)
    a = Action.close(PositionId(999))
    assert bro.dispatch(a, ts=ts, instrument=INSTR) is False


def test_legacy_position_and_pnl_accessors():
    """Cover .position(), .realized_pnl_log() legacy accessors (lines 95, 101)."""
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    assert bro.position() == 0.0
    assert bro.realized_pnl_log() == 0.0
    bro.dispatch(_open_action(), ts=Timestamp.from_ms(base_ms), instrument=INSTR)
    bar = _activate_pending(bro, base_ms)
    # After activation, position() reflects net signed size
    assert bro.position() != 0.0
    pid = next(iter(bro.portfolio().positions.keys()))
    bro.dispatch(Action.close(pid), ts=bar.close_time, instrument=INSTR)
    # realized_pnl_log accessor returns float
    assert isinstance(bro.realized_pnl_log(), float)


def test_finalize_returns_ledger():
    bro = SimBroker()
    ledger = bro.finalize()
    assert ledger.fills == []
    assert ledger.n_open_at_finalize == 0
    assert ledger.config["m_minutes"] == 20


def test_legacy_cancel_method_removes_pending():
    """SimBroker.cancel(order_id) is the legacy single-instrument hook (line 437)."""
    from wagie.core.identity import OrderId
    base_ms = 1_700_000_000_000
    bro = SimBroker()
    bro.dispatch(_open_action(), ts=Timestamp.from_ms(base_ms), instrument=INSTR)
    assert bro.open_orders() == 1
    bro.cancel(OrderId(1))   # position_id == 1 for first open
    assert bro.open_orders() == 0


def test_advance_to_with_short_minute_tuple_skips():
    """If bar.underlying has fewer than m_minutes, advance_to short-circuits
    (lines 252-256). Bar=None case also."""
    bro = SimBroker(m_minutes=20)
    ts = Timestamp.from_ms(1_700_000_000_000)
    # bar=None branch
    assert bro.advance_to(ts, None) == []
    # short minute tuple
    base_ms = 1_700_000_000_000
    short_minutes = tuple(_flat_minute(base_ms, i) for i in range(5))
    short_bar = DecisionBar(
        ts_init=Timestamp.from_ms(base_ms),
        instrument=INSTR,
        open=Price(100.0), high=Price(100.0001),
        low=Price(99.9999), close=Price(100.0),
        volume=Quantity(5.0),
        duration=Duration.from_minutes(5),
        underlying=short_minutes,
    )
    assert bro.advance_to(short_bar.close_time, short_bar) == []


def test_binance_broker_stub():
    """Cover the BinanceBroker stub paths."""
    from wagie.io.brokers import BinanceBroker
    bb = BinanceBroker(foo="bar")
    assert bb.position() == 0.0
    assert bb.open_orders() == 0
    assert bb.realized_pnl_log() == 0.0
    assert bb.portfolio().n_open == 0
    assert bb.advance_to(None, None) == []
    led = bb.finalize()
    assert led.fills == []
    assert led.n_open_at_finalize == 0
    import pytest as _pytest
    with _pytest.raises(NotImplementedError):
        bb.submit(None)
    with _pytest.raises(NotImplementedError):
        bb.cancel(0)
