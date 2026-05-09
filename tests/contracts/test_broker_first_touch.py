"""SimBroker first-touch contracts.

Hand-crafted 20-minute DecisionBars with deterministic OHLC sequences
exercise the triple-barrier scan: TP-only, SL-only, both-in-same-minute
(SL wins), no-touch timeout, and the entry-minute exclusion when a
position opens within the same bar. Also asserts the LONG/SHORT TP/SL
asymmetry (LONG TP above entry, SHORT TP below).
"""

from __future__ import annotations

import math

from wagie.core.action import Action, ExitReason, Side
from wagie.core.event import DecisionBar, MinuteBar
from wagie.core.identity import DEFAULT_INSTRUMENT, InstrumentId
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.time import Duration, Timestamp
from wagie.io.brokers import SimBroker


INSTR: InstrumentId = DEFAULT_INSTRUMENT
M = 20
TP_BPS = 50.0   # 50 bps either side -> ~0.5%
SL_BPS = 50.0
EXP = Duration.from_minutes(M)
ENTRY_PRICE = 100.0


def _minute(ts_ms_start: int, idx: int, *, o: float, h: float, l: float, c: float) -> MinuteBar:
    open_ms = ts_ms_start + idx * 60_000
    return MinuteBar(
        ts_init=Timestamp.from_ms(open_ms),
        instrument=INSTR,
        open=Price(o), high=Price(h), low=Price(l), close=Price(c),
        volume=Quantity(1.0),
        ts_close=Timestamp.from_ms(open_ms + 59_999),
    )


def _flat_minute(ts_ms_start: int, idx: int, price: float = ENTRY_PRICE) -> MinuteBar:
    return _minute(ts_ms_start, idx, o=price,
                   h=price * 1.00001, l=price * 0.99999, c=price)


def _build_bar(ts_ms_start: int, minutes: list[MinuteBar]) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp.from_ms(ts_ms_start),
        instrument=INSTR,
        open=minutes[0].open,
        high=Price(max(float(m.high) for m in minutes)),
        low=Price(min(float(m.low) for m in minutes)),
        close=minutes[-1].close,
        volume=Quantity(sum(float(m.volume) for m in minutes)),
        duration=Duration.from_minutes(M),
        underlying=tuple(minutes),
    )


def _open_long_and_activate(bro: SimBroker, ts_ms_start: int):
    """Submit OPEN long, activate via a flat 20-min bar, return (pid, activation_close_ts)."""
    ts0 = Timestamp.from_ms(ts_ms_start)
    a = Action.open(side=Side.LONG, size=Probability(1.0),
                    take_profit=LogReturn.from_bps(TP_BPS),
                    stop_loss=LogReturn.from_bps(SL_BPS),
                    expiry=EXP)
    assert bro.dispatch(a, ts=ts0, instrument=INSTR) is True
    flat_minutes = [_flat_minute(ts_ms_start, i) for i in range(M)]
    bar = _build_bar(ts_ms_start, flat_minutes)
    bro.advance_to(bar.close_time, bar)
    pid = next(iter(bro.portfolio().positions.keys()))
    return pid, bar.close_time


def _open_short_and_activate(bro: SimBroker, ts_ms_start: int):
    ts0 = Timestamp.from_ms(ts_ms_start)
    a = Action.open(side=Side.SHORT, size=Probability(1.0),
                    take_profit=LogReturn.from_bps(TP_BPS),
                    stop_loss=LogReturn.from_bps(SL_BPS),
                    expiry=EXP)
    assert bro.dispatch(a, ts=ts0, instrument=INSTR) is True
    flat_minutes = [_flat_minute(ts_ms_start, i) for i in range(M)]
    bar = _build_bar(ts_ms_start, flat_minutes)
    bro.advance_to(bar.close_time, bar)
    pid = next(iter(bro.portfolio().positions.keys()))
    return pid, bar.close_time


# ---- LONG ----


def test_long_hits_tp_at_minute_5():
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    pid, _ = _open_long_and_activate(bro, base)

    pos = bro.portfolio().get(pid)
    entry = float(pos.avg_entry_price)
    tp_price = entry * math.exp(float(pos.take_profit_log))

    next_base = base + M * 60_000
    minutes = [_flat_minute(next_base, i) for i in range(M)]
    # at minute 5 high crosses tp_price; everywhere else flat
    minutes[5] = _minute(next_base, 5, o=entry, h=tp_price * 1.0001,
                         l=entry * 0.9999, c=entry)
    bar2 = _build_bar(next_base, minutes)
    bro.advance_to(bar2.close_time, bar2)

    closed = bro.portfolio().get(pid)
    assert closed.closed is True
    assert closed.close_reason == ExitReason.TP
    fills = list(bro.fills())
    assert fills[-1].reason == ExitReason.TP
    assert abs(float(fills[-1].exit_price) - tp_price) < 1e-6


def test_long_hits_sl_at_minute_5():
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    pid, _ = _open_long_and_activate(bro, base)
    pos = bro.portfolio().get(pid)
    entry = float(pos.avg_entry_price)
    sl_price = entry * math.exp(-float(pos.stop_loss_log))

    next_base = base + M * 60_000
    minutes = [_flat_minute(next_base, i) for i in range(M)]
    minutes[5] = _minute(next_base, 5, o=entry, h=entry * 1.0001,
                         l=sl_price * 0.9999, c=entry)
    bar2 = _build_bar(next_base, minutes)
    bro.advance_to(bar2.close_time, bar2)

    closed = bro.portfolio().get(pid)
    assert closed.closed is True
    assert closed.close_reason == ExitReason.SL
    fills = list(bro.fills())
    assert fills[-1].reason == ExitReason.SL


def test_long_both_in_same_minute_sl_wins():
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    pid, _ = _open_long_and_activate(bro, base)
    pos = bro.portfolio().get(pid)
    entry = float(pos.avg_entry_price)
    tp_price = entry * math.exp(float(pos.take_profit_log))
    sl_price = entry * math.exp(-float(pos.stop_loss_log))

    next_base = base + M * 60_000
    minutes = [_flat_minute(next_base, i) for i in range(M)]
    # one wide minute that blows through BOTH barriers
    minutes[5] = _minute(next_base, 5, o=entry,
                         h=tp_price * 1.001, l=sl_price * 0.999, c=entry)
    bar2 = _build_bar(next_base, minutes)
    bro.advance_to(bar2.close_time, bar2)
    closed = bro.portfolio().get(pid)
    assert closed.closed is True
    # Worst-case rule: SL wins
    assert closed.close_reason == ExitReason.SL


def test_long_no_touch_timeout_at_last_minute_close():
    base = 1_700_000_000_000
    # Use M-min expiry so the second bar's close_time triggers timeout
    bro = SimBroker(m_minutes=M, cost=0.0)
    pid, activation_close = _open_long_and_activate(bro, base)
    # Next bar: completely flat — no TP/SL touch. expiry == activation_close,
    # which is <= bar.close_time, so TIMEOUT fires at last_minute.close.
    next_base = base + M * 60_000
    minutes = [_flat_minute(next_base, i) for i in range(M)]
    bar2 = _build_bar(next_base, minutes)
    bro.advance_to(bar2.close_time, bar2)
    closed = bro.portfolio().get(pid)
    assert closed.closed is True
    assert closed.close_reason == ExitReason.TIMEOUT
    fills = list(bro.fills())
    assert fills[-1].reason == ExitReason.TIMEOUT
    last_min = minutes[-1]
    assert abs(float(fills[-1].exit_price) - float(last_min.close)) < 1e-9


def test_entry_minute_excluded_when_pos_opens_in_same_bar():
    """If pos.open_ts >= bar.ts_init, scan starts at minute exec_latency.

    Rig the activation bar so minute 0 (the entry minute itself) would have
    crossed the TP, but the post-entry minutes remain flat. The scan must
    skip minute 0 — so no TP fires this bar.
    """
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0, execution_latency_minutes=1)
    ts0 = Timestamp.from_ms(base)
    a = Action.open(side=Side.LONG, size=Probability(1.0),
                    take_profit=LogReturn.from_bps(TP_BPS),
                    stop_loss=LogReturn.from_bps(SL_BPS),
                    expiry=Duration.from_minutes(M * 5))  # expiry beyond bar
    assert bro.dispatch(a, ts=ts0, instrument=INSTR) is True

    # Build a bar where minute 0 has a giant high (would hit TP), but every
    # other minute is flat at the entry close.
    entry = ENTRY_PRICE
    tp_log = LogReturn.from_bps(TP_BPS)
    tp_price = entry * math.exp(float(tp_log))

    minutes = [_flat_minute(base, i) for i in range(M)]
    # spike on minute 0 — pre-entry, must NOT trigger
    minutes[0] = _minute(base, 0, o=entry, h=tp_price * 1.01,
                         l=entry * 0.999, c=entry)
    bar = _build_bar(base, minutes)
    bro.advance_to(bar.close_time, bar)
    pos = bro.portfolio().get(next(iter(bro.portfolio().positions.keys())))
    # Position should still be open — the TP-touch on minute 0 was excluded.
    assert not pos.closed


# ---- SHORT (asymmetric barrier prices) ----


def test_short_tp_is_below_entry_sl_above():
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    pid, _ = _open_short_and_activate(bro, base)
    pos = bro.portfolio().get(pid)
    entry = float(pos.avg_entry_price)
    # For SHORT: TP hit when low <= entry * exp(-tp); SL hit when high >= entry * exp(+sl)
    tp_price = entry * math.exp(-float(pos.take_profit_log))
    sl_price = entry * math.exp(float(pos.stop_loss_log))
    assert tp_price < entry < sl_price

    # Move price DOWN to TP — SHORT profits
    next_base = base + M * 60_000
    minutes = [_flat_minute(next_base, i) for i in range(M)]
    minutes[5] = _minute(next_base, 5, o=entry, h=entry * 1.0001,
                         l=tp_price * 0.9999, c=entry)
    bar2 = _build_bar(next_base, minutes)
    bro.advance_to(bar2.close_time, bar2)
    closed = bro.portfolio().get(pid)
    assert closed.closed is True
    assert closed.close_reason == ExitReason.TP


def test_short_hits_sl_when_price_rises():
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    pid, _ = _open_short_and_activate(bro, base)
    pos = bro.portfolio().get(pid)
    entry = float(pos.avg_entry_price)
    sl_price = entry * math.exp(float(pos.stop_loss_log))

    next_base = base + M * 60_000
    minutes = [_flat_minute(next_base, i) for i in range(M)]
    minutes[7] = _minute(next_base, 7, o=entry, h=sl_price * 1.0001,
                         l=entry * 0.9999, c=entry)
    bar2 = _build_bar(next_base, minutes)
    bro.advance_to(bar2.close_time, bar2)
    closed = bro.portfolio().get(pid)
    assert closed.closed is True
    assert closed.close_reason == ExitReason.SL
