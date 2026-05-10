"""SimBroker batched-winner sweep + safety-cap-only hold-age contracts.

These cover the two structural changes to SimBroker:
  1. Force-TIMEOUT-at-vertical-barrier removed; strategy now owns hold-age.
     Only the broker's MAX_HOLD_BARS hard safety net closes positions that
     never hit a barrier and never receive an Action.close.
  2. When a TP barrier fires on one position, the broker sweeps every other
     open position on the same instrument and closes any currently profitable
     one with reason BATCHED_WINNER. SL touches do NOT trigger any sweep.
"""

from __future__ import annotations

import math

import pytest

from wagie.core.action import Action, ExitReason, Side
from wagie.core.event import DecisionBar, MinuteBar
from wagie.core.identity import DEFAULT_INSTRUMENT, InstrumentId
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.time import Duration, Timestamp
from wagie.io.brokers import MAX_HOLD_BARS, SimBroker


INSTR: InstrumentId = DEFAULT_INSTRUMENT
M = 20
TP_BPS = 50.0
SL_BPS = 50.0
EXP = Duration.from_minutes(M)
ENTRY_PRICE = 100.0


# ---- bar helpers (mirror test_broker_first_touch.py) ---------------------


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


def _open_long_at(bro: SimBroker, ts_ms_start: int, *, entry_price: float = ENTRY_PRICE):
    """Submit OPEN long, activate via a flat 20-min bar AT `entry_price`."""
    ts0 = Timestamp.from_ms(ts_ms_start)
    a = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(TP_BPS),
        stop_loss=LogReturn.from_bps(SL_BPS),
        expiry=Duration.from_minutes(M * 1000),  # large expiry — irrelevant now
    )
    assert bro.dispatch(a, ts=ts0, instrument=INSTR) is True
    flat_minutes = [_flat_minute(ts_ms_start, i, price=entry_price) for i in range(M)]
    bar = _build_bar(ts_ms_start, flat_minutes)
    bro.advance_to(bar.close_time, bar)
    pid = max(bro.portfolio().positions.keys(), key=lambda p: int(p))
    return pid, bar.close_time


# ---- Change 2: batched winners on TP -------------------------------------


def test_two_longs_both_profitable_second_hits_tp_both_close():
    """Two long positions: pos1 entered at p1=100, pos2 entered at p2=99.7
    (pos2 LOWER, so its TP is also lower — it hits TP first without
    triggering pos1's TP). At pos2's TP price ≈ 100.2, pos1 is profitable
    (log(100.2/100) > 0) so pos1 must be swept as BATCHED_WINNER.
    """
    base = 1_700_000_000_000
    # cost=0 so the eps buffer is 0 — any positive net PnL counts as winning
    bro = SimBroker(m_minutes=M, cost=0.0)

    # Open pos1 at price p1 = 100.0
    pid1, _ = _open_long_at(bro, base, entry_price=100.0)
    pos1 = bro.portfolio().get(pid1)
    p1 = float(pos1.avg_entry_price)

    # Open pos2 at price p2 BELOW p1, but above pos1's SL. With TP=SL=50bps:
    # p1 * exp(-50bps) ≈ 99.5 < p2 < p1 = 100  → pick 99.7.
    p2 = 99.7
    base2 = base + M * 60_000
    pid2, _ = _open_long_at(bro, base2, entry_price=p2)
    pos2 = bro.portfolio().get(pid2)

    # pos2's TP price < pos1's TP price — only pos2 will TP on the next bar.
    pos1_tp = p1 * math.exp(float(pos1.take_profit_log))   # ≈ 100.50
    pos2_tp = p2 * math.exp(float(pos2.take_profit_log))   # ≈ 100.20
    assert pos2_tp < pos1_tp
    assert pos2_tp > p1, "pos1 must be profitable when pos2 TPs"

    base3 = base2 + M * 60_000
    minutes = [_flat_minute(base3, i, price=p2) for i in range(M)]
    # Minute 5: spike high to pos2_tp; pos1's HIGH at pos2_tp (≈100.2) stays
    # below pos1_tp (≈100.5), so pos1 does NOT TP itself.
    minutes[5] = _minute(
        base3, 5, o=p2, h=pos2_tp * 1.0001, l=p2 * 0.9999, c=p2,
    )
    bar3 = _build_bar(base3, minutes)
    bro.advance_to(bar3.close_time, bar3)

    pos1_after = bro.portfolio().get(pid1)
    pos2_after = bro.portfolio().get(pid2)
    assert pos2_after.closed is True
    assert pos2_after.close_reason == ExitReason.TP
    assert pos1_after.closed is True
    assert pos1_after.close_reason == ExitReason.BATCHED_WINNER


def test_two_longs_first_profitable_second_not_only_second_closes():
    """Two longs: pos1 LONG entered at 100.5 with WIDE TP/SL (300bps each)
    so its barriers don't fire during pos2's normal-width activation bar.
    pos2 LONG entered at 100.0 with normal 50bps TP. pos2 TPs at ≈100.5.

    At exit price ≈100.5, pos1 entered at 100.5 has unrealized log-return
    ≈ 0 — NOT profitable net of cost (and we use cost=0 so eps=0 means we
    need strictly positive net pnl). pos1 must NOT be swept.
    """
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)

    # pos1 with WIDE barriers (300bps) opened at 100.5
    ts0 = Timestamp.from_ms(base)
    a1 = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(300.0),
        stop_loss=LogReturn.from_bps(300.0),
        expiry=Duration.from_minutes(M * 1000),
    )
    assert bro.dispatch(a1, ts=ts0, instrument=INSTR) is True
    flat = [_flat_minute(base, i, price=100.5) for i in range(M)]
    bar1 = _build_bar(base, flat)
    bro.advance_to(bar1.close_time, bar1)
    pid1 = next(iter(bro.portfolio().positions.keys()))

    # pos2 with normal 50bps TP opened at 100.0 — its TP fires at ≈100.5
    base2 = base + M * 60_000
    pid2, _ = _open_long_at(bro, base2, entry_price=100.0)
    pos2 = bro.portfolio().get(pid2)
    p2 = float(pos2.avg_entry_price)
    p2_tp = p2 * math.exp(float(pos2.take_profit_log))   # ≈ 100.5012
    # p2_tp ≈ 100.5012, pos1 entered at 100.5 → log(100.5012/100.5) ≈ 1.2e-5
    # > 0 strictly. To make pos1 strictly underwater, enter pos1 above p2_tp.
    # But pos1 must also not have its TP fire during pos2's open-bar (flat
    # at 100.0). pos1's TP price = entry * exp(300bps). With entry=101 → TP
    # at 104, well above 100.0. SL = 101*exp(-300bps) ≈ 98.0, well below.
    # Re-do pos1 at entry 101 to make it reliably underwater at p2_tp ≈ 100.5.

    # Reset by closing pos1 manually (open a fresh broker for clarity)
    bro2 = SimBroker(m_minutes=M, cost=0.0)
    a1b = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(300.0),
        stop_loss=LogReturn.from_bps(300.0),
        expiry=Duration.from_minutes(M * 1000),
    )
    assert bro2.dispatch(a1b, ts=Timestamp.from_ms(base), instrument=INSTR) is True
    flat_b = [_flat_minute(base, i, price=101.0) for i in range(M)]
    bar1b = _build_bar(base, flat_b)
    bro2.advance_to(bar1b.close_time, bar1b)
    pid1b = next(iter(bro2.portfolio().positions.keys()))

    # Open pos2 at 100.0 — pos1's TP=104, SL=98 are far away, so pos1 won't
    # touch its barriers during pos2's flat-100 activation bar.
    pid2b, _ = _open_long_at(bro2, base + M * 60_000, entry_price=100.0)
    pos2b = bro2.portfolio().get(pid2b)
    p2b = float(pos2b.avg_entry_price)
    p2b_tp = p2b * math.exp(float(pos2b.take_profit_log))
    assert p2b_tp < 101.0, "exit price must keep pos1 underwater"

    # Drive pos2 to TP. HIGH must be < pos1's TP (104) and LOW > pos1's SL (98).
    base3 = base + 2 * M * 60_000
    mins = [_flat_minute(base3, i, price=p2b) for i in range(M)]
    mins[5] = _minute(base3, 5, o=p2b, h=p2b_tp * 1.0001, l=p2b * 0.9999, c=p2b)
    bar3 = _build_bar(base3, mins)
    bro2.advance_to(bar3.close_time, bar3)

    pos1b_after = bro2.portfolio().get(pid1b)
    pos2b_after = bro2.portfolio().get(pid2b)
    assert pos2b_after.closed is True
    assert pos2b_after.close_reason == ExitReason.TP
    # pos1 entered at 101, now exit price ≈100.5 → underwater → NOT swept
    assert pos1b_after.closed is False


def test_three_longs_all_profitable_third_hits_tp_all_close():
    """Three longs entered in DECREASING price order: pos1 highest, pos3
    lowest. pos3's TP fires first (since it's at the lowest entry price).
    At pos3's TP exit, pos1 and pos2 are profitable (exit > both entries)
    yet their own TPs (which are higher up) are not touched.

    Constraint: opening a later position at a LOWER price requires earlier
    positions' SL barriers to not fire when the activation bar runs at the
    new (lower) price. We use 50bps SL and step 10bps per position so 3
    positions span 20bps — well within the 50bps SL budget.
    """
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    # pos1 at 100.0, pos2 at 99.9 (-10bps), pos3 at 99.8 (-20bps)
    pid1, _ = _open_long_at(bro, base, entry_price=100.0)
    pos1 = bro.portfolio().get(pid1)
    p1 = float(pos1.avg_entry_price)

    pid2, _ = _open_long_at(bro, base + M * 60_000, entry_price=99.9)
    pos2 = bro.portfolio().get(pid2)
    p2 = float(pos2.avg_entry_price)

    pid3, _ = _open_long_at(bro, base + 2 * M * 60_000, entry_price=99.8)
    pos3 = bro.portfolio().get(pid3)
    p3 = float(pos3.avg_entry_price)

    # pos3's TP price ≈ 99.8 * 1.005 ≈ 100.30, all entries (100, 99.9, 99.8)
    # are < 100.30, so all positions are profitable when pos3 TPs.
    p3_tp = p3 * math.exp(float(pos3.take_profit_log))
    pos1_tp = p1 * math.exp(float(pos1.take_profit_log))   # ≈ 100.50
    pos2_tp = p2 * math.exp(float(pos2.take_profit_log))   # ≈ 100.40
    assert p3_tp < pos2_tp < pos1_tp
    assert p3_tp > p1, "all three must be profitable at p3_tp"

    base4 = base + 3 * M * 60_000
    minutes = [_flat_minute(base4, i, price=p3) for i in range(M)]
    minutes[7] = _minute(
        base4, 7, o=p3, h=p3_tp * 1.0001, l=p3 * 0.9999, c=p3,
    )
    bar4 = _build_bar(base4, minutes)
    bro.advance_to(bar4.close_time, bar4)

    pos1_after = bro.portfolio().get(pid1)
    pos2_after = bro.portfolio().get(pid2)
    pos3_after = bro.portfolio().get(pid3)
    assert pos3_after.closed is True
    assert pos3_after.close_reason == ExitReason.TP
    assert pos1_after.closed is True
    assert pos1_after.close_reason == ExitReason.BATCHED_WINNER
    assert pos2_after.closed is True
    assert pos2_after.close_reason == ExitReason.BATCHED_WINNER


def test_sl_does_not_batch_sweep_winners():
    """When the triggering close is an SL, the broker does NOT sweep
    profitable siblings — losers stay individual and winners keep running.

    Setup: pos1 at 99.5 (LONG, 50bps barriers), pos2 at 100.0 (LONG, 50bps).
    On the test bar a dive to pos2's SL ≈99.5 fires. At that moment pos1
    (entered at 99.5) has unrealized log-return = log(exit/99.5). To make
    pos1 strictly profitable we set the SL trigger LOW slightly BELOW
    99.5 — but that would also hit pos1's SL (99.5*0.995≈99.0). So
    instead: open pos1 LOWER (at 99.0). Then pos1 SL is at 99.0*0.995=98.50
    and TP at 99.0*1.005=99.495. pos2 SL at 100*0.995=99.50.

    A dive to 99.50 (just touching pos2's SL) means LOW=99.50; pos1's TP
    at 99.495 is BELOW 99.50, so pos1 won't TP. pos1's SL at 98.50 is
    far below; safe. At LOW=99.50 pos1's mark = 99.50, log(99.50/99.0)
    ≈ +0.005 > 0, so pos1 is profitable at the moment pos2 SLs.

    Hmm but the broker uses HIGH for LONG TP, and the bar HIGH must be
    LESS than 99.495 for pos1 to not TP. We construct: open=99.495,
    high=99.495, low=pos2_sl*0.9999, close=99.495 — HIGH equals pos1_tp;
    pos1's TP-hit check is `hi >= tp_price` so we need HIGH STRICTLY <
    pos1_tp. Use HIGH=99.49.
    """
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)

    # pos1 LOW
    pid1, _ = _open_long_at(bro, base, entry_price=99.0)
    pos1 = bro.portfolio().get(pid1)
    pos1_tp = float(pos1.avg_entry_price) * math.exp(float(pos1.take_profit_log))
    pos1_sl = float(pos1.avg_entry_price) * math.exp(-float(pos1.stop_loss_log))

    # pos2 HIGHER, to be SLed on the next bar
    base2 = base + M * 60_000
    pid2, _ = _open_long_at(bro, base2, entry_price=100.0)
    pos2 = bro.portfolio().get(pid2)
    pos2_sl = float(pos2.avg_entry_price) * math.exp(-float(pos2.stop_loss_log))

    # Sanity: opening pos2's bar (flat at 100) does NOT fire pos1's TP/SL.
    # pos1 TP=99.495, pos2 open bar runs flat at 100 — high=100.001 >= 99.495
    # WOULD fire pos1's TP. Bug — flat at 100 will TP pos1.
    # Resolution: pos1 had its activation bar earlier; subsequent bars scan
    # min[0..M] for pos1. So pos2's open bar at flat-100 DOES TP pos1.
    # We need pos1's TP barrier above 100.
    # Re-setup: pos1 entered HIGHER, at price s.t. TP > 100, but with WIDE
    # barriers so pos1's SL doesn't fire either.
    #
    # Cleaner restart with WIDE pos1 barriers:
    bro = SimBroker(m_minutes=M, cost=0.0)
    a1 = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(500.0),    # ±500bps barriers
        stop_loss=LogReturn.from_bps(500.0),
        expiry=Duration.from_minutes(M * 1000),
    )
    assert bro.dispatch(a1, ts=Timestamp.from_ms(base), instrument=INSTR) is True
    flat = [_flat_minute(base, i, price=99.0) for i in range(M)]
    bar1 = _build_bar(base, flat)
    bro.advance_to(bar1.close_time, bar1)
    pid1 = next(iter(bro.portfolio().positions.keys()))

    # pos2 LONG entered at 100.0 (50bps barriers, normal helper)
    pid2, _ = _open_long_at(bro, base + M * 60_000, entry_price=100.0)
    pos2 = bro.portfolio().get(pid2)
    pos2_sl = float(pos2.avg_entry_price) * math.exp(-float(pos2.stop_loss_log))

    # On the next bar drive a SL touch on pos2 by diving to pos2_sl.
    # pos1's barriers are at 99.0*exp(±500bps) ≈ (94.2, 104.0) — well outside
    # the dive range, so pos1's barriers won't fire.
    base3 = base + 2 * M * 60_000
    minutes = [_flat_minute(base3, i, price=99.7) for i in range(M)]
    minutes[5] = _minute(
        base3, 5, o=99.7, h=99.71, l=pos2_sl * 0.9999, c=99.7,
    )
    bar3 = _build_bar(base3, minutes)
    bro.advance_to(bar3.close_time, bar3)

    pos1_after = bro.portfolio().get(pid1)
    pos2_after = bro.portfolio().get(pid2)
    assert pos2_after.closed is True
    assert pos2_after.close_reason == ExitReason.SL
    # pos1 entered at 99.0; current price hovered ≈99.7 → profitable. But
    # since this was an SL trigger, pos1 must NOT have been swept.
    assert pos1_after.closed is False


# ---- Change 1: TIMEOUT-at-vertical-barrier removed -----------------------


def test_position_past_expiry_ts_is_not_auto_closed_by_broker():
    """Regression for the removed TIMEOUT branch.

    A position whose `expiry_ts` has already passed must NOT be closed by
    the broker on the next bar. Strategy is now responsible for hold-age
    via Action.close.
    """
    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    # Open with a SHORT expiry so we can quickly cross the vertical barrier
    ts0 = Timestamp.from_ms(base)
    a = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(TP_BPS),
        stop_loss=LogReturn.from_bps(SL_BPS),
        expiry=Duration.from_minutes(M),  # one bar
    )
    assert bro.dispatch(a, ts=ts0, instrument=INSTR) is True
    minutes = [_flat_minute(base, i) for i in range(M)]
    bar = _build_bar(base, minutes)
    bro.advance_to(bar.close_time, bar)
    pid = next(iter(bro.portfolio().positions.keys()))
    pos = bro.portfolio().get(pid)
    assert pos.closed is False

    # Next bar: completely flat, no TP/SL touches; expiry_ts < bar.close_time
    base2 = base + M * 60_000
    minutes2 = [_flat_minute(base2, i) for i in range(M)]
    bar2 = _build_bar(base2, minutes2)
    assert pos.expiry_ts <= bar2.close_time
    bro.advance_to(bar2.close_time, bar2)

    pos_after = bro.portfolio().get(pid)
    assert pos_after.closed is False, (
        "Broker must NOT force-close at vertical barrier — strategy owns hold-age."
    )
    # And the fills history should not reference this position
    assert all(int(f.order_id) != int(pid) for f in bro.fills())


def test_safety_cap_eventually_closes_runaway_position(monkeypatch):
    """The hard MAX_HOLD_BARS safety net DOES close positions that exceed
    the cap (with ExitReason.SAFETY_CAP). Patched to a tiny cap for speed.
    """
    import wagie.io.brokers as brokers_mod
    monkeypatch.setattr(brokers_mod, "MAX_HOLD_BARS", 3)

    base = 1_700_000_000_000
    bro = SimBroker(m_minutes=M, cost=0.0)
    ts0 = Timestamp.from_ms(base)
    a = Action.open(
        side=Side.LONG, size=Probability(1.0),
        take_profit=LogReturn.from_bps(TP_BPS),
        stop_loss=LogReturn.from_bps(SL_BPS),
        expiry=Duration.from_minutes(M * 1000),  # huge — irrelevant
    )
    assert bro.dispatch(a, ts=ts0, instrument=INSTR) is True

    # First bar: activates the position
    minutes = [_flat_minute(base, i) for i in range(M)]
    bar = _build_bar(base, minutes)
    bro.advance_to(bar.close_time, bar)
    pid = next(iter(bro.portfolio().positions.keys()))

    # Drive 5 more flat bars — bars_held will exceed MAX_HOLD_BARS=3
    for k in range(1, 6):
        b_base = base + k * M * 60_000
        ms = [_flat_minute(b_base, i) for i in range(M)]
        bar_k = _build_bar(b_base, ms)
        bro.advance_to(bar_k.close_time, bar_k)

    pos_after = bro.portfolio().get(pid)
    assert pos_after.closed is True
    assert pos_after.close_reason == ExitReason.SAFETY_CAP


def test_max_hold_bars_default_is_large():
    """Sanity: the default safety net is large enough that it doesn't
    interfere with normal backtests."""
    from wagie.io.brokers import MAX_HOLD_BARS as cap
    assert cap >= 10_000
