"""Action vocabulary contracts.

Validates each constructor's ActionKind tag and field subset, the
`is_active` / `is_opening` properties, the modify_stop validation, and
that frozen+slots instances behave as value objects (equality, hashing).
"""

from __future__ import annotations

import pytest

from wagie.core.action import Action, ActionKind, Side
from wagie.core.identity import PositionId
from wagie.core.numeric import LogReturn, Probability
from wagie.core.time import Duration


# ---- helpers ----

TP = LogReturn.from_bps(41.11)
SL = LogReturn.from_bps(41.11)
EXP = Duration.from_minutes(20)
PID = PositionId(7)


# ---- constructors: kind + field subset ----


def test_hold_kind_and_empty_fields():
    a = Action.hold()
    assert a.kind == ActionKind.HOLD
    # all option fields default-None
    assert a.target_position_id is None
    assert a.side is None
    assert a.size is None
    assert a.take_profit is None
    assert a.stop_loss is None
    assert a.expiry is None
    assert a.new_take_profit is None
    assert a.new_stop_loss is None
    assert a.fraction is None


def test_open_fills_side_size_tp_sl_expiry():
    a = Action.open(side=Side.LONG, size=Probability(0.5),
                    take_profit=TP, stop_loss=SL, expiry=EXP)
    assert a.kind == ActionKind.OPEN
    assert a.side == Side.LONG
    assert float(a.size) == 0.5
    assert float(a.take_profit) == float(TP)
    assert float(a.stop_loss) == float(SL)
    assert a.expiry == EXP
    # cross-kind fields should be None
    assert a.target_position_id is None
    assert a.new_take_profit is None
    assert a.new_stop_loss is None
    assert a.fraction is None


def test_long_short_set_side_correctly():
    al = Action.long(size=Probability(1.0), take_profit=TP, stop_loss=SL, expiry=EXP)
    as_ = Action.short(size=Probability(1.0), take_profit=TP, stop_loss=SL, expiry=EXP)
    assert al.kind == ActionKind.OPEN and al.side == Side.LONG
    assert as_.kind == ActionKind.OPEN and as_.side == Side.SHORT


def test_open_default_side_and_size():
    a = Action.open(take_profit=TP, stop_loss=SL, expiry=EXP)
    assert a.side == Side.LONG
    assert float(a.size) == 1.0


def test_scale_in_fields():
    a = Action.scale_in(PID, size=Probability(0.25))
    assert a.kind == ActionKind.SCALE_IN
    assert a.target_position_id == PID
    assert float(a.size) == 0.25
    assert a.side is None and a.take_profit is None and a.expiry is None


def test_scale_out_fields():
    a = Action.scale_out(PID, fraction=Probability(0.5))
    assert a.kind == ActionKind.SCALE_OUT
    assert a.target_position_id == PID
    assert float(a.fraction) == 0.5
    assert a.size is None and a.side is None


def test_close_only_target_id():
    a = Action.close(PID)
    assert a.kind == ActionKind.CLOSE
    assert a.target_position_id == PID
    # everything else None
    for fld in ("side", "size", "take_profit", "stop_loss", "expiry",
                "new_take_profit", "new_stop_loss", "fraction"):
        assert getattr(a, fld) is None


def test_modify_stop_with_both():
    new_tp = LogReturn.from_bps(50)
    new_sl = LogReturn.from_bps(30)
    a = Action.modify_stop(PID, new_take_profit=new_tp, new_stop_loss=new_sl)
    assert a.kind == ActionKind.MODIFY_STOP
    assert a.target_position_id == PID
    assert float(a.new_take_profit) == float(new_tp)
    assert float(a.new_stop_loss) == float(new_sl)


def test_modify_stop_requires_at_least_one():
    with pytest.raises(ValueError):
        Action.modify_stop(PID)
    with pytest.raises(ValueError):
        Action.modify_stop(PID, new_take_profit=None, new_stop_loss=None)


def test_modify_stop_one_only_ok():
    a1 = Action.modify_stop(PID, new_take_profit=LogReturn.from_bps(60))
    a2 = Action.modify_stop(PID, new_stop_loss=LogReturn.from_bps(20))
    assert a1.kind == ActionKind.MODIFY_STOP
    assert a1.new_stop_loss is None
    assert a2.new_take_profit is None


def test_cancel_fields():
    a = Action.cancel(PID)
    assert a.kind == ActionKind.CANCEL
    assert a.target_position_id == PID
    assert a.side is None and a.size is None


# ---- is_active / is_opening ----


def test_is_active_per_kind():
    assert Action.hold().is_active is False
    open_a = Action.open(take_profit=TP, stop_loss=SL, expiry=EXP)
    assert open_a.is_active is True
    assert Action.scale_in(PID, size=Probability(0.5)).is_active is True
    assert Action.scale_out(PID, fraction=Probability(0.5)).is_active is True
    assert Action.close(PID).is_active is True
    assert Action.modify_stop(PID, new_stop_loss=LogReturn.from_bps(20)).is_active is True
    assert Action.cancel(PID).is_active is True


def test_is_opening_per_kind():
    assert Action.hold().is_opening is False
    assert Action.open(take_profit=TP, stop_loss=SL, expiry=EXP).is_opening is True
    assert Action.scale_in(PID, size=Probability(0.5)).is_opening is True
    assert Action.scale_out(PID, fraction=Probability(0.5)).is_opening is False
    assert Action.close(PID).is_opening is False
    assert Action.modify_stop(PID, new_stop_loss=LogReturn.from_bps(20)).is_opening is False
    assert Action.cancel(PID).is_opening is False


# ---- value-object semantics: equality + hashable ----


def test_equal_field_actions_compare_equal():
    a = Action.open(side=Side.LONG, size=Probability(0.5),
                    take_profit=TP, stop_loss=SL, expiry=EXP)
    b = Action.open(side=Side.LONG, size=Probability(0.5),
                    take_profit=TP, stop_loss=SL, expiry=EXP)
    assert a == b
    assert Action.hold() == Action.hold()
    assert Action.close(PID) == Action.close(PID)


def test_unequal_actions_not_equal():
    assert Action.close(PID) != Action.close(PositionId(99))
    assert Action.hold() != Action.cancel(PID)


def test_action_is_hashable_frozen_slots():
    a = Action.close(PID)
    b = Action.close(PID)
    s = {a, b, Action.hold(), Action.cancel(PID)}
    # equal Actions collapse in a set
    assert len(s) == 3
    # hash stable
    assert hash(a) == hash(b)
