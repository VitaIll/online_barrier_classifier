"""Extended TradingConsole coverage — operator command surface.

Each operator action queues a callable on `engine.command_queue`. The engine
applies them via `_drain_console_commands()` between bars. We verify both
the queueing contract and side-effects after drain.

NOTE: bug surface — ``console.set_stop`` exposes ``sl=`` / ``tp=`` (not
``stop_loss=`` / ``take_profit=`` as the spec brief asks); ``console.history``
also requires a ``position_id`` arg. We exercise the actual API and flag the
naming inconsistency in the report.
"""

from __future__ import annotations

import pytest

from wagie.console import TradingConsole
from wagie.core.action import ActionKind
from wagie.core.numeric import LogReturn
from wagie.engine import Engine
from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
from wagie.features.catalog import default_streaming_features
from wagie.io.brokers import SimBroker
from wagie.io.clock import TestClock
from wagie.io.sources import ParquetReplaySource
from wagie.pipeline import (
    LabelBuffer,
    MondrianACICalibrator,
    OnlineARFCorrector,
    Pipeline,
)
from wagie.risk import KillSwitchPolicy, MaxDrawdownPolicy, RiskEngine
from wagie.strategy import PureConformalGate


def _build_engine(parquet):
    base_bar = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features()[:5])
    cuts = RegimeCuts(
        feature="parkinson_var_rolling_mean_24",
        edges=(1e-6, 1e-5),
        labels=("low", "med", "high"),
    )
    rg = RegimeFeature(cuts)
    arf = OnlineARFCorrector()
    aci = MondrianACICalibrator()
    lb = LabelBuffer()
    strat = PureConformalGate(name="pcg", alpha=0.10)
    pipeline = Pipeline([base_bar, fb, rg, arf, aci, lb, strat])
    src = ParquetReplaySource(str(parquet), m_minutes=20)
    bro = SimBroker(m_minutes=20, inventory_cap=5)
    eng = Engine(
        source=src,
        pipeline=pipeline,
        broker=bro,
        clock=TestClock(),
        label_buffer=lb,
        risk_engine=RiskEngine.default(),
        warmup_samples=5,
        capture_events=True,
    )
    return eng


def test_start_returns_self_for_chaining(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng)
    out = console.start()
    assert out is console


def test_flatten_queues_close_per_open_position(synthetic_minute_parquet, monkeypatch):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()

    msg = console.flatten()
    assert "queued" in msg
    # Exactly one callable enqueued (the closure that iterates open positions).
    assert eng.command_queue.qsize() == 1

    # Mock-out broker.dispatch and seed two fake open positions, then drain
    # and verify each gets a CLOSE Action.
    dispatched: list[tuple] = []

    def _fake_dispatch(action, **kw):
        dispatched.append((action.kind, action.target_position_id))
        return True

    class _FakePos:
        def __init__(self, pid, instrument):
            from wagie.core.identity import PositionId
            self.position_id = PositionId(pid)
            self.instrument = instrument

    class _FakePortfolio:
        def __init__(self, positions):
            self.open_positions = positions

        def get(self, pid):
            for p in self.open_positions:
                if int(p.position_id) == int(pid):
                    return p
            return None

    fake_positions = [_FakePos(1, "BTCUSDT"), _FakePos(2, "BTCUSDT")]
    monkeypatch.setattr(eng.broker, "portfolio", lambda: _FakePortfolio(fake_positions))
    monkeypatch.setattr(eng.broker, "dispatch", _fake_dispatch)

    eng._drain_console_commands()
    assert len(dispatched) == 2
    assert all(kind == ActionKind.CLOSE for kind, _ in dispatched)
    assert {int(pid) for _, pid in dispatched} == {1, 2}


def test_flatten_specific_position_id(synthetic_minute_parquet, monkeypatch):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    msg = console.flatten(position_id=7)
    assert "7" in msg

    dispatched = []

    def _fake_dispatch(action, **kw):
        dispatched.append(action.target_position_id)
        return True

    class _FakePortfolio:
        open_positions = []
        def get(self, pid):
            return None  # exercises the empty-positions fallback path

    monkeypatch.setattr(eng.broker, "portfolio", lambda: _FakePortfolio())
    monkeypatch.setattr(eng.broker, "dispatch", _fake_dispatch)
    eng._drain_console_commands()
    # one CLOSE for position 7 was queued (instrument=None per fallback)
    assert len(dispatched) == 1
    assert int(dispatched[0]) == 7


def test_set_stop_queues_modify_stop(synthetic_minute_parquet, monkeypatch):
    """Console exposes set_stop(pid, *, sl=, tp=) — queues a MODIFY_STOP via
    broker._dispatch_modify_stop. (Spec brief used stop_loss=/take_profit=
    names; the actual API uses sl=/tp=. Reported as a naming inconsistency.)
    """
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()

    msg = console.set_stop(13, sl=LogReturn(-0.001), tp=LogReturn(0.002))
    assert "modify_stop" in msg
    assert "13" in msg
    assert eng.command_queue.qsize() == 1

    # When pid does not exist, _do returns early — no exception.
    class _Empty:
        open_positions = []
        def get(self, pid):
            return None
    monkeypatch.setattr(eng.broker, "portfolio", lambda: _Empty())
    eng._drain_console_commands()  # should be a no-op, no error


def test_set_stop_requires_sl_or_tp(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    msg = console.set_stop(1)
    assert "requires" in msg
    # No command queued when invalid.
    assert eng.command_queue.qsize() == 0


def test_kill_returns_engaged_message(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    msg = console.kill("vol spike")
    assert "ENGAGED" in msg
    assert "vol spike" in msg
    eng._drain_console_commands()
    ks = eng.risk_engine.get("kill_switch")
    assert ks is not None and ks.engaged


def test_unkill_disengages_kill_switch(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    console.kill("manual")
    eng._drain_console_commands()
    assert eng.risk_engine.get("kill_switch").engaged is True

    msg = console.unkill()
    assert "DISENGAGED" in msg
    eng._drain_console_commands()
    assert eng.risk_engine.get("kill_switch").engaged is False


def test_risk_add_inserts_policy_between_bars(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    # Make sure starting state has no max_drawdown policy.
    assert eng.risk_engine.get("max_drawdown") is None
    console = TradingConsole(eng).start()

    policy = MaxDrawdownPolicy(max_dd=-0.05)
    msg = console.risk_add(policy)
    assert "added" in msg
    assert "max_drawdown" in msg or "MaxDrawdownPolicy" in msg
    eng._drain_console_commands()
    assert eng.risk_engine.get("max_drawdown") is not None


def test_risk_remove_pulls_policy_out(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    # Default RiskEngine has kill_switch — remove it.
    assert "kill_switch" in eng.risk_engine.policy_names
    console = TradingConsole(eng).start()
    msg = console.risk_remove("kill_switch")
    assert "removed" in msg
    eng._drain_console_commands()
    assert "kill_switch" not in eng.risk_engine.policy_names


def test_history_returns_string_for_unknown_position(synthetic_minute_parquet):
    """`history` requires position_id — we test the no-such-id branch.

    (Spec brief implied a no-arg history(); the actual signature requires pid.)
    """
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    out = console.history(999)
    assert isinstance(out, str)
    assert "no position" in out


def test_save_queues_persistence_call(synthetic_minute_parquet, tmp_path, monkeypatch):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()

    captured: dict = {}

    def _fake_save(pipeline, broker, path, *, config_hash=None, n_seen=0):
        captured["path"] = str(path)
        captured["n_seen"] = int(n_seen)
        return path

    # Patch the save symbol in wagie.persistence (where _do imports it from).
    monkeypatch.setattr("wagie.persistence.save", _fake_save)

    target = tmp_path / "snap.tar.gz"
    msg = console.save(str(target))
    assert "queued" in msg
    eng._drain_console_commands()
    assert captured["path"] == str(target)


def test_concurrent_commands_apply_in_fifo_order(synthetic_minute_parquet):
    """Multiple operator commands queue in-order and flush in FIFO via _drain."""
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()

    # Queue: pause -> kill -> unkill -> resume
    console.pause()
    console.kill("ops")
    console.unkill()
    console.resume()
    assert eng.command_queue.qsize() == 4

    eng._drain_console_commands()
    # Final state reflects the LAST applied command in each axis (FIFO ordering
    # means later commands win).
    assert eng.paused is False
    assert eng.risk_engine.get("kill_switch").engaged is False
    assert eng.command_queue.qsize() == 0


def test_pause_and_resume_round_trip(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    console.pause()
    eng._drain_console_commands()
    assert eng.paused is True
    console.resume()
    eng._drain_console_commands()
    assert eng.paused is False


def test_portfolio_and_risk_list_strings(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    p = console.portfolio()
    assert "Portfolio" in p
    rl = console.risk_list()
    assert "kill_switch" in rl or "max_positions" in rl


def test_risk_list_empty_when_no_policies(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    eng.risk_engine = RiskEngine([])
    console = TradingConsole(eng).start()
    assert "no risk policies" in console.risk_list()


def test_stop_sets_event(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    console.stop()
    assert console._stop_event.is_set()


# -----------------------------------------------------------------------------
# Read-only query happy paths (positions() + history() with non-empty broker).
# -----------------------------------------------------------------------------


def _make_fake_position(pid_int: int):
    """Tiny duck-typed Position the console formatter accepts."""
    from wagie.core.action import Side
    from wagie.core.identity import PositionId
    from wagie.core.numeric import LogReturn
    from wagie.core.time import Timestamp

    class _FakeDuration:
        minutes = 5.0

    class _Fake:
        position_id = PositionId(pid_int)
        side = Side.LONG
        instrument = "BTCUSDT"
        current_size = 1.0
        open_size = 1.0
        avg_entry_price = 100.0
        last_mark_price = 101.0
        unrealized_pnl_log = LogReturn(0.0010)
        mae_log = LogReturn(-0.0005)
        mfe_log = LogReturn(0.0015)
        realized_pnl_log = LogReturn(0.0)
        time_in_trade = _FakeDuration()
        open_ts = Timestamp.from_ms(1_700_000_000_000)
        expiry_ts = Timestamp.from_ms(1_700_001_200_000)
        take_profit_log = LogReturn(0.0041)
        stop_loss_log = LogReturn(-0.0041)
        closed = False
        close_reason = None

    return _Fake()


def test_positions_renders_table_when_open(synthetic_minute_parquet, monkeypatch):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    fake_pos = _make_fake_position(7)

    class _Portfolio:
        is_flat = False
        open_positions = [fake_pos]

        def get(self, pid):
            return fake_pos if int(pid) == 7 else None

    monkeypatch.setattr(eng.broker, "portfolio", lambda: _Portfolio())
    out = console.positions()
    assert "PID" in out and "Side" in out
    assert "LONG" in out


def test_history_renders_known_position(synthetic_minute_parquet, monkeypatch):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    fake_pos = _make_fake_position(11)

    class _Portfolio:
        is_flat = False
        open_positions = [fake_pos]

        def get(self, pid):
            return fake_pos if int(pid) == 11 else None

    monkeypatch.setattr(eng.broker, "portfolio", lambda: _Portfolio())
    out = console.history(11)
    assert "Position 11" in out
    assert "BTCUSDT" in out


def test_set_stop_dispatches_when_position_exists(
    synthetic_minute_parquet, monkeypatch
):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    fake_pos = _make_fake_position(3)

    class _Portfolio:
        is_flat = False
        open_positions = [fake_pos]

        def get(self, pid):
            return fake_pos if int(pid) == 3 else None

    captured = {}

    def _fake_modify(action, *, ts, actor):
        captured["action"] = action
        captured["actor"] = actor
        return True

    monkeypatch.setattr(eng.broker, "portfolio", lambda: _Portfolio())
    monkeypatch.setattr(eng.broker, "_dispatch_modify_stop", _fake_modify)

    console.set_stop(3, sl=LogReturn(-0.0010), tp=LogReturn(0.0030))
    eng._drain_console_commands()
    from wagie.core.action import ActionKind as _AK
    assert captured["action"].kind == _AK.MODIFY_STOP
    assert captured["actor"] == "operator"


def test_kill_with_existing_kill_switch_engages_in_place(synthetic_minute_parquet):
    """If RiskEngine already has a kill_switch, kill() engages it (no duplicate)."""
    eng = _build_engine(synthetic_minute_parquet)
    n_before = len(eng.risk_engine.policies)
    console = TradingConsole(eng).start()
    console.kill("scheduled")
    eng._drain_console_commands()
    assert len(eng.risk_engine.policies) == n_before  # no duplicate added


def test_kill_creates_kill_switch_when_missing(synthetic_minute_parquet):
    """When no kill_switch exists, kill() adds one and engages it."""
    eng = _build_engine(synthetic_minute_parquet)
    eng.risk_engine = RiskEngine([])  # strip the default
    assert eng.risk_engine.get("kill_switch") is None
    console = TradingConsole(eng).start()
    console.kill("emergency")
    eng._drain_console_commands()
    ks = eng.risk_engine.get("kill_switch")
    assert ks is not None and ks.engaged is True
