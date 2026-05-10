"""G10: TradingConsole — operator REPL queues commands; engine applies between bars."""

from __future__ import annotations

from wagie.config import WagieConfig
from wagie.console import TradingConsole
from wagie.core.numeric import LogReturn
from wagie.engine import Engine
from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
from wagie.features.catalog import default_streaming_features
from wagie.io.brokers import SimBroker
from wagie.io.clock import TestClock
from wagie.io.sources import ParquetReplaySource
from wagie.pipeline import LabelBuffer, OnlineARFCorrector, Pipeline
from wagie.risk import KillSwitchPolicy, RiskEngine
from wagie.strategy import ThresholdGate


def _build_engine(parquet):
    base_bar = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features()[:5])
    cuts = RegimeCuts(feature="parkinson_var_rolling_mean_24",
                      edges=(1e-6, 1e-5), labels=("low", "med", "high"))
    rg = RegimeFeature(cuts)
    arf = OnlineARFCorrector()
    lb = LabelBuffer()
    strat = ThresholdGate(name="pcg", tau=0.20)
    pipeline = Pipeline([base_bar, fb, rg, arf, lb, strat])
    src = ParquetReplaySource(str(parquet), m_minutes=20)
    bro = SimBroker(m_minutes=20, inventory_cap=5)
    eng = Engine(source=src, pipeline=pipeline, broker=bro,
                 clock=TestClock(), label_buffer=lb,
                 risk_engine=RiskEngine.default(),
                 warmup_samples=5, capture_events=True)
    return eng


def test_console_queues_kill_command(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    msg = console.kill("vol spike")
    assert "ENGAGED" in msg
    # Process the queued command (would happen at next bar in real run)
    eng._drain_console_commands()
    ks = eng.risk_engine.get("kill_switch")
    assert ks is not None
    assert ks.engaged


def test_console_pause_resume(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    console.pause()
    eng._drain_console_commands()
    assert eng.paused
    console.resume()
    eng._drain_console_commands()
    assert not eng.paused


def test_console_positions_when_flat(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    out = console.positions()
    assert "no open positions" in out


def test_console_help_lists_commands(synthetic_minute_parquet):
    eng = _build_engine(synthetic_minute_parquet)
    console = TradingConsole(eng).start()
    h = console.help()
    assert "positions" in h
    assert "kill" in h
    assert "flatten" in h
    assert "set_stop" in h
