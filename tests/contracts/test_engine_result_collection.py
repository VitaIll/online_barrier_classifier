"""Contract tests for per-bar strategy-state collection on EngineResult.

A strategy that exposes ``get_state() -> dict`` populates the new history
arrays on EngineResult; a strategy without ``get_state`` is silently
ignored (backwards-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from wagie.core.action import Action, Side
from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.pipeline import StageKind
from wagie.core.time import Duration, Timestamp
from wagie.engine import Engine, EngineResult
from wagie.io.brokers import SimBroker
from wagie.io.clock import TestClock
from wagie.pipeline.sealed import Pipeline


_NS_PER_MIN = 60_000_000_000


def _bar(close: float = 100.0, ts_ns: int = _NS_PER_MIN) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(ts_ns), instrument=DEFAULT_INSTRUMENT,
        open=Price(close), high=Price(close * 1.01), low=Price(close * 0.99),
        close=Price(close), volume=Quantity(1.0),
        duration=Duration.from_minutes(20), segment_id=0,
    )


class _OneShotSource:
    """Yield N synthetic bars then stop."""

    def __init__(self, n: int = 10):
        self.n = int(n)
        self.closed = False

    def stream(self):
        for i in range(self.n):
            yield _bar(close=100.0 + i * 0.01,
                       ts_ns=(i + 1) * _NS_PER_MIN)

    def close(self):
        self.closed = True


@dataclass
class _StatefulStrategy:
    """Synthetic strategy that exposes ``get_state()`` with all spec keys.

    The state mutates per-bar via a tick counter so we can verify alignment
    of the engine-collected arrays with the bar index.
    """

    name: str = "stateful"
    kind: StageKind = StageKind.STRATEGY
    _ticks: int = 0

    def transform(self, obs):
        return obs

    def decide(self, obs, ctx):
        self._ticks += 1
        return ()  # Hold every bar — keeps the test focused on collection only.

    def update(self, obs, label=None):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        return None

    def state_hash(self):
        return f"stateful|{self._ticks}".encode()

    def reset(self):
        self._ticks = 0

    def get_state(self) -> dict:
        return {
            "tau": 0.5 + 0.001 * self._ticks,
            "r_hat": 0.10 + 0.0005 * self._ticks,
            "r_star": 0.20,
            "paused": (self._ticks % 4 == 0),
            "sigma_ve": 0.30 + 0.0002 * self._ticks,
            "inv_size": self._ticks // 2,
            "hold_age_max": min(self._ticks, 3),
        }


@dataclass
class _LegacyStrategy:
    """Strategy without ``get_state`` — must not break the engine."""

    name: str = "legacy"
    kind: StageKind = StageKind.STRATEGY

    def transform(self, obs):
        return obs

    def decide(self, obs, ctx):
        return ()

    def update(self, obs, label=None):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        return None

    def state_hash(self):
        return b"legacy"

    def reset(self):
        pass


def _run(strategy, *, n_bars: int = 8, warmup: int = 0) -> EngineResult:
    pipe = Pipeline([strategy])
    src = _OneShotSource(n=n_bars)
    broker = SimBroker(m_minutes=20, inventory_cap=10)
    eng = Engine(
        source=src, pipeline=pipe, broker=broker, clock=TestClock(0),
        warmup_samples=warmup,
    )
    return eng.run()


# --------------------------- new-fields default to empty -------------------


def test_engine_result_new_fields_default_empty() -> None:
    """A fresh EngineResult constructed by hand defaults all new lists to []
    and warmup_calibration to None."""
    from wagie.core.portfolio import Portfolio
    from wagie.io.brokers import BrokerLedger
    res = EngineResult(
        ledger=BrokerLedger(fills=[], n_open_at_finalize=0, config={}),
        n_decisions=0, n_filled=0, n_skipped_warmup=0,
        n_actions_approved=0, n_actions_rejected=0,
        pipeline_state_hash=b"\x00" * 32,
        final_portfolio=Portfolio(),
    )
    assert res.tau_history == []
    assert res.r_hat_ewma_history == []
    assert res.r_star_ewma_history == []
    assert res.paused_history == []
    assert res.sigma_ve_history == []
    assert res.inventory_size_history == []
    assert res.hold_age_max_history == []
    assert res.warmup_calibration is None


# --------------------------- legacy strategy: no break ---------------------


def test_engine_does_not_break_when_strategy_has_no_get_state() -> None:
    """A strategy without ``get_state`` runs fine; arrays stay empty."""
    res = _run(_LegacyStrategy(), n_bars=5)
    assert isinstance(res, EngineResult)
    assert res.tau_history == []
    assert res.r_hat_ewma_history == []
    assert res.r_star_ewma_history == []
    assert res.paused_history == []
    assert res.sigma_ve_history == []
    assert res.inventory_size_history == []
    assert res.hold_age_max_history == []


# --------------------------- stateful strategy: arrays populated -----------


def test_engine_populates_history_lists_for_stateful_strategy() -> None:
    """Every key returned by get_state lands in the matching history list,
    one entry per processed bar (warmup excluded — collected only after
    a successful transform_one)."""
    strat = _StatefulStrategy()
    res = _run(strat, n_bars=8, warmup=0)
    n = 8
    assert len(res.tau_history) == n
    assert len(res.r_hat_ewma_history) == n
    assert len(res.r_star_ewma_history) == n
    assert len(res.paused_history) == n
    assert len(res.sigma_ve_history) == n
    assert len(res.inventory_size_history) == n
    assert len(res.hold_age_max_history) == n
    # tau monotone increasing with the tick counter we baked in.
    assert res.tau_history == sorted(res.tau_history)
    # r_star is constant per spec of the synthetic state.
    assert all(r == pytest.approx(0.20) for r in res.r_star_ewma_history)
    # paused is True every 4th tick (1-indexed counter from decide()).
    assert res.paused_history[3] is True   # tick 4
    assert res.paused_history[0] is False  # tick 1
    # inv_size increments by 1 every 2 ticks, starting at 0.
    assert res.inventory_size_history[0] == 0  # tick 1 -> 1//2 = 0
    assert res.inventory_size_history[1] == 1  # tick 2 -> 2//2 = 1
    # hold_age_max clamped to 3.
    assert max(res.hold_age_max_history) == 3


def test_collection_handles_partial_state_dict() -> None:
    """A strategy returning only a subset of keys still works — missing
    floats become NaN, missing bools become False, missing ints become 0."""
    import math

    @dataclass
    class _PartialStateStrategy:
        name: str = "partial"
        kind: StageKind = StageKind.STRATEGY

        def transform(self, obs):
            return obs

        def decide(self, obs, ctx):
            return ()

        def update(self, obs, label=None):
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            return None

        def state_hash(self):
            return b"partial"

        def reset(self):
            pass

        def get_state(self) -> dict:
            # Only tau + paused — every other key absent.
            return {"tau": 0.42, "paused": True}

    res = _run(_PartialStateStrategy(), n_bars=3)
    assert all(t == pytest.approx(0.42) for t in res.tau_history)
    assert all(p is True for p in res.paused_history)
    # Missing floats default to NaN; missing ints default to 0.
    assert all(math.isnan(x) for x in res.r_hat_ewma_history)
    assert all(math.isnan(x) for x in res.r_star_ewma_history)
    assert all(math.isnan(x) for x in res.sigma_ve_history)
    assert all(x == 0 for x in res.inventory_size_history)
    assert all(x == 0 for x in res.hold_age_max_history)


def test_warmup_calibration_pulled_from_strategy_when_exposed() -> None:
    """If the strategy exposes ``warmup_calibration`` (attribute or callable),
    the engine surfaces it on EngineResult."""

    @dataclass
    class _WithWarmup:
        name: str = "warmup_strat"
        kind: StageKind = StageKind.STRATEGY
        warmup_calibration: dict = field(
            default_factory=lambda: {"isotonic": [0.1, 0.5, 0.9], "n_warmup": 100},
        )

        def transform(self, obs):
            return obs

        def decide(self, obs, ctx):
            return ()

        def update(self, obs, label=None):
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            return None

        def state_hash(self):
            return b"warmup_strat"

        def reset(self):
            pass

    res = _run(_WithWarmup(), n_bars=3)
    assert isinstance(res.warmup_calibration, dict)
    assert res.warmup_calibration.get("n_warmup") == 100


def test_strategy_get_state_failure_does_not_crash_engine() -> None:
    """A buggy ``get_state`` is logged + skipped — the engine keeps going."""

    @dataclass
    class _BoomState:
        name: str = "boom_state"
        kind: StageKind = StageKind.STRATEGY

        def transform(self, obs):
            return obs

        def decide(self, obs, ctx):
            return ()

        def update(self, obs, label=None):
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            return None

        def state_hash(self):
            return b"boom_state"

        def reset(self):
            pass

        def get_state(self):
            raise RuntimeError("get_state boom")

    # Must not raise.
    res = _run(_BoomState(), n_bars=4)
    # No state was collected — arrays empty.
    assert res.tau_history == []
