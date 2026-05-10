"""Engine — sealed event loop with Portfolio + RiskEngine threading.

Per-tick ordering:
    1. DataSource yields bar(t)
    2. Clock.set_time_ns(t)
    3. Process operator console commands (if any pending)
    4. LabelBuffer.maybe_emit(bar) → if a label matured, drive learn_one
       and record (p_online, label, regime_id) for offline calibration metrics.
    5. Broker.advance_to(t, bar) → walks first-touch, emits position events
    6. obs = Pipeline.transform_one(bar, ctx)  ← AFTER step 4 (no leakage)
    7. for each Action in obs.actions:
         RiskEngine.check(action, portfolio, ts) → approved/rejected
         if approved → Broker.dispatch(action)
         if rejected → emit RiskRejected event
    8. Audit + EventLog
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field
from typing import Optional

from wagie.core.action import Action, ActionKind, Side
from wagie.core.event import (
    BarrierTouched, DecisionBar, Event, OrderSubmitted,
    PositionClosed, RiskPolicyChanged, RiskRejected,
)
from wagie.core.identity import OrderId
from wagie.core.observation import Observation
from wagie.core.portfolio import Portfolio
from wagie.core.time import Timestamp
from wagie.io.brokers import BrokerLedger
from wagie.io.clock import Clock, TestClock
from wagie.pipeline.label_buffer import LabelBuffer
from wagie.pipeline.sealed import Pipeline
from wagie.risk import RiskEngine
from wagie.strategy import StrategyContext


logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    ledger: BrokerLedger
    n_decisions: int
    n_filled: int
    n_skipped_warmup: int
    n_actions_approved: int
    n_actions_rejected: int
    pipeline_state_hash: bytes
    final_portfolio: Portfolio
    audit: list[dict] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    # Streaming calibration trace: parallel arrays of (p_online, label, regime_id)
    # captured each time the LabelBuffer emits a matured prediction-label pair.
    # These feed MetricsBattery (Brier / ECE / per-regime calibration).
    p_online_history: list[float] = field(default_factory=list)
    label_history: list[int] = field(default_factory=list)
    regime_history: list[int] = field(default_factory=list)

    @property
    def fills(self) -> list[BarrierTouched]:
        return self.ledger.fills


class Engine:
    """Single event loop. Same code in prod and backtest."""

    def __init__(
        self,
        *,
        source,
        pipeline: Pipeline,
        broker,
        clock: Clock,
        label_buffer: Optional[LabelBuffer] = None,
        risk_engine: Optional[RiskEngine] = None,
        warmup_samples: int = 0,
        capture_audit: bool = False,
        capture_events: bool = False,
    ):
        self.source = source
        self.pipeline = pipeline
        self.broker = broker
        self.clock = clock
        self.label_buffer = label_buffer
        self.risk_engine = risk_engine or RiskEngine.default()
        self.warmup_samples = int(warmup_samples)
        self.capture_audit = bool(capture_audit)
        self.capture_events = bool(capture_events)

        # Console command queue (for G10 — TradingConsole)
        self.command_queue: "queue.Queue" = queue.Queue()
        self.paused: bool = False

        self._n_decisions = 0
        self._n_filled = 0
        self._n_skipped_warmup = 0
        self._n_approved = 0
        self._n_rejected = 0
        self._audit: list[dict] = []
        self._events: list[Event] = []
        self._n_seen = 0
        # Streaming calibration trace, populated when matured labels arrive.
        self._p_online_history: list[float] = []
        self._label_history: list[int] = []
        self._regime_history: list[int] = []

    def run(self) -> EngineResult:
        try:
            for bar in self.source.stream():
                self._process_one(bar)
        finally:
            try:
                self.source.close()
            except Exception:
                pass
        ledger = self.broker.finalize()
        # Drain broker events into engine's event log
        if self.capture_events:
            self._events.extend(getattr(self.broker, "events", []))
        return EngineResult(
            ledger=ledger,
            n_decisions=self._n_decisions,
            n_filled=len(ledger.fills),
            n_skipped_warmup=self._n_skipped_warmup,
            n_actions_approved=self._n_approved,
            n_actions_rejected=self._n_rejected,
            pipeline_state_hash=self.pipeline.state_hash(),
            final_portfolio=self.broker.portfolio() if hasattr(self.broker, "portfolio")
            else Portfolio(),
            audit=self._audit,
            events=self._events,
            p_online_history=list(self._p_online_history),
            label_history=list(self._label_history),
            regime_history=list(self._regime_history),
        )

    def _drain_console_commands(self) -> None:
        """Process any operator commands in the queue, between bars."""
        while True:
            try:
                cmd = self.command_queue.get_nowait()
            except queue.Empty:
                return
            try:
                cmd(self)
            except Exception as e:
                logger.warning(f"console command failed: {e}")

    def _process_one(self, bar: DecisionBar) -> None:
        ts = bar.close_time
        self.clock.set_time_ns(ts.ns)

        # 3. Process operator commands (G10)
        self._drain_console_commands()

        # 4. Drive delayed-label learning BEFORE prediction
        if self.label_buffer is not None:
            emitted = self.label_buffer.maybe_emit(bar)
            if emitted is not None:
                obs_prev, y_prev = emitted
                self.pipeline.learn_one(obs_prev, y_prev)
                # Capture (p_online, label, regime) for calibration metrics.
                p_prev = obs_prev.p_online
                if p_prev is not None:
                    try:
                        self._p_online_history.append(float(p_prev))
                        self._label_history.append(int(y_prev))
                        r = obs_prev.regime_id
                        self._regime_history.append(int(r) if r is not None else -1)
                    except (TypeError, ValueError):
                        pass

        # 5. Broker resolves any open positions; emits position events
        fills = self.broker.advance_to(ts, bar)
        self._n_filled += len(fills)

        # 6. Pipeline forward pass
        self._n_seen += 1
        if self._n_seen <= self.warmup_samples:
            obs = Observation(bar=bar)
            try:
                self.pipeline.transform_one(obs, self._make_ctx(ts))
            except Exception:
                pass
            self._n_skipped_warmup += 1
            return

        if self.paused:
            # Skip strategy emit while paused; broker still resolves barriers
            return

        obs = Observation(bar=bar)
        ctx = self._make_ctx(ts)
        try:
            obs = self.pipeline.transform_one(obs, ctx)
        except Exception as e:
            logger.warning(f"transform_one failed: {e}")
            return

        # 7. Dispatch each Action through RiskEngine -> Broker
        actions = obs.actions or ()
        portfolio = self.broker.portfolio() if hasattr(self.broker, "portfolio") \
            else Portfolio()
        for action in actions:
            if not isinstance(action, Action):
                continue
            if action.kind == ActionKind.HOLD:
                continue
            check = self.risk_engine.check(action, portfolio, ts)
            if not check.approved:
                self._n_rejected += 1
                if self.capture_events:
                    self._events.append(RiskRejected(
                        ts_init=ts, instrument=bar.instrument,
                        action_kind=str(action.kind),
                        reason=check.reason,
                        policy_name=check.policy_name,
                    ))
                if check.suggested_action is None:
                    continue
                action = check.suggested_action
            ok = self.broker.dispatch(
                action, ts=ts, instrument=bar.instrument,
                features=obs.features.to_dict() if action.kind == ActionKind.OPEN else {},
            )
            if ok:
                self._n_approved += 1
                if action.kind == ActionKind.OPEN:
                    self._n_decisions += 1

        if self.capture_audit:
            self._audit.append({
                "ts_ns": ts.ns,
                "p_offline": float(obs.p_offline) if obs.p_offline is not None else None,
                "p_online": float(obs.p_online) if obs.p_online is not None else None,
                "regime_id": obs.regime_id,
                "n_actions": len(actions),
                "n_open_after": portfolio.n_open,
                "n_fills_this_tick": len(fills),
            })

    def _make_ctx(self, ts: Timestamp) -> StrategyContext:
        portfolio = (self.broker.portfolio()
                     if hasattr(self.broker, "portfolio") else Portfolio())
        return StrategyContext(
            portfolio=portfolio,
            clock_ns=ts.ns,
            risk_engine_view=self.risk_engine.policy_names,
        )


__all__ = ["Engine", "EngineResult"]
