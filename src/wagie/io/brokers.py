"""SimBroker — event-driven triple-barrier with full Position tracking.

Consumes Action; emits position lifecycle events (PositionOpened, ScaledIn,
ScaledOut, Closed). Maintains a live Portfolio (read-only snapshot via
.portfolio()).

Per P3: positions keyed by PositionId; Portfolio.net_position(InstrumentId)
aggregates over instrument. Per P1: default inventory_cap = 5.

Tie-break — when SL and TP both fire in the same minute the order is
ambiguous; configurable via `tie_break`:
    - "pessimistic_sl_first" (default): SL wins. The conservative audit-safe
      choice but systematically overstates losses for upside-excursion labels.
    - "optimistic_tp_first": TP wins. Symmetric counterpart for falsification.
    - "probabilistic_hl_bridge": Bernoulli(0.5) per collision (deterministic
      seed via the bar's open timestamp). Treats both barriers as equiprobable
      first-touches when path inside the minute is unknown.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional

from wagie.core.action import Action, ActionKind, ExitReason, Side
from wagie.core.event import (
    BarrierTouched,
    DecisionBar,
    Event,
    MinuteBar,
    OrderSubmitted,
    PositionClosed,
    PositionOpened,
    PositionScaledIn,
    PositionScaledOut,
    PositionStopModified,
)
from wagie.core.identity import InstrumentId, OrderId, PositionId
from wagie.core.numeric import LogReturn, Price, Probability, Quantity
from wagie.core.portfolio import Portfolio
from wagie.core.position import Position
from wagie.core.time import Duration, Timestamp


# A small dataclass mirror of config.BrokerCostModel so SimBroker has zero
# pydantic dependency at import time. config.BrokerCostModel duck-types into
# this when the engine is wired.
@dataclass(frozen=True)
class _CostSpec:
    taker_fee_bps: float = 1.0
    spread_bps: float = 0.0
    slippage_bps_per_unit_size: float = 0.0

    def cost_log_per_side(self, size: float = 1.0) -> float:
        bps = (
            float(self.taker_fee_bps)
            + 0.5 * float(self.spread_bps)
            + float(self.slippage_bps_per_unit_size) * float(size)
        )
        return bps * 1e-4


TieBreak = Literal[
    "pessimistic_sl_first", "optimistic_tp_first", "probabilistic_hl_bridge"
]


@dataclass
class _PendingOpen:
    position_id: PositionId
    instrument: InstrumentId
    decision: Action       # the OPEN action that produced this
    submit_ts: Timestamp
    features_at_open: dict
    expiry: Duration


@dataclass
class _PendingScaleIn:
    target_position_id: PositionId
    size: Probability
    submit_ts: Timestamp


class SimBroker:
    """Event-driven triple-barrier sim with first-class Position tracking.

    Per-tick (advance_to(bar)):
      1. Walk this bar's M minutes for first-touch on every open Position
      2. Activate pending opens at this bar's first-minute close
      3. Mark-to-market the live Portfolio
    """

    def __init__(
        self,
        *,
        m_minutes: int = 20,
        cost: float = 1e-4,                # log-return per side (legacy flat)
        cost_model: Optional[object] = None,  # _CostSpec OR config.BrokerCostModel
        execution_latency_minutes: int = 1,
        inventory_cap: int = 5,            # P1: default 5
        stop_loss_log: Optional[float] = 0.0041113,  # None ⇒ no stop-loss
        tie_break: TieBreak = "pessimistic_sl_first",
    ):
        self.m_minutes = int(m_minutes)
        # legacy flat cost (kept for backwards-compat); cost_model takes precedence
        self.cost = LogReturn(float(cost))
        self.cost_model: _CostSpec = (
            cost_model if cost_model is not None
            else _CostSpec(taker_fee_bps=float(cost) * 1e4)
        )
        self.execution_latency_minutes = int(execution_latency_minutes)
        self.inventory_cap = int(inventory_cap)
        # stop_loss_log: None means "no stop loss" — broker skips SL branch
        # in first-touch scan and the position carries an inf SL.
        self.stop_loss_log: Optional[float] = (
            None if stop_loss_log is None else float(stop_loss_log)
        )
        if tie_break not in (
            "pessimistic_sl_first",
            "optimistic_tp_first",
            "probabilistic_hl_bridge",
        ):
            raise ValueError(f"unknown tie_break {tie_break!r}")
        self.tie_break: TieBreak = tie_break

        self._next_position_id: int = 1
        self._next_order_id: int = 1
        self._pending_opens: list[_PendingOpen] = []
        self._pending_scale_ins: list[_PendingScaleIn] = []

        # Live state
        self._portfolio: Portfolio = Portfolio()
        self._closed_history: list[BarrierTouched] = []
        self._events: list[Event] = []
        self._global_minute_idx: int = 0

    # ---- cost helpers ----

    def _cost_log_for(self, size: float) -> float:
        """Per-side cost in log-units, scaled by position size."""
        try:
            return float(self.cost_model.cost_log_per_side(float(size)))
        except (AttributeError, TypeError):
            return float(self.cost)

    # ---- public observability (read-only) ----

    def portfolio(self) -> Portfolio:
        """Snapshot of the current Portfolio. Strategies see this via ctx."""
        return self._portfolio

    def position(self) -> float:
        """Legacy: net signed size across all positions (single-instrument compat)."""
        return sum(p.signed_size for p in self._portfolio.open_positions)

    def open_orders(self) -> int:
        return self._portfolio.n_open + len(self._pending_opens) + len(self._pending_scale_ins)

    def realized_pnl_log(self) -> float:
        return float(self._portfolio.realized_pnl_log)

    def fills(self) -> Iterator[BarrierTouched]:
        yield from self._closed_history

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    # ---- Action dispatch (G9) ----

    def dispatch(
        self,
        action: Action,
        *,
        ts: Timestamp,
        instrument: InstrumentId,
        features: Optional[dict] = None,
    ) -> bool:
        """Route a single Action. Returns True if the broker accepted it."""
        if action.kind == ActionKind.HOLD:
            return True
        if action.kind == ActionKind.OPEN:
            return self._dispatch_open(action, ts=ts, instrument=instrument,
                                        features=features or {})
        if action.kind == ActionKind.SCALE_IN:
            return self._dispatch_scale_in(action, ts=ts)
        if action.kind == ActionKind.SCALE_OUT:
            return self._dispatch_scale_out(action, ts=ts)
        if action.kind == ActionKind.CLOSE:
            return self._dispatch_close(action, ts=ts, reason=ExitReason.MANUAL)
        if action.kind == ActionKind.MODIFY_STOP:
            return self._dispatch_modify_stop(action, ts=ts, actor="strategy")
        if action.kind == ActionKind.CANCEL:
            return self._dispatch_cancel(action, ts=ts)
        return False

    def _dispatch_open(self, action, *, ts, instrument, features):
        if self.open_orders() >= self.inventory_cap:
            return False
        pid = PositionId(self._next_position_id)
        self._next_position_id += 1
        self._pending_opens.append(_PendingOpen(
            position_id=pid,
            instrument=instrument,
            decision=action,
            submit_ts=ts,
            features_at_open=dict(features),
            expiry=action.expiry or Duration.from_minutes(self.m_minutes),
        ))
        return True

    def _dispatch_scale_in(self, action, *, ts):
        if action.target_position_id is None or action.size is None:
            return False
        if self._portfolio.get(action.target_position_id) is None:
            return False
        self._pending_scale_ins.append(_PendingScaleIn(
            target_position_id=action.target_position_id,
            size=action.size,
            submit_ts=ts,
        ))
        return True

    def _dispatch_scale_out(self, action, *, ts):
        if action.target_position_id is None or action.fraction is None:
            return False
        pos = self._portfolio.get(action.target_position_id)
        if pos is None or pos.closed:
            return False
        size_to_remove = float(pos.current_size) * float(action.fraction)
        if size_to_remove <= 0.0:
            return False
        # Use the position's last_mark_price as the exit price (assumes can
        # exit at last mark; real broker would use bid/ask).
        new_pos = pos.with_scale_out(
            size_removed=Quantity(size_to_remove),
            exit_price=pos.last_mark_price,
            ts=ts,
            cost_log=LogReturn(self._cost_log_for(size_to_remove)),
        )
        self._portfolio = self._portfolio.with_position(new_pos)
        # Realize the partial PnL into the portfolio's cumulative
        delta = float(new_pos.realized_pnl_log) - float(pos.realized_pnl_log)
        self._portfolio = self._portfolio.with_realized(LogReturn(delta))
        # Emit event
        self._events.append(PositionScaledOut(
            ts_init=ts, instrument=pos.instrument,
            position_id=int(pos.position_id),
            size_removed=size_to_remove,
            fill_price=float(pos.last_mark_price),
            realized_pnl_log=delta,
        ))
        if new_pos.closed:
            self._emit_close_event(new_pos, ExitReason.MANUAL, ts)
        return True

    def _dispatch_close(self, action, *, ts, reason: ExitReason):
        if action.target_position_id is None:
            return False
        pos = self._portfolio.get(action.target_position_id)
        if pos is None or pos.closed:
            return False
        new_pos = pos.with_close(
            exit_price=pos.last_mark_price,
            ts=ts,
            reason=reason,
            cost_log=LogReturn(self._cost_log_for(float(pos.current_size))),
        )
        self._portfolio = self._portfolio.with_position(new_pos)
        delta = float(new_pos.realized_pnl_log) - float(pos.realized_pnl_log)
        self._portfolio = self._portfolio.with_realized(LogReturn(delta))
        self._emit_close_event(new_pos, reason, ts)
        return True

    def _dispatch_modify_stop(self, action, *, ts, actor: str):
        if action.target_position_id is None:
            return False
        pos = self._portfolio.get(action.target_position_id)
        if pos is None or pos.closed:
            return False
        new_pos = pos.with_modified_stops(
            take_profit_log=action.new_take_profit,
            stop_loss_log=action.new_stop_loss,
            ts=ts,
        )
        self._portfolio = self._portfolio.with_position(new_pos)
        self._events.append(PositionStopModified(
            ts_init=ts, instrument=pos.instrument,
            position_id=int(pos.position_id),
            old_stop_loss=float(pos.stop_loss_log),
            new_stop_loss=float(new_pos.stop_loss_log),
            old_take_profit=float(pos.take_profit_log),
            new_take_profit=float(new_pos.take_profit_log),
            actor=actor,
        ))
        return True

    def _dispatch_cancel(self, action, *, ts):
        if action.target_position_id is None:
            return False
        before = len(self._pending_opens)
        self._pending_opens = [
            p for p in self._pending_opens if p.position_id != action.target_position_id
        ]
        return len(self._pending_opens) < before

    # ---- engine integration: advance_to ----

    def advance_to(self, ts: Timestamp, bar: Optional[DecisionBar]) -> list[BarrierTouched]:
        if bar is None:
            return []
        minutes: tuple = bar.underlying
        if not minutes or len(minutes) < self.m_minutes:
            self._global_minute_idx += len(minutes) if minutes else self.m_minutes
            return []

        new_fills: list[BarrierTouched] = []

        # Step A: activate pending opens
        if self._pending_opens and self._portfolio.n_open < self.inventory_cap:
            still_pending: list[_PendingOpen] = []
            for po in self._pending_opens:
                if self._portfolio.n_open >= self.inventory_cap:
                    still_pending.append(po)
                    continue
                fill_idx = max(0, self.execution_latency_minutes - 1)
                if fill_idx >= len(minutes):
                    still_pending.append(po)
                    continue
                m: MinuteBar = minutes[fill_idx]
                ep = float(m.close)
                if not (math.isfinite(ep) and ep > 0):
                    continue
                d = po.decision
                # Resolve stop-loss: action overrides; otherwise broker default
                # (which may be None ⇒ "no stop loss" ⇒ +inf in log-units).
                if d.stop_loss is not None:
                    sl_log = d.stop_loss
                elif self.stop_loss_log is None:
                    sl_log = LogReturn(float("inf"))
                else:
                    sl_log = LogReturn(self.stop_loss_log)
                pos = Position.open_at(
                    position_id=po.position_id,
                    instrument=po.instrument,
                    side=d.side or Side.LONG,
                    size=Quantity(float(d.size or 1.0)),
                    entry_price=Price(ep),
                    ts=m.close_time,
                    take_profit_log=d.take_profit or LogReturn.from_bps(41),
                    stop_loss_log=sl_log,
                    expiry=po.expiry,
                )
                # Attach features_at_open onto the position via fills (we keep
                # them on the close event for ARF training).
                self._portfolio = self._portfolio.with_position(pos)
                self._events.append(PositionOpened(
                    ts_init=m.close_time, instrument=po.instrument,
                    position_id=int(po.position_id),
                    side=int(pos.side),
                    size=float(pos.current_size),
                    entry_price=float(pos.avg_entry_price),
                ))
            self._pending_opens = still_pending

        # Step A2: activate pending scale-ins (use bar's first-minute close as fill)
        if self._pending_scale_ins:
            still_pending: list[_PendingScaleIn] = []
            for ps in self._pending_scale_ins:
                pos = self._portfolio.get(ps.target_position_id)
                if pos is None or pos.closed:
                    continue
                if self.execution_latency_minutes - 1 < len(minutes):
                    m = minutes[max(0, self.execution_latency_minutes - 1)]
                    fill_price = Price(float(m.close))
                    size_added = Quantity(float(ps.size) * float(pos.open_size))
                    new_pos = pos.with_scale_in(
                        size_added=size_added, price=fill_price, ts=m.close_time,
                    )
                    self._portfolio = self._portfolio.with_position(new_pos)
                    self._events.append(PositionScaledIn(
                        ts_init=m.close_time, instrument=pos.instrument,
                        position_id=int(pos.position_id),
                        size_added=float(size_added),
                        fill_price=float(fill_price),
                        new_avg_entry_price=float(new_pos.avg_entry_price),
                    ))
                else:
                    still_pending.append(ps)
            self._pending_scale_ins = still_pending

        # Step B: walk minutes; first-touch on every open position
        for pos_id in list(self._portfolio.positions.keys()):
            pos = self._portfolio.get(pos_id)
            if pos is None or pos.closed:
                continue
            ep = float(pos.avg_entry_price)
            sl_finite = math.isfinite(float(pos.stop_loss_log))
            if pos.side == Side.LONG:
                tp_price = ep * math.exp(float(pos.take_profit_log))
                sl_price = (ep * math.exp(-float(pos.stop_loss_log))
                             if sl_finite else 0.0)
            else:
                tp_price = ep * math.exp(-float(pos.take_profit_log))
                sl_price = (ep * math.exp(float(pos.stop_loss_log))
                             if sl_finite else float("inf"))
            # Determine when to start scanning — for newly-opened positions only
            # scan minutes AFTER the entry minute. We use a simple heuristic:
            # if pos opened earlier than this bar, scan all minutes.
            scan_start = 0
            if pos.open_ts >= bar.ts_init:
                # opened in this bar; scan after entry minute
                scan_start = max(0, self.execution_latency_minutes)
            exited = False
            for i in range(scan_start, len(minutes)):
                mb = minutes[i]
                hi, lo = float(mb.high), float(mb.low)
                if pos.side == Side.LONG:
                    sl_hit = sl_finite and (lo <= sl_price)
                    tp_hit = hi >= tp_price
                else:
                    sl_hit = sl_finite and (hi >= sl_price)
                    tp_hit = lo <= tp_price
                if sl_hit and tp_hit:
                    reason, exit_px = self._resolve_tie_break(
                        sl_price=sl_price, tp_price=tp_price, mb=mb,
                    )
                    self._barrier_close(pos, mb, reason, exit_px, ts)
                    exited = True
                    break
                if sl_hit:
                    self._barrier_close(pos, mb, ExitReason.SL, sl_price, ts)
                    exited = True
                    break
                if tp_hit:
                    self._barrier_close(pos, mb, ExitReason.TP, tp_price, ts)
                    exited = True
                    break
            if exited:
                new_fills.append(self._closed_history[-1])
                continue
            # Not exited via barriers; check expiry
            if pos.expiry_ts <= bar.close_time:
                # Find the minute closest to expiry in this bar
                last_mb = minutes[-1]
                self._barrier_close(pos, last_mb, ExitReason.TIMEOUT,
                                    float(last_mb.close), ts)
                new_fills.append(self._closed_history[-1])
                continue
            # Mark-to-market with this bar's close
            new_pos = pos.with_mark(Price(float(bar.close)), bar.close_time)
            self._portfolio = self._portfolio.with_position(new_pos)

        # Step C: mark-to-market refresh + high-water mark
        self._portfolio = self._portfolio.with_mark(
            mark_prices={bar.instrument: Price(float(bar.close))},
            ts=bar.close_time,
        )

        self._global_minute_idx += len(minutes)
        return new_fills

    def _resolve_tie_break(
        self,
        *,
        sl_price: float,
        tp_price: float,
        mb: MinuteBar,
    ) -> tuple[ExitReason, float]:
        """Pick (reason, exit_price) when SL and TP both fire in the same minute.

        See module docstring for semantics. Probabilistic mode draws Bernoulli(0.5)
        from a per-minute deterministic seed (the minute's open ts) so backtests
        are reproducible across runs.
        """
        if self.tie_break == "optimistic_tp_first":
            return ExitReason.TP, tp_price
        if self.tie_break == "probabilistic_hl_bridge":
            seed = int(mb.ts_init.ns) ^ 0x5A5A5A5A
            rng = random.Random(seed)
            if rng.random() < 0.5:
                return ExitReason.TP, tp_price
            return ExitReason.SL, sl_price
        # default — pessimistic
        return ExitReason.SL, sl_price

    def _barrier_close(self, pos: Position, mb: MinuteBar, reason: ExitReason,
                       exit_price: float, ts: Timestamp) -> None:
        cost_log = LogReturn(self._cost_log_for(float(pos.open_size)))
        new_pos = pos.with_close(
            exit_price=Price(float(exit_price)),
            ts=mb.close_time,
            reason=reason,
            cost_log=cost_log,
        )
        delta = float(new_pos.realized_pnl_log) - float(pos.realized_pnl_log)
        self._portfolio = self._portfolio.with_position(new_pos)
        self._portfolio = self._portfolio.with_realized(LogReturn(delta))
        # BarrierTouched feeds the metrics path; PositionClosed feeds the event log.
        fill = BarrierTouched(
            ts_init=mb.close_time, instrument=pos.instrument,
            order_id=OrderId(int(pos.position_id)),
            entry_ts=pos.open_ts, exit_ts=mb.close_time,
            entry_price=pos.avg_entry_price, exit_price=Price(float(exit_price)),
            fill_size=Quantity(float(pos.open_size)),
            side=int(pos.side), reason=reason,
            pnl_log_gross=LogReturn(delta + 2.0 * float(cost_log) * float(pos.open_size)),
            pnl_log_net=LogReturn(delta),
            features_at_open={},
            y_matured=1 if reason == ExitReason.TP else 0,
        )
        self._closed_history.append(fill)
        self._emit_close_event(new_pos, reason, mb.close_time)

    def _emit_close_event(self, pos: Position, reason: ExitReason, ts: Timestamp) -> None:
        cost_log = self._cost_log_for(float(pos.open_size))
        self._events.append(PositionClosed(
            ts_init=ts, instrument=pos.instrument,
            position_id=int(pos.position_id),
            side=int(pos.side),
            fill_size=float(pos.open_size),
            entry_price=float(pos.avg_entry_price),
            exit_price=float(pos.last_mark_price),
            entry_ts=pos.open_ts,
            exit_ts=ts,
            reason=str(reason),
            pnl_log_gross=float(pos.realized_pnl_log) + 2.0 * float(cost_log) * float(pos.open_size),
            pnl_log_net=float(pos.realized_pnl_log),
            y_matured=1 if reason == ExitReason.TP else 0,
        ))

    def cancel(self, order_id: OrderId) -> None:
        """Cancel a pending open by order_id (= position_id, single-instrument)."""
        self._pending_opens = [p for p in self._pending_opens
                                if int(p.position_id) != int(order_id)]

    # ---- finalize ----

    def finalize(self) -> "BrokerLedger":
        return BrokerLedger(
            fills=list(self._closed_history),
            n_open_at_finalize=self._portfolio.n_open,
            config={
                "m_minutes": self.m_minutes,
                "cost": float(self.cost),
                "execution_latency_minutes": self.execution_latency_minutes,
                "inventory_cap": self.inventory_cap,
                "stop_loss_log": self.stop_loss_log,
                "tie_break": self.tie_break,
                "cost_model": {
                    "taker_fee_bps": float(getattr(self.cost_model, "taker_fee_bps", float(self.cost) * 1e4)),
                    "spread_bps": float(getattr(self.cost_model, "spread_bps", 0.0)),
                    "slippage_bps_per_unit_size": float(
                        getattr(self.cost_model, "slippage_bps_per_unit_size", 0.0)
                    ),
                },
            },
        )


@dataclass
class BrokerLedger:
    fills: list[BarrierTouched]
    n_open_at_finalize: int
    config: dict


class BinanceBroker:
    """STUB. Real implementation deferred."""

    def __init__(self, **kw):
        self._kw = kw

    def submit(self, order):
        raise NotImplementedError

    def cancel(self, order_id):
        raise NotImplementedError

    def position(self) -> float:
        return 0.0

    def open_orders(self) -> int:
        return 0

    def realized_pnl_log(self) -> float:
        return 0.0

    def portfolio(self) -> Portfolio:
        return Portfolio()

    def advance_to(self, ts, bar):
        return []

    def finalize(self):
        return BrokerLedger(fills=[], n_open_at_finalize=0, config={})


__all__ = ["SimBroker", "BinanceBroker", "BrokerLedger"]
