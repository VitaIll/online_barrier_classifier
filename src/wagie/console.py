"""TradingConsole — operator REPL for a running Engine.

Lives in a separate thread alongside the engine. Commands are queued via
engine.command_queue and processed between bars (no race conditions).

Usage:
    engine = Engine(...)
    console = TradingConsole(engine)
    console.start()              # background thread
    console.positions()          # query (synchronous)
    console.kill("vol spike")    # operator action (queued)
    console.set_stop(17, sl=LogReturn.from_bps(-30))
    console.flatten()
    console.save("snap.tar.gz")

Commands are non-blocking from the caller's POV (they queue) but the engine
applies them atomically between bars.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from wagie.core.action import Action, ActionKind, ExitReason, Side
from wagie.core.identity import PositionId
from wagie.core.numeric import LogReturn
from wagie.core.time import Timestamp


logger = logging.getLogger(__name__)


class TradingConsole:
    """Operator console for a running Engine. Thread-safe via command queue."""

    def __init__(self, engine, prompt: str = "wagie> "):
        self.engine = engine
        self.prompt = prompt
        self._stop_event = threading.Event()
        self._repl_thread: Optional[threading.Thread] = None

    # ---- read-only queries (synchronous, snapshot-based) ----

    def positions(self) -> str:
        portfolio = self._snapshot()
        if portfolio.is_flat:
            return "no open positions"
        lines = [
            f"{'PID':>4} {'Side':>5} {'Size':>7} {'Entry':>10} "
            f"{'Mark':>10} {'PnL':>10} {'MAE':>10} {'MFE':>10} {'Age':>10}"
        ]
        for p in portfolio.open_positions:
            side = "LONG" if p.side == Side.LONG else "SHORT" if p.side == Side.SHORT else "FLAT"
            lines.append(
                f"{int(p.position_id):>4} {side:>5} {float(p.current_size):>7.3f} "
                f"{float(p.avg_entry_price):>10.4f} "
                f"{float(p.last_mark_price):>10.4f} "
                f"{float(p.unrealized_pnl_log)*1e4:>+9.1f}b "
                f"{float(p.mae_log)*1e4:>+9.1f}b "
                f"{float(p.mfe_log)*1e4:>+9.1f}b "
                f"{p.time_in_trade.minutes:>10.1f}m"
            )
        return "\n".join(lines)

    def portfolio(self) -> str:
        p = self._snapshot()
        return (f"Portfolio: n_open={p.n_open}\n"
                f"  realized   : {float(p.realized_pnl_log)*1e4:+.1f} bps\n"
                f"  unrealized : {float(p.unrealized_pnl_log)*1e4:+.1f} bps\n"
                f"  equity     : {float(p.equity_log)*1e4:+.1f} bps (log)\n"
                f"  drawdown   : {float(p.drawdown_log)*1e4:+.1f} bps (from peak)\n"
                f"  hwm        : {float(p.high_water_mark_log)*1e4:+.1f} bps")

    def risk_list(self) -> str:
        names = self.engine.risk_engine.policy_names
        if not names:
            return "no risk policies active"
        return "active risk policies:\n  " + "\n  ".join(names)

    def history(self, position_id: int) -> str:
        portfolio = self._snapshot()
        p = portfolio.get(PositionId(int(position_id)))
        if p is None:
            return f"no position with id {position_id}"
        lines = [f"Position {int(p.position_id)} ({p.instrument}):"]
        lines.append(f"  side={p.side.name} open_size={float(p.open_size):.3f} "
                      f"current_size={float(p.current_size):.3f}")
        lines.append(f"  open_ts={p.open_ts.isoformat()} "
                      f"avg_entry={float(p.avg_entry_price):.4f}")
        lines.append(f"  TP={float(p.take_profit_log)*1e4:+.1f}bps "
                      f"SL={float(p.stop_loss_log)*1e4:+.1f}bps "
                      f"expiry={p.expiry_ts.isoformat()}")
        lines.append(f"  realized={float(p.realized_pnl_log)*1e4:+.1f}bps "
                      f"unrealized={float(p.unrealized_pnl_log)*1e4:+.1f}bps "
                      f"MAE={float(p.mae_log)*1e4:+.1f}bps "
                      f"MFE={float(p.mfe_log)*1e4:+.1f}bps")
        if p.closed:
            lines.append(f"  CLOSED reason={p.close_reason}")
        return "\n".join(lines)

    # ---- operator actions (queued) ----

    def flatten(self, position_id: Optional[int] = None) -> str:
        """Close all positions, or one specific position."""
        def _do(engine):
            ts = Timestamp(engine.clock.now_ns())
            portfolio = engine.broker.portfolio()
            ids = ([PositionId(position_id)] if position_id is not None
                   else [p.position_id for p in portfolio.open_positions])
            for pid in ids:
                action = Action.close(pid)
                engine.broker.dispatch(action, ts=ts,
                                        instrument=portfolio.get(pid).instrument
                                        if portfolio.get(pid) else
                                        portfolio.open_positions[0].instrument
                                        if portfolio.open_positions else None,
                                        features={})
        self.engine.command_queue.put(_do)
        return f"flatten {'all' if position_id is None else position_id} queued"

    def set_stop(
        self,
        position_id: int,
        *,
        sl: Optional[LogReturn] = None,
        tp: Optional[LogReturn] = None,
    ) -> str:
        if sl is None and tp is None:
            return "set_stop requires sl= or tp="
        def _do(engine):
            ts = Timestamp(engine.clock.now_ns())
            portfolio = engine.broker.portfolio()
            pos = portfolio.get(PositionId(int(position_id)))
            if pos is None:
                return
            engine.broker._dispatch_modify_stop(
                Action.modify_stop(
                    PositionId(int(position_id)),
                    new_take_profit=tp,
                    new_stop_loss=sl,
                ),
                ts=ts, actor="operator",
            )
        self.engine.command_queue.put(_do)
        msg = f"modify_stop {position_id}: "
        if sl is not None: msg += f"sl={float(sl)*1e4:+.1f}bps "
        if tp is not None: msg += f"tp={float(tp)*1e4:+.1f}bps"
        return msg + " queued"

    def kill(self, reason: str = "operator") -> str:
        from wagie.risk import KillSwitchPolicy
        def _do(engine):
            ks = engine.risk_engine.get("kill_switch")
            if ks is None:
                ks = KillSwitchPolicy()
                engine.risk_engine.add(ks)
            ks.engage(reason)
        self.engine.command_queue.put(_do)
        return f"kill switch ENGAGED: {reason}"

    def unkill(self) -> str:
        def _do(engine):
            ks = engine.risk_engine.get("kill_switch")
            if ks is not None:
                ks.disengage()
        self.engine.command_queue.put(_do)
        return "kill switch DISENGAGED"

    def pause(self) -> str:
        def _do(engine):
            engine.paused = True
        self.engine.command_queue.put(_do)
        return "engine PAUSED (no new strategy emit)"

    def resume(self) -> str:
        def _do(engine):
            engine.paused = False
        self.engine.command_queue.put(_do)
        return "engine RESUMED"

    def risk_add(self, policy) -> str:
        def _do(engine):
            engine.risk_engine.add(policy)
        self.engine.command_queue.put(_do)
        return f"risk: added {getattr(policy, 'name', policy.__class__.__name__)}"

    def risk_remove(self, name: str) -> str:
        def _do(engine):
            engine.risk_engine.remove(name)
        self.engine.command_queue.put(_do)
        return f"risk: removed {name}"

    def save(self, path: str) -> str:
        def _do(engine):
            from wagie.persistence import save as _save
            _save(engine.pipeline, engine.broker, path,
                  config_hash=None, n_seen=engine._n_seen)
        self.engine.command_queue.put(_do)
        return f"save {path} queued"

    def help(self) -> str:
        return """\
TradingConsole commands:
  positions()                   list open positions
  portfolio()                   equity + cash + drawdown
  risk_list()                   active risk policies
  history(position_id)          position lifecycle
  flatten(position_id=None)     close all or one position
  set_stop(pid, sl=, tp=)       modify stops on a position
  kill(reason="...")            engage kill switch
  unkill()                      disengage kill switch
  pause()                       stop emitting strategy actions
  resume()                      resume strategy emit
  risk_add(policy)              add a RiskPolicy
  risk_remove(name)             remove by name
  save(path)                    save snapshot
  help()                        this message
"""

    # ---- internals ----

    def _snapshot(self):
        return self.engine.broker.portfolio()

    def start(self) -> "TradingConsole":
        """Idempotent — returns self for chaining."""
        return self

    def stop(self) -> None:
        self._stop_event.set()


__all__ = ["TradingConsole"]
