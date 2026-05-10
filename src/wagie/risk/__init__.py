"""RiskEngine — composable, hot-swappable safety policies.

Strategy emits Sequence[Action]; Engine routes each through RiskEngine.check();
on rejection, optionally substitutes the policy's suggested_action; on approval,
dispatches to broker.

Policies are mutable (rate counters, kill engage) and pluggable. Operators add/
remove/replace at runtime via TradingConsole. Every change is audited via
RiskPolicyChanged events.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from wagie.core.action import Action, ActionKind
from wagie.core.numeric import LogReturn
from wagie.core.portfolio import Portfolio
from wagie.core.time import Timestamp


@dataclass(frozen=True, slots=True)
class RiskCheck:
    """Outcome of a policy check. If approved=False, optional suggested_action
    asks the engine to substitute (e.g., scale-down a too-large order)."""

    approved: bool
    reason: str = ""
    policy_name: str = ""
    suggested_action: Optional[Action] = None


@runtime_checkable
class RiskPolicy(Protocol):
    """A pluggable risk policy. Mutable state (rate counters, kill engaged)
    is allowed."""

    name: str

    def check(self, action: Action, portfolio: Portfolio, ts: Timestamp) -> RiskCheck: ...

    def reset(self) -> None: ...


# -----------------------------------------------------------------------------
# Concrete policies
# -----------------------------------------------------------------------------


@dataclass
class MaxPositionsPolicy:
    """Cap on number of concurrently open positions."""

    name: str = "max_positions"
    max_open: int = 5

    def check(self, action: Action, portfolio: Portfolio, ts: Timestamp) -> RiskCheck:
        if action.kind == ActionKind.OPEN and portfolio.n_open >= self.max_open:
            return RiskCheck(False, f"max_open={self.max_open} reached", self.name)
        return RiskCheck(True, policy_name=self.name)

    def reset(self) -> None:
        return None


@dataclass
class MaxDrawdownPolicy:
    """Block new opens when drawdown breaches threshold."""

    name: str = "max_drawdown"
    max_dd: float = -0.10              # in log-units; e.g. -0.10 ≈ -10%

    def check(self, action: Action, portfolio: Portfolio, ts: Timestamp) -> RiskCheck:
        if action.kind not in (ActionKind.OPEN, ActionKind.SCALE_IN):
            return RiskCheck(True, policy_name=self.name)
        dd = float(portfolio.drawdown_log)
        if dd < self.max_dd:
            return RiskCheck(
                False,
                f"drawdown {dd:+.4f} < limit {self.max_dd:+.4f}",
                self.name,
            )
        return RiskCheck(True, policy_name=self.name)

    def reset(self) -> None:
        return None


@dataclass
class MaxLossPerPositionPolicy:
    """Cap on stop-loss size per opened position. Suggests scaled-down stop."""

    name: str = "max_loss_per_position"
    max_loss: float = 0.005   # in log-units (0.005 ≈ 50 bps)

    def check(self, action: Action, portfolio: Portfolio, ts: Timestamp) -> RiskCheck:
        if action.kind != ActionKind.OPEN or action.stop_loss is None:
            return RiskCheck(True, policy_name=self.name)
        sl_abs = abs(float(action.stop_loss))
        if sl_abs > self.max_loss:
            # Suggest scaled-down stop
            from dataclasses import replace
            suggested = replace(action, stop_loss=LogReturn(self.max_loss))
            return RiskCheck(
                False,
                f"stop_loss {sl_abs:.4f} > max {self.max_loss:.4f}",
                self.name,
                suggested,
            )
        return RiskCheck(True, policy_name=self.name)

    def reset(self) -> None:
        return None


@dataclass
class MaxOrderRatePolicy:
    """Throttle order submissions per time window."""

    name: str = "max_order_rate"
    max_per_minute: int = 10
    _recent: collections.deque = field(default_factory=collections.deque)

    def check(self, action: Action, portfolio: Portfolio, ts: Timestamp) -> RiskCheck:
        if action.kind not in (ActionKind.OPEN, ActionKind.SCALE_IN):
            return RiskCheck(True, policy_name=self.name)
        cutoff_ns = ts.ns - 60_000_000_000
        while self._recent and self._recent[0] < cutoff_ns:
            self._recent.popleft()
        if len(self._recent) >= self.max_per_minute:
            return RiskCheck(
                False,
                f"order rate {len(self._recent)}/min >= max {self.max_per_minute}",
                self.name,
            )
        self._recent.append(ts.ns)
        return RiskCheck(True, policy_name=self.name)

    def reset(self) -> None:
        self._recent.clear()


@dataclass
class KillSwitchPolicy:
    """Operator-engaged blanket block on opening/scaling actions.

    When engaged, only CLOSE/CANCEL/MODIFY_STOP/HOLD are allowed. Per P7
    decision, this DOES NOT auto-flatten — operator must explicitly close.
    """

    name: str = "kill_switch"
    engaged: bool = False
    reason: str = ""

    def engage(self, reason: str = "") -> None:
        self.engaged = True
        self.reason = reason

    def disengage(self) -> None:
        self.engaged = False
        self.reason = ""

    def check(self, action: Action, portfolio: Portfolio, ts: Timestamp) -> RiskCheck:
        if not self.engaged:
            return RiskCheck(True, policy_name=self.name)
        if action.kind in (ActionKind.HOLD, ActionKind.CLOSE,
                           ActionKind.CANCEL, ActionKind.MODIFY_STOP,
                           ActionKind.SCALE_OUT):
            return RiskCheck(True, policy_name=self.name)
        return RiskCheck(False, f"kill switch engaged: {self.reason}", self.name)

    def reset(self) -> None:
        self.disengage()

    @classmethod
    def default_armed(cls, *, armed: bool = True, reason: str = "armed-default") -> "KillSwitchPolicy":
        """Live-default factory: returns an armed kill switch (engaged=False
        but `armed` flag tracks operator intent — for live-trading scaffolding
        the operator pre-installs a kill switch that is reachable but not yet
        firing). When `armed=False` it returns a fully disengaged policy."""
        ks = cls()
        if armed:
            # We DO NOT auto-engage — armed means "wired and ready", not "blocking".
            # Operators flip via .engage(reason).
            ks.reason = reason
        return ks


# -----------------------------------------------------------------------------
# RiskEngine
# -----------------------------------------------------------------------------


class RiskEngine:
    """Composable, mutable, observable risk policy chain.

    Policies are evaluated in order; the first rejection wins. Operators
    mutate via add/remove/replace; every change is logged.
    """

    def __init__(self, policies: Optional[list[RiskPolicy]] = None):
        self.policies: list[RiskPolicy] = list(policies or [])
        self.changes: list[dict] = []

    def check(self, action: Action, portfolio: Portfolio, ts: Timestamp) -> RiskCheck:
        for p in self.policies:
            r = p.check(action, portfolio, ts)
            if not r.approved:
                return r
        return RiskCheck(True)

    # ---- mutation (operator-driven) ----

    def add(self, policy: RiskPolicy) -> None:
        # If a policy of the same name exists, replace it
        existing = self.get(policy.name)
        if existing is not None:
            self.replace(policy.name, policy)
            return
        self.policies.append(policy)
        self.changes.append({"op": "add", "name": policy.name, "ts": time.time_ns()})

    def remove(self, name: str) -> bool:
        before = len(self.policies)
        self.policies = [p for p in self.policies if p.name != name]
        removed = len(self.policies) < before
        if removed:
            self.changes.append({"op": "remove", "name": name, "ts": time.time_ns()})
        return removed

    def replace(self, name: str, new: RiskPolicy) -> bool:
        replaced = False
        for i, p in enumerate(self.policies):
            if p.name == name:
                self.policies[i] = new
                replaced = True
                break
        if replaced:
            self.changes.append({"op": "replace", "name": name, "ts": time.time_ns()})
        return replaced

    def get(self, name: str) -> Optional[RiskPolicy]:
        return next((p for p in self.policies if p.name == name), None)

    def reset_all(self) -> None:
        for p in self.policies:
            p.reset()

    @classmethod
    def default(cls) -> "RiskEngine":
        """Sensible defaults wired for realistic backtests:

        1. MaxPositionsPolicy(max_open=5) — inventory cap
        2. MaxDrawdownPolicy(-0.30) — block opens after 30% peak-to-trough drawdown
        3. MaxLossPerPositionPolicy(0.05) — cap stop-loss size at 500 bps
        4. KillSwitchPolicy.default_armed() — wired and ready for live use

        These match the audit recommendation that default backtests have
        realistic safety wired (offline Sharpe collapse can be partly traced
        to absent drawdown protection).
        """
        return cls([
            MaxPositionsPolicy(max_open=5),
            MaxDrawdownPolicy(max_dd=-0.30),
            MaxLossPerPositionPolicy(max_loss=0.05),
            KillSwitchPolicy.default_armed(armed=True),
        ])

    @property
    def policy_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.policies)


__all__ = [
    "RiskCheck",
    "RiskPolicy",
    "RiskEngine",
    "MaxPositionsPolicy",
    "MaxDrawdownPolicy",
    "MaxLossPerPositionPolicy",
    "MaxOrderRatePolicy",
    "KillSwitchPolicy",
]
