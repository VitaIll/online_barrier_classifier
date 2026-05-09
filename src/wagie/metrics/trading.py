"""Trading metrics computed from a list of fills (BarrierTouched events).

This is the SINGLE source of truth for trading metrics in `wagie`. Sharpe is
annualized at the M-minute bar cadence; PSR follows Bailey & López de Prado
(2012); CDaR is the 5%-tail conditional drawdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats

from wagie.core.event import BarrierTouched as Fill


@dataclass
class TradingMetrics:
    n_trades: int
    n_tp: int
    n_sl: int
    n_timeout: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_log_return: float
    total_pct_return: float
    sharpe: float
    probabilistic_sharpe: float
    sortino: float
    max_drawdown_log: float
    cdar_5pct_log: float

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades, "n_tp": self.n_tp, "n_sl": self.n_sl,
            "n_timeout": self.n_timeout, "hit_rate": self.hit_rate,
            "avg_win": self.avg_win, "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "total_log_return": self.total_log_return,
            "total_pct_return": self.total_pct_return,
            "sharpe": self.sharpe, "probabilistic_sharpe": self.probabilistic_sharpe,
            "sortino": self.sortino, "max_drawdown_log": self.max_drawdown_log,
            "cdar_5pct_log": self.cdar_5pct_log,
        }


def compute_trading_metrics(
    fills: Sequence[Fill], *, m_minutes: int = 20,
) -> TradingMetrics:
    n = len(fills)
    if n == 0:
        return TradingMetrics(
            n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
            hit_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            total_log_return=0.0, total_pct_return=0.0,
            sharpe=0.0, probabilistic_sharpe=0.5, sortino=0.0,
            max_drawdown_log=0.0, cdar_5pct_log=0.0,
        )

    pnl = np.array([float(f.pnl_log_net) for f in fills], dtype=float)
    bars_per_year = 365.0 * 24.0 * 60.0 / m_minutes
    avg_bars_per_trade = float(np.mean([
        max(1.0, (f.exit_ts.ns - f.entry_ts.ns) / 60_000_000_000.0)
        if (getattr(f, "entry_ts", None) and getattr(f, "exit_ts", None))
        else float(m_minutes)
        for f in fills
    ]))
    trades_per_year = bars_per_year / max(avg_bars_per_trade, 1.0)

    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
    sr_trade = mean_pnl / max(std_pnl, 1e-12)
    sharpe = sr_trade * math.sqrt(trades_per_year)

    if n > 3:
        skew = float(scipy_stats.skew(pnl))
        kurt = float(scipy_stats.kurtosis(pnl, fisher=False))
    else:
        skew, kurt = 0.0, 3.0
    psr_denom = max(1.0 - skew * sr_trade + (kurt - 1.0) / 4.0 * sr_trade ** 2, 1e-12)
    psr = (
        float(scipy_stats.norm.cdf(sr_trade * math.sqrt(max(n - 1, 1)) / math.sqrt(psr_denom)))
        if n > 1 else 0.5
    )

    downside = pnl[pnl < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-12
    sortino = (mean_pnl / max(downside_std, 1e-12)) * math.sqrt(trades_per_year)

    eq = np.cumsum(pnl)
    cummax = np.maximum.accumulate(eq)
    drawdown = eq - cummax
    max_dd = float(-drawdown.min()) if len(drawdown) else 0.0
    n_tail = max(1, int(0.05 * len(drawdown)))
    cdar_5 = float(-np.mean(np.sort(drawdown)[:n_tail])) if len(drawdown) else 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    profit_factor = float(wins.sum() / max(-losses.sum(), 1e-12)) if len(losses) else float("inf")

    total_log = float(eq[-1]) if len(eq) else 0.0
    total_pct = float(math.exp(total_log) - 1.0)

    return TradingMetrics(
        n_trades=int(n),
        n_tp=int(sum(1 for f in fills if str(f.reason) == "tp")),
        n_sl=int(sum(1 for f in fills if str(f.reason) == "sl")),
        n_timeout=int(sum(1 for f in fills if str(f.reason) == "timeout")),
        hit_rate=float((pnl > 0).mean()),
        avg_win=avg_win, avg_loss=avg_loss, profit_factor=profit_factor,
        total_log_return=total_log, total_pct_return=total_pct,
        sharpe=float(sharpe), probabilistic_sharpe=float(psr),
        sortino=float(sortino),
        max_drawdown_log=max_dd, cdar_5pct_log=cdar_5,
    )


def deflated_sharpe(
    observed_sharpe: float, var_trial_sharpes: float,
    n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
) -> tuple[float, float]:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio. Returns (p, sr_star)."""
    gamma = 0.5772156649015329
    e = math.e
    sr_star = math.sqrt(max(var_trial_sharpes, 0.0)) * (
        (1.0 - gamma) * scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
        + gamma * scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    )
    psr_denom = max(
        1.0 - skew * observed_sharpe + (kurt - 1.0) / 4.0 * observed_sharpe ** 2,
        1e-12,
    )
    z = (observed_sharpe - sr_star) * math.sqrt(max(n_obs - 1, 1)) / math.sqrt(psr_denom)
    return float(scipy_stats.norm.cdf(z)), float(sr_star)


__all__ = ["TradingMetrics", "compute_trading_metrics", "deflated_sharpe"]
