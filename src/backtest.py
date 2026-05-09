"""Inventory-aware triple-barrier backtest harness.

Implements the "diligent trader" idealization from López de Prado (2018) Ch. 11:
flawless execution but realistic costs and a strict no-overlap (unit-position)
inventory cap. The first-touch logic mirrors `backtesting.py`'s
`_Broker._process_orders` (pattern-only; AGPL code not vendored). Metrics
follow Bailey & López de Prado (2012, 2014) with Probabilistic and Deflated
Sharpe Ratios that explicitly correct for non-Gaussian PnL distributions.

This module is the economic accept-gate for the autonomous loop. Any modeling
round that improves probability quality must also survive the harness on the
test split before it can be `accept`ed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import math
import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

ExitReason = Literal["tp", "sl", "timeout"]


@dataclass(frozen=True)
class Trade:
    k_open: int                # decision-boundary index at entry
    k_close: int               # decision-boundary index at exit (>= k_open + 1)
    n_open: int                # 1-minute index at entry (= k_open * M)
    n_close: int               # 1-minute index at exit
    entry_price: float
    exit_price: float
    exit_reason: ExitReason
    pnl_log_gross: float       # ln(exit/entry)
    pnl_log_net: float         # gross minus round-trip cost
    p_signal: float            # P(y=1) at entry
    bars_held: int             # n_close - n_open


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series          # cumulative net log-PnL indexed by k
    metrics: dict[str, float]
    config: dict


# -----------------------------------------------------------------------------
# Core simulation
# -----------------------------------------------------------------------------

def simulate_inventory_aware(
    boundaries: pd.DataFrame,
    minute_close: np.ndarray,
    minute_high: np.ndarray,
    minute_low: np.ndarray,
    p: np.ndarray,
    *,
    tau_open: float = 0.5,
    M: int = 10,
    phi: float = 0.0025,
    c_stop: float = 0.0023,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Simulate an inventory-aware long-only triple-barrier strategy.

    At each decision boundary `k` with `p[k] > tau_open` and flat inventory,
    open a long at `minute_close[n_k]` (n_k = k*M). Walk minutes
    `[n_k+1, n_k+M]` looking for first touch:
        - `low[n] <= entry · exp(-c_stop)`     -> exit_reason='sl',  exit_price=sl
        - `high[n] >= entry · exp(+phi)`       -> exit_reason='tp',  exit_price=tp
        - `n == n_k + M`                        -> exit_reason='timeout', exit_price=close[n]

    On same-bar collision (TP and SL pierced in the same 1m bar) SL wins
    (pessimistic). New signals during an open trade are skipped (inventory cap).

    Net per-trade log return: `ln(exit/entry) - 2 · cost_bps · 1e-4`.

    Parameters
    ----------
    boundaries : pd.DataFrame
        Must contain columns ``k`` and ``ts`` (no other constraints; one row
        per decision boundary in chronological order).
    minute_close, minute_high, minute_low : np.ndarray
        1-minute price arrays. ``boundaries["k"][i] * M`` indexes into these.
    p : np.ndarray
        Predicted P(y=1) at each boundary. Same length as `boundaries`.
    tau_open : float
        Decision threshold for opening a long.
    M : int
        Decision interval (minutes per boundary).
    phi : float
        Take-profit barrier in log-return units (e.g. 0.0025 = 25 bps).
    c_stop : float
        Stop-loss barrier in log-return units (positive number).
    cost_bps : float
        Per-side transaction-cost in basis points; round-trip = 2 · cost_bps.

    Returns
    -------
    BacktestResult
    """
    n_boundaries = len(boundaries)
    if len(p) != n_boundaries:
        raise ValueError(f"p length {len(p)} != boundaries length {n_boundaries}")
    if not (np.isfinite(p)).all():
        raise ValueError("p contains non-finite values")
    M = int(M)

    trades: list[Trade] = []
    cooldown_until_n = -1  # last 1m index inclusive in which inventory is held

    for k in range(n_boundaries):
        if p[k] <= tau_open:
            continue
        n_k = int(boundaries["k"].iat[k]) * M  # canonical: k * M
        if n_k <= cooldown_until_n:
            continue                      # still in trade
        if n_k + M >= len(minute_close):
            break                         # not enough future bars

        entry_price = float(minute_close[n_k])
        if not math.isfinite(entry_price) or entry_price <= 0:
            continue
        tp_price = entry_price * math.exp(phi)
        sl_price = entry_price * math.exp(-c_stop)

        # Slice future bars [n_k+1, n_k+M] inclusive on both ends.
        future_high = minute_high[n_k + 1 : n_k + 1 + M]
        future_low = minute_low[n_k + 1 : n_k + 1 + M]

        sl_hits = future_low <= sl_price
        tp_hits = future_high >= tp_price

        # `argmax` returns 0 when no True (need explicit empty-mask handling).
        sl_first = int(np.argmax(sl_hits)) if sl_hits.any() else M
        tp_first = int(np.argmax(tp_hits)) if tp_hits.any() else M

        # Pessimistic: SL wins on tie.
        if sl_first <= tp_first and sl_first < M:
            exit_reason: ExitReason = "sl"
            exit_idx = n_k + 1 + sl_first
            exit_price = sl_price
        elif tp_first < sl_first:
            exit_reason = "tp"
            exit_idx = n_k + 1 + tp_first
            exit_price = tp_price
        else:
            exit_reason = "timeout"
            exit_idx = n_k + M
            exit_price = float(minute_close[exit_idx])

        pnl_log_gross = float(math.log(exit_price / entry_price))
        pnl_log_net = pnl_log_gross - 2.0 * cost_bps * 1e-4

        # Map exit_idx back to the next decision boundary (k_close >= k+1).
        k_close = (exit_idx + M - 1) // M
        if k_close <= k:
            k_close = k + 1
        if k_close >= n_boundaries:
            k_close = n_boundaries - 1

        trades.append(
            Trade(
                k_open=int(k),
                k_close=int(k_close),
                n_open=int(n_k),
                n_close=int(exit_idx),
                entry_price=entry_price,
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                pnl_log_gross=pnl_log_gross,
                pnl_log_net=pnl_log_net,
                p_signal=float(p[k]),
                bars_held=int(exit_idx - n_k),
            )
        )
        cooldown_until_n = int(exit_idx)

    trade_df = (
        pd.DataFrame([asdict(t) for t in trades])
        if trades
        else pd.DataFrame(
            columns=[
                "k_open", "k_close", "n_open", "n_close", "entry_price",
                "exit_price", "exit_reason", "pnl_log_gross", "pnl_log_net",
                "p_signal", "bars_held",
            ]
        )
    )

    # Equity curve at boundary cadence: cumulative net log-PnL.
    equity_log = np.zeros(n_boundaries, dtype=float)
    for t in trades:
        # Apply trade's PnL from k_close onward (PnL realized at exit).
        equity_log[t.k_close:] += t.pnl_log_net
    equity = pd.Series(equity_log, index=boundaries["k"].to_numpy(), name="equity_log")

    metrics = compute_backtest_metrics(trade_df, equity, M=M)
    config = {
        "tau_open": float(tau_open),
        "M": int(M),
        "phi": float(phi),
        "c_stop": float(c_stop),
        "cost_bps": float(cost_bps),
        "n_boundaries": int(n_boundaries),
    }
    return BacktestResult(trades=trade_df, equity=equity, metrics=metrics, config=config)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_backtest_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    *,
    M: int,
) -> dict[str, float]:
    """Return the minimum-viable metric set defined by the round-1 hypothesis."""
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "total_log_return": 0.0, "total_pct_return": 0.0,
            "ann_log_return": 0.0, "ann_pct_return": 0.0,
            "sharpe": 0.0, "probabilistic_sharpe": 0.5,
            "sortino": 0.0, "calmar": 0.0,
            "max_drawdown_log": 0.0, "cdar_5pct_log": 0.0,
            "hit_rate": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "avg_bars_held": 0.0, "trades_per_year_est": 0.0,
        }

    pnl = trades["pnl_log_net"].to_numpy(dtype=float)
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1)) if n > 1 else 0.0

    # Annualization: M-min decision bars, 24*60/M bars/day, 365 days/yr.
    bars_per_year = 365.0 * 24.0 * 60.0 / M
    avg_bars_per_trade = float(trades["bars_held"].mean())
    trades_per_year = bars_per_year / max(avg_bars_per_trade, 1.0)

    # Per-trade Sharpe (no annualization).
    sr_trade = mean_pnl / max(std_pnl, 1e-12)
    sharpe = sr_trade * math.sqrt(trades_per_year)

    # Probabilistic Sharpe vs SR*=0 (Bailey & López de Prado 2012, eq 1).
    if n > 3:
        skew = float(stats.skew(pnl))
        kurt = float(stats.kurtosis(pnl, fisher=False))  # full kurtosis (Gaussian = 3)
    else:
        skew, kurt = 0.0, 3.0
    psr_denom = max(1.0 - skew * sr_trade + (kurt - 1.0) / 4.0 * sr_trade ** 2, 1e-12)
    psr = float(stats.norm.cdf(sr_trade * math.sqrt(max(n - 1, 1)) / math.sqrt(psr_denom))) if n > 1 else 0.5

    # Sortino vs MAR=0 (downside deviation).
    downside = pnl[pnl < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside, ddof=1))
    else:
        downside_std = 1e-12
    sortino = (mean_pnl / max(downside_std, 1e-12)) * math.sqrt(trades_per_year)

    # Drawdown (log-equity).
    eq = equity.to_numpy(dtype=float)
    cummax = np.maximum.accumulate(eq)
    drawdown = eq - cummax  # <= 0
    max_drawdown = float(-drawdown.min()) if len(drawdown) else 0.0
    # CDaR_5%: mean of the worst 5% of (negative) drawdowns.
    n_tail = max(1, int(0.05 * len(drawdown)))
    cdar_5 = float(-np.mean(np.sort(drawdown)[:n_tail])) if len(drawdown) else 0.0

    total_log = float(eq[-1]) if len(eq) else 0.0
    total_pct = float(math.exp(total_log) - 1.0)
    n_periods_year = bars_per_year / max(len(eq), 1)
    ann_log = total_log * n_periods_year
    ann_pct = float(math.exp(ann_log) - 1.0)
    calmar = ann_log / max(max_drawdown, 1e-12)

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    profit_factor = float(wins.sum() / max(-losses.sum(), 1e-12)) if len(losses) else float("inf")

    return {
        "n_trades": int(n),
        "n_tp": int((trades["exit_reason"] == "tp").sum()),
        "n_sl": int((trades["exit_reason"] == "sl").sum()),
        "n_timeout": int((trades["exit_reason"] == "timeout").sum()),
        "trades_per_year_est": float(trades_per_year),
        "avg_bars_held": float(avg_bars_per_trade),
        "total_log_return": float(total_log),
        "total_pct_return": float(total_pct),
        "ann_log_return": float(ann_log),
        "ann_pct_return": float(ann_pct),
        "sharpe": float(sharpe),
        "probabilistic_sharpe": float(psr),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown_log": float(max_drawdown),
        "cdar_5pct_log": float(cdar_5),
        "hit_rate": float((pnl > 0).mean()),
        "profit_factor": float(profit_factor),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def deflated_sharpe(
    observed_sharpe: float,
    var_trial_sharpes: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt: float,
) -> tuple[float, float]:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Returns (DSR, SR_star) where DSR ∈ [0, 1] is the probability that the
    observed Sharpe exceeds the expected maximum SR among `n_trials`
    independent zero-skill strategies with cross-trial variance
    `var_trial_sharpes`. Reject null at DSR > 0.95.

    `kurt` is full kurtosis (Gaussian = 3). `n_obs` is the number of return
    observations contributing to `observed_sharpe`.
    """
    gamma = 0.5772156649015329  # Euler-Mascheroni
    e = math.e
    sr_star = math.sqrt(max(var_trial_sharpes, 0.0)) * (
        (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    )
    psr_denom = max(
        1.0 - skew * observed_sharpe + (kurt - 1.0) / 4.0 * observed_sharpe ** 2,
        1e-12,
    )
    z = (observed_sharpe - sr_star) * math.sqrt(max(n_obs - 1, 1)) / math.sqrt(psr_denom)
    return float(stats.norm.cdf(z)), float(sr_star)


# -----------------------------------------------------------------------------
# Diagnostic plots (visual-first reporting per CONSTITUTION V)
# -----------------------------------------------------------------------------

def plot_backtest(result: BacktestResult, out_path) -> None:
    """Equity, drawdown, trade-PnL distribution, confidence-vs-realized."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    # Equity curve.
    ax = axes[0, 0]
    eq = result.equity
    ax.plot(eq.index, eq.values, color="#1B6B33", linewidth=1.3)
    ax.axhline(0, color="#999", linewidth=0.6)
    ax.set_xlabel("Decision boundary k")
    ax.set_ylabel("Cumulative net log-PnL")
    ax.set_title(f"Equity curve  ({result.metrics['n_trades']} trades, "
                 f"PSR={result.metrics['probabilistic_sharpe']:.2f}, "
                 f"Sharpe={result.metrics['sharpe']:.2f})")
    ax.grid(alpha=0.25)

    # Drawdown.
    ax = axes[0, 1]
    eq_arr = result.equity.to_numpy()
    cummax = np.maximum.accumulate(eq_arr)
    dd = eq_arr - cummax
    ax.fill_between(result.equity.index, dd, 0, color="#7A1B1B", alpha=0.4)
    ax.plot(result.equity.index, dd, color="#7A1B1B", linewidth=0.8)
    ax.set_xlabel("Decision boundary k")
    ax.set_ylabel("Drawdown (log)")
    ax.set_title(f"Drawdown   max={result.metrics['max_drawdown_log']:.4f},  "
                 f"CDaR(5%)={result.metrics['cdar_5pct_log']:.4f}")
    ax.grid(alpha=0.25)

    # Trade PnL distribution by exit reason.
    ax = axes[1, 0]
    if not result.trades.empty:
        bins = np.linspace(
            result.trades["pnl_log_net"].min() - 1e-6,
            result.trades["pnl_log_net"].max() + 1e-6,
            40,
        )
        for reason, color in [
            ("tp", "#1B6B33"),
            ("sl", "#7A1B1B"),
            ("timeout", "#7A5C00"),
        ]:
            sub = result.trades[result.trades["exit_reason"] == reason]
            if len(sub):
                ax.hist(
                    sub["pnl_log_net"], bins=bins, alpha=0.55,
                    color=color, label=f"{reason}  n={len(sub)}",
                    edgecolor="white", linewidth=0.4,
                )
        ax.axvline(0, color="black", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Net log-return per trade")
    ax.set_ylabel("Count")
    ax.set_title(f"Trade PnL by exit reason   "
                 f"hit-rate={result.metrics['hit_rate']:.2f},  "
                 f"PF={result.metrics['profit_factor']:.2f}")
    ax.grid(alpha=0.25)

    # Confidence vs realized.
    ax = axes[1, 1]
    if not result.trades.empty:
        colors = result.trades["exit_reason"].map(
            {"tp": "#1B6B33", "sl": "#7A1B1B", "timeout": "#7A5C00"}
        )
        ax.scatter(
            result.trades["p_signal"],
            result.trades["pnl_log_net"],
            c=colors, alpha=0.55, s=18, edgecolors="white", linewidths=0.3,
        )
        ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"Predicted $P(y=1)$ at entry")
    ax.set_ylabel("Realized net log-return")
    ax.set_title("Confidence vs realized PnL")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Backtest result   "
        f"τ_open={result.config['tau_open']:.2f}  "
        f"φ={result.config['phi']:.4f}  "
        f"c_stop={result.config['c_stop']:.4f}  "
        f"cost_bps={result.config['cost_bps']:.1f}",
        fontsize=11, weight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


# -----------------------------------------------------------------------------
# Bootstrap no-skill null
# -----------------------------------------------------------------------------

def bootstrap_no_skill_pvalue(
    boundaries: pd.DataFrame,
    minute_close: np.ndarray,
    minute_high: np.ndarray,
    minute_low: np.ndarray,
    *,
    n_trades_observed: int,
    sharpe_observed: float,
    M: int = 10,
    phi: float = 0.0025,
    c_stop: float = 0.0023,
    cost_bps: float = 0.0,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Bootstrap p-value vs random-entry null.

    Generates `n_bootstrap` random predictions whose entry-rate matches the
    observed strategy, runs the harness on each, and reports the fraction of
    null Sharpes >= `sharpe_observed`. Use sparingly (each bootstrap replays
    the simulation; budget aware).
    """
    rng = np.random.default_rng(seed)
    n_boundaries = len(boundaries)
    entry_rate = n_trades_observed / max(n_boundaries, 1)
    null_sharpes = []
    for _ in range(n_bootstrap):
        p_null = (rng.random(n_boundaries) < entry_rate).astype(float)
        # Set tau_open just below 1 so True maps to "trade".
        result = simulate_inventory_aware(
            boundaries, minute_close, minute_high, minute_low,
            p_null,
            tau_open=0.5, M=M, phi=phi, c_stop=c_stop, cost_bps=cost_bps,
        )
        null_sharpes.append(result.metrics["sharpe"])
    null_sharpes = np.asarray(null_sharpes)
    p_value = float((null_sharpes >= sharpe_observed).mean())
    return {
        "p_value": p_value,
        "null_sharpe_mean": float(null_sharpes.mean()),
        "null_sharpe_std": float(null_sharpes.std()),
        "null_sharpe_q95": float(np.quantile(null_sharpes, 0.95)),
        "n_bootstrap": int(n_bootstrap),
    }
