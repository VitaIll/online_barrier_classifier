"""Contract tests for `wagie.metrics.trading`.

Covers:
  - Empty fills → all-zero TradingMetrics (PSR fallback = 0.5).
  - Hand-crafted PnL series with verifiable Sharpe / PSR (cross-checked vs scipy).
  - All-positive, all-negative, identical-PnL edge cases.
  - Profit factor goes to +inf when there are no losses.
  - n_tp/n_sl/n_timeout counters dispatch on ExitReason.
  - `deflated_sharpe` returns sane (p, sr_star) and respects monotonicity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from wagie.core.action import ExitReason
from wagie.core.event import BarrierTouched
from wagie.core.identity import OrderId
from wagie.core.numeric import LogReturn, Price, Quantity
from wagie.core.time import Timestamp
from wagie.metrics import TradingMetrics, compute_trading_metrics, deflated_sharpe


# ---------- helpers --------------------------------------------------------

_NS_PER_MIN = 60_000_000_000


def _fill(
    pnl: float,
    *,
    reason: ExitReason = ExitReason.TP,
    entry_min: int = 0,
    exit_min: int = 20,
) -> BarrierTouched:
    """Build a minimal BarrierTouched with controlled net log-PnL and bar age."""
    return BarrierTouched(
        ts_init=Timestamp(exit_min * _NS_PER_MIN),
        order_id=OrderId(int(entry_min)),
        entry_ts=Timestamp(entry_min * _NS_PER_MIN),
        exit_ts=Timestamp(exit_min * _NS_PER_MIN),
        entry_price=Price(100.0),
        exit_price=Price(100.0 * math.exp(pnl)),
        fill_size=Quantity(1.0),
        side=1,
        reason=reason,
        pnl_log_gross=LogReturn(pnl),
        pnl_log_net=LogReturn(pnl),
    )


# ---------- empty / one fill -----------------------------------------------


def test_empty_fills_returns_zero_metrics() -> None:
    tm = compute_trading_metrics([])
    assert isinstance(tm, TradingMetrics)
    assert tm.n_trades == 0
    assert tm.n_tp == tm.n_sl == tm.n_timeout == 0
    assert tm.hit_rate == 0.0
    assert tm.avg_win == tm.avg_loss == 0.0
    assert tm.profit_factor == 0.0
    assert tm.sharpe == 0.0
    assert tm.probabilistic_sharpe == 0.5
    assert tm.sortino == 0.0
    assert tm.max_drawdown_log == 0.0
    assert tm.cdar_5pct_log == 0.0


def test_single_fill_psr_fallback_is_half() -> None:
    """n=1 → std undefined; PSR fallback = 0.5 by contract."""
    tm = compute_trading_metrics([_fill(0.01)])
    assert tm.n_trades == 1
    assert tm.probabilistic_sharpe == 0.5
    # std=0 path: sharpe explodes via 1e-12 epsilon — just check finite, > 0.
    assert math.isfinite(tm.sharpe) and tm.sharpe > 0.0


# ---------- Sharpe / PSR formula cross-check -------------------------------


def test_sharpe_and_psr_match_closed_form() -> None:
    """Hand-crafted PnL: 8 trades, mean/std known. Cross-check Sharpe & PSR vs
    the explicit Bailey-López de Prado formula using scipy.stats."""
    pnls = [0.01, -0.005, 0.012, -0.003, 0.008, 0.004, -0.002, 0.006]
    fills = [_fill(p, entry_min=i * 20, exit_min=(i + 1) * 20) for i, p in enumerate(pnls)]
    tm = compute_trading_metrics(fills, m_minutes=20)

    arr = np.array(pnls)
    mean_p = arr.mean()
    std_p = arr.std(ddof=1)
    sr_trade = mean_p / max(std_p, 1e-12)

    bars_per_year = 365.0 * 24.0 * 60.0 / 20.0
    avg_bars = 20.0  # each trade is exactly 20 minutes
    trades_per_year = bars_per_year / avg_bars
    expected_sharpe = sr_trade * math.sqrt(trades_per_year)

    assert tm.sharpe == pytest.approx(expected_sharpe, rel=1e-9)

    skew = float(scipy_stats.skew(arr))
    kurt = float(scipy_stats.kurtosis(arr, fisher=False))
    denom = max(1.0 - skew * sr_trade + (kurt - 1.0) / 4.0 * sr_trade ** 2, 1e-12)
    expected_psr = float(
        scipy_stats.norm.cdf(sr_trade * math.sqrt(len(arr) - 1) / math.sqrt(denom))
    )
    assert tm.probabilistic_sharpe == pytest.approx(expected_psr, rel=1e-9)
    assert 0.0 <= tm.probabilistic_sharpe <= 1.0


def test_identical_pnls_have_zero_dispersion_but_finite_sharpe() -> None:
    """std=0 path: Sharpe uses 1e-12 epsilon → very large but finite."""
    fills = [_fill(0.01) for _ in range(5)]
    tm = compute_trading_metrics(fills)
    assert tm.hit_rate == 1.0
    assert math.isfinite(tm.sharpe)
    # All wins → no loss array → profit_factor returns +inf.
    assert tm.profit_factor == float("inf")


# ---------- all-positive / all-negative ------------------------------------


def test_all_positive_pnl() -> None:
    fills = [_fill(0.01), _fill(0.02), _fill(0.005), _fill(0.015), _fill(0.008)]
    tm = compute_trading_metrics(fills)
    assert tm.hit_rate == 1.0
    assert tm.avg_loss == 0.0
    assert tm.avg_win > 0.0
    assert tm.profit_factor == float("inf")
    assert tm.max_drawdown_log == pytest.approx(0.0, abs=1e-12)
    assert tm.total_log_return == pytest.approx(sum([0.01, 0.02, 0.005, 0.015, 0.008]))
    assert tm.total_pct_return == pytest.approx(math.exp(tm.total_log_return) - 1.0)


def test_all_negative_pnl() -> None:
    pnls = [-0.01, -0.02, -0.005, -0.015, -0.008]
    fills = [_fill(p) for p in pnls]
    tm = compute_trading_metrics(fills)
    assert tm.hit_rate == 0.0
    assert tm.avg_win == 0.0
    assert tm.avg_loss < 0.0
    # losses non-empty → finite profit_factor (= 0 since wins.sum() == 0)
    assert tm.profit_factor == 0.0
    assert tm.max_drawdown_log > 0.0
    # eq is monotonically decreasing → cummax = eq[0], so max_dd = -(eq[-1]-eq[0])
    eq = np.cumsum(pnls)
    expected_max_dd = float(-(eq.min() - eq[0]))
    assert tm.max_drawdown_log == pytest.approx(expected_max_dd)


# ---------- counters / drawdown --------------------------------------------


@pytest.mark.parametrize(
    "reason,attr",
    [(ExitReason.TP, "n_tp"), (ExitReason.SL, "n_sl"), (ExitReason.TIMEOUT, "n_timeout")],
)
def test_reason_counters(reason: ExitReason, attr: str) -> None:
    fills = [_fill(0.001, reason=reason) for _ in range(4)]
    tm = compute_trading_metrics(fills)
    assert getattr(tm, attr) == 4


def test_drawdown_basic() -> None:
    """eq = cumsum, drawdown = eq - cummax → known max DD."""
    pnls = [0.10, -0.05, -0.03, 0.02, -0.10, 0.05]
    fills = [_fill(p) for p in pnls]
    tm = compute_trading_metrics(fills)
    eq = np.cumsum(pnls)
    cummax = np.maximum.accumulate(eq)
    expected_dd = float(-(eq - cummax).min())
    assert tm.max_drawdown_log == pytest.approx(expected_dd)


def test_no_entry_or_exit_ts_uses_default_bar_age() -> None:
    """Fills without entry/exit ts should fall back to m_minutes for bar age."""
    f = BarrierTouched(
        ts_init=Timestamp(0),
        order_id=OrderId(0),
        entry_ts=None,
        exit_ts=None,
        entry_price=Price(100.0),
        exit_price=Price(100.0),
        fill_size=Quantity(1.0),
        side=1,
        reason=ExitReason.TP,
        pnl_log_net=LogReturn(0.01),
    )
    tm = compute_trading_metrics([f, f], m_minutes=20)
    assert tm.n_trades == 2
    assert math.isfinite(tm.sharpe)


# ---------- deflated_sharpe ------------------------------------------------


def test_deflated_sharpe_returns_pair_and_finite() -> None:
    p, sr_star = deflated_sharpe(
        observed_sharpe=2.0,
        var_trial_sharpes=0.25,
        n_trials=20,
        n_obs=200,
        skew=0.0,
        kurt=3.0,
    )
    assert 0.0 <= p <= 1.0
    assert math.isfinite(sr_star) and sr_star > 0.0


def test_deflated_sharpe_higher_observed_yields_higher_p() -> None:
    p_lo, _ = deflated_sharpe(0.5, 0.1, 10, 100)
    p_hi, _ = deflated_sharpe(2.5, 0.1, 10, 100)
    assert p_hi > p_lo


def test_deflated_sharpe_zero_variance_gives_zero_sr_star() -> None:
    _, sr_star = deflated_sharpe(1.0, 0.0, 10, 100)
    assert sr_star == 0.0


# ---------- TradingMetrics.to_dict round trip ------------------------------


def test_trading_metrics_to_dict_keys() -> None:
    tm = compute_trading_metrics([_fill(0.01), _fill(-0.01)])
    d = tm.to_dict()
    expected = {
        "n_trades", "n_tp", "n_sl", "n_timeout", "hit_rate",
        "avg_win", "avg_loss", "profit_factor",
        "total_log_return", "total_pct_return",
        "sharpe", "probabilistic_sharpe", "sortino",
        "max_drawdown_log", "cdar_5pct_log",
        # CI block (round-040 additive port from src/bootstrap.py).
        "sharpe_ci_lo", "sharpe_ci_hi",
        "sortino_ci_lo", "sortino_ci_hi",
        "max_dd_ci_lo", "max_dd_ci_hi",
        "block_length", "n_bootstrap",
    }
    assert set(d) == expected
    # When compute_ci=False (default) the CI fields are None.
    for k in ("sharpe_ci_lo", "sharpe_ci_hi", "sortino_ci_lo",
              "sortino_ci_hi", "max_dd_ci_lo", "max_dd_ci_hi",
              "block_length", "n_bootstrap"):
        assert d[k] is None
