"""Contract tests for `wagie.charts.ChartBattery` and the theme module.

ChartBattery is the SINGLE chart battery: render against any EngineResult,
get back a documented key-set of PNG paths. These tests pin down:

  - the rendered key set always contains equity_curve / drawdown / pnl_distribution
  - reliability + p_histogram only appear when y/p are passed (and non-empty)
  - coverage_bars only when coverage_per_alpha is non-empty
  - quantile_drift only when q_history is non-empty
  - every produced PNG is non-empty (>1KB)
  - empty fills do not crash
  - apply_theme() actually mutates plt.rcParams
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

# Force a non-interactive backend for any direct pyplot use in this file.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from wagie.charts import ChartBattery
from wagie.charts.theme import PALETTE, apply_theme, figsize
from wagie.core.event import BarrierTouched
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import LogReturn
from wagie.core.portfolio import Portfolio
from wagie.core.time import Timestamp
from wagie.engine import EngineResult
from wagie.io.brokers import BrokerLedger


# -----------------------------------------------------------------
# Stub builders
# -----------------------------------------------------------------

def _make_fill(pnl_log_net: float) -> BarrierTouched:
    """Build a minimal BarrierTouched event with the specified net log-return."""
    return BarrierTouched(
        ts_init=Timestamp(0),
        instrument=DEFAULT_INSTRUMENT,
        pnl_log_net=LogReturn(float(pnl_log_net)),
    )


def _engine_result(pnls: Iterable[float]) -> EngineResult:
    fills = [_make_fill(p) for p in pnls]
    ledger = BrokerLedger(fills=fills, n_open_at_finalize=0, config={})
    return EngineResult(
        ledger=ledger,
        n_decisions=len(fills),
        n_filled=len(fills),
        n_skipped_warmup=0,
        n_actions_approved=len(fills),
        n_actions_rejected=0,
        pipeline_state_hash=b"\x00" * 32,
        final_portfolio=Portfolio(),
    )


# -----------------------------------------------------------------
# render_all key-set contract
# -----------------------------------------------------------------

def test_render_all_minimum_keys(tmp_path: Path) -> None:
    """Without optional inputs only the trading triplet appears."""
    res = _engine_result([0.001, -0.002, 0.0015, -0.0005, 0.003])
    out = ChartBattery().render_all(res, tmp_path / "charts")
    assert set(out.keys()) == {"equity_curve", "drawdown", "pnl_distribution"}
    for p in out.values():
        assert p.is_file()
        assert p.stat().st_size > 1024  # > 1KB


def test_render_all_adds_calibration_when_y_p_given(tmp_path: Path) -> None:
    res = _engine_result([0.001, -0.001, 0.002])
    out = ChartBattery().render_all(
        res,
        tmp_path / "charts",
        y_true=[0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        p_pred=[0.1, 0.8, 0.7, 0.2, 0.6, 0.3, 0.4, 0.9, 0.55, 0.15],
    )
    assert "reliability" in out
    assert "p_histogram" in out
    assert out["reliability"].stat().st_size > 1024
    assert out["p_histogram"].stat().st_size > 1024


def test_render_all_skips_calibration_when_y_p_empty(tmp_path: Path) -> None:
    """Empty arrays are equivalent to omitting them — no calibration charts."""
    res = _engine_result([0.001, -0.001])
    out = ChartBattery().render_all(
        res, tmp_path / "charts", y_true=[], p_pred=[],
    )
    assert "reliability" not in out
    assert "p_histogram" not in out


def test_render_all_adds_coverage_bars(tmp_path: Path) -> None:
    res = _engine_result([0.001])
    coverage = [
        {"alpha": 0.10, "empirical": 0.91, "target": 0.90, "gap": 0.01, "n": 100},
        {"alpha": 0.20, "empirical": 0.78, "target": 0.80, "gap": -0.02, "n": 100},
    ]
    out = ChartBattery().render_all(
        res, tmp_path / "charts", coverage_per_alpha=coverage,
    )
    assert "coverage_bars" in out
    assert out["coverage_bars"].stat().st_size > 1024


def test_render_all_skips_coverage_when_empty_or_none(tmp_path: Path) -> None:
    res = _engine_result([0.001])
    o1 = ChartBattery().render_all(res, tmp_path / "c1", coverage_per_alpha=None)
    o2 = ChartBattery().render_all(res, tmp_path / "c2", coverage_per_alpha=[])
    assert "coverage_bars" not in o1
    assert "coverage_bars" not in o2


def test_render_all_adds_quantile_drift(tmp_path: Path) -> None:
    res = _engine_result([0.001])
    out = ChartBattery().render_all(
        res, tmp_path / "charts", q_history=[0.1, 0.12, 0.11, 0.13, 0.14, 0.12],
    )
    assert "quantile_drift" in out
    assert out["quantile_drift"].stat().st_size > 1024


def test_render_all_skips_quantile_drift_when_empty(tmp_path: Path) -> None:
    res = _engine_result([0.001])
    o1 = ChartBattery().render_all(res, tmp_path / "q1", q_history=None)
    o2 = ChartBattery().render_all(res, tmp_path / "q2", q_history=[])
    assert "quantile_drift" not in o1
    assert "quantile_drift" not in o2


def test_render_all_full_optional_set(tmp_path: Path) -> None:
    """All seven documented chart keys when every optional input is provided."""
    res = _engine_result([0.001, -0.002, 0.0015])
    out = ChartBattery().render_all(
        res, tmp_path / "charts",
        y_true=[0, 1, 1, 0, 1],
        p_pred=[0.2, 0.8, 0.7, 0.3, 0.6],
        coverage_per_alpha=[
            {"alpha": 0.10, "empirical": 0.9, "target": 0.9, "gap": 0.0, "n": 50},
        ],
        q_history=[0.1, 0.12, 0.13, 0.11, 0.14],
    )
    assert set(out.keys()) == {
        "equity_curve", "drawdown", "pnl_distribution",
        "reliability", "p_histogram",
        "coverage_bars", "quantile_drift",
    }


# -----------------------------------------------------------------
# Empty fills + edge cases
# -----------------------------------------------------------------

def test_render_all_empty_fills_does_not_crash(tmp_path: Path) -> None:
    res = _engine_result([])
    out = ChartBattery().render_all(res, tmp_path / "charts")
    # Three placeholder PNGs still produced
    assert set(out.keys()) == {"equity_curve", "drawdown", "pnl_distribution"}
    for p in out.values():
        assert p.is_file()
        assert p.stat().st_size > 1024


def test_render_all_creates_out_dir(tmp_path: Path) -> None:
    res = _engine_result([0.001])
    out_dir = tmp_path / "deep" / "nested" / "charts"
    ChartBattery().render_all(res, out_dir)
    assert out_dir.is_dir()


def test_n_calibration_bins_threading(tmp_path: Path) -> None:
    """The configured bin count flows through to the reliability diagram."""
    res = _engine_result([0.001])
    out = ChartBattery(n_calibration_bins=5).render_all(
        res, tmp_path / "charts",
        y_true=[0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        p_pred=[0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.15, 0.85],
    )
    assert out["reliability"].is_file()


# -----------------------------------------------------------------
# Theme module
# -----------------------------------------------------------------

def test_apply_theme_mutates_rc_params() -> None:
    # Reset to defaults, snapshot, apply theme, verify mutation
    plt.rcdefaults()
    before = dict(plt.rcParams)
    apply_theme(plt)
    after = dict(plt.rcParams)
    assert before != after, "apply_theme should change at least one rcParam"
    # Spot-check several values the theme is documented to set
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["legend.frameon"] is False
    assert plt.rcParams["figure.facecolor"] == PALETTE["background"]


def test_palette_keys_present() -> None:
    for k in ("primary", "secondary", "accent", "warning",
              "muted", "background", "grid"):
        assert k in PALETTE
        v = PALETTE[k]
        assert isinstance(v, str)
        assert v.startswith("#")


def test_figsize_returns_known_kinds() -> None:
    assert figsize("single") == (6.0, 4.0)
    assert figsize("wide") == (9.0, 4.0)
    assert figsize("tall") == (6.0, 6.0)
    assert figsize("panel") == (10.0, 7.0)
    # Unknown kind falls back to single (documented default)
    assert figsize("unknown") == (6.0, 4.0)
