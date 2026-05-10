"""Contract tests for ``wagie.metrics.MetricsBattery`` + ``MetricsReport``.

Covers:
  - compute() populates trading from result.ledger.fills always.
  - Calibration metrics computed only when y_true + p_pred provided.
  - Defaults (Brier=0, ECE=0, ROC=0.5, PR=0) when optional inputs missing.
  - Empty-fill EngineResult yields defaults across the report.
  - The engine's streaming p_online_history / label_history feed compute()
    automatically — assert non-zero Brier on a fixture with that history.
  - to_dict() and to_json() round-trip cleanly.
  - Per-regime stratification when regime_ids supplied (or routed via engine).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from wagie.core.action import ExitReason
from wagie.core.event import BarrierTouched
from wagie.core.identity import OrderId
from wagie.core.numeric import LogReturn, Price, Quantity
from wagie.core.portfolio import Portfolio
from wagie.core.time import Timestamp
from wagie.engine import EngineResult
from wagie.io.brokers import BrokerLedger
from wagie.metrics import MetricsBattery, MetricsReport


# ---------- helpers --------------------------------------------------------

_NS_PER_MIN = 60_000_000_000


def _fill(pnl: float, *, reason: ExitReason = ExitReason.TP, order_id: int = 0) -> BarrierTouched:
    return BarrierTouched(
        ts_init=Timestamp(20 * _NS_PER_MIN),
        order_id=OrderId(order_id),
        entry_ts=Timestamp(0),
        exit_ts=Timestamp(20 * _NS_PER_MIN),
        entry_price=Price(100.0),
        exit_price=Price(100.0 * math.exp(pnl)),
        fill_size=Quantity(1.0),
        side=1,
        reason=reason,
        pnl_log_gross=LogReturn(pnl),
        pnl_log_net=LogReturn(pnl),
    )


def _make_result(
    fills: list[BarrierTouched],
    n_decisions: int = 10,
    *,
    p_online_history: list[float] | None = None,
    label_history: list[int] | None = None,
    regime_history: list[int] | None = None,
) -> EngineResult:
    """Minimal EngineResult stub. We only need ledger.fills and a few counters."""
    ledger = BrokerLedger(fills=fills, n_open_at_finalize=0, config={})
    return EngineResult(
        ledger=ledger,
        n_decisions=n_decisions,
        n_filled=len(fills),
        n_skipped_warmup=0,
        n_actions_approved=len(fills),
        n_actions_rejected=0,
        pipeline_state_hash=b"\x00" * 32,
        final_portfolio=Portfolio(),
        p_online_history=p_online_history or [],
        label_history=label_history or [],
        regime_history=regime_history or [],
    )


# ---------- defaults / empty -----------------------------------------------


def test_metrics_report_defaults() -> None:
    r = MetricsReport()
    assert r.brier == 0.0
    assert r.ece == 0.0
    assert r.reliability == []
    assert r.calibration_by_regime == []
    assert r.trading == {}
    assert r.roc_auc == 0.5
    assert r.pr_auc == 0.0
    assert r.n_decisions == 0
    assert r.pipeline_state_hash == ""


def test_battery_with_empty_fills_yields_defaults_for_optional() -> None:
    res = _make_result(fills=[], n_decisions=0)
    rep = MetricsBattery().compute(res)
    # trading dict populated even if empty (n_trades=0 path).
    assert rep.trading["n_trades"] == 0
    assert rep.trading["sharpe"] == 0.0
    assert rep.trading["probabilistic_sharpe"] == 0.5
    # No labels supplied → calibration defaults preserved.
    assert rep.brier == 0.0
    assert rep.ece == 0.0
    assert rep.reliability == []
    assert rep.roc_auc == 0.5
    assert rep.pr_auc == 0.0


# ---------- trading populated from fills -----------------------------------


def test_trading_populated_from_fills() -> None:
    fills = [_fill(0.01, order_id=i) for i in range(5)] + [
        _fill(-0.005, order_id=99, reason=ExitReason.SL),
    ]
    res = _make_result(fills, n_decisions=20)
    rep = MetricsBattery().compute(res)
    assert rep.trading["n_trades"] == 6
    assert rep.trading["n_tp"] == 5
    assert rep.trading["n_sl"] == 1
    assert rep.n_decisions == 20
    assert rep.n_filled == 6


# ---------- calibration only when labels provided --------------------------


def test_calibration_skipped_when_only_y_true() -> None:
    res = _make_result(fills=[_fill(0.01)])
    rep = MetricsBattery().compute(res, y_true=[1, 0, 1])
    # p_pred missing → calibration block skipped.
    assert rep.brier == 0.0
    assert rep.ece == 0.0
    assert rep.reliability == []


def test_calibration_skipped_when_only_p_pred() -> None:
    res = _make_result(fills=[_fill(0.01)])
    rep = MetricsBattery().compute(res, p_pred=[0.5, 0.5])
    assert rep.brier == 0.0


def test_calibration_skipped_when_y_true_empty() -> None:
    res = _make_result(fills=[_fill(0.01)])
    rep = MetricsBattery().compute(res, y_true=[], p_pred=[])
    assert rep.brier == 0.0
    assert rep.ece == 0.0


def test_calibration_populated_when_both_provided() -> None:
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int).tolist()
    res = _make_result(fills=[])
    rep = MetricsBattery(n_calibration_bins=10).compute(
        res, y_true=y, p_pred=p.tolist(),
    )
    assert 0.0 <= rep.brier <= 1.0
    assert 0.0 <= rep.ece <= 1.0
    assert len(rep.reliability) > 0
    # Each reliability entry is a dict with the right keys.
    for b in rep.reliability:
        assert set(b.keys()) == {"p_mean", "y_mean", "n"}
    # Diagnostic ranking metrics also populated when labels present.
    assert 0.0 <= rep.roc_auc <= 1.0
    assert 0.0 <= rep.pr_auc <= 1.0


# ---------- engine streaming history routes through automatically ----------


def test_calibration_routes_engine_history_when_no_explicit_inputs() -> None:
    """The engine populates p_online_history / label_history; MetricsBattery
    should pick them up without the caller explicitly threading them."""
    rng = np.random.default_rng(1)
    n = 200
    p = rng.uniform(0.05, 0.95, n).tolist()
    y = [int(rng.uniform() < pi) for pi in p]
    res = _make_result(
        fills=[],
        p_online_history=p,
        label_history=y,
    )
    rep = MetricsBattery().compute(res)
    assert rep.brier > 0.0, "engine history should drive non-zero Brier"
    assert rep.ece >= 0.0


def test_calibration_per_regime_stratification() -> None:
    """When regime_ids align with y_true / p_pred, calibration_by_regime
    contains one entry per regime with sufficient samples."""
    rng = np.random.default_rng(2)
    n = 300
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int).tolist()
    regimes = (rng.integers(0, 3, n)).tolist()
    res = _make_result(fills=[])
    rep = MetricsBattery().compute(
        res, y_true=y, p_pred=p.tolist(), regime_ids=regimes,
    )
    assert len(rep.calibration_by_regime) >= 2
    rids = {entry["regime_id"] for entry in rep.calibration_by_regime}
    assert rids.issubset({0, 1, 2})
    for entry in rep.calibration_by_regime:
        assert entry["n"] >= 2
        assert 0.0 <= entry["brier"] <= 1.0
        assert 0.0 <= entry["ece"] <= 1.0


def test_calibration_per_regime_via_engine_history() -> None:
    """regime_history on the result also feeds per-regime stratification."""
    rng = np.random.default_rng(3)
    n = 200
    p = rng.uniform(0.05, 0.95, n).tolist()
    y = [int(rng.uniform() < pi) for pi in p]
    regimes = rng.integers(0, 2, n).tolist()
    res = _make_result(
        fills=[],
        p_online_history=p,
        label_history=y,
        regime_history=regimes,
    )
    rep = MetricsBattery().compute(res)
    assert len(rep.calibration_by_regime) >= 1


# ---------- provenance / hash ----------------------------------------------


def test_pipeline_state_hash_truncated_to_16_hex() -> None:
    """compute() stores the first 16 hex chars of the hash."""
    res = _make_result(fills=[])
    res.pipeline_state_hash = b"\xab\xcd\xef\x01" * 8  # 32 bytes
    rep = MetricsBattery().compute(res)
    assert rep.pipeline_state_hash == "abcdef01abcdef01"
    assert len(rep.pipeline_state_hash) == 16


def test_n_actions_counters_propagated() -> None:
    res = _make_result(fills=[_fill(0.01)])
    res.n_actions_approved = 7
    res.n_actions_rejected = 3
    rep = MetricsBattery().compute(res)
    assert rep.n_actions_approved == 7
    assert rep.n_actions_rejected == 3


# ---------- to_dict / to_json round-trip -----------------------------------


def test_to_dict_contains_all_top_level_keys() -> None:
    rep = MetricsReport()
    d = rep.to_dict()
    expected_keys = {
        "brier", "ece", "reliability", "calibration_by_regime", "trading",
        "roc_auc", "pr_auc",
        "n_decisions", "n_filled", "n_actions_approved", "n_actions_rejected",
        "pipeline_state_hash",
        # CI / pre-registration block (round-040 additive port).
        "sharpe_ci", "brier_ci", "ece_ci", "roc_ci_delong", "pr_ci",
        "n_trials", "dsr", "pbo",
        "accepted", "blocked_reasons",
        # Adaptive-threshold strategy extension (additive — round-NEW).
        "controller", "drift", "inventory", "warmup_calibration",
    }
    assert set(d.keys()) == expected_keys


def test_to_json_round_trip() -> None:
    res = _make_result(fills=[_fill(0.01), _fill(-0.005)])
    rep = MetricsBattery().compute(
        res,
        y_true=[1, 0, 1, 0],
        p_pred=[0.7, 0.3, 0.8, 0.2],
    )
    s = rep.to_json()
    parsed = json.loads(s)
    assert parsed["trading"]["n_trades"] == 2
    assert parsed["brier"] == pytest.approx(rep.brier)
    assert parsed["ece"] == pytest.approx(rep.ece)
    # Ensure indent kwarg is honored (default = 2 → multiline).
    assert "\n" in s


def test_to_json_indent_zero_compact() -> None:
    rep = MetricsReport()
    s = rep.to_json(indent=0)
    parsed = json.loads(s)
    assert parsed["brier"] == 0.0
