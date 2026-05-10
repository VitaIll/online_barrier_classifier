"""Contract tests for MetricsReport additive extensions.

Pinned behavior:
  - MetricsReport defaults the four new optional dicts to None.
  - to_dict() emits the new keys (None passes through as JSON null).
  - MetricsBattery.compute() populates controller / drift / inventory /
    warmup_calibration when the EngineResult carries the matching arrays.
  - When no per-bar arrays are present, the new fields stay None — no
    spurious empty dicts that pollute the JSON.
  - The pause_spans encoder collapses contiguous True runs.
  - The rolling-Brier helper produces an array aligned with the input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from wagie.core.portfolio import Portfolio
from wagie.engine import EngineResult
from wagie.io.brokers import BrokerLedger
from wagie.metrics import MetricsBattery, MetricsReport


def _empty_result(**kw) -> EngineResult:
    return EngineResult(
        ledger=BrokerLedger(fills=[], n_open_at_finalize=0, config={}),
        n_decisions=0, n_filled=0, n_skipped_warmup=0,
        n_actions_approved=0, n_actions_rejected=0,
        pipeline_state_hash=b"\x00" * 32,
        final_portfolio=Portfolio(),
        **kw,
    )


# ---------------- defaults ---------------------------------------------------


def test_metrics_report_new_fields_default_to_none() -> None:
    rep = MetricsReport()
    assert rep.controller is None
    assert rep.drift is None
    assert rep.inventory is None
    assert rep.warmup_calibration is None


def test_to_dict_includes_new_keys() -> None:
    d = MetricsReport().to_dict()
    for key in ("controller", "drift", "inventory", "warmup_calibration"):
        assert key in d, f"to_dict missing extension key {key!r}"
        assert d[key] is None  # default


def test_to_json_round_trips_new_keys_as_null() -> None:
    s = MetricsReport().to_json()
    parsed = json.loads(s)
    for key in ("controller", "drift", "inventory", "warmup_calibration"):
        assert key in parsed
        assert parsed[key] is None


# ---------------- compute populates blocks when arrays present --------------


def test_compute_populates_controller_block_from_history() -> None:
    """When the engine carries the per-bar history arrays, ``controller``
    contains tau_traj, r_hat_traj, r_star_traj, pause_spans, sigma_ve_dist,
    sigma_max."""
    res = _empty_result(
        tau_history=[0.5, 0.51, 0.52, 0.53],
        r_hat_ewma_history=[0.1, 0.11, 0.12, 0.13],
        r_star_ewma_history=[0.2, 0.2, 0.2, 0.2],
        paused_history=[False, True, True, False],
        sigma_ve_history=[0.30, 0.32, 0.40, 0.31],
    )
    rep = MetricsBattery().compute(res)
    assert rep.controller is not None
    c = rep.controller
    assert c["tau_traj"] == [0.5, 0.51, 0.52, 0.53]
    assert c["r_hat_traj"] == [0.1, 0.11, 0.12, 0.13]
    assert c["r_star_traj"] == [0.2, 0.2, 0.2, 0.2]
    assert c["sigma_ve_dist"] == [0.30, 0.32, 0.40, 0.31]
    assert c["sigma_max"] == pytest.approx(0.40)
    # Contiguous True run from index 1 (inclusive) to 3 (exclusive).
    assert c["pause_spans"] == [(1, 3)]


def test_compute_controller_pause_spans_handles_multiple_runs() -> None:
    res = _empty_result(
        paused_history=[True, False, True, True, False, True],
    )
    rep = MetricsBattery().compute(res)
    assert rep.controller is not None
    assert rep.controller["pause_spans"] == [(0, 1), (2, 4), (5, 6)]


def test_compute_controller_none_when_no_history() -> None:
    """All controller arrays empty + no engine state => None."""
    res = _empty_result()
    rep = MetricsBattery().compute(res)
    assert rep.controller is None


def test_compute_inventory_block_from_history() -> None:
    res = _empty_result(
        inventory_size_history=[0, 1, 1, 2, 2],
        hold_age_max_history=[0, 1, 2, 3, 4],
    )
    rep = MetricsBattery().compute(res)
    assert rep.inventory is not None
    assert rep.inventory["size_traj"] == [0, 1, 1, 2, 2]
    assert rep.inventory["hold_age_max_traj"] == [0, 1, 2, 3, 4]


def test_compute_inventory_none_when_arrays_empty() -> None:
    res = _empty_result()
    rep = MetricsBattery().compute(res)
    assert rep.inventory is None


def test_compute_warmup_calibration_pass_through() -> None:
    payload = {"isotonic": [0.1, 0.5, 0.9], "n_warmup": 250}
    res = _empty_result(warmup_calibration=payload)
    rep = MetricsBattery().compute(res)
    assert rep.warmup_calibration == payload
    # Defensive copy — mutating the report's dict must not affect the result.
    rep.warmup_calibration["n_warmup"] = 999
    assert res.warmup_calibration["n_warmup"] == 250


def test_compute_warmup_calibration_none_when_absent() -> None:
    res = _empty_result()
    rep = MetricsBattery().compute(res)
    assert rep.warmup_calibration is None


# ---------------- drift block: rolling brier --------------------------------


def test_compute_drift_block_rolling_brier_alignment() -> None:
    """rolling_window defaults to 1000; with fewer samples than the window,
    the rolling array is filled with NaN (alignment guarantee)."""
    rng = np.random.default_rng(0)
    n = 1500
    p = rng.uniform(0.05, 0.95, n).tolist()
    y = [int(rng.uniform() < pi) for pi in p]
    res = _empty_result(
        p_online_history=p,
        label_history=y,
    )
    rep = MetricsBattery(compute_ci=False).compute(res)
    assert rep.drift is not None
    d = rep.drift
    assert d["rolling_window"] == 1000
    # Length of the rolling array equals the matured-trace length.
    assert len(d["brier_online_rolling"]) == n
    # First 999 positions are NaN; tail is finite.
    assert all(math.isnan(x) for x in d["brier_online_rolling"][:999])
    assert all(math.isfinite(x) for x in d["brier_online_rolling"][999:])
    # baseline_brier matches a single-shot Brier on the same data.
    expected = float(np.mean((np.asarray(p) - np.asarray(y, dtype=float)) ** 2))
    assert d["baseline_brier"] == pytest.approx(expected, rel=1e-6)
    # No offline trace was passed — array stays empty.
    assert d["brier_offline_rolling"] == []


def test_compute_drift_none_when_no_calibration_history() -> None:
    res = _empty_result()
    rep = MetricsBattery(compute_ci=False).compute(res)
    assert rep.drift is None


def test_compute_drift_short_trace_yields_all_nan_rolling() -> None:
    """When the matured trace is shorter than the window, the rolling array
    is all NaN (still aligned to bar index)."""
    res = _empty_result(
        p_online_history=[0.5] * 10,
        label_history=[1] * 10,
    )
    rep = MetricsBattery(compute_ci=False).compute(res)
    assert rep.drift is not None
    assert len(rep.drift["brier_online_rolling"]) == 10
    assert all(math.isnan(x) for x in rep.drift["brier_online_rolling"])
    # baseline_brier is still computable.
    assert rep.drift["baseline_brier"] == pytest.approx(0.25)


# ---------------- key set check on extended to_dict -------------------------


def test_to_dict_key_set_contains_new_extensions() -> None:
    rep = MetricsReport()
    d = rep.to_dict()
    expected_extra = {"controller", "drift", "inventory", "warmup_calibration"}
    assert expected_extra.issubset(set(d.keys()))


def test_compute_populates_all_blocks_together() -> None:
    """Smoke: a fully-populated EngineResult lights up every new block."""
    rng = np.random.default_rng(7)
    n = 1200
    p = rng.uniform(0.05, 0.95, n).tolist()
    y = [int(rng.uniform() < pi) for pi in p]
    res = _empty_result(
        p_online_history=p, label_history=y,
        tau_history=[0.5] * 4,
        r_hat_ewma_history=[0.1] * 4,
        r_star_ewma_history=[0.2] * 4,
        paused_history=[False, True, False, False],
        sigma_ve_history=[0.3, 0.4, 0.5, 0.35],
        inventory_size_history=[0, 1, 1, 0],
        hold_age_max_history=[0, 1, 2, 0],
        warmup_calibration={"snapshot": True},
    )
    rep = MetricsBattery(compute_ci=False).compute(res)
    assert rep.controller is not None
    assert rep.drift is not None
    assert rep.inventory is not None
    assert rep.warmup_calibration == {"snapshot": True}
