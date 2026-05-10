"""Contract tests for the additive RebuildBundle extensions.

The bundle is what the renderer pickles under ``state/rebuild_bundle.pkl``
to allow per-section rebuilds without re-running the engine. Per the new
schema, the bundle must:
  - default the new history arrays to empty + warmup_calibration to None
  - round-trip new arrays through pickle/unpickle
  - capture them when an EngineResult-shaped object is passed to
    ``_build_bundle``
  - surface them on ``_ReplayEngineResult.from_bundle`` so emitters can
    consume them during ``rebuild()``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from wagie.reporting.manifest import RunMeta
from wagie.reporting.renderer import (
    REBUILD_BUNDLE_NAME,
    RebuildBundle,
    ReportRenderer,
    _build_bundle,
    _ReplayEngineResult,
)


@dataclass
class _FakeFill:
    pnl_log_net: float = 0.0


@dataclass
class _FakeLedger:
    fills: list = field(default_factory=list)


@dataclass
class _FakeEngineWithState:
    p_online_history: list = field(default_factory=list)
    label_history: list = field(default_factory=list)
    regime_history: list = field(default_factory=list)
    ledger: _FakeLedger = field(default_factory=_FakeLedger)
    n_decisions: int = 0
    n_filled: int = 0
    n_skipped_warmup: int = 0
    n_actions_approved: int = 0
    n_actions_rejected: int = 0
    pipeline_state_hash: bytes = b"\xde\xad\xbe\xef" * 8
    tau_history: list = field(default_factory=list)
    r_hat_ewma_history: list = field(default_factory=list)
    r_star_ewma_history: list = field(default_factory=list)
    paused_history: list = field(default_factory=list)
    sigma_ve_history: list = field(default_factory=list)
    inventory_size_history: list = field(default_factory=list)
    hold_age_max_history: list = field(default_factory=list)
    warmup_calibration: Optional[dict] = None


def _run_meta() -> RunMeta:
    return RunMeta(
        run_id="test-rid", spec_name="bundle-ext", spec_hash="cafebabe",
        mode="backtest", accepted=True,
    )


# --------------------------- defaults ---------------------------------------


def test_rebuild_bundle_new_fields_default() -> None:
    """RebuildBundle constructed with only the required fields defaults the
    new arrays to empty lists / None."""
    b = RebuildBundle(metrics={}, spec_dict={}, run_meta={})
    assert b.tau_history == []
    assert b.r_hat_ewma_history == []
    assert b.r_star_ewma_history == []
    assert b.paused_history == []
    assert b.sigma_ve_history == []
    assert b.inventory_size_history == []
    assert b.hold_age_max_history == []
    assert b.warmup_calibration is None


# --------------------------- pickle round-trip ------------------------------


def test_rebuild_bundle_pickle_round_trip_preserves_new_arrays(tmp_path: Path) -> None:
    """Pickle the bundle to disk, reload, assert every new array survives
    intact AND is equivalent to the input (defensive equality)."""
    b = RebuildBundle(
        metrics={"brier": 0.21}, spec_dict={"name": "x"}, run_meta={},
        tau_history=[0.5, 0.51, 0.52],
        r_hat_ewma_history=[0.10, 0.11, 0.12],
        r_star_ewma_history=[0.20, 0.20, 0.20],
        paused_history=[False, True, False],
        sigma_ve_history=[0.3, 0.31, 0.32],
        inventory_size_history=[0, 1, 1],
        hold_age_max_history=[0, 1, 2],
        warmup_calibration={"isotonic": [0.1, 0.5, 0.9], "n_warmup": 100},
    )
    fp = tmp_path / "bundle.pkl"
    with fp.open("wb") as fh:
        pickle.dump(b, fh, protocol=4)
    with fp.open("rb") as fh:
        b2 = pickle.load(fh)

    assert b2.tau_history == b.tau_history
    assert b2.r_hat_ewma_history == b.r_hat_ewma_history
    assert b2.r_star_ewma_history == b.r_star_ewma_history
    assert b2.paused_history == b.paused_history
    assert b2.sigma_ve_history == b.sigma_ve_history
    assert b2.inventory_size_history == b.inventory_size_history
    assert b2.hold_age_max_history == b.hold_age_max_history
    assert b2.warmup_calibration == b.warmup_calibration


# --------------------------- _build_bundle captures from engine ------------


def test_build_bundle_captures_strategy_state_from_engine_result() -> None:
    eng = _FakeEngineWithState(
        p_online_history=[0.4, 0.5, 0.6],
        label_history=[0, 1, 1],
        regime_history=[0, 0, 1],
        ledger=_FakeLedger(fills=[_FakeFill(pnl_log_net=0.001)]),
        tau_history=[0.50, 0.55, 0.60],
        r_hat_ewma_history=[0.10, 0.11, 0.12],
        r_star_ewma_history=[0.20, 0.20, 0.20],
        paused_history=[False, False, True],
        sigma_ve_history=[0.30, 0.31, 0.32],
        inventory_size_history=[0, 1, 0],
        hold_age_max_history=[0, 1, 2],
        warmup_calibration={"n_warmup": 50},
    )
    b = _build_bundle(
        metrics={}, spec_dict={}, run_meta=_run_meta(),
        use_plotly=False, engine_result=eng,
    )
    assert b.tau_history == [0.50, 0.55, 0.60]
    assert b.r_hat_ewma_history == [0.10, 0.11, 0.12]
    assert b.r_star_ewma_history == [0.20, 0.20, 0.20]
    assert b.paused_history == [False, False, True]
    assert b.sigma_ve_history == [0.30, 0.31, 0.32]
    assert b.inventory_size_history == [0, 1, 0]
    assert b.hold_age_max_history == [0, 1, 2]
    assert b.warmup_calibration == {"n_warmup": 50}


def test_build_bundle_tolerates_engine_without_new_attrs() -> None:
    """Old EngineResult-shaped objects without the new attributes still
    yield a valid (empty) bundle for the new fields."""

    @dataclass
    class _LegacyEngine:
        p_online_history: list = field(default_factory=list)
        label_history: list = field(default_factory=list)
        regime_history: list = field(default_factory=list)
        ledger: _FakeLedger = field(default_factory=_FakeLedger)

    b = _build_bundle(
        metrics={}, spec_dict={}, run_meta=_run_meta(),
        use_plotly=False, engine_result=_LegacyEngine(),
    )
    assert b.tau_history == []
    assert b.warmup_calibration is None


# --------------------------- _ReplayEngineResult.from_bundle ---------------


def test_replay_engine_result_surfaces_new_history_lists() -> None:
    b = RebuildBundle(
        metrics={"pipeline_state_hash": "abcd" * 16,
                 "n_decisions": 10, "n_filled": 5,
                 "n_actions_approved": 5, "n_actions_rejected": 1},
        spec_dict={}, run_meta={},
        tau_history=[0.5, 0.6],
        r_hat_ewma_history=[0.1, 0.2],
        r_star_ewma_history=[0.20, 0.21],
        paused_history=[False, True],
        sigma_ve_history=[0.3, 0.4],
        inventory_size_history=[0, 1],
        hold_age_max_history=[0, 1],
        warmup_calibration={"n_warmup": 100},
    )
    replay = _ReplayEngineResult.from_bundle(b)
    assert replay.tau_history == [0.5, 0.6]
    assert replay.r_hat_ewma_history == [0.1, 0.2]
    assert replay.r_star_ewma_history == [0.20, 0.21]
    assert replay.paused_history == [False, True]
    assert replay.sigma_ve_history == [0.3, 0.4]
    assert replay.inventory_size_history == [0, 1]
    assert replay.hold_age_max_history == [0, 1]
    assert replay.warmup_calibration == {"n_warmup": 100}


# --------------------------- end-to-end via ReportRenderer ------------------


def _fake_metrics() -> dict:
    return {
        "brier": 0.21, "ece": 0.034,
        "trading": {
            "n_trades": 0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "sharpe": 0.0, "probabilistic_sharpe": 0.5, "sortino": 0.0,
            "hit_rate": 0.0, "profit_factor": 1.0,
            "max_drawdown_log": 0.0, "cdar_5pct_log": 0.0,
            "total_log_return": 0.0, "total_pct_return": 0.0,
        },
        "n_decisions": 0, "n_filled": 0,
        "n_actions_approved": 0, "n_actions_rejected": 0,
        "roc_auc": 0.5, "pr_auc": 0.0,
        "pipeline_state_hash": "ab" * 16,
        "calibration_by_regime": [],
        "accepted": True, "blocked_reasons": [],
    }


def test_renderer_persists_extended_bundle_through_full_render(
    tmp_path: Path,
) -> None:
    """End-to-end: ``render`` must persist a RebuildBundle with the new
    arrays + warmup_calibration; reloading from disk preserves them."""
    eng = _FakeEngineWithState(
        p_online_history=[0.4, 0.5],
        label_history=[0, 1],
        regime_history=[0, 1],
        ledger=_FakeLedger(fills=[]),
        tau_history=[0.5, 0.55],
        r_hat_ewma_history=[0.1, 0.11],
        r_star_ewma_history=[0.2, 0.2],
        paused_history=[False, True],
        sigma_ve_history=[0.3, 0.31],
        inventory_size_history=[0, 1],
        hold_age_max_history=[0, 1],
        warmup_calibration={"isotonic": [0.1, 0.5, 0.9]},
    )
    rr = ReportRenderer(report_root=tmp_path, enable_archive=False)
    rr.render(
        engine_result=eng,
        metrics=_fake_metrics(),
        spec_dict={"name": "ext-bundle", "seed": 42,
                   "wagie": {"data": {"parquet_path": "x.parquet",
                                       "m_minutes": 20}}},
        run_meta=_run_meta(),
        use_plotly=False,
    )
    bundle_path = tmp_path / "state" / REBUILD_BUNDLE_NAME
    assert bundle_path.is_file()
    with bundle_path.open("rb") as fh:
        loaded = pickle.load(fh)
    assert isinstance(loaded, RebuildBundle)
    assert loaded.tau_history == [0.5, 0.55]
    assert loaded.r_hat_ewma_history == [0.1, 0.11]
    assert loaded.paused_history == [False, True]
    assert loaded.warmup_calibration == {"isotonic": [0.1, 0.5, 0.9]}
