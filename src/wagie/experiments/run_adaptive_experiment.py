"""End-to-end runner for the composite adaptive-threshold trading experiment.

Two-pass design:
  1. **Pre-pass** (warmup window only): stream warmup bars through the full
     pipeline using a no-trade strategy (ThresholdGate with τ=2.0). Harvests
     ``p_online_history``, ``sigma_ve_history``, ``label_history`` from
     ``EngineResult`` for calibration.
  2. **Main pass** (full window: warmup + trading): rebuild the pipeline with
     the calibrated CompositeAdaptiveStrategy. ``warmup_samples`` skips
     strategy actions during the warmup region so ARF re-warms naturally
     before trading begins.

Outputs:
  - ``artifacts/adaptive_experiment/metrics.json`` — full MetricsReport
  - ``artifacts/adaptive_experiment/index.html`` — rendered report with
    Controller / Inventory / Drift / Trading / Calibration sections.
  - ``artifacts/adaptive_experiment/state/rebuild_bundle.pkl`` — for
    per-section rebuilds.

Usage:
  python -m wagie.experiments.run_adaptive_experiment

Reads its window bounds from ``artifacts/experiment_window.json`` (produced
by ``wagie.experiments.window_selector``).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from wagie.config import (
    ARFSubConfig, BrokerConfig, CVConfig, DataConfig, ModelConfig,
    RuntimeConfig, StrategyConfig, WagieConfig,
)
from wagie.engine import EngineResult
from wagie.experiments.warmup_calibration import (
    CalibrationError, WarmupCalibration, calibrate_from_warmup,
)
from wagie.features import BaseBarFeatures, FeatureBuilder
from wagie.features.catalog import default_streaming_features
from wagie.metrics import MetricsBattery
from wagie.reporting.manifest import RunMeta
from wagie.reporting.renderer import ReportRenderer
from wagie.run import run as wagie_run


def _build_streaming_features():
    """Construct the BaseBarFeatures + FeatureBuilder pair that the trained
    offline model expects. Without these the pipeline produces only raw OHLCV
    bar fields and the predictor reads NaN for every engineered feature."""
    return BaseBarFeatures(), FeatureBuilder(default_streaming_features())


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — match the dataset / model in the cwd repo
# ---------------------------------------------------------------------------

DATA_PARQUET = Path("data/cleansed_data/BTCUSDT/1m.parquet")
CATBOOST_MODEL = Path("artifacts/offline_model/model.cbm")
SELECTED_FEATURES = Path("artifacts/offline_model/selected_features.json")
WINDOW_JSON = Path("artifacts/experiment_window.json")
REPORT_ROOT = Path("artifacts/adaptive_experiment")

# Strategy / barrier defaults from the existing baseline spec
TP_SL_LOG = 0.0041113   # ≈ 41 bps symmetric barrier
COST_BPS = 1.0
EXPIRY_MIN = 20         # one 20-min bar
M_MINUTES = 20
INVENTORY_CAP = 50      # generous: don't artificially throttle
# EV-breakeven floor = 0.5 + cost/barrier ≈ 0.524 for symmetric 41-bp barriers.
# However the offline model on this window never produces p_online > 0.50,
# so an EV-strict floor would refuse every trade (correct but uninformative).
# We relax to TAU_FLOOR = 0.0 so the controller can find an operating point
# that matches r* (entry rate ≈ event rate). This is per the user's design
# directive: "entry rate should be calibrated to actual event rate".
# Trades will be EV-negative on average given the cost/barrier ratio, but the
# experiment then exercises the controller / inventory / batched-exit machinery
# and the resulting loss is an honest measurement of the model's edge gap.
TAU_FLOOR = 0.0
EV_BREAKEVEN_TAU = 0.5 + (COST_BPS * 1e-4) / TP_SL_LOG  # ≈ 0.524 — reported, not enforced


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _make_config(
    *,
    start_ts_ms: int,
    end_ts_ms: int,
    strategy_kind: str,
    strategy_tau: Optional[float] = None,
    strategy_extra: Optional[dict] = None,
    warmup_samples: int = 0,
) -> WagieConfig:
    return WagieConfig(
        data=DataConfig(
            parquet_path=str(DATA_PARQUET),
            m_minutes=M_MINUTES,
            start_ts_ms=int(start_ts_ms),
            end_ts_ms=int(end_ts_ms),
            symbol="BTCUSDT", venue="binance",
        ),
        model=ModelConfig(
            catboost_path=str(CATBOOST_MODEL),
            catboost_ensemble_n=3,        # 3-member virtual ensemble (model_seed{0,1,2}.cbm); enables σ_VE
            selected_features_path=str(SELECTED_FEATURES),
            arf=ARFSubConfig(
                n_models=10, max_features="sqrt", lambda_value=6.0, seed=42,
            ),
        ),
        strategy=StrategyConfig(
            kind=strategy_kind,
            tau=strategy_tau,
            k=None,
            extra=dict(strategy_extra or {}),
        ),
        broker=BrokerConfig(
            take_profit_log=TP_SL_LOG,
            stop_loss_log=TP_SL_LOG,
            cost_bps=COST_BPS,
            tie_break="pessimistic_sl_first",
            execution_latency_minutes=1,
            expiry_minutes=EXPIRY_MIN,
            inventory_cap=INVENTORY_CAP,
            label_alpha=TP_SL_LOG,
        ),
        cv=CVConfig(),
        runtime=RuntimeConfig(
            seed=42, warmup_samples=warmup_samples, capture_audit=False,
        ),
    )


# ---------------------------------------------------------------------------
# Calibration adapter — re-shape EngineResult arrays into warmup_calibration
# ---------------------------------------------------------------------------


def _calibrate_from_engine_result(
    pre_result: EngineResult,
    *,
    training_label_ttbarrier: list[int],
    quantile_sigma: float = 0.75,
    stability_window: int = 50,
    max_warmup_bars: Optional[int] = None,
) -> WarmupCalibration:
    """Wrap calibrate_from_warmup with a duck-typed Observation iterator built
    from the pre-pass EngineResult arrays."""

    class _DuckObs:
        def __init__(self, p_online: float, sigma_ve: float, p_offline: float):
            self.p_online = p_online
            self.sigma_ve = sigma_ve
            self.p_offline = p_offline

    # Per the existing engine, p_online_history and sigma_ve_history align
    # with label_history bar-for-bar (matured pairs only — pre-warmup labels
    # may not have matured yet, so we use the LabelBuffer-matured trace).
    p_online = list(pre_result.p_online_history or [])
    labels = list(pre_result.label_history or [])
    sigmas = list(pre_result.sigma_ve_history or [])

    # If sigma_ve_history is shorter (it's per-bar-with-strategy-state, not
    # per-matured-pair), we fall back to a placeholder — the warmup_calibration
    # module needs a sigma_ve per bar but for σ_max we only need the distribution
    # so we trim/pad to the shorter length conservatively.
    n = min(len(p_online), len(labels))
    if n == 0:
        raise CalibrationError(
            "pre-pass EngineResult has no matured (p_online, label) pairs — "
            "warmup window may be too short relative to the labeling horizon"
        )
    p_online = p_online[:n]
    labels = labels[:n]
    if len(sigmas) >= n:
        sigmas = sigmas[:n]
    elif len(sigmas) > 0:
        # Pad the missing positions at the front with the median observed σ_VE
        # (defensive — non-strategy pre-pass shouldn't trip this).
        med = float(np.median(sigmas))
        sigmas = [med] * (n - len(sigmas)) + list(sigmas)
    else:
        # No σ_VE captured at all — treat as 0 (won't gate anything).
        sigmas = [0.0] * n

    pairs = [
        (_DuckObs(p_online=p, sigma_ve=s, p_offline=p), int(y))
        for p, s, y in zip(p_online, sigmas, labels)
    ]

    return calibrate_from_warmup(
        pairs,
        training_label_ttbarrier=training_label_ttbarrier,
        quantile_sigma=quantile_sigma,
        stability_window=stability_window,
        stability_threshold=0.05,
        min_warmup_bars=min(50, n),
        max_warmup_bars=max_warmup_bars,
        tau_floor=TAU_FLOOR,
        r_min=0.02,
        t_max_quantile=0.95,
    )


# ---------------------------------------------------------------------------
# Training-set time-to-barrier extraction (for t_max default)
# ---------------------------------------------------------------------------


def _training_ttbarrier(parquet_path: Path, *, max_n: int = 50_000) -> list[int]:
    """Best-effort: with H=1 horizon labels in this repo, time-to-first-barrier
    is essentially uniform at 1 bar for hits and undefined for misses. The
    strategy's t_max should be a *liquidation* parameter, not a labeling one.

    Default heuristic: t_max = 4 × labeling horizon = 4 bars. That gives the
    strategy ~80 minutes max hold at 20m bars, consistent with "exit fast" per
    the user's design.
    """
    return [1, 2, 3, 4, 4, 4]  # quantile_0.95 → 4


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run_experiment(
    window_json_path: Path = WINDOW_JSON,
    report_root: Path = REPORT_ROOT,
    *,
    log_level: int = logging.INFO,
) -> Path:
    """Run the full two-pass experiment and render the report.

    Returns path to ``index.html``.
    """
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("=" * 70)
    logger.info("Composite Adaptive-Threshold Experiment")
    logger.info("=" * 70)

    # 1. Load window bounds
    if not window_json_path.is_file():
        raise FileNotFoundError(f"missing window file: {window_json_path}")
    win = json.loads(window_json_path.read_text())
    logger.info("window: %s → %s (%d bars, warmup ends %s)",
                win["start_iso"], win["end_iso"], win["n_bars"], win["warmup_end_iso"])
    logger.info("event rate: %.4f, up_frac: %.3f, parkinson_cv: %.2f",
                win["event_rate"], win["up_frac"], win["parkinson_cv"])

    # ---- Pre-pass: warmup window with no-trade strategy --------------------
    logger.info("--- PRE-PASS: warmup observation pass (no trades) ---")
    pre_cfg = _make_config(
        start_ts_ms=win["start_ts"],
        end_ts_ms=win["warmup_end_ts"],
        strategy_kind="threshold_gate",
        strategy_tau=2.0,            # never triggers — observation only
        strategy_extra={},
        warmup_samples=0,
    )
    bb, fb = _build_streaming_features()
    pre_result = wagie_run(pre_cfg, feature_builder=fb, base_bar=bb)
    logger.info("pre-pass complete: %d decisions, %d matured labels",
                pre_result.n_decisions, len(pre_result.label_history))

    # ---- Calibrate ---------------------------------------------------------
    training_tt = _training_ttbarrier(DATA_PARQUET)
    try:
        calibration = _calibrate_from_engine_result(
            pre_result, training_label_ttbarrier=training_tt,
            quantile_sigma=0.75, stability_window=50, max_warmup_bars=200,
        )
    except CalibrationError as e:
        logger.error("calibration failed: %s", e)
        raise

    cal_dict = calibration.to_dict()
    logger.info("calibration: τ_init=%.4f, σ_max=%.4f, r*_init=%.4f, t_max=%d, "
                "brier_baseline=%.4f, converged=%s, bars=%d",
                cal_dict["tau_init"], cal_dict["sigma_max"], cal_dict["r_star_init"],
                cal_dict["t_max"], cal_dict["brier_baseline"], cal_dict["converged"],
                cal_dict["warmup_bars_used"])

    # ---- Main pass: full window with composite_adaptive --------------------
    logger.info("--- MAIN PASS: full window with composite_adaptive strategy ---")
    n_warmup_bars = max(1, int(round(
        (win["warmup_end_ts"] - win["start_ts"]) / (M_MINUTES * 60_000)
    )))
    logger.info("main-pass warmup_samples=%d (skip strategy during ARF re-warm)", n_warmup_bars)

    controller_kwargs = {
        "tau": float(calibration.tau_init),
        "sigma_max": float(calibration.sigma_max),
        "r_star_init": float(calibration.r_star_init),
        "tau_floor": TAU_FLOOR,
        "tau_ceil": 0.95,
        # Faster controller: with a well-calibrated offline model, p_online
        # concentrates near the base rate so τ must track narrowly. Original
        # gamma=0.005 / dtau_max=0.001 took ~700 bars to traverse 0.2 of τ
        # range — too slow to catch the rare high-p_online peaks. 4x faster
        # response keeps adaptation honest without becoming jumpy.
        "gamma": 0.02,
        "lambda_decay": 4e-3,    # half-life ≈ 175 bars (~2.4 days at 20m)
        "dtau_max": 0.005,
        "r_min": 0.02,
        "r_resume": 0.05,
    }
    main_cfg = _make_config(
        start_ts_ms=win["start_ts"],
        end_ts_ms=win["end_ts"],
        strategy_kind="composite_adaptive",
        strategy_tau=None,
        strategy_extra={
            "controller_kwargs": controller_kwargs,
            "t_max": int(calibration.t_max),
        },
        warmup_samples=n_warmup_bars,
    )
    bb, fb = _build_streaming_features()
    main_result = wagie_run(main_cfg, feature_builder=fb, base_bar=bb)

    # Inject the warmup_calibration snapshot for the report.
    main_result.warmup_calibration = cal_dict

    logger.info("main pass complete: %d decisions, %d filled, %d open at end",
                main_result.n_decisions, main_result.n_filled,
                main_result.ledger.n_open_at_finalize if main_result.ledger else 0)

    # ---- Compute metrics + render report -----------------------------------
    logger.info("--- METRICS + REPORT ---")
    metrics_report = MetricsBattery(m_minutes=M_MINUTES, compute_ci=False).compute(main_result)

    spec_dict = {
        "name": "composite_adaptive_experiment",
        "window": win,
        "controller_kwargs": controller_kwargs,
        "warmup_calibration": cal_dict,
        "wagie": main_cfg.model_dump(mode="json"),
    }
    run_meta = RunMeta(
        run_id="adaptive-exp-v1",
        spec_name="composite_adaptive",
        spec_hash="adaptive001",
        mode="backtest",
        accepted=True,
    )

    report_root.mkdir(parents=True, exist_ok=True)
    renderer = ReportRenderer(report_root=report_root, enable_archive=False)
    index_path = renderer.render(
        engine_result=main_result,
        metrics=metrics_report.to_dict(),
        spec_dict=spec_dict,
        run_meta=run_meta,
        use_plotly=True,
        title="Composite Adaptive-Threshold — 2024-01-22 → 2024-02-05",
    )

    # Also dump a top-line summary JSON
    summary = {
        "window": win,
        "warmup_calibration": cal_dict,
        "controller_kwargs": controller_kwargs,
        "n_decisions": main_result.n_decisions,
        "n_filled": main_result.n_filled,
        "n_actions_approved": main_result.n_actions_approved,
        "n_actions_rejected": main_result.n_actions_rejected,
        "metrics_summary": {
            "brier": metrics_report.brier,
            "ece": metrics_report.ece,
            "sharpe": metrics_report.trading.get("sharpe", 0.0),
            "n_trades": metrics_report.trading.get("n_trades", 0),
            "hit_rate": metrics_report.trading.get("hit_rate", 0.0),
            "total_log_return": metrics_report.trading.get("total_log_return", 0.0),
            "max_drawdown_log": metrics_report.trading.get("max_drawdown_log", 0.0),
        },
    }
    (report_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    logger.info("report rendered: %s", index_path)
    logger.info("summary saved: %s", report_root / "experiment_summary.json")
    return index_path


def main(argv: Optional[list[str]] = None) -> int:
    try:
        run_experiment()
        return 0
    except Exception as e:
        logger.exception("experiment failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
