"""ARF-bypass variant of the adaptive-threshold experiment.

Strategy reads ``p_offline`` directly (not ``p_online``). ARF still runs as a
pipeline stage (so the engine's matured-trace metrics still capture it for the
report's drift section), but the trade decision uses the raw offline-ensemble
probability.

Calibration for τ_init / σ_max / r*_init is done by direct CatBoost prediction
on the warmup-window features (not via the engine pre-pass) since we need
p_offline, not p_online.

Usage:
  python scripts/run_experiment_offline.py [--window WINDOW_JSON] [--out OUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from wagie.experiments.run_adaptive_experiment import (
    EV_BREAKEVEN_TAU, M_MINUTES, REPORT_ROOT, TAU_FLOOR, WINDOW_JSON,
    _make_config, _build_streaming_features,
)
from wagie.metrics import MetricsBattery
from wagie.reporting.manifest import RunMeta
from wagie.reporting.renderer import ReportRenderer
from wagie.run import run as wagie_run

logger = logging.getLogger(__name__)

DATA_PARQUET = Path("data/model_data/BTCUSDT/bars_20m_features.parquet")
MODEL_DIR = Path("artifacts/offline_model")


def _calibrate_offline(window: dict, *, ensemble_n: int = 3,
                       quantile_sigma: float = 0.75, r_min: float = 0.02,
                       t_max_default: int = 4) -> dict:
    """Direct CatBoost prediction on the warmup window for τ_init / σ_max / r*."""
    logger.info("calibrating offline (direct CatBoost ensemble on warmup features)")
    # Load 3-member ensemble
    members = []
    for i in range(ensemble_n):
        path = MODEL_DIR / f"model_seed{i}.cbm"
        if not path.is_file():
            raise FileNotFoundError(f"missing ensemble member: {path}")
        m = CatBoostClassifier(); m.load_model(str(path))
        members.append(m)
    feat_cols = list(members[0].feature_names_)

    df = pl.read_parquet(DATA_PARQUET)
    # Cast open_time to int64 for safety (some files are f64)
    df = df.with_columns(open_time=pl.col("open_time").cast(pl.Int64))
    warmup_df = df.filter(
        (pl.col("open_time") >= int(window["start_ts"]))
        & (pl.col("open_time") < int(window["warmup_end_ts"]))
    )
    logger.info("warmup window rows: %d", len(warmup_df))
    if len(warmup_df) < 50:
        raise RuntimeError(f"warmup too short ({len(warmup_df)} rows)")

    X = warmup_df.select(feat_cols).to_numpy()
    y = warmup_df["label"].to_numpy().astype(float)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid].astype(int)
    if len(X) == 0:
        raise RuntimeError("no valid (label, features) pairs in warmup window")

    ps = np.stack([m.predict_proba(X)[:, 1] for m in members], axis=0)
    p_offline = ps.mean(axis=0)
    sigma_ve = ps.std(axis=0, ddof=0)

    # r*_init = warmup empirical event rate
    r_star_init = float(y.mean())
    if r_star_init < r_min:
        raise RuntimeError(f"warmup event rate {r_star_init:.4f} < r_min={r_min}")

    # τ_init = quantile(p_offline, 1 - r_star_init), bounded by floor
    tau_init = float(np.quantile(p_offline, 1.0 - r_star_init))
    tau_init = max(tau_init, TAU_FLOOR)

    # σ_max = quantile(sigma_ve, 0.75); fall back to inf if degenerate
    if sigma_ve.max() > 0:
        sigma_max = float(np.quantile(sigma_ve, quantile_sigma))
        if sigma_max >= sigma_ve.max() * 0.95:
            logger.warning("sigma_max gate would reject few bars; setting to inf")
            sigma_max = float("inf")
    else:
        sigma_max = float("inf")

    out = {
        "tau_init": tau_init,
        "sigma_max": sigma_max,
        "r_star_init": r_star_init,
        "t_max": int(t_max_default),
        "warmup_bars_used": int(len(y)),
        "p_offline_min": float(p_offline.min()),
        "p_offline_max": float(p_offline.max()),
        "p_offline_mean": float(p_offline.mean()),
        "p_offline_q90": float(np.quantile(p_offline, 0.9)),
        "sigma_ve_max": float(sigma_ve.max()),
        "ensemble_n": ensemble_n,
    }
    logger.info("calibration: tau_init=%.4f, sigma_max=%s, r*_init=%.4f, t_max=%d, "
                "p_offline range=[%.4f, %.4f] q90=%.4f",
                out["tau_init"],
                f"{out['sigma_max']:.4f}" if out['sigma_max'] != float('inf') else "inf",
                out["r_star_init"], out["t_max"],
                out["p_offline_min"], out["p_offline_max"], out["p_offline_q90"])
    return out


def run_experiment_offline(window_json_path: Path, report_root: Path,
                           *, tau_pin: float | None = None) -> Path:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("=" * 70)
    logger.info("ARF-BYPASS Experiment: strategy reads p_offline directly")
    logger.info("=" * 70)

    win = json.loads(window_json_path.read_text())
    logger.info("window: %s -> %s (%d days, %d bars)",
                win["start_iso"], win["end_iso"], win["window_days"], win["n_bars"])

    cal = _calibrate_offline(win)
    if tau_pin is not None:
        logger.warning("tau-pin override: setting tau=%.4f and clamping controller "
                       "(floor=ceil=tau, no adaptation)", tau_pin)
        cal["tau_init"] = float(tau_pin)
        cal["tau_pinned"] = float(tau_pin)

    n_warmup_bars = max(1, int(round(
        (win["warmup_end_ts"] - win["start_ts"]) / (M_MINUTES * 60_000)
    )))

    if tau_pin is not None:
        # Clamp the controller — τ will not move from its initial value.
        controller_kwargs = {
            "tau": float(tau_pin),
            "sigma_max": float(cal["sigma_max"]),
            "r_star_init": float(cal["r_star_init"]),
            "tau_floor": float(tau_pin),
            "tau_ceil": float(tau_pin),
            "gamma": 0.0, "lambda_decay": 4e-3, "dtau_max": 0.0,
            "r_min": 0.02, "r_resume": 0.05,
        }
    else:
        controller_kwargs = {
            "tau": float(cal["tau_init"]),
            "sigma_max": float(cal["sigma_max"]),
            "r_star_init": float(cal["r_star_init"]),
            "tau_floor": TAU_FLOOR,
            "tau_ceil": 0.95,
            "gamma": 0.02,
            "lambda_decay": 4e-3,
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
            "t_max": int(cal["t_max"]),
            "signal_source": "offline",   # <<< the key change
        },
        warmup_samples=n_warmup_bars,
    )
    logger.info("main pass: signal_source='offline', warmup_samples=%d", n_warmup_bars)

    bb, fb = _build_streaming_features()
    main_result = wagie_run(main_cfg, feature_builder=fb, base_bar=bb)
    main_result.warmup_calibration = cal
    logger.info("done: %d decisions, %d filled, %d open at end",
                main_result.n_decisions, main_result.n_filled,
                main_result.ledger.n_open_at_finalize if main_result.ledger else 0)

    # Metrics + report
    metrics_report = MetricsBattery(m_minutes=M_MINUTES, compute_ci=False).compute(main_result)
    spec_dict = {
        "name": "composite_adaptive_OFFLINE_signal",
        "window": win,
        "controller_kwargs": controller_kwargs,
        "warmup_calibration": cal,
        "wagie": main_cfg.model_dump(mode="json"),
    }
    run_meta = RunMeta(
        run_id="adaptive-exp-offline-v1",
        spec_name="composite_adaptive_offline",
        spec_hash="adaptive_off_001",
        mode="backtest",
        accepted=True,
    )
    report_root.mkdir(parents=True, exist_ok=True)
    renderer = ReportRenderer(report_root=report_root, enable_archive=False)
    index_path = renderer.render(
        engine_result=main_result,
        metrics=metrics_report.to_dict(),
        spec_dict=spec_dict, run_meta=run_meta, use_plotly=True,
        title=f"ARF-Bypass — {win['start_iso']} -> {win['end_iso']}",
    )

    # Summary
    t = metrics_report.trading
    summary = {
        "window": win,
        "warmup_calibration": cal,
        "controller_kwargs": controller_kwargs,
        "n_decisions": main_result.n_decisions,
        "n_filled": main_result.n_filled,
        "metrics_summary": {
            "brier": metrics_report.brier, "ece": metrics_report.ece,
            "roc_auc": metrics_report.roc_auc, "pr_auc": metrics_report.pr_auc,
            "n_trades": t.get("n_trades", 0),
            "hit_rate": t.get("hit_rate", 0.0),
            "total_log_return": t.get("total_log_return", 0.0),
            "total_pct_return": t.get("total_pct_return", 0.0),
            "max_drawdown_log": t.get("max_drawdown_log", 0.0),
            "sharpe": t.get("sharpe", 0.0),
            "profit_factor": t.get("profit_factor", 0.0),
            "n_tp": t.get("n_tp", 0),
            "n_sl": t.get("n_sl", 0),
        },
    }
    (report_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("summary saved: %s", report_root / "experiment_summary.json")
    return index_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=Path, default=WINDOW_JSON)
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/adaptive_experiment_OFFLINE"))
    p.add_argument("--tau-pin", type=float, default=None,
                   help="Pin τ_init at this value (overrides warmup calibration). "
                        "Disables controller adaptation by clamping floor=ceil=tau-pin.")
    args = p.parse_args(argv)
    try:
        run_experiment_offline(args.window, args.out, tau_pin=args.tau_pin)
        return 0
    except Exception as e:
        logger.exception("experiment failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
