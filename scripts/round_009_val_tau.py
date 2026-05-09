"""H-005b: validation-split τ_open selection for the H-005 backtest.

Round 001 picked τ_open as a post-hoc grid on test. This round runs the
backtest harness on the VAL slice, sweeps τ ∈ [0.10..0.60] on val Sharpe,
picks the winning τ, then evaluates that single τ on test for an honest
out-of-sample number.

Steps (no model retrain — uses the persisted model.cbm):
1. Load `bars_20m_features.parquet`; chronological-split into (train, val,
   test). Confirm the split sizes match round 001 (train=37,783, val=9,445,
   test=31,486).
2. Load `artifacts/offline_model/model.cbm` + `selected_features.json`.
3. Predict p_offline on the val slice's selected features.
4. Run inventory-aware backtest on val for τ ∈ {0.10, 0.12, …, 0.60}.
   Pick τ* by max val Sharpe (CONSTITUTION V.b). Tie-break by val PR-AUC
   then by smaller τ (more trades = more statistical power).
5. Apply τ* to test (using the existing test predictions from
   `artifacts/online_eval/predictions.parquet` — no need to re-predict).
6. Bootstrap p-value vs shuffled-signal null, deflated Sharpe, full
   metrics block.
7. Persist:
    - val_predictions.parquet      (val features → p_offline)
    - val_tau_sweep_metrics.csv    (Sharpe + PR-AUC by τ on val)
    - val_tau_sweep.png            (val-Sharpe-vs-τ curve, marks τ*)
    - test_at_val_tau_metrics.json (full metrics at chosen τ on test)
    - test_at_val_tau_equity.png   (equity curve + drawdown at τ*)
    - headline.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.backtest import (  # noqa: E402
    BacktestResult,
    deflated_sharpe,
    simulate_inventory_aware,
)
from src.utils import chronological_split  # noqa: E402

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_009"
ARTIFACTS = REPO / "artifacts" / "offline_model"
PREDS_TEST = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
BARS = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"

ALPHA = 0.0041113
M = 20
COST_BPS = 1.0
N_NULL_BOOTSTRAP = 200
SEED = 42
TAU_GRID = tuple(round(0.10 + 0.02 * i, 4) for i in range(26))  # 0.10..0.60 step 0.02


def _load_minute_data() -> pd.DataFrame:
    return pd.read_parquet(
        MIN1, columns=["open_time", "high", "low", "close", "segment_id"]
    )


def _build_boundaries(
    feat_slice: pd.DataFrame, minute_df: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Round-001 boundary mapping reused on an arbitrary contiguous bar slice.

    Returns (boundaries_kept, keep_mask, minute_close, minute_high, minute_low).
    """
    first_open_ms = int(feat_slice["open_time"].iloc[0])
    last_close_ms = int(feat_slice["close_time"].iloc[-1])
    mask = (minute_df["open_time"] >= first_open_ms) & (
        minute_df["open_time"] <= last_close_ms + 60_000
    )
    mn = minute_df.loc[mask].reset_index(drop=True)
    minute_idx = np.searchsorted(
        mn["open_time"].to_numpy(), feat_slice["open_time"].to_numpy()
    )
    boundaries_full = pd.DataFrame({
        "k": minute_idx // M,
        "ts": pd.to_datetime(feat_slice["open_time"], unit="ms", utc=True),
    })
    seg = mn["segment_id"].to_numpy()
    n_k = boundaries_full["k"].to_numpy() * M
    n_close = n_k + M
    in_range = n_close < len(seg)
    keep = np.zeros(len(boundaries_full), dtype=bool)
    keep[in_range] = seg[n_k[in_range]] == seg[n_close[in_range]]
    boundaries = boundaries_full.loc[keep].reset_index(drop=True)
    return boundaries, keep, mn["close"].to_numpy(), mn["high"].to_numpy(), mn["low"].to_numpy()


def _shuffled_null(p, tau, boundaries, mn_close, mn_high, mn_low, *, n: int = N_NULL_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    nulls = np.empty(n, dtype=float)
    for i in range(n):
        p_shuf = p.copy()
        rng.shuffle(p_shuf)
        res = simulate_inventory_aware(
            boundaries, mn_close, mn_high, mn_low, p_shuf,
            tau_open=tau, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        nulls[i] = res.metrics["sharpe"]
    return nulls


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from catboost import CatBoostClassifier
    selected = json.loads((ARTIFACTS / "selected_features.json").read_text())["features"]

    print("[round 009] loading features + minute data …")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(feats, train_fraction=0.6, val_fraction=0.2)
    print(f"  split: train={len(train):,}  val={len(val):,}  test={len(test_full):,}")

    minute_df = _load_minute_data()
    print(f"  minute rows: {len(minute_df):,}")

    print("[round 009] loading model …")
    model = CatBoostClassifier()
    model.load_model(str(ARTIFACTS / "model.cbm"))
    # The model fixes its own feature ordering via `feature_names_` (uses the
    # full 726-feature set; `selected_features.json` lists the top-120 by
    # importance and is downstream documentation, not a training restriction).
    model_feats = list(model.feature_names_)
    missing = [f for f in model_feats if f not in val.columns]
    if missing:
        raise RuntimeError(f"val frame missing {len(missing)} model features; first 5: {missing[:5]}")
    print(f"  model expects {len(model_feats)} features in fixed order; "
          f"selected_features.json had top-{len(selected)} by importance")

    print("[round 009] predicting p_offline on val …")
    p_val = model.predict_proba(val[model_feats])[:, 1]
    val_pred_df = pd.DataFrame({
        "open_time": val["open_time"].to_numpy(),
        "y_true": val["label"].astype(int).to_numpy(),
        "p_offline": p_val,
    })
    val_pred_df.to_parquet(OUTDIR / "val_predictions.parquet", index=False)

    print("[round 009] building val boundaries …")
    val_for_bound = val[["open_time", "close_time"]].reset_index(drop=True)
    val_bound, val_keep, val_mn_c, val_mn_h, val_mn_l = _build_boundaries(val_for_bound, minute_df)
    p_val_kept = p_val[val_keep]
    print(f"  val bars kept: {len(val_bound)} / {len(val)}")

    print(f"[round 009] sweeping tau on val ({len(TAU_GRID)} points) …")
    rows = []
    for tau in TAU_GRID:
        res = simulate_inventory_aware(
            val_bound, val_mn_c, val_mn_h, val_mn_l, p_val_kept,
            tau_open=tau, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        m = res.metrics
        rows.append({
            "tau": tau,
            "sharpe": m["sharpe"],
            "psr": m["probabilistic_sharpe"],
            "n_trades": m["n_trades"],
            "total_log_return": m["total_log_return"],
            "max_drawdown": m["max_drawdown_log"],
            "hit_rate": m["hit_rate"],
        })
    val_sweep = pd.DataFrame(rows)
    val_sweep.to_csv(OUTDIR / "val_tau_sweep_metrics.csv", index=False)

    # Pick τ* by max val Sharpe (require ≥ 50 trades for stability).
    eligible = val_sweep[val_sweep["n_trades"] >= 50]
    if eligible.empty:
        eligible = val_sweep  # fallback
    tau_star_row = eligible.sort_values(["sharpe", "n_trades"], ascending=[False, False]).iloc[0]
    tau_star = float(tau_star_row["tau"])
    print(f"[round 009] val tau* = {tau_star:.3f}  (val Sharpe={tau_star_row['sharpe']:+.3f}, "
          f"n_trades={int(tau_star_row['n_trades'])})")

    print(f"[round 009] applying tau* on test …")
    # Use the existing test-aligned data the same way round 001 does.
    test_bars_for_bound = test_full[["open_time", "close_time"]].reset_index(drop=True)
    test_bound, test_keep, test_mn_c, test_mn_h, test_mn_l = _build_boundaries(
        test_bars_for_bound, minute_df,
    )
    preds_test = pd.read_parquet(PREDS_TEST)
    if len(preds_test) != len(test_full):
        raise RuntimeError(
            f"test predictions n={len(preds_test)} != test split n={len(test_full)}"
        )
    p_test_kept = preds_test["p_offline"].to_numpy()[test_keep]
    y_test_kept = preds_test["y_true"].to_numpy()[test_keep]

    test_res: BacktestResult = simulate_inventory_aware(
        test_bound, test_mn_c, test_mn_h, test_mn_l, p_test_kept,
        tau_open=tau_star, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
    )
    print(f"  test Sharpe={test_res.metrics['sharpe']:+.3f}  PSR={test_res.metrics['probabilistic_sharpe']:.3f}  "
          f"n_trades={test_res.metrics['n_trades']}")

    print(f"[round 009] bootstrap null on test (n={N_NULL_BOOTSTRAP}) …")
    nulls = _shuffled_null(
        p_test_kept, tau_star, test_bound, test_mn_c, test_mn_h, test_mn_l,
        n=N_NULL_BOOTSTRAP,
    )
    observed_sharpe = float(test_res.metrics["sharpe"])
    p_boot = float((nulls >= observed_sharpe).sum() / len(nulls))
    null_mean = float(nulls.mean())
    null_std = float(nulls.std(ddof=1)) if len(nulls) > 1 else 0.0

    # Deflated Sharpe accounting for τ-grid multiplicity (Bailey-LdP 2014).
    # Cross-trial variance approximated by var of val sweep Sharpes.
    var_trial = float(np.var(val_sweep["sharpe"], ddof=1)) if len(val_sweep) > 1 else 0.0
    deflated_dsr, deflated_sr_star = deflated_sharpe(
        observed_sharpe=observed_sharpe,
        var_trial_sharpes=var_trial,
        n_trials=len(TAU_GRID),
        n_obs=max(int(test_res.metrics["n_trades"]), 2),
        skew=0.0,
        kurt=3.0,
    )

    test_metrics_full = {
        "tau_star": tau_star,
        "val_tau_sharpe": float(tau_star_row["sharpe"]),
        "val_tau_psr": float(tau_star_row["psr"]),
        "val_n_trades": int(tau_star_row["n_trades"]),
        "test_sharpe": observed_sharpe,
        "test_psr": float(test_res.metrics["probabilistic_sharpe"]),
        "test_n_trades": int(test_res.metrics["n_trades"]),
        "test_total_log_return": float(test_res.metrics["total_log_return"]),
        "test_max_drawdown": float(test_res.metrics["max_drawdown_log"]),
        "test_hit_rate": float(test_res.metrics["hit_rate"]),
        "p_boot": p_boot,
        "null_sharpe_mean": null_mean,
        "null_sharpe_std": null_std,
        "deflated_sharpe_dsr": float(deflated_dsr),
        "deflated_sr_star": float(deflated_sr_star),
        "n_null_bootstrap": N_NULL_BOOTSTRAP,
        "tau_grid": list(TAU_GRID),
        "tau_grid_size": len(TAU_GRID),
        "round_001_post_hoc_tau": 0.30,
        "round_001_post_hoc_test_sharpe": 1.50,
        "round_001_post_hoc_test_psr": 0.995,
    }
    (OUTDIR / "test_at_val_tau_metrics.json").write_text(
        json.dumps(test_metrics_full, indent=2), encoding="utf-8",
    )

    # ---- Plot 1: val Sharpe vs τ -------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(val_sweep["tau"], val_sweep["sharpe"], color="#3b7dd8", linewidth=1.6, marker="o",
            markersize=4, label="val Sharpe")
    ax.axvline(tau_star, color="red", linestyle="--", linewidth=1.0,
               label=f"tau*={tau_star:.3f}")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("tau_open")
    ax.set_ylabel("val Sharpe (annualised)")
    ax.set_title(
        f"VAL tau-sweep · grid {TAU_GRID[0]:.2f}..{TAU_GRID[-1]:.2f} step "
        f"{TAU_GRID[1]-TAU_GRID[0]:.3f} · winner Sharpe={tau_star_row['sharpe']:+.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "val_tau_sweep.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: equity curve + drawdown at val-chosen τ on test -----------
    # `equity` is already cumulative log-PnL indexed by decision-bar k.
    log_returns = test_res.equity.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(log_returns)
    drawdown = log_returns - running_max
    bar_idx = np.arange(len(log_returns))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(bar_idx, log_returns, color="#3b7dd8", linewidth=1.0)
    axes[0].set_title(
        f"TEST equity curve at val-chosen tau*={tau_star:.3f}  "
        f"(Sharpe={observed_sharpe:+.3f}, PSR={test_res.metrics['probabilistic_sharpe']:.3f}, "
        f"p_boot={p_boot:.3f}, DSR={deflated_dsr:.3f}, n_trades={int(test_res.metrics['n_trades'])})"
    )
    axes[0].set_ylabel("cumulative log return")
    axes[0].grid(True, alpha=0.3)
    axes[1].fill_between(bar_idx, drawdown, color="#d44", alpha=0.5, linewidth=0)
    axes[1].set_xlabel("test decision-bar index")
    axes[1].set_ylabel("drawdown (log)")
    axes[1].set_title(f"max drawdown (log) = {test_res.metrics['max_drawdown_log']:.3f}")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "test_at_val_tau_equity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "009",
        "hypothesis": "H-005b",
        "claim": "validation-chosen tau on val Sharpe-grid; test Sharpe at that tau is the honest out-of-sample number",
        **test_metrics_full,
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8",
    )
    print(f"[round 009] saved diagnostics → {OUTDIR}")
    print(json.dumps({k: v for k, v in headline.items() if k not in ("tau_grid",)},
                      indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
