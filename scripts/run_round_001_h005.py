"""H-005 round 001 — first PnL number for the offline+online stack.

Loads the persisted offline+online predictions (`artifacts/online_eval/
predictions.parquet`) plus the test slice of the bar / minute parquets, runs
`simulate_inventory_aware` with symmetric triple-barriers φ = c_stop = α =
0.0041113 (the calibrated 90%-quantile of train log-excursions), realistic
1bp per-side cost, across a τ_open ∈ {0.10, 0.15, 0.20, 0.30, 0.50} sweep on
both signals. Reports headline metrics + a shuffled-signal random-entry null
(per Bailey/Borwein/López de Prado 2014). Saves a 5-panel comparison plot.

This is intentionally a glue script — it does not retrain anything, it does
not touch label construction, and it produces no model artifacts. Its only
output is a metrics CSV, an MLflow run, and a comparison plot under
`RESEARCH/diagrams/round_001/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.backtest import (  # noqa: E402
    simulate_inventory_aware, plot_backtest, BacktestResult,
)

PREDS = REPO / "artifacts" / "online_eval" / "predictions.parquet"
BARS = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"

ALPHA = 0.0041113497         # from data/model_data/BTCUSDT/feature_metadata.json
M = 20                       # decision_interval, minutes
COST_BPS = 1.0               # per-side; round-trip = 2 * COST_BPS * 1e-4
TAU_GRID = (0.10, 0.15, 0.20, 0.30, 0.50)
NULL_BOOTSTRAP = 200
SEED = 42


def load_aligned() -> dict:
    """Return aligned arrays + boundaries DataFrame for the 31486-row test slice.

    Per CODE-SCOUT (round 001): map decision-bar k → minute-index using
    `open_time` searchsort, then drop bars whose [n_k+1, n_k+M] lookahead window
    crosses a segment boundary (the single 80-min major gap).
    """
    preds = pd.read_parquet(PREDS)
    bars = pd.read_parquet(BARS, columns=["open_time", "close_time"])
    mn_full = pd.read_parquet(
        MIN1, columns=["open_time", "high", "low", "close", "segment_id"]
    )

    n_test = len(preds)
    test_bars = bars.iloc[-n_test:].reset_index(drop=True)
    first_open_ms = int(test_bars["open_time"].iloc[0])
    last_close_ms = int(test_bars["close_time"].iloc[-1])

    mask = (mn_full["open_time"] >= first_open_ms) & (
        mn_full["open_time"] <= last_close_ms + 60_000
    )
    mn_test = mn_full.loc[mask].reset_index(drop=True)

    minute_idx = np.searchsorted(
        mn_test["open_time"].to_numpy(),
        test_bars["open_time"].to_numpy(),
    )
    boundaries_full = pd.DataFrame({
        "k": minute_idx // M,
        "ts": pd.to_datetime(test_bars["open_time"], unit="ms", utc=True),
    })

    seg = mn_test["segment_id"].to_numpy()
    n_k_arr = boundaries_full["k"].to_numpy() * M
    n_close_arr = n_k_arr + M
    in_range = n_close_arr < len(seg)
    keep = np.zeros(len(boundaries_full), dtype=bool)
    keep[in_range] = seg[n_k_arr[in_range]] == seg[n_close_arr[in_range]]

    boundaries = boundaries_full.loc[keep].reset_index(drop=True)
    return {
        "boundaries": boundaries,
        "p_offline": preds["p_offline"].to_numpy()[keep],
        "p_final": preds["p_final"].to_numpy()[keep],
        "y_true": preds["y_true"].to_numpy()[keep],
        "minute_close": mn_test["close"].to_numpy(),
        "minute_high": mn_test["high"].to_numpy(),
        "minute_low": mn_test["low"].to_numpy(),
        "n_test_full": n_test,
        "n_test_kept": int(keep.sum()),
        "ts_first": str(boundaries["ts"].iloc[0]),
        "ts_last": str(boundaries["ts"].iloc[-1]),
    }


def run_one(p: np.ndarray, tau: float, data: dict) -> BacktestResult:
    return simulate_inventory_aware(
        data["boundaries"],
        data["minute_close"], data["minute_high"], data["minute_low"],
        p,
        tau_open=tau, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
    )


def shuffled_signal_null(
    p: np.ndarray, tau: float, data: dict, *, n: int = NULL_BOOTSTRAP, seed: int = SEED
) -> dict:
    """Per-tau random-entry null: shuffle the p vector, recompute metrics.

    Preserves entry-rate (number of crossings) and trade-mechanics; randomizes
    only the signal-to-bar alignment. Drift inherited automatically — that's
    the point per THEORIST round 001 (BTC 2024 +120% drift would inflate any
    always-on-style null). Returns the full sharpes array so the caller can
    compute exact bootstrap p-values without resimulating.
    """
    rng = np.random.default_rng(seed)
    null_sharpes = np.empty(n, dtype=float)
    null_pnls = np.empty(n, dtype=float)
    null_n_trades = np.empty(n, dtype=int)
    for i in range(n):
        p_shuffled = p.copy()
        rng.shuffle(p_shuffled)
        res = run_one(p_shuffled, tau, data)
        null_sharpes[i] = res.metrics["sharpe"]
        null_pnls[i] = res.metrics["total_log_return"]
        null_n_trades[i] = res.metrics["n_trades"]
    return {
        "null_sharpe_mean": float(null_sharpes.mean()),
        "null_sharpe_std": float(null_sharpes.std(ddof=1)) if n > 1 else 0.0,
        "null_sharpe_q05": float(np.quantile(null_sharpes, 0.05)),
        "null_sharpe_q95": float(np.quantile(null_sharpes, 0.95)),
        "null_total_log_return_mean": float(null_pnls.mean()),
        "null_n_trades_mean": float(null_n_trades.mean()),
        "null_sharpes": null_sharpes,
    }


def bootstrap_p_value(observed: float, null_samples: np.ndarray) -> float:
    n = len(null_samples)
    return float((null_samples >= observed).sum() / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "RESEARCH" / "diagrams" / "round_001"))
    ap.add_argument("--null-bootstrap", type=int, default=NULL_BOOTSTRAP)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[H-005] loading aligned data …")
    data = load_aligned()
    print(f"  test slice: {data['n_test_full']} bars (predictions);"
          f" {data['n_test_kept']} kept after gap-window filter")
    print(f"  date range: {data['ts_first']} -> {data['ts_last']}")
    print(f"  minute array length: {len(data['minute_close'])}")
    print(f"  alpha (= phi = c_stop): {ALPHA:.7f}; M={M}; cost_bps={COST_BPS}")

    rows = []
    headline_results: dict = {}

    for tau in TAU_GRID:
        for label, p in (("offline", data["p_offline"]), ("final", data["p_final"])):
            res = run_one(p, tau, data)
            null = shuffled_signal_null(
                p, tau, data, n=args.null_bootstrap, seed=SEED + int(tau * 100),
            )
            p_value = bootstrap_p_value(res.metrics["sharpe"], null["null_sharpes"])

            row = {
                "tau_open": tau,
                "signal": label,
                **res.metrics,
                "shuffled_null_sharpe_mean": null["null_sharpe_mean"],
                "shuffled_null_sharpe_std": null["null_sharpe_std"],
                "shuffled_null_sharpe_q95": null["null_sharpe_q95"],
                "shuffled_null_total_log_return_mean": null["null_total_log_return_mean"],
                "shuffled_null_n_trades_mean": null["null_n_trades_mean"],
                "bootstrap_p_value": p_value,
            }
            rows.append(row)
            headline_results[(tau, label)] = res
            print(f"  tau={tau:.2f} signal={label:7s} | "
                  f"n={res.metrics['n_trades']:4d} | "
                  f"Sharpe={res.metrics['sharpe']:+.3f} | "
                  f"PSR={res.metrics['probabilistic_sharpe']:.3f} | "
                  f"PnL_log={res.metrics['total_log_return']:+.4f} | "
                  f"hit-rate={res.metrics['hit_rate']:.3f} | "
                  f"max-DD={res.metrics['max_drawdown_log']:.4f} | "
                  f"null_sharpe_mean={null['null_sharpe_mean']:+.3f} | "
                  f"p_boot={p_value:.3f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "tau_sweep_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[H-005] wrote {csv_path}")

    # -- Plot 1: 4-panel detail at the headline (τ=0.20, signal=final). -------
    headline_tau = 0.20
    detail_path = out_dir / f"backtest_detail_tau{int(headline_tau*100):02d}_final.png"
    plot_backtest(headline_results[(headline_tau, "final")], detail_path)
    print(f"[H-005] wrote {detail_path}")

    # -- Plot 2: τ-sweep comparison (offline vs final vs null). ---------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    # Pivot to {tau, offline, final, null_offline_mean, null_final_mean}.
    pivot = df.pivot(index="tau_open", columns="signal", values="sharpe")
    null_pivot = df.pivot(index="tau_open", columns="signal", values="shuffled_null_sharpe_mean")
    null_q95_pivot = df.pivot(index="tau_open", columns="signal", values="shuffled_null_sharpe_q95")
    n_pivot = df.pivot(index="tau_open", columns="signal", values="n_trades")
    pnl_pivot = df.pivot(index="tau_open", columns="signal", values="total_log_return")
    pval_pivot = df.pivot(index="tau_open", columns="signal", values="bootstrap_p_value")

    ax = axes[0, 0]
    ax.plot(pivot.index, pivot["offline"], "-o", color="#1B6B33", label="p_offline (strategy)")
    ax.plot(pivot.index, pivot["final"], "-s", color="#1B3F6B", label="p_final = p_online (strategy)")
    ax.plot(null_pivot.index, null_pivot["offline"], "--^", color="#1B6B33", alpha=0.55,
            label="shuffled-null mean (offline)")
    ax.plot(null_pivot.index, null_pivot["final"], "--v", color="#1B3F6B", alpha=0.55,
            label="shuffled-null mean (final)")
    ax.plot(null_q95_pivot.index, null_q95_pivot["offline"], ":", color="#1B6B33", alpha=0.4,
            label="null q95 (offline)")
    ax.plot(null_q95_pivot.index, null_q95_pivot["final"], ":", color="#1B3F6B", alpha=0.4,
            label="null q95 (final)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\tau_{open}$")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Sharpe vs τ — strategy vs shuffled-signal null\n"
                 "(null inherits BTC 2024 drift; gap = real edge)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(n_pivot.index, n_pivot["offline"], "-o", color="#1B6B33", label="p_offline")
    ax.plot(n_pivot.index, n_pivot["final"], "-s", color="#1B3F6B", label="p_final")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_{open}$")
    ax.set_ylabel("# trades (log scale)")
    ax.set_title("Trade count — offline overpredicts → more crossings")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(pnl_pivot.index, pnl_pivot["offline"], "-o", color="#1B6B33", label="p_offline")
    ax.plot(pnl_pivot.index, pnl_pivot["final"], "-s", color="#1B3F6B", label="p_final")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\tau_{open}$")
    ax.set_ylabel("Total log-PnL on test split")
    ax.set_title("Total net log-PnL across τ sweep")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(pval_pivot.index, pval_pivot["offline"], "-o", color="#1B6B33", label="p_offline")
    ax.plot(pval_pivot.index, pval_pivot["final"], "-s", color="#1B3F6B", label="p_final")
    ax.axhline(0.05, color="#7A1B1B", linestyle="--", linewidth=0.7, label="α=0.05")
    ax.axhline(0.20, color="#7A5C00", linestyle="--", linewidth=0.7, label="α=0.20")
    ax.set_xlabel(r"$\tau_{open}$")
    ax.set_ylabel("Bootstrap p-value (vs shuffled-signal null)")
    ax.set_title("Bootstrap p-value vs τ — lower = stronger evidence of edge")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.02, 1.02)

    fig.suptitle(
        f"H-005 round 001 — τ sweep on real BTCUSDT 2024 test stream\n"
        f"n_test={data['n_test_kept']}, "
        f"date={data['ts_first'][:10]} → {data['ts_last'][:10]}, "
        f"φ=c_stop=α={ALPHA:.5f}, cost={COST_BPS}bp/side, "
        f"null=shuffled-signal×{NULL_BOOTSTRAP}",
        fontsize=11, weight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    sweep_path = out_dir / "tau_sweep_summary.png"
    fig.savefig(sweep_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[H-005] wrote {sweep_path}")

    # -- MLflow logging --------------------------------------------------------
    if not args.no_mlflow:
        try:
            from src import mlflow_utils
            with mlflow_utils.round_run(
                round_id="001",
                hypothesis_id="H-005",
                fast_mode=False,
                data_start=data["ts_first"],
                data_end=data["ts_last"],
                extra_tags={
                    "tagged": "backtest",
                    "alpha_phi_c_stop": str(ALPHA),
                    "cost_bps_per_side": str(COST_BPS),
                    "n_test_kept": str(data["n_test_kept"]),
                    "n_test_full": str(data["n_test_full"]),
                    "null_bootstrap": str(NULL_BOOTSTRAP),
                },
            ):
                import mlflow
                # Per-row metrics with descriptive keys.
                for r in rows:
                    tau_id = int(r["tau_open"] * 100)
                    sig = r["signal"]
                    for mk in (
                        "n_trades", "sharpe", "probabilistic_sharpe", "total_log_return",
                        "hit_rate", "max_drawdown_log", "cdar_5pct_log", "profit_factor",
                        "shuffled_null_sharpe_mean", "shuffled_null_sharpe_q95",
                        "bootstrap_p_value",
                    ):
                        v = r.get(mk)
                        try:
                            v = float(v)
                        except Exception:
                            continue
                        if np.isfinite(v):
                            mlflow.log_metric(f"tau{tau_id:02d}.{sig}.{mk}", v)
                mlflow_utils.log_artifact_safely(csv_path)
                mlflow_utils.log_artifact_safely(detail_path)
                mlflow_utils.log_artifact_safely(sweep_path)
                print("[H-005] MLflow run logged")
        except Exception as exc:
            print(f"[H-005] MLflow logging skipped: {exc}")

    # JSON dump of headline numbers for the LEDGER / report.
    headline = {
        "headline_tau": headline_tau,
        "n_test_full": data["n_test_full"],
        "n_test_kept": data["n_test_kept"],
        "alpha_phi_c_stop": ALPHA,
        "cost_bps_per_side": COST_BPS,
        "ts_first": data["ts_first"],
        "ts_last": data["ts_last"],
        "rows": [
            {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
             for k, v in r.items()}
            for r in rows
        ],
    }
    json_path = out_dir / "headline.json"
    json_path.write_text(json.dumps(headline, indent=2, default=str), encoding="utf-8")
    print(f"[H-005] wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
