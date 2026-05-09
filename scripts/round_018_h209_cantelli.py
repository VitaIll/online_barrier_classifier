"""Round 018 — H-209 Cantelli-bound sized entries via virtual-ensemble sigma.

Strategy: open long when p_offline > tau (round-017's val-chosen tau=0.20),
size by `clip(p_lower_cantelli - tau, 0, 1)` with
`p_lower = max(0, p_offline - k · sigma_epistemic)` from Cantelli's
inequality. Sweep k ∈ {0, 0.5, 1.0, 1.5, 2.0} on val Sharpe with
n_trades >= 50 stability filter.

Architecture: still two-layer; we use offline `p_offline` + virtual-ensemble
sigma_epistemic from the persisted CatBoost (langevin=True, 625 trees).
This is NOT a sibling-style combiner — sigma is a UQ feature on the
offline output, not a parallel signal.

Reuses round-015 cached val/test unified prediction parquets for p_offline,
p_online, q_lo. Uses round-017's c_stop=inf config (the no-stop run that
unlocked positive offline Sharpe). Compares against round-017's
baseline_offline_tau as the binary baseline.

Outputs under RESEARCH/diagrams/phase_A/round_018/.
"""

from __future__ import annotations

import json
import sys
import time
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

from catboost import CatBoostClassifier  # noqa: E402

from src.backtest import (  # noqa: E402
    cscv_pbo,
    deflated_sharpe,
    simulate_inventory_aware_sized,
    walk_forward_backtest,
)
from src.uncertainty import predict_with_decomposed_uq  # noqa: E402
from src.utils import chronological_split  # noqa: E402

ALPHA = 0.0041113
M = 20
COST_BPS = 1.0
C_STOP = float("inf")  # round-017 winner
TAU_OPEN = 0.20  # round-017 val-chosen tau* for baseline_offline_tau
K_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
N_VIRTUAL = 10
N_NULL_BOOTSTRAP = 200
SEED = 42
N_FOLDS = 5
CSCV_N_CHUNKS = 16

OUTDIR = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_018"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ARTIFACTS = REPO / "artifacts" / "offline_model"
ONLINE_PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
ROUND017_TABLE = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_017" / "phase_A_table.csv"
SIGMA_CACHE_VAL = PHASE_A_DIR / "val_sigma_epistemic.parquet"
SIGMA_CACHE_TEST = PHASE_A_DIR / "test_sigma_epistemic.parquet"


def _load_minute() -> pd.DataFrame:
    return pd.read_parquet(
        MIN1, columns=["open_time", "high", "low", "close", "segment_id"]
    )


def _build_boundaries(feat_slice: pd.DataFrame, minute_df: pd.DataFrame):
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
    return (
        boundaries, keep,
        mn["close"].to_numpy(), mn["high"].to_numpy(), mn["low"].to_numpy(),
    )


def _compute_sigma_or_load(
    cache_path: Path, X_df: pd.DataFrame, model: CatBoostClassifier,
) -> pd.DataFrame:
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if len(df) == len(X_df):
            return df

    print(f"[round 018] computing virtual-ensemble UQ on n={len(X_df)} rows ...")
    t0 = time.time()
    feats = list(model.feature_names_)
    uq = predict_with_decomposed_uq(model, X_df[feats], n_virtual=N_VIRTUAL)
    df = pd.DataFrame({
        "p_mean_vens": uq.p_mean,
        "sigma_total": uq.sigma_total,
        "sigma_data": uq.sigma_data,
        "sigma_epistemic": uq.sigma_epistemic,
    })
    df.to_parquet(cache_path, index=False)
    print(f"  cached {cache_path.name} (took {time.time() - t0:.1f}s)")
    return df


def cantelli_size(
    p_offline: np.ndarray, sigma_epistemic: np.ndarray, *, tau: float, k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Open signal + size for Cantelli-bound sizing.

    open_signal = (p_offline > tau) (entry gate, same as baseline_offline)
    size        = clip(p_lower - tau, 0, 1)
    p_lower     = max(0, p_offline - k * sigma_epistemic)
    """
    p_lower = np.maximum(0.0, p_offline - k * sigma_epistemic)
    open_signal = (p_offline > tau).astype(float)
    size = np.clip(p_lower - tau, 0.0, 1.0)
    # When we don't open, size doesn't matter; force to 0 for cleanliness.
    size = size * open_signal
    return open_signal, size


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 018] ===== H-209 Cantelli-bound sized entries =====")
    print(f"[round 018] config: tau_open={TAU_OPEN}  c_stop={C_STOP}  "
          f"k_grid={K_GRID}  n_virtual={N_VIRTUAL}")
    t_main = time.time()

    print("[round 018] loading splits + minute data ...")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    minute_df = _load_minute()

    val_unified = pd.read_parquet(PHASE_A_DIR / "val_predictions_unified.parquet")
    test_unified = pd.read_parquet(PHASE_A_DIR / "test_predictions_unified.parquet")

    val_for_bound = val[["open_time", "close_time"]].reset_index(drop=True)
    val_bound, val_keep, val_mn_c, val_mn_h, val_mn_l = _build_boundaries(
        val_for_bound, minute_df,
    )
    test_for_bound = test_full[["open_time", "close_time"]].reset_index(drop=True)
    test_bound, test_keep, test_mn_c, test_mn_h, test_mn_l = _build_boundaries(
        test_for_bound, minute_df,
    )
    print(f"  val kept: {int(val_keep.sum())}  test kept: {int(test_keep.sum())}")

    # Load model + compute sigma_epistemic on val + test (cached).
    print("[round 018] loading offline CatBoost ...")
    model = CatBoostClassifier()
    model.load_model(str(ARTIFACTS / "model.cbm"))
    val_kept_features = val.iloc[val_keep].reset_index(drop=True)
    test_kept_features = test_full.iloc[test_keep].reset_index(drop=True)

    val_sigma = _compute_sigma_or_load(SIGMA_CACHE_VAL, val_kept_features, model)
    test_sigma = _compute_sigma_or_load(SIGMA_CACHE_TEST, test_kept_features, model)
    print(f"  val sigma_epistemic: mean={val_sigma['sigma_epistemic'].mean():.4f} "
          f"std={val_sigma['sigma_epistemic'].std():.4f}  "
          f"q10/50/90={val_sigma['sigma_epistemic'].quantile(0.1):.4f}/"
          f"{val_sigma['sigma_epistemic'].quantile(0.5):.4f}/"
          f"{val_sigma['sigma_epistemic'].quantile(0.9):.4f}")
    print(f"  test sigma_epistemic: mean={test_sigma['sigma_epistemic'].mean():.4f} "
          f"std={test_sigma['sigma_epistemic'].std():.4f}")

    val_p_off = val_unified["p_offline"].to_numpy(dtype=float)
    test_p_off = test_unified["p_offline"].to_numpy(dtype=float)
    val_sig_eps = val_sigma["sigma_epistemic"].to_numpy(dtype=float)
    test_sig_eps = test_sigma["sigma_epistemic"].to_numpy(dtype=float)

    # Sanity: virtual-ensemble p_mean should be very close to p_offline cached.
    diff = float(np.abs(val_unified["p_offline"].to_numpy() - val_sigma["p_mean_vens"].to_numpy()).max())
    print(f"  |p_offline_cached - p_mean_vens| max on val: {diff:.4f} "
          f"(should be <0.05 for sanity)")

    # Sweep k on val.
    print("\n[round 018] val sweep k_grid:")
    sweep_rows = []
    for k_v in K_GRID:
        open_sig, size = cantelli_size(val_p_off, val_sig_eps, tau=TAU_OPEN, k=k_v)
        res = simulate_inventory_aware_sized(
            val_bound, val_mn_c, val_mn_h, val_mn_l, open_sig, size,
            M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
        )
        m = res.metrics
        sweep_rows.append({
            "k": float(k_v),
            "sharpe": m["sharpe"], "psr": m["probabilistic_sharpe"],
            "n_trades": m["n_trades"],
            "total_log_return": m["total_log_return"],
            "max_drawdown_log": m["max_drawdown_log"],
            "hit_rate": m["hit_rate"],
            "mean_size": float(size[size > 0].mean()) if (size > 0).any() else 0.0,
            "n_above_tau": int((open_sig > 0).sum()),
        })
        print(f"  k={k_v:.1f}: val Sharpe={m['sharpe']:+.3f}  n_trades={m['n_trades']}  "
              f"mean_size={sweep_rows[-1]['mean_size']:.3f}")
    val_sweep = pd.DataFrame(sweep_rows)
    val_sweep.to_csv(OUTDIR / "val_sweep_k.csv", index=False)

    eligible = val_sweep[val_sweep["n_trades"] >= 50]
    if eligible.empty:
        eligible = val_sweep
    winning_row = eligible.sort_values(
        ["sharpe", "n_trades"], ascending=[False, False]
    ).iloc[0]
    k_star = float(winning_row["k"])
    print(f"\n[round 018] val k* = {k_star:.2f}  "
          f"val Sharpe={winning_row['sharpe']:+.3f}  n={int(winning_row['n_trades'])}")

    # Apply k* to test.
    open_sig_test, size_test = cantelli_size(
        test_p_off, test_sig_eps, tau=TAU_OPEN, k=k_star,
    )
    res_test = simulate_inventory_aware_sized(
        test_bound, test_mn_c, test_mn_h, test_mn_l, open_sig_test, size_test,
        M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
    )
    test_sharpe = float(res_test.metrics["sharpe"])
    print(f"\n[round 018] test Sharpe at k*={k_star}: {test_sharpe:+.3f}  "
          f"n={res_test.metrics['n_trades']}  PSR={res_test.metrics['probabilistic_sharpe']:.3f}  "
          f"hit_rate={res_test.metrics['hit_rate']:.3f}")

    # Bootstrap null: shuffle p_offline AND sigma jointly (random-entry baseline).
    print("[round 018] shuffled-signal bootstrap null (n=200) ...")
    rng = np.random.default_rng(SEED + 209)
    nulls = np.empty(N_NULL_BOOTSTRAP, dtype=float)
    for i in range(N_NULL_BOOTSTRAP):
        perm = rng.permutation(len(test_p_off))
        p_shuf = test_p_off[perm]
        sig_shuf = test_sig_eps[perm]
        open_n, size_n = cantelli_size(p_shuf, sig_shuf, tau=TAU_OPEN, k=k_star)
        rn = simulate_inventory_aware_sized(
            test_bound, test_mn_c, test_mn_h, test_mn_l, open_n, size_n,
            M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
        )
        nulls[i] = rn.metrics["sharpe"]
    p_boot = float((nulls >= test_sharpe).sum() / len(nulls))
    print(f"  null Sharpe mean={nulls.mean():+.3f} std={nulls.std(ddof=1):+.3f}  p_boot={p_boot:.3f}")

    # DSR within k-grid.
    var_within = float(np.var(val_sweep["sharpe"], ddof=1))
    dsr_within, sr_star = deflated_sharpe(
        observed_sharpe=test_sharpe,
        var_trial_sharpes=var_within,
        n_trials=len(K_GRID),
        n_obs=max(int(res_test.metrics["n_trades"]), 2),
        skew=0.0, kurt=3.0,
    )
    print(f"  DSR_within={dsr_within:.3f}  SR*_within={sr_star:.3f} "
          f"(var_trial_within={var_within:.3f})")

    # Walk-forward.
    wf = walk_forward_backtest(
        test_bound, test_mn_c, test_mn_h, test_mn_l,
        open_sig_test, size_test,
        n_folds=N_FOLDS, M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
    )
    wf_mean = float(wf["sharpe"].mean())
    wf_std = float(wf["sharpe"].std(ddof=1)) if len(wf) > 1 else 0.0
    print(f"  WF Sharpe = {wf_mean:+.3f} ± {wf_std:.3f}")

    # CSCV PBO over k-grid.
    print("[round 018] CSCV PBO over k-grid ...")
    rmat_rows = []
    for k_v in K_GRID:
        open_n, size_n = cantelli_size(test_p_off, test_sig_eps, tau=TAU_OPEN, k=k_v)
        rn = simulate_inventory_aware_sized(
            test_bound, test_mn_c, test_mn_h, test_mn_l, open_n, size_n,
            M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
        )
        eq = rn.equity.to_numpy(dtype=float)
        rmat_rows.append(np.diff(eq, prepend=0.0))
    rmat = np.vstack(rmat_rows)
    cscv_out = cscv_pbo(rmat, n_chunks=CSCV_N_CHUNKS)
    print(f"  PBO={cscv_out['pbo']:.3f}  median_logit={cscv_out['median_logit']:+.3f}  "
          f"modal IS-best k={K_GRID[cscv_out['is_best_strategy_modal_index']]}")

    # Compare to round-017 baseline_offline_tau (binary, no sizing).
    if ROUND017_TABLE.exists():
        r017 = pd.read_csv(ROUND017_TABLE)
        offline_row = r017[r017["strategy"] == "baseline_offline_tau"].iloc[0]
        baseline_sharpe = float(offline_row["test_sharpe"])
        baseline_n = int(offline_row["test_n_trades"])
        print(f"\n[round 018] vs round-017 binary baseline_offline_tau "
              f"(c_stop=inf, tau*=0.20): Sharpe={baseline_sharpe:+.3f}  n={baseline_n}")
        print(f"  ΔSharpe = {test_sharpe - baseline_sharpe:+.3f}  "
              f"Δn_trades = {res_test.metrics['n_trades'] - baseline_n}")
    else:
        baseline_sharpe = float("nan")
        baseline_n = 0

    # Plots.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(val_sweep["k"], val_sweep["sharpe"], color="#3b7dd8", marker="o",
            markersize=7, linewidth=1.6, label="val")
    ax.axvline(k_star, color="red", linestyle="--", linewidth=1.0,
               label=f"k* = {k_star}")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("k (Cantelli k)")
    ax.set_ylabel("val Sharpe")
    ax.set_title(f"H-209 Cantelli sized — val sweep k_grid (tau_open={TAU_OPEN}, c_stop=inf)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "val_k_sweep.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2.4, 1]})
    eq = res_test.equity.to_numpy(dtype=float)
    bar_idx = np.arange(len(eq))
    running_max = np.maximum.accumulate(eq)
    dd = eq - running_max
    axes[0].plot(bar_idx, eq, color="#1B6B33", linewidth=1.0,
                 label=f"H-209 sized k*={k_star}  Sharpe={test_sharpe:+.3f}")
    if not np.isnan(baseline_sharpe):
        axes[0].axhline(0, color="gray", linewidth=0.6)
        axes[0].set_title(
            f"H-209 Cantelli sized vs round-017 binary baseline (Sharpe={baseline_sharpe:+.3f})"
        )
    else:
        axes[0].set_title(f"H-209 Cantelli sized — test equity (k*={k_star})")
    axes[0].set_ylabel("cumulative log return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].fill_between(bar_idx, dd, color="#7A1B1B", alpha=0.3)
    axes[1].set_xlabel("test bar idx")
    axes[1].set_ylabel("drawdown (log)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "test_equity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # σ_epistemic vs realized PnL by trade.
    if len(res_test.trades) > 0:
        trades = res_test.trades.copy()
        # Map entry σ via k_open.
        trades["entry_sigma"] = test_sig_eps[trades["k_open"].to_numpy()]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(trades["entry_sigma"], trades["pnl_log_net"], s=3, alpha=0.4)
        axes[0].axhline(0, color="gray", linewidth=0.6)
        axes[0].set_xlabel("entry sigma_epistemic")
        axes[0].set_ylabel("realized log PnL net")
        axes[0].set_title("σ_epistemic vs realized PnL (per trade)")
        axes[0].grid(alpha=0.3)
        # Bin sigma into deciles, show mean PnL.
        try:
            trades["sigma_decile"] = pd.qcut(trades["entry_sigma"], 10,
                                              labels=False, duplicates="drop")
            decile_pnl = trades.groupby("sigma_decile").agg(
                mean_pnl=("pnl_log_net", "mean"),
                hit_rate=("pnl_log_net", lambda s: (s > 0).mean()),
                n=("pnl_log_net", "size"),
            ).reset_index()
            ax2 = axes[1]
            ax2.bar(decile_pnl["sigma_decile"], decile_pnl["mean_pnl"],
                     color="#3b7dd8", alpha=0.8, edgecolor="black", linewidth=0.4)
            ax2.axhline(0, color="gray", linewidth=0.6)
            ax2.set_xlabel("sigma_epistemic decile (low → high)")
            ax2.set_ylabel("mean log PnL net")
            ax2.set_title("Per-σ-decile mean PnL")
            ax2.grid(alpha=0.3, axis="y")
            decile_pnl.to_csv(OUTDIR / "sigma_decile_pnl.csv", index=False)
        except Exception as e:
            print(f"  decile plot skipped: {e}")
        fig.tight_layout()
        fig.savefig(OUTDIR / "sigma_vs_pnl.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    headline = {
        "round_id": "018",
        "phase": "A",
        "phase_round": "h209_cantelli",
        "claim": "H-209 Cantelli-bound sized entries via virtual-ensemble sigma",
        "alpha": ALPHA, "M": M, "cost_bps": COST_BPS, "c_stop": "inf",
        "tau_open": TAU_OPEN,
        "k_grid": list(K_GRID),
        "n_virtual": N_VIRTUAL,
        "k_star": k_star,
        "val_sharpe_at_kstar": float(winning_row["sharpe"]),
        "val_n_trades_at_kstar": int(winning_row["n_trades"]),
        "test_sharpe": test_sharpe,
        "test_psr": float(res_test.metrics["probabilistic_sharpe"]),
        "test_n_trades": int(res_test.metrics["n_trades"]),
        "test_total_log_return": float(res_test.metrics["total_log_return"]),
        "test_max_drawdown_log": float(res_test.metrics["max_drawdown_log"]),
        "test_hit_rate": float(res_test.metrics["hit_rate"]),
        "p_boot": p_boot,
        "null_sharpe_mean": float(nulls.mean()),
        "null_sharpe_std": float(nulls.std(ddof=1)) if len(nulls) > 1 else 0.0,
        "deflated_sharpe_within_kgrid": float(dsr_within),
        "sr_star_within": float(sr_star),
        "walk_forward_sharpe_mean": wf_mean,
        "walk_forward_sharpe_std": wf_std,
        "cscv_pbo_within_kgrid": float(cscv_out["pbo"]),
        "cscv_modal_is_best_k": float(K_GRID[cscv_out['is_best_strategy_modal_index']]),
        "round_017_baseline_offline_sharpe": baseline_sharpe,
        "round_017_baseline_offline_n_trades": baseline_n,
        "delta_sharpe_vs_round_017_baseline": (
            test_sharpe - baseline_sharpe if not np.isnan(baseline_sharpe) else None
        ),
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 018] DONE in {headline['wall_clock_s']:.1f}s "
          f"(test Sharpe {test_sharpe:+.3f}; ΔvsR017 binary baseline = "
          f"{headline['delta_sharpe_vs_round_017_baseline']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
