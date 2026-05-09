"""Round 030 — H-182 conditional Sharpe by feature regime (parkinson × Hurst).

Per parkinson_var tercile × Hurst tercile cell, compute Sharpe with
stationary-block-bootstrap CI (H-190). 9 cells × Sharpe + CI.

Strategy-side analog of `calibration_by_regime`. Highlights where alpha
(if any) is concentrated. BH-FDR over 9 cells.

Outputs under RESEARCH/diagrams/round_030/.
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

from src.bootstrap import bootstrap_sharpe, optimal_block_length  # noqa: E402
from src.features import compute_hurst_rs  # noqa: E402
from src.utils import chronological_split  # noqa: E402

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_030"
TRADE_PARQ = REPO / "RESEARCH" / "diagrams" / "round_021" / "trade_postmortem.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    significant_sorted = p_sorted <= thresholds
    if not significant_sorted.any():
        return np.zeros(n, dtype=bool)
    last = np.where(significant_sorted)[0].max()
    significant = np.zeros(n, dtype=bool)
    significant[order[:last + 1]] = True
    return significant


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 030] ===== H-182 conditional Sharpe by parkinson × Hurst =====")
    t_main = time.time()

    if not TRADE_PARQ.exists():
        print(f"[round 030] ERROR: round-021 trade parquet missing")
        return 1
    trades = pd.read_parquet(TRADE_PARQ)
    sub = trades[trades["strategy"] == "baseline_offline_tau"].copy()
    print(f"[round 030] baseline_offline_tau trades: {len(sub)}")

    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    test_start = int(test_full.index.min())
    sub["feats_idx"] = sub["k_open"].astype(int) + test_start
    sub = sub[sub["feats_idx"] < len(feats)].reset_index(drop=True)

    # Compute Hurst on the full feats series so we can fit terciles on TRAIN only.
    print("[round 030] computing Hurst on full feats (window=128) ...")
    return_col = "return" if "return" in feats.columns else "log_return"
    hurst_full = compute_hurst_rs(feats[return_col].fillna(0.0), window=128)
    print(f"  hurst_full finite count: {hurst_full.notna().sum()}")

    # Train-only terciles for both regime signals.
    train_mask = feats.index < int(test_full.index.min() - len(val))
    pk_train = feats.loc[train_mask, "parkinson_var_rolling_mean_24"].dropna()
    h_train = hurst_full.loc[train_mask].dropna()

    if len(pk_train) < 100 or len(h_train) < 100:
        print("[round 030] insufficient training samples for terciles")
        return 1

    pk_edges = np.quantile(pk_train, [1/3, 2/3])
    h_edges = np.quantile(h_train, [1/3, 2/3])
    print(f"  parkinson tercile edges: {pk_edges}")
    print(f"  Hurst tercile edges: {h_edges}")

    # Assign each trade to (pk_tercile, h_tercile).
    sub["entry_pk"] = feats.loc[sub["feats_idx"], "parkinson_var_rolling_mean_24"].to_numpy()
    sub["entry_hurst"] = hurst_full.loc[sub["feats_idx"]].to_numpy()
    sub["pk_tercile"] = np.digitize(sub["entry_pk"], pk_edges)
    sub["hurst_tercile"] = np.digitize(sub["entry_hurst"], h_edges)
    sub["cell"] = sub["pk_tercile"].astype(str) + "_" + sub["hurst_tercile"].astype(str)
    print(f"  trades with valid (pk, hurst): {sub.dropna(subset=['entry_hurst']).shape[0]}")

    # Per-cell stats with bootstrap CI.
    print("\n[round 030] per-cell Sharpe with stationary-block-bootstrap CI:")
    rows = []
    for pk_t in [0, 1, 2]:
        for h_t in [0, 1, 2]:
            cell_sub = sub[(sub["pk_tercile"] == pk_t) & (sub["hurst_tercile"] == h_t)]
            cell_sub = cell_sub.dropna(subset=["entry_hurst"])
            n = len(cell_sub)
            if n < 30:
                rows.append({
                    "pk_tercile": pk_t, "hurst_tercile": h_t, "n_trades": n,
                    "sharpe_point": float("nan"),
                    "sharpe_ci_lo": float("nan"),
                    "sharpe_ci_hi": float("nan"),
                    "p_value_vs_zero": float("nan"),
                    "ci_excludes_zero": False,
                })
                continue
            pnl = cell_sub["pnl_log_net"].to_numpy(dtype=float)
            try:
                bl = optimal_block_length(pnl) if len(pnl) >= 32 else 1
                out = bootstrap_sharpe(pnl, n_resamples=2000, block_length=bl,
                                         seed=42, return_samples=True)
                # p-value: 2 * P(samples opposite of point sign)
                samples = out["samples"]
                if out["estimate"] >= 0:
                    p_vs_zero = 2 * float((samples <= 0).mean())
                else:
                    p_vs_zero = 2 * float((samples >= 0).mean())
                p_vs_zero = min(1.0, p_vs_zero)
                ci_excl = bool(out["ci_lo"] > 0 or out["ci_hi"] < 0)
            except Exception as e:
                print(f"  cell ({pk_t}, {h_t}) err: {e}")
                continue
            rows.append({
                "pk_tercile": pk_t, "hurst_tercile": h_t, "n_trades": n,
                "sharpe_point": out["estimate"],
                "sharpe_ci_lo": out["ci_lo"],
                "sharpe_ci_hi": out["ci_hi"],
                "block_length": int(bl),
                "p_value_vs_zero": float(p_vs_zero),
                "ci_excludes_zero": ci_excl,
            })
            print(f"  cell pk={pk_t}, hurst={h_t}: n={n}, "
                  f"Sharpe={out['estimate']:+.4f} [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]  "
                  f"p={p_vs_zero:.3f}  CI-excl-zero={ci_excl}")

    df = pd.DataFrame(rows).dropna(subset=["sharpe_point"])
    # BHY-FDR correction over 9 cells.
    p_vals = df["p_value_vs_zero"].to_numpy()
    df["bh_significant"] = benjamini_hochberg(p_vals, alpha=0.05)
    df.to_csv(OUTDIR / "h182_per_cell_sharpe.csv", index=False)
    print(f"\n[round 030] cells with CI excluding zero: {df['ci_excludes_zero'].sum()}")
    print(f"[round 030] cells significant after BH-FDR alpha=0.05: {df['bh_significant'].sum()}")
    falsifier_passed = bool(df["bh_significant"].any())
    print(f"[round 030] BACKLOG H-182 falsifier "
          f"({'PASSES' if falsifier_passed else 'TRIGGERS'}): "
          f"≥ 1 cell BH-FDR-significant? {falsifier_passed}")

    # Plot: 3x3 heatmap of Sharpe (per-cell).
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap of point Sharpe.
    grid = np.full((3, 3), np.nan)
    grid_n = np.zeros((3, 3), dtype=int)
    grid_sig = np.zeros((3, 3), dtype=bool)
    for _, row in df.iterrows():
        i = int(row["pk_tercile"])
        j = int(row["hurst_tercile"])
        grid[i, j] = row["sharpe_point"]
        grid_n[i, j] = int(row["n_trades"])
        grid_sig[i, j] = bool(row["bh_significant"])

    ax = axes[0]
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-np.nanmax(np.abs(grid)),
                     vmax=np.nanmax(np.abs(grid)), aspect="auto")
    for i in range(3):
        for j in range(3):
            txt = f"S={grid[i,j]:+.3f}\nn={grid_n[i,j]}"
            if grid_sig[i, j]:
                txt += "\n*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                     color="black" if abs(grid[i,j] or 0) < 0.02 else "white")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["low Hurst", "mid Hurst", "high Hurst"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["low pk_var", "mid pk_var", "high pk_var"])
    ax.set_title("H-182 per-cell Sharpe (per-bar; * = BH-FDR significant)")
    plt.colorbar(im, ax=ax)

    # Bar chart with CIs.
    ax = axes[1]
    df_plot = df.copy()
    df_plot["cell_label"] = df_plot.apply(
        lambda r: f"pk{int(r['pk_tercile'])}_h{int(r['hurst_tercile'])}", axis=1,
    )
    y = np.arange(len(df_plot))
    err_lo = df_plot["sharpe_point"] - df_plot["sharpe_ci_lo"]
    err_hi = df_plot["sharpe_ci_hi"] - df_plot["sharpe_point"]
    colors = ["#1B6B33" if s else "#888" for s in df_plot["ci_excludes_zero"]]
    ax.barh(y, df_plot["sharpe_point"], xerr=[err_lo, err_hi], color=colors,
              alpha=0.85, edgecolor="black", linewidth=0.4, capsize=4)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["cell_label"])
    ax.set_xlabel("per-bar Sharpe (95% block-bootstrap CI)")
    ax.set_title("H-182 cell ranking")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Round 030 — H-182 Conditional Sharpe by parkinson × Hurst regime")
    fig.tight_layout()
    fig.savefig(OUTDIR / "h182_panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "030",
        "phase": "D",
        "phase_round": "h182_conditional_sharpe",
        "claim": "H-182 conditional Sharpe by parkinson × Hurst regime",
        "n_cells_finite": int(len(df)),
        "n_cells_ci_excludes_zero": int(df["ci_excludes_zero"].sum()),
        "n_cells_bh_significant": int(df["bh_significant"].sum()),
        "falsifier_passed": bool(falsifier_passed),
        "best_cell": df.loc[df["sharpe_point"].idxmax()].to_dict() if len(df) > 0 else None,
        "worst_cell": df.loc[df["sharpe_point"].idxmin()].to_dict() if len(df) > 0 else None,
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 030] DONE in {headline['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
