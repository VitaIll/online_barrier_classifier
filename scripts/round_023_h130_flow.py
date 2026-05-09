"""Round 023 — H-130 prep + run.

Aggregates `taker_buy_base`, `taker_buy_quote`, `volume`, `quote_volume`
from `data/cleansed_data/BTCUSDT/1m.parquet` to 20m bars, joins with
`bars_20m_features.parquet`, runs the H-130 flow features
(compute_taker_buy_ratio + compute_signed_dollar_flow), reports label
correlation + pairwise correlation with existing features.

The 20m flow aggregation is a sum over 20 1m bars per decision boundary
(strict CONSTITUTION I.6: bar k uses minute bars [k·M, (k+1)·M)). Output
parquet is persisted at `data/model_data/BTCUSDT/bars_20m_flow.parquet`
for future feature_build retrains to include H-130 by default.

Outputs under RESEARCH/diagrams/round_023/.
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

from src.features import (  # noqa: E402
    compute_signed_dollar_flow,
    compute_taker_buy_ratio,
)
from src.utils import chronological_split  # noqa: E402

M = 20
OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_023"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
FLOW_PARQ = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_flow.parquet"


def aggregate_flow_1m_to_20m(min_df: pd.DataFrame, feat_open_times: pd.Series) -> pd.DataFrame:
    """Aggregate taker_buy_base, taker_buy_quote, volume, quote_volume from
    1m to 20m bars by summing over each [k·M, (k+1)·M) window indexed by
    feat_open_times.

    Strict CONSTITUTION I.6: the 20m bar starting at `open_time` aggregates
    the 1m bars whose open_time ∈ [open_time, open_time + 20*60_000).
    """
    needed_cols = ["open_time", "taker_buy_base", "taker_buy_quote",
                    "volume", "quote_volume"]
    missing = [c for c in needed_cols if c not in min_df.columns]
    if missing:
        raise ValueError(f"1m parquet missing columns: {missing}")

    # Sort 1m by open_time.
    mn = min_df[needed_cols].sort_values("open_time").reset_index(drop=True)

    # Build search keys for each 20m bar: [open_time_k, open_time_k + 20*60_000)
    feat_open_ms = feat_open_times.to_numpy(dtype=np.int64)
    bar_window_ms = 20 * 60_000
    end_ms = feat_open_ms + bar_window_ms

    mn_open = mn["open_time"].to_numpy(dtype=np.int64)
    starts = np.searchsorted(mn_open, feat_open_ms, side="left")
    ends = np.searchsorted(mn_open, end_ms, side="left")

    # Cumulative sums for fast range-summing.
    cum_tbb = np.concatenate(([0.0], np.cumsum(mn["taker_buy_base"].to_numpy(dtype=float))))
    cum_tbq = np.concatenate(([0.0], np.cumsum(mn["taker_buy_quote"].to_numpy(dtype=float))))
    cum_v = np.concatenate(([0.0], np.cumsum(mn["volume"].to_numpy(dtype=float))))
    cum_qv = np.concatenate(([0.0], np.cumsum(mn["quote_volume"].to_numpy(dtype=float))))

    out = pd.DataFrame({
        "open_time": feat_open_ms,
        "taker_buy_base": cum_tbb[ends] - cum_tbb[starts],
        "taker_buy_quote": cum_tbq[ends] - cum_tbq[starts],
        "volume": cum_v[ends] - cum_v[starts],
        "quote_volume": cum_qv[ends] - cum_qv[starts],
    })
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 023] ===== H-130 prep: aggregate taker_buy 1m → 20m =====")
    t_main = time.time()

    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    print(f"[round 023] loaded bars_20m_features ({len(feats)})")

    if FLOW_PARQ.exists():
        print(f"[round 023] loading cached aggregation: {FLOW_PARQ}")
        flow_20m = pd.read_parquet(FLOW_PARQ)
    else:
        print("[round 023] aggregating taker_buy 1m → 20m ...")
        t0 = time.time()
        min_df = pd.read_parquet(MIN1, columns=[
            "open_time", "taker_buy_base", "taker_buy_quote",
            "volume", "quote_volume",
        ])
        flow_20m = aggregate_flow_1m_to_20m(min_df, feats["open_time"])
        FLOW_PARQ.parent.mkdir(parents=True, exist_ok=True)
        flow_20m.to_parquet(FLOW_PARQ, index=False)
        print(f"  aggregation took {time.time() - t0:.1f}s; saved to {FLOW_PARQ}")

    # Sanity: row counts match.
    assert len(flow_20m) == len(feats), (
        f"row count mismatch: flow={len(flow_20m)} vs feats={len(feats)}"
    )

    # Compute H-130 features.
    print("\n[round 023] computing H-130 features ...")
    flow_features = compute_taker_buy_ratio(flow_20m, windows=[12, 24, 48, 96])
    sdf_features = compute_signed_dollar_flow(flow_20m, windows=[12, 24, 48])
    new_features = pd.concat([flow_features, sdf_features], axis=1)
    print(f"  computed {new_features.shape[1]} new features")

    # Test-slice diagnostics.
    test_idx = test_full.index
    new_test = new_features.loc[test_idx]
    label = test_full["label"]

    summary_rows = []
    from scipy.stats import pearsonr, spearmanr
    for col in new_test.columns:
        x = new_test[col].to_numpy(dtype=float)
        y = label.to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            continue
        x_m = x[mask]
        y_m = y[mask]
        try:
            pr = pearsonr(x_m, y_m)[0]
        except Exception:
            pr = float("nan")
        try:
            sr = spearmanr(x_m, y_m)[0]
        except Exception:
            sr = float("nan")
        summary_rows.append({
            "feature": col, "n_finite": int(mask.sum()),
            "mean": float(np.mean(x_m)), "std": float(np.std(x_m)),
            "q10": float(np.quantile(x_m, 0.1)),
            "median": float(np.quantile(x_m, 0.5)),
            "q90": float(np.quantile(x_m, 0.9)),
            "pearson_with_label": float(pr),
            "spearman_with_label": float(sr),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "h130_feature_summary.csv", index=False)
    print("\n[round 023] H-130 feature summary on test slice:")
    print(summary_df[["feature", "n_finite", "mean", "std", "q10", "median",
                       "q90", "pearson_with_label", "spearman_with_label"]].to_string(index=False))

    # Pairwise corr with existing top features.
    EXISTING_TOP = [
        "minute_sin", "return_rolling_mean_8", "return_rolling_mean_12",
        "minute_cos", "parkinson_var_rolling_mean_2",
        "parkinson_var_rolling_mean_24", "return", "high", "low", "close",
    ]
    existing_in_test = [c for c in EXISTING_TOP if c in test_full.columns]
    pair_rows = []
    for new_col in new_test.columns:
        for old_col in existing_in_test:
            x = new_test[new_col].to_numpy(dtype=float)
            y = test_full[old_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 100:
                continue
            try:
                rho = pearsonr(x[mask], y[mask])[0]
            except Exception:
                rho = float("nan")
            pair_rows.append({
                "new_feature": new_col, "existing_feature": old_col,
                "pearson": float(rho),
                "abs_pearson": float(abs(rho)) if np.isfinite(rho) else float("nan"),
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUTDIR / "pairwise_corr_with_existing.csv", index=False)
    if not pair_df.empty:
        print("\n[round 023] top 10 absolute pairwise corr with existing top features:")
        print(pair_df.sort_values("abs_pearson", ascending=False).head(10).to_string(index=False))

    # Plot: distributions.
    n_features = len(new_test.columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for i, col in enumerate(new_test.columns):
        ax = axes[i // n_cols, i % n_cols]
        x = new_test[col].dropna()
        if len(x) > 0:
            x_clipped = x.clip(x.quantile(0.001), x.quantile(0.999))
            ax.hist(x_clipped, bins=40, color="#3b7dd8", alpha=0.85,
                     edgecolor="white", linewidth=0.4)
            ax.set_title(col, fontsize=8)
            ax.grid(alpha=0.3)
    for i in range(n_features, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")
    fig.suptitle("Round 023 — H-130 derived flow feature distributions on test slice")
    fig.tight_layout()
    fig.savefig(OUTDIR / "h130_feature_distributions.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "023",
        "phase": "D",
        "phase_round": "h130_flow_features",
        "claim": "H-130 prep + compute on real BTCUSDT data",
        "n_features": int(n_features),
        "summary_top5_by_abs_pearson": (
            summary_df.assign(abs_pearson=summary_df["pearson_with_label"].abs())
                       .sort_values("abs_pearson", ascending=False)
                       .head(5)
                       .drop(columns=["abs_pearson"])
                       .to_dict(orient="records")
        ),
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 023] DONE in {headline['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
