"""Round 022 — H-040 + H-115 + H-130 + H-131 feature computation + smoke test
on real BTCUSDT 20m bars.

Computes the new compute_* functions on the actual cleansed/feature-built
data and reports:
- distribution stats (mean, std, q10/50/90)
- pointwise label correlation (Pearson + Spearman) on the test slice
- pairwise correlation with existing top-importance features

Does NOT retrain CatBoost — paired retrain is a separate (expensive) round.
This round documents the new features so a future round can pick which
ones to include in a retrain. Also serves as a smoke test that the
features work on real data with realistic shapes.

Outputs under RESEARCH/diagrams/round_022/.
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
    compute_bpv_ratio,
    compute_dfa_alpha,
    compute_hurst_rs,
    compute_permutation_entropy,
    compute_sample_entropy,
    compute_signed_dollar_flow,
    compute_signed_semivariance,
    compute_taker_buy_ratio,
    compute_vol_of_vol,
)
from src.utils import chronological_split  # noqa: E402

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_022"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 022] ===== H-040 + H-115 + H-130 + H-131 features on real data =====")
    t_main = time.time()

    feats = pd.read_parquet(FEAT)
    print(f"[round 022] loaded bars_20m_features: shape={feats.shape}  cols[:5]={list(feats.columns[:5])}")

    # Chronological split.
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )

    # Find the log-return column.
    return_col = None
    for cand in ["log_return", "ret_close", "return", "log_ret_close"]:
        if cand in feats.columns:
            return_col = cand
            break
    if return_col is None:
        # Construct from close.
        if "close" in feats.columns:
            print("[round 022] constructing log_return from close ...")
            feats = feats.copy()
            feats["log_return"] = np.log(feats["close"] / feats["close"].shift(1))
            return_col = "log_return"
        else:
            raise RuntimeError("No return column found and no 'close' column.")

    log_r = feats[return_col]
    print(f"[round 022] using return col '{return_col}' (n={log_r.notna().sum()})")

    # H-040: Hurst R/S, DFA, sample entropy on rolling windows.
    print("\n[round 022] H-040: Hurst R/S, DFA, sample entropy ...")
    t0 = time.time()
    hurst = compute_hurst_rs(log_r.fillna(0.0), window=128)
    dfa = compute_dfa_alpha(log_r.fillna(0.0), window=128)
    sampen = compute_sample_entropy(log_r.fillna(0.0), window=128, m=2, r_factor=0.2)
    print(f"  H-040 took {time.time() - t0:.1f}s")

    # H-115: Permutation entropy.
    print("\n[round 022] H-115: permutation entropy ...")
    t0 = time.time()
    pe = compute_permutation_entropy(log_r.fillna(0.0), window=128, m=3, tau=1)
    print(f"  H-115 took {time.time() - t0:.1f}s")

    # H-130: derived flow features (need taker_buy_base, volume).
    flow_df = pd.DataFrame()
    sdf_df = pd.DataFrame()
    if {"taker_buy_base", "volume"}.issubset(feats.columns):
        print("\n[round 022] H-130: derived flow features ...")
        flow_df = compute_taker_buy_ratio(feats, windows=[12, 24, 48])
        if {"taker_buy_quote", "quote_volume"}.issubset(feats.columns):
            sdf_df = compute_signed_dollar_flow(feats, windows=[12, 24])
    else:
        print("\n[round 022] H-130 skipped: taker_buy_base/volume columns absent")

    # H-131: BPV/RV ratio + signed semivariance + vov.
    print("\n[round 022] H-131: BPV/RV ratio + SV asymmetry + vov ...")
    bpv_df = compute_bpv_ratio(log_r.fillna(0.0), windows=[12, 24, 48])
    sv_df = compute_signed_semivariance(log_r.fillna(0.0), windows=[12, 24, 48])
    vov = compute_vol_of_vol(log_r.fillna(0.0), vol_window=24, vov_window=24)

    # Combine new features into one DataFrame.
    new_features = pd.concat([
        hurst.rename("hurst_rs_128"),
        dfa.rename("dfa_alpha_128"),
        sampen.rename("sampen_128_m2"),
        pe.rename("permen_128_m3_tau1"),
        flow_df,
        sdf_df,
        bpv_df,
        sv_df,
        vov.rename("vov_24_24"),
    ], axis=1)
    print(f"[round 022] new feature panel: shape={new_features.shape}  cols={list(new_features.columns)}")

    # Distribution stats on test slice.
    test_df_slice = test_full.copy()
    test_idx = test_df_slice.index
    new_features_test = new_features.loc[test_idx]
    label = test_df_slice["label"] if "label" in test_df_slice.columns else None

    summary_rows = []
    if label is not None:
        from scipy.stats import pearsonr, spearmanr
        for col in new_features_test.columns:
            x = new_features_test[col].to_numpy(dtype=float)
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
                "feature": col,
                "n_finite": int(mask.sum()),
                "mean": float(np.mean(x_m)),
                "std": float(np.std(x_m)),
                "q10": float(np.quantile(x_m, 0.1)),
                "median": float(np.quantile(x_m, 0.5)),
                "q90": float(np.quantile(x_m, 0.9)),
                "pearson_with_label": float(pr),
                "spearman_with_label": float(sr),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "feature_distributions.csv", index=False)
    print("\n[round 022] feature distributions on test slice:")
    if not summary_df.empty:
        print(summary_df[["feature", "n_finite", "mean", "std", "q10", "median",
                            "q90", "pearson_with_label", "spearman_with_label"]].to_string(index=False))

    # Pairwise correlation with existing top-importance features (if available).
    EXISTING_TOP = [
        "minute_sin", "return_rolling_mean_8", "return_rolling_mean_12",
        "minute_cos", "parkinson_var_rolling_mean_2",
        "parkinson_var_rolling_mean_24",
    ]
    existing_in_test = [c for c in EXISTING_TOP if c in test_df_slice.columns]
    print(f"\n[round 022] checking pairwise corr with existing top-importance features: {existing_in_test}")
    pair_rows = []
    for new_col in new_features_test.columns:
        for old_col in existing_in_test:
            x = new_features_test[new_col].to_numpy(dtype=float)
            y = test_df_slice[old_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 100:
                continue
            try:
                from scipy.stats import pearsonr
                rho = pearsonr(x[mask], y[mask])[0]
            except Exception:
                rho = float("nan")
            pair_rows.append({
                "new_feature": new_col,
                "existing_feature": old_col,
                "pearson": float(rho),
                "abs_pearson": float(abs(rho)) if np.isfinite(rho) else float("nan"),
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUTDIR / "pairwise_corr_with_existing.csv", index=False)
    if not pair_df.empty:
        print("\n[round 022] top 10 absolute pairwise correlations (new vs existing):")
        print(pair_df.sort_values("abs_pearson", ascending=False).head(10).to_string(index=False))

    # Plot: distributions.
    n_features = len(new_features_test.columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for i, col in enumerate(new_features_test.columns):
        ax = axes[i // n_cols, i % n_cols]
        x = new_features_test[col].dropna()
        if len(x) > 0:
            x_clipped = x.clip(x.quantile(0.001), x.quantile(0.999))
            ax.hist(x_clipped, bins=40, color="#3b7dd8", alpha=0.85, edgecolor="white",
                     linewidth=0.4)
            ax.set_title(col, fontsize=8)
            ax.grid(alpha=0.3)
    # Blank unused subplots.
    for i in range(n_features, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")
    fig.suptitle("Round 022 — new H-040/115/130/131 feature distributions on test slice")
    fig.tight_layout()
    fig.savefig(OUTDIR / "feature_distributions.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "022",
        "phase": "D",
        "phase_round": "feature_compute_smoke",
        "claim": "H-040 + H-115 + H-130 + H-131 compute_* functions land + smoke run on real BTCUSDT data",
        "n_features_added": int(n_features),
        "n_test_samples": int(len(test_df_slice)),
        "feature_summary_top5_by_abs_pearson": (
            summary_df.assign(abs_pearson=summary_df["pearson_with_label"].abs())
                       .sort_values("abs_pearson", ascending=False)
                       .head(5)
                       .drop(columns=["abs_pearson"])
                       .to_dict(orient="records")
            if not summary_df.empty else []
        ),
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 022] DONE in {headline['wall_clock_s']:.1f}s; "
          f"new features: {n_features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
