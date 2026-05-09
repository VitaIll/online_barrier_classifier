"""H-106 diagnostic: per-regime calibration curves on the real test stream.

Joins `artifacts/online_eval/predictions.parquet` (y_true, p_offline,
p_online) with `data/model_data/BTCUSDT/bars_20m_features.parquet`
(parkinson_var_rolling_mean_24) on the test slice, buckets the regime
signal into terciles, and produces:

- 2x3 grid of per-regime calibration curves (rows: p_offline, p_online;
  columns: low/med/high vol regimes), with ECE annotation and reference
  diagonal.
- Bottom-row text panel with per-regime base rate / mean predicted /
  Brier / ECE for each predictor.
- A sidecar JSON `headline.json` with the full per-regime metric dict
  for later consumption (LEDGER row, follow-up rounds).

This is the first round to use the H-106 utilities on real data and
gives an honest diagnostic of where the offline/online layers are mis-
calibrated by regime — the gap H-202..H-208 must close.

Outputs:
  RESEARCH/diagrams/round_005/calibration_by_regime.png
  RESEARCH/diagrams/round_005/headline.json
  RESEARCH/diagrams/round_005/threshold_sweep.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils import (  # noqa: E402
    DEFAULT_REGIME_LABELS,
    calibration_by_regime,
    chronological_split,
    compute_all_metrics,
    threshold_analysis,
)

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_005"
PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
REGIME_COL = "parkinson_var_rolling_mean_24"


def _load_test_slice() -> pd.DataFrame:
    preds = pd.read_parquet(PRED, columns=["k", "y_true", "p_offline", "p_online"])
    feat_cols = ["open_time", REGIME_COL]
    feats = pd.read_parquet(FEAT, columns=feat_cols)
    # The test split is the last 31486 rows of the feature parquet (matches
    # round-001's setup; verified by chronological_split row counts).
    _, _, test_feats = chronological_split(feats, train_fraction=0.6, val_fraction=0.2)
    test_feats = test_feats.reset_index(drop=True)
    if len(test_feats) != len(preds):
        raise RuntimeError(
            f"row-count mismatch: predictions n={len(preds)}, test_feats n={len(test_feats)}"
        )
    out = preds.copy()
    out[REGIME_COL] = test_feats[REGIME_COL].to_numpy()
    return out


def _calibration_curve(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Return (mean_predicted_per_bin, mean_observed_per_bin, count_per_bin)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    means_p, means_y, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        means_p.append(float(p[mask].mean()))
        means_y.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(means_p), np.array(means_y), np.array(counts)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = _load_test_slice()
    y = df["y_true"].to_numpy().astype(int)
    vol = df[REGIME_COL].to_numpy()

    # Bucket once so panels share regime membership.
    buckets = pd.qcut(vol, 3, labels=list(DEFAULT_REGIME_LABELS))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    headline = {
        "round_id": "005",
        "hypothesis": "H-106",
        "claim": "regime-stratified calibration helpers land + first per-regime calibration plot on real test stream",
        "n_test": int(len(df)),
        "regime_signal": REGIME_COL,
        "n_regimes": 3,
        "predictors": {},
    }

    for row, predictor in enumerate(["p_offline", "p_online"]):
        p = df[predictor].to_numpy()
        global_metrics = compute_all_metrics(y, p)
        per_regime = calibration_by_regime(y, p, vol, n_bins=10, n_regimes=3)
        headline["predictors"][predictor] = {
            "global": global_metrics,
            "per_regime": per_regime,
        }
        for col, regime in enumerate(DEFAULT_REGIME_LABELS):
            ax = axes[row, col]
            mask = np.asarray(buckets == regime)
            mp, my, counts = _calibration_curve(y[mask], p[mask], n_bins=10)
            stats = per_regime.get(regime, {})
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=0.8)
            if len(mp):
                # Marker size scales with bin count.
                sizes = 30 + 80 * (counts / counts.max())
                ax.scatter(mp, my, s=sizes, color="#3b7dd8" if row == 0 else "#56b870",
                           edgecolor="black", alpha=0.85, zorder=3)
                ax.plot(mp, my, color="#3b7dd8" if row == 0 else "#56b870",
                        alpha=0.6, linewidth=1.2, zorder=2)
            ax.set_xlim(0, max(0.4, mp.max() if len(mp) else 0.4))
            ax.set_ylim(0, max(0.4, my.max() if len(my) else 0.4))
            ax.grid(True, alpha=0.3)
            base = stats.get("base_rate", float("nan"))
            mean_p = stats.get("mean_predicted", float("nan"))
            ece = stats.get("ece", float("nan"))
            ax.set_title(
                f"{predictor} · {regime}-vol\n"
                f"n={int(stats.get('n_samples', 0)):,}  "
                f"base={base:.3f}  mean_p={mean_p:.3f}  ECE={ece:.3f}",
                fontsize=10,
            )
            if row == 1:
                ax.set_xlabel("mean predicted probability")
            if col == 0:
                ax.set_ylabel("observed positive rate")

    fig.suptitle(
        "Per-regime calibration on real test stream (n=31,486; "
        "regime = parkinson_var_rolling_mean_24 terciles)\n"
        "Top: offline CatBoost ; Bottom: online ARF (post-conformal layer)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = OUTDIR / "calibration_by_regime.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Threshold analysis CSV side product.
    rows = []
    for predictor in ["p_offline", "p_online"]:
        p = df[predictor].to_numpy()
        ts = threshold_analysis(y, p, thresholds=np.linspace(0.0, 0.6, 25))
        ts["predictor"] = predictor
        rows.append(ts)
    pd.concat(rows, ignore_index=True).to_csv(OUTDIR / "threshold_sweep.csv", index=False)

    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )

    print(f"saved: {out_png}")
    # Compact summary to stdout — full JSON is too verbose.
    for predictor, blk in headline["predictors"].items():
        print(f"  {predictor}: global ECE={blk['global']['ece']:.4f}  Brier={blk['global']['brier_score']:.4f}")
        for regime in DEFAULT_REGIME_LABELS:
            r = blk["per_regime"].get(regime, {})
            if r:
                print(
                    f"    {regime:>4} n={r['n_samples']:5d}  base={r['base_rate']:.3f}  "
                    f"mean_p={r['mean_predicted']:.3f}  ECE={r['ece']:.4f}  Brier={r['brier']:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
