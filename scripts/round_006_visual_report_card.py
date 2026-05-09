"""H-107 diagnostic: end-to-end visual report card using the new plotting helpers.

Exercises every helper in src/plotting.py against real data:

- 2x2 grid: overall calibration (offline + online) + threshold curves +
  feature-importance bar chart (top-25 from the persisted offline model).
- Separate per-regime calibration figure using `plot_calibration_by_regime`,
  to confirm parity with round 005's hand-rolled per-regime plot.

The point: if round 005's per-regime numbers replicate via the new helpers,
the helpers are wired correctly; the helpers can then go into the
notebooks and future rounds without re-implementing the same plotting
boilerplate every time.

Outputs:
  RESEARCH/diagrams/round_006/visual_report_card.png   (2x2 overview)
  RESEARCH/diagrams/round_006/calibration_by_regime_via_helper.png
  RESEARCH/diagrams/round_006/headline.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.plotting import (  # noqa: E402
    plot_calibration_by_regime,
    plot_calibration_curve,
    plot_feature_importance,
    plot_threshold_curves,
)
from src.utils import (  # noqa: E402
    chronological_split,
    compute_all_metrics,
    threshold_analysis,
)

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_006"
PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
META = REPO / "data" / "model_data" / "BTCUSDT" / "feature_metadata.json"
SELECTED = REPO / "artifacts" / "offline_model" / "selected_features.json"
MODEL = REPO / "artifacts" / "offline_model" / "model.cbm"
REGIME_COL = "parkinson_var_rolling_mean_24"


def _load_predictions_with_regime() -> pd.DataFrame:
    preds = pd.read_parquet(PRED, columns=["k", "y_true", "p_offline", "p_online"])
    feats = pd.read_parquet(FEAT, columns=["open_time", REGIME_COL])
    _, _, test_feats = chronological_split(feats, train_fraction=0.6, val_fraction=0.2)
    test_feats = test_feats.reset_index(drop=True)
    if len(test_feats) != len(preds):
        raise RuntimeError(
            f"row mismatch: predictions n={len(preds)}, test_feats n={len(test_feats)}"
        )
    out = preds.copy()
    out[REGIME_COL] = test_feats[REGIME_COL].to_numpy()
    return out


def _load_offline_model_importances():
    """Return (importances, feature_names) for the persisted offline model.

    Prefer artifacts/offline_model/feature_importance.csv if present (it's
    written by offline_train.ipynb and avoids the catboost-import dance);
    fall back to loading model.cbm + selected_features.json otherwise.
    """
    fi_csv = REPO / "artifacts" / "offline_model" / "feature_importance.csv"
    if fi_csv.exists():
        df = pd.read_csv(fi_csv)
        # Heuristic: pick the first object/string col as feature, first numeric as importance.
        feat_col = next(c for c in df.columns if df[c].dtype == object)
        imp_col = next(c for c in df.columns if c != feat_col and np.issubdtype(df[c].dtype, np.number))
        return df[imp_col].to_numpy(dtype=float), df[feat_col].tolist()
    # Fallback path
    from catboost import CatBoostClassifier
    if not SELECTED.exists():
        raise FileNotFoundError(
            f"selected features not at {SELECTED}; can't load offline model importances"
        )
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    feature_names = selected if isinstance(selected, list) else selected.get("feature_names", [])
    m = CatBoostClassifier()
    m.load_model(str(MODEL))
    return np.asarray(m.get_feature_importance(), dtype=float), list(feature_names)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = _load_predictions_with_regime()
    y = df["y_true"].to_numpy().astype(int)
    p_off = df["p_offline"].to_numpy()
    p_on = df["p_online"].to_numpy()

    importances, feat_names = _load_offline_model_importances()

    # --- 2x2 visual report card ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    plot_calibration_curve(y, p_off, ax=axes[0, 0], label="offline", color="#3b7dd8")
    axes[0, 0].set_title(f"Offline calibration · {axes[0, 0].get_title()}")

    plot_calibration_curve(y, p_on, ax=axes[0, 1], label="online", color="#56b870")
    axes[0, 1].set_title(f"Online calibration · {axes[0, 1].get_title()}")

    plot_feature_importance(
        importances, feat_names, top_n=25, ax=axes[1, 0],
        title=f"Offline model: top 25 / {len(feat_names)} features",
    )

    ts = threshold_analysis(y, p_off)
    plot_threshold_curves(ts, ax=axes[1, 1], title="Threshold sweep · offline (p_offline)")

    fig.suptitle(
        f"Round 006 visual report card — n_test={len(df):,}, "
        f"base_rate={y.mean():.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_main = OUTDIR / "visual_report_card.png"
    fig.savefig(out_main, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Per-regime calibration via helper (parity vs round 005) ---
    vol = df[REGIME_COL].to_numpy()
    fig_off = plot_calibration_by_regime(y, p_off, vol, n_bins=10, n_regimes=3)
    fig_off.suptitle("Offline · per-regime calibration (via plot_calibration_by_regime helper)",
                     fontsize=11, y=1.02)
    out_reg_off = OUTDIR / "calibration_by_regime_offline.png"
    fig_off.savefig(out_reg_off, dpi=110, bbox_inches="tight")
    plt.close(fig_off)

    fig_on = plot_calibration_by_regime(y, p_on, vol, n_bins=10, n_regimes=3)
    fig_on.suptitle("Online · per-regime calibration (via plot_calibration_by_regime helper)",
                    fontsize=11, y=1.02)
    out_reg_on = OUTDIR / "calibration_by_regime_online.png"
    fig_on.savefig(out_reg_on, dpi=110, bbox_inches="tight")
    plt.close(fig_on)

    headline = {
        "round_id": "006",
        "hypothesis": "H-107",
        "claim": "plot helpers in src/plotting.py produce parity diagnostics and pass smoke tests on real data",
        "n_test": int(len(df)),
        "base_rate": float(y.mean()),
        "offline_global": compute_all_metrics(y, p_off),
        "online_global": compute_all_metrics(y, p_on),
        "top_5_features": [
            {"name": str(feat_names[i]), "importance": float(importances[i])}
            for i in np.argsort(importances)[::-1][:5]
        ],
        "outputs": [str(out_main.relative_to(REPO)),
                    str(out_reg_off.relative_to(REPO)),
                    str(out_reg_on.relative_to(REPO))],
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    print(f"saved: {out_main}")
    print(f"saved: {out_reg_off}")
    print(f"saved: {out_reg_on}")
    print(json.dumps({k: v for k, v in headline.items() if k != "top_5_features"},
                     indent=2))
    print("top_5_features:")
    for f in headline["top_5_features"]:
        print(f"  {f['name']:60s}  {f['importance']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
