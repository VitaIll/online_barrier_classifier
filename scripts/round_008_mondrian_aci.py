"""H-203 diagnostic: Mondrian-ACI vs plain ACI vs Mondrian-LAC on real stream.

Three-way comparison of per-regime coverage gap on the test slice:

  - Plain ACI  (round 007): single global q_t, online update.
  - Mondrian-ACI (THIS round): per-regime q_t, online update per regime.
  - Mondrian-LAC (round 002): per-regime q_hat from a held-out cal set (batch).

Setup matches round-002 / round-007:
  - Same chronological 30/70 cal/eval split.
  - Same regime signal: parkinson_var_rolling_mean_24 terciles.
  - Tercile boundaries fit on the EVAL slice (same as round 002 / 007).

Falsifier: at α=0.10, Mondrian-ACI per-regime gap |target − empirical| ≤ 5pp
on every regime (tighter than plain ACI's ±6pp).

Outputs:
  RESEARCH/diagrams/round_008/per_regime_q_trajectories.png
  RESEARCH/diagrams/round_008/three_way_gap_comparison.png
  RESEARCH/diagrams/round_008/headline.json
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

from src.conformal import (  # noqa: E402
    aci_mondrian_stream,
    aci_stream,
)
from src.utils import (  # noqa: E402
    DEFAULT_REGIME_LABELS,
    chronological_split,
)

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_008"
PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
REGIME_COL = "parkinson_var_rolling_mean_24"

CAL_FRAC = 0.30
ALPHAS = (0.05, 0.10, 0.20)
GAMMA = 0.01
Q_INIT_DEFAULT = 0.5


def _load_with_regime() -> pd.DataFrame:
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


def _split_cal_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_cal = int(round(len(df) * CAL_FRAC))
    return df.iloc[:n_cal].copy(), df.iloc[n_cal:].copy()


def _per_regime_coverage(
    sets: np.ndarray,
    y: np.ndarray,
    regime: np.ndarray,
    labels: tuple[str, ...] = DEFAULT_REGIME_LABELS,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in labels:
        mask = np.asarray(regime == r)
        if mask.sum() == 0:
            continue
        sub_sets = sets[mask]
        sub_y = y[mask]
        cov = float(np.mean([sub_sets[i, sub_y[i]] for i in range(len(sub_y))]))
        out[r] = cov
    return out


def _warm_q_per_regime(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    regime_cal: np.ndarray,
    alpha: float,
    gamma: float,
) -> dict[str, float]:
    """Run plain ACI on EACH regime's cal substream to warm its q_init.

    This gives Mondrian-ACI a calibrated starting q per regime, fair to
    round-002 Mondrian-LAC (which gets per-regime q_hat from cal).
    """
    out: dict[str, float] = {}
    for r in DEFAULT_REGIME_LABELS:
        mask = np.asarray(regime_cal == r)
        if mask.sum() < 50:
            out[r] = Q_INIT_DEFAULT
            continue
        warm = aci_stream(
            p_cal[mask], y_cal[mask], alpha=alpha, gamma=gamma, q_init=Q_INIT_DEFAULT,
        )
        out[r] = float(warm["q_history"][-1])
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = _load_with_regime()
    cal, ev = _split_cal_eval(df)

    # Regime tagging on each split (terciles fit ONCE on full df for determinism).
    full_buckets = pd.qcut(df[REGIME_COL].to_numpy(), 3, labels=list(DEFAULT_REGIME_LABELS))
    cal_buckets = full_buckets[: len(cal)].to_numpy().astype(str)
    ev_buckets = full_buckets[len(cal):].to_numpy().astype(str)

    headline = {
        "round_id": "008",
        "hypothesis": "H-203",
        "claim": "Mondrian-ACI closes the per-regime coverage gap that plain ACI leaves open under regime drift",
        "n_cal": len(cal),
        "n_eval": len(ev),
        "alphas": list(ALPHAS),
        "gamma": GAMMA,
        "results": {},
    }

    # --- Plot 1: per-regime q trajectories (just for α=0.10, both predictors) ---
    fig_q, axes_q = plt.subplots(2, 3, figsize=(14, 6.5), sharey=True)

    # --- Plot 2: 3-way gap comparison (α=0.10 + 0.20) ---
    fig_g, axes_g = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Round-002 Mondrian-LAC numbers (from LEDGER, signed gaps target − empirical)
    round_002_mondrian_gap = {
        ("p_offline", 0.10): {"low": +0.034, "med": -0.012, "high": -0.005},
        ("p_online", 0.10): {"low": +0.020, "med": -0.005, "high": -0.034},
        ("p_offline", 0.20): {"low": -0.058, "med": -0.020, "high": -0.010},
        ("p_online", 0.20): {"low": -0.079, "med": -0.025, "high": -0.012},
    }

    for predictor_idx, (predictor, color_off) in enumerate([
        ("p_offline", "#3b7dd8"),
        ("p_online", "#56b870"),
    ]):
        p_cal = cal[predictor].to_numpy()
        y_cal = cal["y_true"].to_numpy().astype(int)
        p_ev = ev[predictor].to_numpy()
        y_ev = ev["y_true"].to_numpy().astype(int)

        per_predictor: dict = {"alpha": {}}

        for alpha in ALPHAS:
            # Plain ACI (round-007 setup): single q warmed on full cal stream.
            warm_global = aci_stream(
                p_cal, y_cal, alpha=alpha, gamma=GAMMA, q_init=Q_INIT_DEFAULT,
            )
            q_global_init = float(warm_global["q_history"][-1])
            plain = aci_stream(
                p_ev, y_ev, alpha=alpha, gamma=GAMMA, q_init=q_global_init,
            )

            # Mondrian-ACI: per-regime q warmed on per-regime cal substreams.
            q_init_by_regime = _warm_q_per_regime(
                p_cal, y_cal, cal_buckets, alpha=alpha, gamma=GAMMA,
            )
            mondrian = aci_mondrian_stream(
                p_ev, y_ev, ev_buckets, alpha=alpha, gamma=GAMMA,
                q_init=Q_INIT_DEFAULT, q_init_by_regime=q_init_by_regime,
            )

            target = 1.0 - alpha
            plain_per_regime_cov = _per_regime_coverage(
                plain["sets"], y_ev, ev_buckets, DEFAULT_REGIME_LABELS,
            )
            mondrian_per_regime_cov = _per_regime_coverage(
                mondrian["sets"], y_ev, ev_buckets, DEFAULT_REGIME_LABELS,
            )

            plain_gap = {r: target - c for r, c in plain_per_regime_cov.items()}
            mondrian_gap = {r: target - c for r, c in mondrian_per_regime_cov.items()}

            per_predictor["alpha"][f"{alpha:.2f}"] = {
                "plain_aci_marginal": plain["coverage"],
                "mondrian_aci_marginal": mondrian["coverage"],
                "plain_aci_per_regime_gap": plain_gap,
                "mondrian_aci_per_regime_gap": mondrian_gap,
                "mondrian_aci_q_init_by_regime": q_init_by_regime,
                "mondrian_aci_q_final_by_regime": {
                    r: float(traj[-1]) if len(traj) else q_init_by_regime[r]
                    for r, traj in mondrian["q_history_by_regime"].items()
                },
            }

            # Plot 1 — only for α=0.10
            if alpha == 0.10:
                for regime_idx, regime in enumerate(DEFAULT_REGIME_LABELS):
                    ax = axes_q[predictor_idx, regime_idx]
                    # Plain ACI restricted to this regime's slice (subset of global q)
                    mask = np.asarray(ev_buckets == regime)
                    if mask.sum() == 0:
                        ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
                        continue
                    plain_q_in_regime = plain["q_history"][mask]
                    mondrian_traj = mondrian["q_history_by_regime"].get(regime, [])
                    ax.plot(plain_q_in_regime, color="gray", alpha=0.7, linewidth=0.8,
                            label="plain ACI (global q)")
                    ax.plot(mondrian_traj, color=color_off, alpha=0.95, linewidth=1.0,
                            label="Mondrian-ACI (per-regime q)")
                    ax.set_title(
                        f"{predictor} · {regime}-vol  (n={int(mask.sum()):,})",
                        fontsize=9,
                    )
                    if regime_idx == 0:
                        ax.set_ylabel("q_t")
                    if predictor_idx == 1:
                        ax.set_xlabel("step within regime")
                    ax.grid(True, alpha=0.3)
                    if predictor_idx == 0 and regime_idx == 2:
                        ax.legend(loc="best", fontsize=8)

            # Plot 2 — α=0.10 (left) and α=0.20 (right)
            if alpha in (0.10, 0.20):
                ax_idx = 0 if alpha == 0.10 else 1
                ax = axes_g[ax_idx]
                regimes = list(DEFAULT_REGIME_LABELS)
                xs = np.arange(len(regimes))
                offset = -0.30 if predictor == "p_offline" else +0.04
                width = 0.08

                aci_vals = [plain_gap.get(r, 0) for r in regimes]
                m_aci_vals = [mondrian_gap.get(r, 0) for r in regimes]
                m_lac_vals = [
                    round_002_mondrian_gap[(predictor, alpha)][r] for r in regimes
                ]
                color_aci = color_off
                ax.bar(xs + offset + 0 * width, aci_vals, width=width,
                       color=color_aci, alpha=0.4,
                       label=f"plain ACI · {predictor}" if alpha == 0.10 else None)
                ax.bar(xs + offset + 1 * width, m_aci_vals, width=width,
                       color=color_aci, alpha=1.0,
                       label=f"Mondrian-ACI · {predictor}" if alpha == 0.10 else None)
                ax.bar(xs + offset + 2 * width, m_lac_vals, width=width,
                       color=color_aci, hatch="///", edgecolor="black", alpha=0.5,
                       label=f"Mondrian-LAC R002 · {predictor}" if alpha == 0.10 else None)

        headline["results"][predictor] = per_predictor

    # Plot 1 dressing
    fig_q.suptitle(
        "Per-regime q_t trajectory — Mondrian-ACI vs plain-ACI restricted to regime  (α=0.10, γ=0.01)",
        y=1.02, fontsize=11,
    )
    fig_q.tight_layout()
    out_q = OUTDIR / "per_regime_q_trajectories.png"
    fig_q.savefig(out_q, dpi=110, bbox_inches="tight")
    plt.close(fig_q)

    # Plot 2 dressing
    for ax_idx, alpha in enumerate((0.10, 0.20)):
        ax = axes_g[ax_idx]
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(0.05, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(len(DEFAULT_REGIME_LABELS)))
        ax.set_xticklabels(list(DEFAULT_REGIME_LABELS))
        ax.set_title(f"per-regime gap (target − empirical) at α={alpha:.2f}")
        ax.set_xlabel("vol regime")
        if ax_idx == 0:
            ax.set_ylabel("gap (positive = under-coverage)")
        ax.grid(True, alpha=0.3, axis="y")
    axes_g[0].legend(loc="best", fontsize=7, ncol=1)
    fig_g.suptitle(
        "Three-way per-regime gap comparison: plain ACI vs Mondrian-ACI vs round-002 Mondrian-LAC",
        y=1.02, fontsize=11,
    )
    fig_g.tight_layout()
    out_g = OUTDIR / "three_way_gap_comparison.png"
    fig_g.savefig(out_g, dpi=110, bbox_inches="tight")
    plt.close(fig_g)

    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )

    print(f"saved: {out_q}")
    print(f"saved: {out_g}")
    print()
    for predictor, blk in headline["results"].items():
        print(f"  {predictor}:")
        for alpha_key, m in blk["alpha"].items():
            print(
                f"    alpha={alpha_key}  marginal: plain={m['plain_aci_marginal']:.4f}  "
                f"mondrian={m['mondrian_aci_marginal']:.4f}"
            )
            for r in DEFAULT_REGIME_LABELS:
                pg = m["plain_aci_per_regime_gap"].get(r, float("nan"))
                mg = m["mondrian_aci_per_regime_gap"].get(r, float("nan"))
                print(
                    f"      {r:>4}  plain gap={pg:+.4f}  mondrian gap={mg:+.4f}  "
                    f"(improvement: {abs(pg) - abs(mg):+.4f})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
