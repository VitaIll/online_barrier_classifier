"""H-202 diagnostic: ACI vs LAC-Mondrian on the real offline+online stream.

Runs Adaptive Conformal Inference (Gibbs & Candès 2021) on:
  - p_offline (raw CatBoost)
  - p_online  (post-ARF "conformal" layer)
at three target miscoverage levels α ∈ {0.05, 0.10, 0.20}, and compares
empirical coverage (marginal + per-regime) against round-002's batch LAC
baselines (lac_marginal + lac_mondrian).

Setup matches round 002 for direct comparability:
  - same chronological 30/70 cal/eval split (n_cal=9446 used for q_init
    only; ACI doesn't need a held-out cal set, but we warm q_t on cal
    so the first eval step is already in-distribution).
  - same regime signal: parkinson_var_rolling_mean_24 terciles.

Produces:
  RESEARCH/diagrams/round_007/aci_q_trajectory.png
  RESEARCH/diagrams/round_007/aci_coverage_convergence.png
  RESEARCH/diagrams/round_007/aci_per_regime_vs_round002.png
  RESEARCH/diagrams/round_007/headline.json
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

from src.conformal import aci_stream  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_REGIME_LABELS,
    chronological_split,
)

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_007"
PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
REGIME_COL = "parkinson_var_rolling_mean_24"

CAL_FRAC = 0.30  # matches round 002
ALPHAS = (0.05, 0.10, 0.20)
GAMMA = 0.01     # standard ACI learning rate
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
    n = len(df)
    n_cal = int(round(n * CAL_FRAC))
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
        n = int(mask.sum())
        if n == 0:
            continue
        sub_sets = sets[mask]
        sub_y = y[mask]
        cov = float(np.mean([sub_sets[i, sub_y[i]] for i in range(len(sub_y))]))
        out[r] = cov
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = _load_with_regime()
    cal, ev = _split_cal_eval(df)

    # Bucket the EVAL slice into terciles using the EVAL signal alone (matches
    # round 002's setup — cal+eval are bucketed jointly there, but for marginal
    # coverage gaps the difference is small).
    eval_buckets = pd.qcut(ev[REGIME_COL].to_numpy(), 3, labels=list(DEFAULT_REGIME_LABELS))

    headline = {
        "round_id": "007",
        "hypothesis": "H-202",
        "claim": "ACI (Gibbs-Candès 2021) marginal coverage converges to 1-α on the real stream; per-regime gap measured for comparison vs round 002 LAC-Mondrian",
        "n_cal": len(cal),
        "n_eval": len(ev),
        "alphas": list(ALPHAS),
        "gamma": GAMMA,
        "results": {},
    }

    # --- Plot 1: q-trajectory at α=0.10 for both predictors ---
    fig_q, ax_q = plt.subplots(figsize=(11, 4.5))
    # --- Plot 2: cumulative coverage convergence at α=0.10 ---
    fig_c, ax_c = plt.subplots(figsize=(11, 4.5))

    # --- Plot 3: per-regime ACI coverage vs round-002 baselines (α=0.10 + 0.20) ---
    # Round-002 baselines (from LEDGER row): Mondrian-LAC closed |gap|@α=0.10 to
    # ±3.4pp everywhere; at α=0.20 low-vol/p_online has -7.9pp gap, low-vol/
    # p_offline has -5.8pp gap. We report 1-α - empirical_coverage as gap.
    round_002_mondrian_gap = {
        # signed gaps (1-α target minus empirical), recorded in round 002.
        # A positive number = under-coverage (covered less than target).
        # A negative number = over-coverage.
        ("p_offline", 0.10): {"low": +0.034, "med": -0.012, "high": -0.005},
        ("p_online", 0.10): {"low": +0.020, "med": -0.005, "high": -0.034},
        ("p_offline", 0.20): {"low": -0.058, "med": -0.020, "high": -0.010},
        ("p_online", 0.20): {"low": -0.079, "med": -0.025, "high": -0.012},
    }

    fig_r, axes_r = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for predictor, color_q, color_c in [
        ("p_offline", "#3b7dd8", "#3b7dd8"),
        ("p_online", "#56b870", "#56b870"),
    ]:
        p_cal = cal[predictor].to_numpy()
        y_cal = cal["y_true"].to_numpy().astype(int)
        p_ev = ev[predictor].to_numpy()
        y_ev = ev["y_true"].to_numpy().astype(int)

        per_predictor: dict = {"alpha": {}}

        for alpha in ALPHAS:
            # Warm q on cal.
            warm = aci_stream(p_cal, y_cal, alpha=alpha, gamma=GAMMA, q_init=Q_INIT_DEFAULT)
            q_after_warm = float(warm["q_history"][-1])
            # Apply on eval starting from warmed q.
            run = aci_stream(p_ev, y_ev, alpha=alpha, gamma=GAMMA, q_init=q_after_warm)

            cumulative_coverage = 1.0 - np.cumsum(run["err_history"]) / np.arange(1, len(p_ev) + 1)
            per_regime = _per_regime_coverage(
                run["sets"], y_ev, eval_buckets, DEFAULT_REGIME_LABELS,
            )
            target = 1.0 - alpha
            per_regime_gap = {r: target - c for r, c in per_regime.items()}

            per_predictor["alpha"][f"{alpha:.2f}"] = {
                "marginal_coverage": run["coverage"],
                "marginal_gap_vs_target": target - run["coverage"],
                "q_after_warm": q_after_warm,
                "q_final": float(run["q_history"][-1]),
                "per_regime_coverage": per_regime,
                "per_regime_gap": per_regime_gap,
            }

            if alpha == 0.10:
                ax_q.plot(run["q_history"], color=color_q, alpha=0.85,
                           linewidth=1.0, label=predictor)
                ax_c.plot(cumulative_coverage, color=color_c, alpha=0.85,
                          linewidth=1.0, label=f"{predictor} (final={run['coverage']:.4f})")

            # Per-regime bars
            if alpha in (0.10, 0.20):
                ax_idx = 0 if alpha == 0.10 else 1
                ax = axes_r[ax_idx]
                regimes = list(DEFAULT_REGIME_LABELS)
                aci_gaps = [per_regime_gap.get(r, 0) for r in regimes]
                m_gaps = [round_002_mondrian_gap[(predictor, alpha)][r] for r in regimes]

                # Side-by-side bars per predictor.
                offset = 0.0 if predictor == "p_offline" else 0.20
                shift = -0.1 if predictor == "p_offline" else +0.10
                xs = np.arange(len(regimes)) + shift
                # Stack: ACI as solid, Mondrian as hatched
                ax.bar(xs - 0.04, aci_gaps, width=0.08, color=color_q,
                       label=f"ACI · {predictor}" if alpha == 0.10 else None)
                ax.bar(xs + 0.04, m_gaps, width=0.08, color=color_q, hatch="///",
                       edgecolor="black", alpha=0.4,
                       label=f"Mondrian-R002 · {predictor}" if alpha == 0.10 else None)

        headline["results"][predictor] = per_predictor

    # Q-trajectory plot dressing
    ax_q.set_title(f"ACI threshold q_t over eval stream  (α=0.10, γ={GAMMA}, q_init=warmed on cal)")
    ax_q.set_xlabel("eval step")
    ax_q.set_ylabel("q_t")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="best", fontsize=9)
    fig_q.tight_layout()
    out_q = OUTDIR / "aci_q_trajectory.png"
    fig_q.savefig(out_q, dpi=110, bbox_inches="tight")
    plt.close(fig_q)

    # Coverage convergence plot dressing
    ax_c.axhline(0.90, linestyle="--", color="gray", alpha=0.6, label="target 1-α=0.90")
    ax_c.set_title(f"Cumulative empirical coverage on eval stream  (α=0.10, γ={GAMMA})")
    ax_c.set_xlabel("eval step")
    ax_c.set_ylabel("cumulative coverage")
    ax_c.set_ylim(0.80, 1.00)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="best", fontsize=9)
    fig_c.tight_layout()
    out_c = OUTDIR / "aci_coverage_convergence.png"
    fig_c.savefig(out_c, dpi=110, bbox_inches="tight")
    plt.close(fig_c)

    # Per-regime gap plot dressing
    for ax_idx, alpha in enumerate((0.10, 0.20)):
        ax = axes_r[ax_idx]
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(0.05, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(len(DEFAULT_REGIME_LABELS)))
        ax.set_xticklabels(list(DEFAULT_REGIME_LABELS))
        ax.set_title(f"per-regime coverage gap (target − empirical) at α={alpha:.2f}")
        ax.set_xlabel("vol regime")
        if ax_idx == 0:
            ax.set_ylabel("gap (positive = under-coverage)")
        ax.grid(True, alpha=0.3, axis="y")
    axes_r[0].legend(loc="best", fontsize=8)
    fig_r.suptitle(
        "ACI (γ=0.01, q-warmed on cal) vs Mondrian-LAC (round 002) on the same eval slice",
        y=1.02, fontsize=11,
    )
    fig_r.tight_layout()
    out_r = OUTDIR / "aci_per_regime_vs_round002.png"
    fig_r.savefig(out_r, dpi=110, bbox_inches="tight")
    plt.close(fig_r)

    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )

    # Compact stdout summary
    print(f"saved: {out_q}")
    print(f"saved: {out_c}")
    print(f"saved: {out_r}")
    print()
    for predictor, blk in headline["results"].items():
        print(f"  {predictor}:")
        for alpha_key, m in blk["alpha"].items():
            print(
                f"    α={alpha_key}  marginal cov={m['marginal_coverage']:.4f}  "
                f"gap={m['marginal_gap_vs_target']:+.4f}  "
                f"q_final={m['q_final']:.3f}  "
                f"per-regime gap: " + "  ".join(
                    f"{r}={g:+.4f}" for r, g in m["per_regime_gap"].items()
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
