"""Round 028 — H-192 CSCV PBO refinement.

Refinement of H-113 (already accepted; `cscv_pbo` lives in src/backtest.py).
Pins the protocol: S=16 partitions (Bailey et al. default), per-partition
(IS-rank, OOS-rank) scatter, and a regression check that the round-015
cross-strategy PBO=0.000 is reproduced exactly.

Per BACKLOG H-192 falsifier: existing `cscv_pbo` helper produces PBO ≠
0.000 cross-strategy on round-015 inputs → regression bug.

Outputs under RESEARCH/diagrams/round_028/.
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

from src.backtest import (  # noqa: E402
    cscv_pbo,
    simulate_inventory_aware_sized,
)
from src.strategies import (  # noqa: E402
    baseline_offline_tau,
    baseline_online_tau,
    conformal_gate_tau,
    mondrian_aci_size,
    null_random_at_rate,
)
from src.utils import chronological_split  # noqa: E402

ALPHA = 0.0041113
M = 20
COST_BPS = 1.0
SEED = 42
S_PARTITIONS = 16  # H-192 spec

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_028"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ROUND015_TABLE = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_015" / "phase_A_table.csv"
ROUND015_CROSS = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_015" / "cscv_cross_strategy.json"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"


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


def _strategy_returns(
    fn, kwargs, extras, pred_df, boundaries, mn_c, mn_h, mn_l, c_stop=ALPHA,
) -> np.ndarray:
    out = fn(pred_df, **kwargs, **extras)
    res = simulate_inventory_aware_sized(
        boundaries, mn_c, mn_h, mn_l, out.open_signal, out.size,
        M=M, phi=ALPHA, c_stop=c_stop, cost_bps=COST_BPS,
    )
    eq = res.equity.to_numpy(dtype=float)
    return np.diff(eq, prepend=0.0) if len(eq) else np.zeros(0)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 028] ===== H-192 CSCV PBO refinement (S=16 protocol) =====")
    t_main = time.time()

    print("[round 028] loading splits + cached predictions ...")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    minute_df = _load_minute()
    test_unified = pd.read_parquet(PHASE_A_DIR / "test_predictions_unified.parquet")
    test_for_bound = test_full[["open_time", "close_time"]].reset_index(drop=True)
    test_bound, test_keep, test_mn_c, test_mn_h, test_mn_l = _build_boundaries(
        test_for_bound, minute_df,
    )

    r015 = pd.read_csv(ROUND015_TABLE)
    knobs = {row["strategy"]: float(row["knob_star"]) for _, row in r015.iterrows()}

    specs = [
        ("baseline_offline_tau", baseline_offline_tau, {"tau": knobs["baseline_offline_tau"]}, {}),
        ("baseline_online_tau", baseline_online_tau, {"tau": knobs["baseline_online_tau"]}, {}),
        ("conformal_gate_tau", conformal_gate_tau, {"tau": knobs["conformal_gate_tau"]}, {"alpha_level": "10"}),
        ("mondrian_aci_size", mondrian_aci_size, {"k": knobs["mondrian_aci_size"]}, {"alpha_level": "10"}),
        ("null_random_at_rate", null_random_at_rate, {"target_rate": knobs["null_random_at_rate"]}, {"seed": SEED}),
    ]

    # Build returns matrix at round-015's c_stop=ALPHA setting (the canonical reference).
    print("\n[round 028] building cross-strategy returns matrix at c_stop=ALPHA ...")
    rmat = np.vstack([
        _strategy_returns(fn, kwargs, extras, test_unified, test_bound,
                            test_mn_c, test_mn_h, test_mn_l, c_stop=ALPHA)
        for name, fn, kwargs, extras in specs
    ])
    print(f"  rmat shape: {rmat.shape}  (n_strategies x n_periods)")

    # Compute CSCV PBO with S=16 (the round-015 setup).
    out = cscv_pbo(rmat, n_chunks=S_PARTITIONS)
    print(f"\n[round 028] cross-strategy CSCV (S={S_PARTITIONS}, c_stop=ALPHA):")
    print(f"  PBO={out['pbo']:.6f}  n_combinations={out['n_combinations']}  "
          f"chunk_size={out['chunk_size']}")
    print(f"  median_logit={out['median_logit']:+.4f}  "
          f"median_rel_rank={out['median_rel_rank']:.4f}")
    print(f"  is_best_modal_index={out['is_best_strategy_modal_index']} "
          f"({specs[out['is_best_strategy_modal_index']][0]})")
    print(f"  is_best_counts: {out['is_best_strategy_counts']}")

    # Regression check: round-015's cross PBO = 0.000.
    r015_cross = json.loads(ROUND015_CROSS.read_text())
    expected_pbo = float(r015_cross["pbo"])
    expected_modal = r015_cross["modal_is_best_strategy"]
    pbo_diff = abs(out["pbo"] - expected_pbo)
    modal_match = (specs[out["is_best_strategy_modal_index"]][0] == expected_modal)
    print(f"\n[round 028] regression check vs round-015:")
    print(f"  expected PBO = {expected_pbo:.6f}; got {out['pbo']:.6f}; "
          f"diff = {pbo_diff:.6e}")
    print(f"  expected modal = {expected_modal}; got "
          f"{specs[out['is_best_strategy_modal_index']][0]}; match = {modal_match}")
    regression_passed = pbo_diff < 1e-9 and modal_match

    # H-192 protocol artifact: per-partition (IS-rank, OOS-rank) scatter.
    # Reconstruct from `out["logits"]` and `out["rel_ranks"]`.
    logits = out["logits"]
    rel_ranks = out["rel_ranks"]
    n_strat = rmat.shape[0]
    # Note: cscv_pbo returns the relative rank of the IS-best strategy on OOS.
    # rel_rank ∈ (0, 1); logit = log(rel_rank / (1 - rel_rank)).

    # Plot 1: PBO histogram (logits) + the median line.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.hist(logits, bins=40, color="#3b7dd8", alpha=0.85, edgecolor="white",
             linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.0,
                label=f"λ=0 (PBO={out['pbo']:.4f})")
    ax.axvline(np.median(logits), color="black", linestyle=":", linewidth=1.0,
                label=f"median={np.median(logits):+.3f}")
    ax.set_xlabel("logit ω")
    ax.set_ylabel("count")
    ax.set_title(f"H-192 PBO histogram, S={S_PARTITIONS} ({out['n_combinations']:,} combos)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: relative rank histogram on its native [0, 1] scale.
    ax = axes[1]
    ax.hist(rel_ranks, bins=40, color="#1B6B33", alpha=0.85, edgecolor="white",
             linewidth=0.4)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.0,
                label="median ω = 0.5")
    ax.axvline(np.median(rel_ranks), color="black", linestyle=":", linewidth=1.0,
                label=f"median={np.median(rel_ranks):.3f}")
    ax.set_xlabel("relative OOS rank ω")
    ax.set_ylabel("count")
    ax.set_title("Relative-rank distribution (1.0 = best, 0 = worst)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: per-partition modal IS-best counts (bars).
    ax = axes[2]
    counts = out["is_best_strategy_counts"]
    names = [s[0] for s in specs]
    colors = ["#1B6B33", "#1B3F6B", "#6B1B6B", "#1B6B6B", "#888888"]
    ax.barh(np.arange(len(names)), counts, color=colors, alpha=0.85,
             edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i, c in enumerate(counts):
        ax.text(c, i, f"  {c:,}", va="center", fontsize=8)
    ax.set_xlabel(f"# IS-best wins (out of {sum(counts):,})")
    ax.set_title("Modal IS-best across CSCV partitions")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Round 028 — H-192 CSCV PBO refinement (S=16)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "h192_cscv_panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Save outputs
    summary = {
        "n_strategies": n_strat,
        "S_partitions": S_PARTITIONS,
        "n_combinations": int(out["n_combinations"]),
        "chunk_size": int(out["chunk_size"]),
        "pbo": float(out["pbo"]),
        "median_logit": float(out["median_logit"]),
        "median_rel_rank": float(out["median_rel_rank"]),
        "is_best_modal_strategy": specs[out["is_best_strategy_modal_index"]][0],
        "is_best_counts": dict(zip(names, [int(c) for c in counts])),
        "regression_check": {
            "expected_pbo": expected_pbo,
            "got_pbo": float(out["pbo"]),
            "pbo_diff": float(pbo_diff),
            "expected_modal": expected_modal,
            "got_modal": specs[out["is_best_strategy_modal_index"]][0],
            "passed": bool(regression_passed),
        },
    }
    (OUTDIR / "h192_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8",
    )
    pd.DataFrame({"logit": logits, "rel_rank": rel_ranks}).to_csv(
        OUTDIR / "h192_per_partition.csv", index=False,
    )

    headline = {
        "round_id": "028",
        "phase": "B",
        "phase_round": "h192_cscv_refinement",
        "claim": "H-192 CSCV PBO refinement; S=16 protocol pinned + regression check",
        "summary": summary,
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 028] DONE in {headline['wall_clock_s']:.1f}s; "
          f"regression check {'PASSED' if regression_passed else 'FAILED'}")
    return 0 if regression_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
