"""Round 031 — low-vol-gated strategy variant exploiting H-182 finding.

H-182 (round 030) showed baseline_offline_tau Sharpe = +0.61 per-bar in
low-parkinson-var cells; flat-to-negative in mid/high-vol. This round
implements the actionable variant: trade ONLY when entry parkinson_var
is in the bottom tercile (train-only fit).

Strategy: gate baseline_offline_tau (tau=0.20, c_stop=∞) by
`parkinson_var_rolling_mean_24 < pk_train_q33`. All other config matches
round 017.

Compare to round-017 ungated baseline:
- Annualized Sharpe
- Per-bar Sharpe with stationary block-bootstrap CI (H-190)
- DSR with the additional knob (1 gate vs 26 tau)
- CSCV PBO

Outputs under RESEARCH/diagrams/round_031/.
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
    deflated_sharpe,
    simulate_inventory_aware_sized,
    walk_forward_backtest,
)
from src.bootstrap import bootstrap_sharpe, optimal_block_length  # noqa: E402
from src.utils import chronological_split  # noqa: E402

ALPHA = 0.0041113
M = 20
COST_BPS = 1.0
C_STOP = float("inf")
TAU_OPEN = 0.20  # round-017 winner
SEED = 42
N_NULL_BOOTSTRAP = 200
N_FOLDS = 5

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_031"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"

REGIME_FEATURE = "parkinson_var_rolling_mean_24"


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


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 031] ===== low-vol-gated baseline_offline_tau =====")
    t_main = time.time()

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
    print(f"[round 031] test bars kept: {int(test_keep.sum())}")

    pk_train = train[REGIME_FEATURE].dropna().to_numpy()
    pk_q33 = float(np.quantile(pk_train, 1.0 / 3.0))
    pk_q67 = float(np.quantile(pk_train, 2.0 / 3.0))
    print(f"[round 031] train-only parkinson tercile edges: q33={pk_q33:.3e}  q67={pk_q67:.3e}")

    test_pk = test_full[REGIME_FEATURE].iloc[test_keep].to_numpy()
    print(f"[round 031] test parkinson stats: mean={test_pk.mean():.3e}  "
          f"q33-frac={(test_pk < pk_q33).mean():.3f}  q67-frac={(test_pk < pk_q67).mean():.3f}")

    p_off = test_unified["p_offline"].to_numpy(dtype=float)

    # Three variants: ungated, low-vol-only, low+mid-vol.
    variants = [
        ("ungated", np.ones(len(test_pk), dtype=bool)),
        ("low_vol_only", test_pk < pk_q33),
        ("low_plus_mid_vol", test_pk < pk_q67),
        ("high_vol_only", test_pk >= pk_q67),
    ]

    rows = []
    test_results = {}
    for name, mask in variants:
        open_signal = (p_off > TAU_OPEN).astype(float) * mask.astype(float)
        size = open_signal  # unit position
        res = simulate_inventory_aware_sized(
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            open_signal, size,
            M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
        )
        eq = res.equity.to_numpy(dtype=float)
        per_bar_r = np.diff(eq, prepend=0.0) if len(eq) else np.zeros(0)
        bl = optimal_block_length(per_bar_r) if len(per_bar_r) >= 32 else 1
        sh = bootstrap_sharpe(per_bar_r, n_resamples=2000, block_length=bl,
                                seed=SEED)

        # Walk-forward.
        try:
            wf = walk_forward_backtest(
                test_bound, test_mn_c, test_mn_h, test_mn_l,
                open_signal, size,
                n_folds=N_FOLDS, M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
            )
            wf_mean = float(wf["sharpe"].mean())
            wf_std = float(wf["sharpe"].std(ddof=1)) if len(wf) > 1 else 0.0
        except Exception:
            wf_mean = float("nan")
            wf_std = float("nan")

        # Bootstrap null: shuffle p_offline jointly with vol mask.
        rng = np.random.default_rng(SEED + abs(hash(name)) % 1000)
        nulls = np.empty(N_NULL_BOOTSTRAP, dtype=float)
        for i in range(N_NULL_BOOTSTRAP):
            perm = rng.permutation(len(p_off))
            p_shuf = p_off[perm]
            mask_shuf = mask[perm]
            os_n = (p_shuf > TAU_OPEN).astype(float) * mask_shuf.astype(float)
            res_n = simulate_inventory_aware_sized(
                test_bound, test_mn_c, test_mn_h, test_mn_l,
                os_n, os_n,
                M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
            )
            nulls[i] = res_n.metrics["sharpe"]
        observed_ann = float(res.metrics["sharpe"])
        p_boot = float((nulls >= observed_ann).sum() / len(nulls))

        rows.append({
            "variant": name,
            "n_bars_traded": int(open_signal.sum()),
            "n_trades": int(res.metrics["n_trades"]),
            "n_tp": int(res.metrics["n_tp"]),
            "n_timeout": int(res.metrics["n_timeout"]),
            "annualized_sharpe": observed_ann,
            "psr": float(res.metrics["probabilistic_sharpe"]),
            "per_bar_sharpe": sh["estimate"],
            "per_bar_sharpe_ci_lo": sh["ci_lo"],
            "per_bar_sharpe_ci_hi": sh["ci_hi"],
            "block_length": int(bl),
            "ci_excludes_zero": bool(sh["ci_lo"] > 0 or sh["ci_hi"] < 0),
            "total_log_return": float(res.metrics["total_log_return"]),
            "max_drawdown_log": float(res.metrics["max_drawdown_log"]),
            "hit_rate": float(res.metrics["hit_rate"]),
            "wf_sharpe_mean": wf_mean,
            "wf_sharpe_std": wf_std,
            "p_boot": p_boot,
            "null_sharpe_mean": float(nulls.mean()),
        })
        test_results[name] = res
        print(f"  {name}: annualized Sharpe={observed_ann:+.3f}  "
              f"per-bar={sh['estimate']:+.4f} [{sh['ci_lo']:+.4f},{sh['ci_hi']:+.4f}] "
              f"CI-excl-zero={rows[-1]['ci_excludes_zero']}  n_trades={rows[-1]['n_trades']}  "
              f"p_boot={p_boot:.3f}  WF={wf_mean:+.3f}±{wf_std:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "low_vol_gated_variants.csv", index=False)

    # CSCV PBO across the 4 variants.
    rmat = np.vstack([
        np.diff(test_results[name].equity.to_numpy(dtype=float), prepend=0.0)
        for name, _ in variants
    ])
    cscv = cscv_pbo(rmat, n_chunks=16)
    print(f"\n[round 031] cross-variant CSCV PBO={cscv['pbo']:.3f}  "
          f"modal IS-best={variants[cscv['is_best_strategy_modal_index']][0]}")

    # Equity overlay.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.4, 1]})
    colors = {
        "ungated": "#666666",
        "low_vol_only": "#1B6B33",
        "low_plus_mid_vol": "#1B3F6B",
        "high_vol_only": "#7A1B1B",
    }
    for name, _ in variants:
        eq = test_results[name].equity.to_numpy(dtype=float)
        running_max = np.maximum.accumulate(eq)
        dd = eq - running_max
        bar_idx = np.arange(len(eq))
        row = next(r for r in rows if r["variant"] == name)
        label = f"{name}: ann_Sharpe={row['annualized_sharpe']:+.3f}  n={row['n_trades']}"
        axes[0].plot(bar_idx, eq, color=colors[name], linewidth=1.0, alpha=0.95,
                      label=label)
        axes[1].fill_between(bar_idx, dd, color=colors[name], alpha=0.25,
                              linewidth=0)
    axes[0].axhline(0, color="gray", linewidth=0.6)
    axes[0].set_title("Round 031 — low-vol-gated baseline_offline_tau (c_stop=inf, tau=0.20)")
    axes[0].set_ylabel("cumulative log return")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("test decision-bar index")
    axes[1].set_ylabel("drawdown (log)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "low_vol_gated_equity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Per-variant per-bar Sharpe with CIs.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    err_lo = df["per_bar_sharpe"] - df["per_bar_sharpe_ci_lo"]
    err_hi = df["per_bar_sharpe_ci_hi"] - df["per_bar_sharpe"]
    for i, row in df.iterrows():
        c = "#1B6B33" if row["ci_excludes_zero"] and row["per_bar_sharpe"] > 0 else \
            "#7A1B1B" if row["ci_excludes_zero"] and row["per_bar_sharpe"] < 0 else "#888"
        ax.plot([x[i], x[i]], [row["per_bar_sharpe_ci_lo"], row["per_bar_sharpe_ci_hi"]],
                  color=c, linewidth=5, alpha=0.7)
        ax.scatter([x[i]], [row["per_bar_sharpe"]], color="#333", s=80, zorder=10)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("per-bar Sharpe (95% block-bootstrap CI)")
    ax.set_title("Round 031 — per-variant Sharpe CIs")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTDIR / "per_variant_sharpe_cis.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "031",
        "phase": "A_followup",
        "phase_round": "low_vol_gate",
        "claim": "low-vol-gated baseline_offline_tau exploits H-182 finding",
        "config": {"tau_open": TAU_OPEN, "c_stop": "inf",
                    "pk_q33": pk_q33, "pk_q67": pk_q67},
        "results": rows,
        "cross_variant_pbo": float(cscv["pbo"]),
        "cross_variant_modal_is_best": variants[cscv["is_best_strategy_modal_index"]][0],
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 031] DONE in {headline['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
