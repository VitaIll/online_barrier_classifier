"""Round 020 — H-190 + H-191 + H-193 bootstrap-library audit.

Re-renders round-017's accepted Phase-A numbers with proper CIs from the
new src/bootstrap.py library:
  - H-190 stationary block bootstrap (Politis-Romano + Politis-White) for
    Sharpe / Sortino / max-DD per strategy.
  - H-191 DeLong (1988) CI for ROC-AUC and stratified Boyd-Eng-Page (2013)
    for PR-AUC, on the offline + online predictors.
  - H-193 sequential-bootstrap-with-overlap CI for per-trade Sharpe
    (LdP §4.5.1 trade uniqueness).

No model retrain. No notebook. Reuses round-017 cached unified prediction
parquets and re-runs the 5 strategies under c_stop=inf to recover per-bar
returns and per-trade outputs, then bootstraps each.

Outputs under RESEARCH/diagrams/round_020/.
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

from src.backtest import simulate_inventory_aware_sized  # noqa: E402
from src.bootstrap import (  # noqa: E402
    bootstrap_max_drawdown,
    bootstrap_metric,
    bootstrap_sharpe,
    bootstrap_sortino,
    bootstrap_trade_metric_overlap_aware,
    delong_roc_auc_ci,
    optimal_block_length,
    stratified_bootstrap_pr_auc,
    wilson_interval,
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
C_STOP = float("inf")  # round-017 winner config
N_RESAMPLES_RETURNS = 3000
N_RESAMPLES_AUC = 1000
SEED = 42

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_020"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ROUND017_TABLE = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_017" / "phase_A_table.csv"
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


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 020] ===== Bootstrap-library audit (H-190 + H-191 + H-193) =====")
    t_main = time.time()

    print("[round 020] loading splits + minute data + cached unified predictions ...")
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
    n_minutes = int(len(test_mn_c))
    print(f"  test_unified n={len(test_unified)}  minute n={n_minutes}")

    # round-017 val-chosen knobs (from the table).
    if not ROUND017_TABLE.exists():
        raise FileNotFoundError(f"Round-017 table not found: {ROUND017_TABLE}")
    r017 = pd.read_csv(ROUND017_TABLE)
    print("  loaded round-017 phase_A_table.csv")

    # Strategy specs with round-017 winning knobs.
    strategy_winners = {
        row["strategy"]: float(row["knob_star"]) for _, row in r017.iterrows()
    }
    print(f"  knob*: {strategy_winners}")

    specs = [
        {"name": "baseline_offline_tau", "fn": baseline_offline_tau,
         "extras": {}, "knob_name": "tau"},
        {"name": "baseline_online_tau", "fn": baseline_online_tau,
         "extras": {}, "knob_name": "tau"},
        {"name": "conformal_gate_tau", "fn": conformal_gate_tau,
         "extras": {"alpha_level": "10"}, "knob_name": "tau"},
        {"name": "mondrian_aci_size", "fn": mondrian_aci_size,
         "extras": {"alpha_level": "10"}, "knob_name": "k"},
        {"name": "null_random_at_rate", "fn": null_random_at_rate,
         "extras": {"seed": SEED}, "knob_name": "target_rate"},
    ]

    # ======================================================================
    # H-190 — stationary block bootstrap on Sharpe / Sortino / max-DD
    # ======================================================================
    print("\n[round 020] H-190: stationary block bootstrap on per-bar returns ...")
    h190_rows = []
    bar_returns_per_strategy = {}
    trades_per_strategy = {}
    for spec in specs:
        name = spec["name"]
        knob = strategy_winners[name]
        out = spec["fn"](test_unified, **{spec["knob_name"]: knob}, **spec["extras"])
        res = simulate_inventory_aware_sized(
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            out.open_signal, out.size,
            M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
        )
        eq = res.equity.to_numpy(dtype=float)
        per_bar_r = np.diff(eq, prepend=0.0) if len(eq) else np.zeros(0)
        bar_returns_per_strategy[name] = per_bar_r
        trades_per_strategy[name] = res.trades.copy()

        bl = optimal_block_length(per_bar_r) if len(per_bar_r) >= 8 else 1
        sh = bootstrap_sharpe(per_bar_r, n_resamples=N_RESAMPLES_RETURNS,
                                seed=SEED, block_length=bl)
        so = bootstrap_sortino(per_bar_r, n_resamples=N_RESAMPLES_RETURNS,
                                seed=SEED, block_length=bl)
        mdd = bootstrap_max_drawdown(per_bar_r, n_resamples=N_RESAMPLES_RETURNS,
                                       seed=SEED, block_length=bl)
        contains_null_sharpe = bool(sh["ci_lo"] <= 0.0 <= sh["ci_hi"])
        h190_rows.append({
            "strategy": name,
            "knob_star": knob,
            "n_trades": int(res.metrics["n_trades"]),
            "block_length": int(bl),
            "sharpe_point": sh["estimate"],
            "sharpe_ci_lo": sh["ci_lo"],
            "sharpe_ci_hi": sh["ci_hi"],
            "sharpe_se": sh["se"],
            "sharpe_contains_null": contains_null_sharpe,
            "sortino_point": so["estimate"],
            "sortino_ci_lo": so["ci_lo"],
            "sortino_ci_hi": so["ci_hi"],
            "max_dd_point": mdd["estimate"],
            "max_dd_ci_lo": mdd["ci_lo"],
            "max_dd_ci_hi": mdd["ci_hi"],
        })
        print(f"  {name}: bl={bl}  Sharpe={sh['estimate']:+.3f} "
              f"[{sh['ci_lo']:+.3f}, {sh['ci_hi']:+.3f}] "
              f"contains_null={contains_null_sharpe}")
    h190_df = pd.DataFrame(h190_rows)
    h190_df.to_csv(OUTDIR / "h190_stationary_block_sharpe.csv", index=False)

    # ======================================================================
    # H-191 — DeLong ROC-AUC + stratified PR-AUC + Wilson per-bin calibration
    # ======================================================================
    print("\n[round 020] H-191: DeLong ROC-AUC + stratified PR-AUC + Wilson calib ...")
    y_true = test_unified["y_true"].to_numpy().astype(int)
    h191_rows = []
    for predictor in ["p_offline", "p_online"]:
        scores = test_unified[predictor].to_numpy(dtype=float)
        roc = delong_roc_auc_ci(y_true, scores, confidence=0.95)
        pr = stratified_bootstrap_pr_auc(
            y_true, scores, n_resamples=N_RESAMPLES_AUC, seed=SEED, confidence=0.95,
        )
        h191_rows.append({
            "predictor": predictor,
            "roc_auc_point": roc["estimate"],
            "roc_auc_ci_lo": roc["ci_lo"],
            "roc_auc_ci_hi": roc["ci_hi"],
            "roc_auc_se": roc["se"],
            "pr_auc_point": pr["estimate"],
            "pr_auc_ci_lo": pr["ci_lo"],
            "pr_auc_ci_hi": pr["ci_hi"],
            "pr_auc_se": pr["se"],
            "n_pos": roc["n_pos"],
            "n_neg": roc["n_neg"],
        })
        print(f"  {predictor}: ROC={roc['estimate']:.4f} "
              f"[{roc['ci_lo']:.4f}, {roc['ci_hi']:.4f}]  "
              f"PR={pr['estimate']:.4f} [{pr['ci_lo']:.4f}, {pr['ci_hi']:.4f}]")
    pd.DataFrame(h191_rows).to_csv(OUTDIR / "h191_auc_cis.csv", index=False)

    # Per-bin Wilson calibration CIs for p_offline and p_online (10 equal-width bins).
    print("[round 020] H-191: per-bin Wilson calibration ...")
    calib_rows = []
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for predictor in ["p_offline", "p_online"]:
        scores = test_unified[predictor].to_numpy(dtype=float)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                in_bin = (scores >= lo) & (scores <= hi)
            else:
                in_bin = (scores >= lo) & (scores < hi)
            n_total = int(in_bin.sum())
            if n_total == 0:
                calib_rows.append({
                    "predictor": predictor, "bin_lo": lo, "bin_hi": hi,
                    "n_total": 0, "n_pos": 0, "p_obs": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "mean_predicted": float("nan"),
                })
                continue
            n_pos = int(y_true[in_bin].sum())
            wilson_lo, wilson_hi = wilson_interval(n_pos, n_total, confidence=0.95)
            calib_rows.append({
                "predictor": predictor, "bin_lo": lo, "bin_hi": hi,
                "n_total": n_total, "n_pos": n_pos,
                "p_obs": n_pos / n_total,
                "ci_lo": wilson_lo, "ci_hi": wilson_hi,
                "mean_predicted": float(scores[in_bin].mean()),
            })
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(OUTDIR / "h191_calibration_wilson.csv", index=False)
    print(f"  per-bin Wilson CI rows: {len(calib_df)}")

    # ======================================================================
    # H-193 — sequential-bootstrap CI on per-trade Sharpe (overlap-aware)
    # ======================================================================
    print("\n[round 020] H-193: sequential-bootstrap on per-trade Sharpe ...")
    h193_rows = []
    for spec in specs:
        name = spec["name"]
        trades = trades_per_strategy[name]
        if len(trades) < 10:
            h193_rows.append({
                "strategy": name, "n_trades": int(len(trades)),
                "naive_sharpe_ci_lo": float("nan"),
                "naive_sharpe_ci_hi": float("nan"),
                "seq_sharpe_ci_lo": float("nan"),
                "seq_sharpe_ci_hi": float("nan"),
                "mean_uniqueness": float("nan"),
                "ci_width_ratio_seq_over_naive": float("nan"),
            })
            continue
        n_open = trades["n_open"].to_numpy(dtype=np.int64)
        n_close = trades["n_close"].to_numpy(dtype=np.int64)
        pnl = trades["pnl_log_net"].to_numpy(dtype=float)

        def sharpe_fn(r: np.ndarray) -> float:
            s = float(r.std(ddof=1))
            return float(r.mean() / max(s, 1e-12))

        out_seq = bootstrap_trade_metric_overlap_aware(
            pnl, n_open, n_close, n_minutes, sharpe_fn,
            n_resamples=2000, seed=SEED, scheme="sequential_bootstrap",
        )
        out_naive = bootstrap_trade_metric_overlap_aware(
            pnl, n_open, n_close, n_minutes, sharpe_fn,
            n_resamples=2000, seed=SEED, scheme="naive",
        )
        width_seq = out_seq["ci_hi"] - out_seq["ci_lo"]
        width_naive = out_naive["ci_hi"] - out_naive["ci_lo"]
        ratio = width_seq / max(width_naive, 1e-12)
        h193_rows.append({
            "strategy": name, "n_trades": int(len(trades)),
            "naive_sharpe_ci_lo": out_naive["ci_lo"],
            "naive_sharpe_ci_hi": out_naive["ci_hi"],
            "seq_sharpe_ci_lo": out_seq["ci_lo"],
            "seq_sharpe_ci_hi": out_seq["ci_hi"],
            "naive_sharpe_se": out_naive["se"],
            "seq_sharpe_se": out_seq["se"],
            "mean_uniqueness": out_seq["mean_uniqueness"],
            "median_uniqueness": out_seq["median_uniqueness"],
            "ci_width_ratio_seq_over_naive": ratio,
        })
        print(f"  {name}: ū_mean={out_seq['mean_uniqueness']:.3f}  "
              f"naive=[{out_naive['ci_lo']:+.3f}, {out_naive['ci_hi']:+.3f}]  "
              f"seq=[{out_seq['ci_lo']:+.3f}, {out_seq['ci_hi']:+.3f}]  "
              f"ratio={ratio:.2f}")
    pd.DataFrame(h193_rows).to_csv(OUTDIR / "h193_sequential_trade_bootstrap.csv", index=False)

    # ======================================================================
    # Plots
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5))
    h190_df_plot = h190_df.copy()
    x = np.arange(len(h190_df_plot))
    points = h190_df_plot["sharpe_point"].to_numpy()
    lo = h190_df_plot["sharpe_ci_lo"].to_numpy()
    hi = h190_df_plot["sharpe_ci_hi"].to_numpy()
    contains_null = h190_df_plot["sharpe_contains_null"].to_numpy()
    for i in range(len(x)):
        c = "#888" if contains_null[i] else "#1B6B33"
        ax.plot([x[i], x[i]], [lo[i], hi[i]], color=c, linewidth=5, alpha=0.7)
        ax.scatter([x[i]], [points[i]], color="#333", s=60, zorder=10)
        ax.plot([x[i] - 0.1, x[i] + 0.1], [lo[i], lo[i]], color=c, linewidth=2)
        ax.plot([x[i] - 0.1, x[i] + 0.1], [hi[i], hi[i]], color=c, linewidth=2)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="zero")
    ax.set_xticks(x)
    ax.set_xticklabels(h190_df_plot["strategy"], rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("test Sharpe (95% CI, stationary block bootstrap)")
    ax.set_title("H-190 stationary-block-bootstrap Sharpe CIs (round-017 strategies, c_stop=inf)\n"
                 "green = CI excludes zero · gray = CI contains zero")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "h190_sharpe_cis.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for predictor, color in [("p_offline", "#1B6B33"), ("p_online", "#1B3F6B")]:
        sub = calib_df[calib_df["predictor"] == predictor].dropna(subset=["p_obs"])
        x = sub["mean_predicted"].to_numpy()
        y = sub["p_obs"].to_numpy()
        lo = sub["ci_lo"].to_numpy()
        hi = sub["ci_hi"].to_numpy()
        axes[0].errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=4,
                          color=color, markersize=5, label=predictor, linewidth=1.4)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("mean predicted in bin")
    axes[0].set_ylabel("observed P(y=1) (Wilson 95% CI)")
    axes[0].set_title("H-191 calibration with per-bin Wilson 95% CI")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    aucs = pd.DataFrame(h191_rows)
    x = np.arange(len(aucs))
    w = 0.35
    axes[1].bar(x - w/2, aucs["roc_auc_point"], w, color="#1B6B33", alpha=0.85,
                 label="ROC-AUC (DeLong)",
                 yerr=[aucs["roc_auc_point"] - aucs["roc_auc_ci_lo"],
                       aucs["roc_auc_ci_hi"] - aucs["roc_auc_point"]],
                 capsize=4, edgecolor="black", linewidth=0.4)
    axes[1].bar(x + w/2, aucs["pr_auc_point"], w, color="#1B3F6B", alpha=0.85,
                 label="PR-AUC (stratified)",
                 yerr=[aucs["pr_auc_point"] - aucs["pr_auc_ci_lo"],
                       aucs["pr_auc_ci_hi"] - aucs["pr_auc_point"]],
                 capsize=4, edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(aucs["predictor"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("AUC")
    axes[1].set_title("H-191 AUCs with closed-form / stratified 95% CIs")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTDIR / "h191_auc_calibration.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    h193_df = pd.DataFrame(h193_rows).dropna(subset=["mean_uniqueness"])
    x = np.arange(len(h193_df))
    w = 0.35
    naive_widths = h193_df["naive_sharpe_ci_hi"] - h193_df["naive_sharpe_ci_lo"]
    seq_widths = h193_df["seq_sharpe_ci_hi"] - h193_df["seq_sharpe_ci_lo"]
    ax.bar(x - w/2, naive_widths, w, color="#888", alpha=0.85,
            label="naive trade-bootstrap", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, seq_widths, w, color="#1B6B33", alpha=0.85,
            label="sequential-bootstrap (uniqueness-weighted)",
            edgecolor="black", linewidth=0.4)
    for i, (xi, mu) in enumerate(zip(x, h193_df["mean_uniqueness"])):
        ax.text(xi, max(naive_widths.iloc[i], seq_widths.iloc[i]) * 1.05,
                f"ū={mu:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(h193_df["strategy"], rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Sharpe 95% CI width")
    ax.set_title("H-193 per-trade Sharpe-CI: naive vs sequential (overlap-aware)\n"
                 "ū = mean trade uniqueness; lower ū → more overlap")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTDIR / "h193_trade_bootstrap_widths.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "020",
        "phase": "B",
        "phase_round": "bootstrap_audit",
        "claim": "H-190 + H-191 + H-193 bootstrap library; CIs on round-017 numbers",
        "h190_strategies": h190_rows,
        "h191_aucs": h191_rows,
        "h191_calibration_n_bins": int(n_bins),
        "h193_per_trade": h193_rows,
        "any_sharpe_excludes_zero": bool(
            any(not r["sharpe_contains_null"] for r in h190_rows)
        ),
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    excluding = [r["strategy"] for r in h190_rows if not r["sharpe_contains_null"]]
    print(f"\n[round 020] DONE in {headline['wall_clock_s']:.1f}s; "
          f"strategies whose Sharpe-CI EXCLUDES zero: "
          f"{excluding if excluding else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
