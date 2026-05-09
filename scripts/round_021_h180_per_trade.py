"""Round 021 — H-180 per-trade attribution dataframe (substrate for Ask 7).

Wraps `simulate_inventory_aware_sized` outputs to emit a per-trade parquet
with the H-180 schema: entry_bar, exit_bar, entry_p_offline, entry_p_online,
entry_sigma_epistemic, entry_q_t (per α=0.10), entry_regime,
realized_pnl_bp, bars_held, exit_reason, cost_paid_bp, plus the
within-trade timestamp + entry-hour + dow features for H-306 postmortem.

Run for the 5 round-017 strategies + the round-018 H-209 sized variant
(which exists at the per-trade level even if the headline Sharpe was
negative). The output parquets are the substrate every future
strategy round must diff against (per H-306 spawning rule).

Outputs under RESEARCH/diagrams/round_021/.
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
C_STOP = float("inf")
SEED = 42
TAU_OPEN_FOR_SIZED = 0.20  # round-017 winner

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_021"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ROUND017_TABLE = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_017" / "phase_A_table.csv"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
SIGMA_CACHE = PHASE_A_DIR / "test_sigma_epistemic.parquet"


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


def enrich_trades(
    trades_df: pd.DataFrame, test_unified: pd.DataFrame, sigma_eps: np.ndarray,
    test_regime: np.ndarray, boundaries: pd.DataFrame, strategy_name: str,
) -> pd.DataFrame:
    """H-180 schema: enrich the raw Trade DataFrame with predictor + UQ + regime
    columns sampled at entry.
    """
    if len(trades_df) == 0:
        return pd.DataFrame()
    enriched = trades_df.copy()
    enriched["strategy"] = strategy_name
    k_open = enriched["k_open"].to_numpy(dtype=int)
    enriched["entry_p_offline"] = test_unified["p_offline"].to_numpy()[k_open]
    enriched["entry_p_online"] = test_unified["p_online"].to_numpy()[k_open]
    enriched["entry_sigma_epistemic"] = sigma_eps[k_open]
    enriched["entry_q_lo_05"] = test_unified["q_lo_05"].to_numpy()[k_open]
    enriched["entry_q_lo_10"] = test_unified["q_lo_10"].to_numpy()[k_open]
    enriched["entry_q_lo_20"] = test_unified["q_lo_20"].to_numpy()[k_open]
    enriched["entry_in_set_05"] = test_unified["in_set_05"].to_numpy()[k_open]
    enriched["entry_in_set_10"] = test_unified["in_set_10"].to_numpy()[k_open]
    enriched["entry_in_set_20"] = test_unified["in_set_20"].to_numpy()[k_open]
    enriched["entry_regime"] = test_regime[k_open]
    enriched["realized_pnl_bp"] = enriched["pnl_log_net"] * 1e4
    enriched["cost_paid_bp"] = 2.0 * COST_BPS  # round-trip cost in bps
    enriched["entry_ts"] = boundaries["ts"].to_numpy()[k_open]
    enriched["entry_hour_utc"] = pd.to_datetime(enriched["entry_ts"]).dt.hour
    enriched["entry_dow"] = pd.to_datetime(enriched["entry_ts"]).dt.dayofweek
    return enriched


def cantelli_size(
    p_offline: np.ndarray, sigma_epistemic: np.ndarray, *, tau: float, k: float,
) -> tuple[np.ndarray, np.ndarray]:
    p_lower = np.maximum(0.0, p_offline - k * sigma_epistemic)
    open_signal = (p_offline > tau).astype(float)
    size = np.clip(p_lower - tau, 0.0, 1.0) * open_signal
    return open_signal, size


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 021] ===== H-180 per-trade attribution dataframe =====")
    t_main = time.time()

    print("[round 021] loading splits + minute data + cached predictions ...")
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
    test_kept_features = test_full.iloc[test_keep].reset_index(drop=True)
    test_regime_kept = test_unified["regime_id"].to_numpy(dtype=int)
    sigma_eps = pd.read_parquet(SIGMA_CACHE)["sigma_epistemic"].to_numpy(dtype=float)
    print(f"  test n={len(test_unified)}  sigma n={len(sigma_eps)}")

    r017 = pd.read_csv(ROUND017_TABLE)
    knobs = {row["strategy"]: float(row["knob_star"]) for _, row in r017.iterrows()}

    specs = [
        {"name": "baseline_offline_tau", "fn": baseline_offline_tau,
         "kwargs": {"tau": knobs["baseline_offline_tau"]}, "extras": {}},
        {"name": "baseline_online_tau", "fn": baseline_online_tau,
         "kwargs": {"tau": knobs["baseline_online_tau"]}, "extras": {}},
        {"name": "conformal_gate_tau", "fn": conformal_gate_tau,
         "kwargs": {"tau": knobs["conformal_gate_tau"]},
         "extras": {"alpha_level": "10"}},
        {"name": "mondrian_aci_size", "fn": mondrian_aci_size,
         "kwargs": {"k": knobs["mondrian_aci_size"]},
         "extras": {"alpha_level": "10"}},
        {"name": "null_random_at_rate", "fn": null_random_at_rate,
         "kwargs": {"target_rate": knobs["null_random_at_rate"]},
         "extras": {"seed": SEED}},
    ]

    all_trades = []
    for spec in specs:
        out = spec["fn"](test_unified, **spec["kwargs"], **spec["extras"])
        res = simulate_inventory_aware_sized(
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            out.open_signal, out.size,
            M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
        )
        enriched = enrich_trades(
            res.trades, test_unified, sigma_eps, test_regime_kept, test_bound,
            spec["name"],
        )
        if len(enriched) > 0:
            all_trades.append(enriched)
            print(f"  {spec['name']}: enriched {len(enriched)} trades")

    # Plus H-209 sized variant for completeness.
    p_off = test_unified["p_offline"].to_numpy(dtype=float)
    open_sig, size = cantelli_size(p_off, sigma_eps, tau=TAU_OPEN_FOR_SIZED, k=0.0)
    res_h209 = simulate_inventory_aware_sized(
        test_bound, test_mn_c, test_mn_h, test_mn_l, open_sig, size,
        M=M, phi=ALPHA, c_stop=C_STOP, cost_bps=COST_BPS,
    )
    enriched_h209 = enrich_trades(
        res_h209.trades, test_unified, sigma_eps, test_regime_kept, test_bound,
        "h209_cantelli_k0",
    )
    if len(enriched_h209) > 0:
        all_trades.append(enriched_h209)
        print(f"  h209_cantelli_k0: enriched {len(enriched_h209)} trades")

    if not all_trades:
        raise RuntimeError("No trades collected from any strategy.")
    full_df = pd.concat(all_trades, ignore_index=True)
    full_df.to_parquet(OUTDIR / "trade_postmortem.parquet", index=False)
    print(f"\n[round 021] wrote per-trade parquet: {len(full_df)} rows, "
          f"{full_df['strategy'].nunique()} strategies")

    # Summary slice tables (lightweight previews of H-181/H-182 slices).
    summary_rows = []
    for strat, sub in full_df.groupby("strategy"):
        # Exit reason mix
        exit_mix = sub["exit_reason"].value_counts(normalize=True).to_dict()
        # Hold time per exit reason
        hold_by_exit = sub.groupby("exit_reason")["bars_held"].mean().to_dict()
        # p_signal decile hit rate
        try:
            sub2 = sub.copy()
            sub2["p_signal_decile"] = pd.qcut(sub2["p_signal"], 10, labels=False,
                                                duplicates="drop")
            top_decile_hit = float(
                (sub2[sub2["p_signal_decile"] == sub2["p_signal_decile"].max()]
                  ["realized_pnl_bp"] > 0).mean()
            )
            bot_decile_hit = float(
                (sub2[sub2["p_signal_decile"] == sub2["p_signal_decile"].min()]
                  ["realized_pnl_bp"] > 0).mean()
            )
        except Exception:
            top_decile_hit = float("nan")
            bot_decile_hit = float("nan")
        # Per-regime mean PnL
        regime_mean_pnl = sub.groupby("entry_regime")["pnl_log_net"].mean().to_dict()
        # Sigma decile mean PnL
        try:
            sub3 = sub.copy()
            sub3["sigma_decile"] = pd.qcut(sub3["entry_sigma_epistemic"], 10,
                                             labels=False, duplicates="drop")
            sigma_top_pnl = float(
                sub3[sub3["sigma_decile"] == sub3["sigma_decile"].max()]
                ["pnl_log_net"].mean()
            )
            sigma_bot_pnl = float(
                sub3[sub3["sigma_decile"] == sub3["sigma_decile"].min()]
                ["pnl_log_net"].mean()
            )
        except Exception:
            sigma_top_pnl = float("nan")
            sigma_bot_pnl = float("nan")
        summary_rows.append({
            "strategy": strat,
            "n_trades": len(sub),
            "frac_tp": exit_mix.get("tp", 0.0),
            "frac_sl": exit_mix.get("sl", 0.0),
            "frac_timeout": exit_mix.get("timeout", 0.0),
            "median_hold_tp": hold_by_exit.get("tp", float("nan")),
            "median_hold_timeout": hold_by_exit.get("timeout", float("nan")),
            "top_decile_p_signal_hit": top_decile_hit,
            "bottom_decile_p_signal_hit": bot_decile_hit,
            "top_minus_bottom_p_signal_hit": top_decile_hit - bot_decile_hit,
            "regime_0_mean_pnl": regime_mean_pnl.get(0, float("nan")),
            "regime_1_mean_pnl": regime_mean_pnl.get(1, float("nan")),
            "regime_2_mean_pnl": regime_mean_pnl.get(2, float("nan")),
            "sigma_top_decile_mean_pnl": sigma_top_pnl,
            "sigma_bottom_decile_mean_pnl": sigma_bot_pnl,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "postmortem_summary.csv", index=False)
    print("\n[round 021] postmortem summary:")
    print(summary_df.to_string(index=False))

    # Diagnostic plots: 4-up panel per strategy stack.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for strat, sub in full_df.groupby("strategy"):
        if len(sub) < 10:
            continue
        # Exit reason pie chart-like (skip; use bar instead)
        # 1) p_signal decile hit-rate
        try:
            sub2 = sub.copy()
            sub2["p_signal_decile"] = pd.qcut(sub2["p_signal"], 10,
                                                labels=False, duplicates="drop")
            decile_hit = sub2.groupby("p_signal_decile").apply(
                lambda d: (d["realized_pnl_bp"] > 0).mean(), include_groups=False,
            )
            axes[0, 0].plot(decile_hit.index, decile_hit.values, marker="o",
                             markersize=4, label=strat, linewidth=1.4, alpha=0.8)
        except Exception:
            pass
        # 2) per-regime mean PnL
        regime_pnl = sub.groupby("entry_regime")["realized_pnl_bp"].mean()
        axes[0, 1].plot(regime_pnl.index, regime_pnl.values, marker="o",
                         markersize=6, label=strat, linewidth=1.4, alpha=0.8)
        # 3) hold-time histogram — use step type to avoid label-per-bin pollution.
        axes[1, 0].hist(sub["bars_held"].clip(0, 30), bins=30,
                          alpha=0.6, label=strat, histtype="step", linewidth=1.5)
        # 4) σ_epistemic decile mean PnL
        try:
            sub3 = sub.copy()
            sub3["sigma_decile"] = pd.qcut(sub3["entry_sigma_epistemic"], 10,
                                             labels=False, duplicates="drop")
            sig_pnl = sub3.groupby("sigma_decile")["realized_pnl_bp"].mean()
            axes[1, 1].plot(sig_pnl.index, sig_pnl.values, marker="o",
                             markersize=4, label=strat, linewidth=1.4, alpha=0.8)
        except Exception:
            pass

    axes[0, 0].axhline(0, color="gray", linewidth=0.6)
    axes[0, 0].set_xlabel("p_signal decile (low → high)")
    axes[0, 0].set_ylabel("hit rate")
    axes[0, 0].set_title("Hit rate by p_signal decile")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=7)

    axes[0, 1].axhline(0, color="gray", linewidth=0.6)
    axes[0, 1].set_xlabel("entry vol regime (0=low, 1=med, 2=high)")
    axes[0, 1].set_ylabel("mean PnL (bp)")
    axes[0, 1].set_title("Mean PnL by entry regime")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(loc="best", fontsize=7)

    axes[1, 0].set_xlabel("bars held")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_title("Hold-time histogram (clipped at 30)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axhline(0, color="gray", linewidth=0.6)
    axes[1, 1].set_xlabel("σ_epistemic decile")
    axes[1, 1].set_ylabel("mean PnL (bp)")
    axes[1, 1].set_title("Mean PnL by σ_epistemic decile")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(loc="best", fontsize=7)

    fig.suptitle("Round 021 — H-180 per-trade postmortem (substrate)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "postmortem_panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Per-hour Sharpe (24 buckets)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for strat in summary_df["strategy"]:
        sub = full_df[full_df["strategy"] == strat]
        if len(sub) < 50:
            continue
        hour_pnl = sub.groupby("entry_hour_utc")["realized_pnl_bp"].agg(["mean", "count"])
        hour_pnl = hour_pnl[hour_pnl["count"] >= 10]
        ax.plot(hour_pnl.index, hour_pnl["mean"], marker="o", markersize=4,
                  label=strat, linewidth=1.4, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("entry hour UTC")
    ax.set_ylabel("mean PnL (bp)")
    ax.set_title("Mean PnL by entry hour — H-306 hour-of-day slice")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "hourly_pnl.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "021",
        "phase": "D",  # Per round-016 plan, postmortem is Phase D
        "phase_round": "h180_per_trade_substrate",
        "claim": "H-180 per-trade attribution dataframe + diagnostic slice tables",
        "n_trades_per_strategy": {
            r["strategy"]: int(r["n_trades"]) for _, r in summary_df.iterrows()
        },
        "summary_rows": summary_df.to_dict(orient="records"),
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 021] DONE in {headline['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
