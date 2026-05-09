"""Phase A round 2 — strategies 4-7 + walk-forward + multi-strategy DSR.

Builds on round-011's unified prediction parquets (artifacts/phase_A/{val,test}_
predictions_unified.parquet) and runs the remaining four §A.2 strategies under
the same val→test discipline, plus a 5-fold walk-forward per strategy and the
multi-strategy deflated Sharpe deflation that round-011 explicitly deferred.

Strategies covered this round (in addition to the 3 already in round 011):
    4. combined_stacked_tau   — logistic on (p_off, p_on) fit on val
    5. conformal_gate_tau     — (p_offline > τ) AND (in_set_10 == 1)
    6. mondrian_aci_size      — open at singleton {1}; size by confidence_score
    7. null_random_at_rate    — random entries at target rate (the no-skill comparator)

The full §A.6 table (all 7 strategies) is rebuilt and persisted.

Outputs (under `RESEARCH/diagrams/phase_A/round_012/`):
    val_sweep_<strategy>.csv          — per-strategy val sweep
    val_tau_sweep_round2.png          — val Sharpe vs knob, all 7 strategies
    equity_overlay_round2.png         — test equity at val-chosen knob
    walk_forward_per_strategy.csv     — per-fold Sharpe table
    walk_forward_sharpe_bars.png      — bar chart, mean ± std
    phase_A_partial_v2.csv            — 7-row table with multi-strategy DSR
    headline.json                     — round summary
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

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
    BacktestResult,
    deflated_sharpe,
    simulate_inventory_aware,
    simulate_inventory_aware_sized,
    walk_forward_backtest,
)
from src.strategies import (  # noqa: E402
    StrategyOutput,
    baseline_offline_tau,
    baseline_online_tau,
    combined_avg_tau,
    combined_stacked_tau,
    conformal_gate_tau,
    fit_stacker,
    mondrian_aci_size,
    null_random_at_rate,
)
from src.utils import chronological_split  # noqa: E402

# ----------------------------------------------------------------------------
# Constants — preserved from round-001/009/011 so results stay comparable.
# ----------------------------------------------------------------------------
ALPHA = 0.0041113
M = 20
COST_BPS = 1.0
N_NULL_BOOTSTRAP = 200
SEED = 42
TAU_GRID = tuple(round(0.10 + 0.02 * i, 4) for i in range(26))
K_GRID = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
N_FOLDS = 5

OUTDIR = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_012"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ONLINE_PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"


# ----------------------------------------------------------------------------
# Boundary mapping (reused from round-011)
# ----------------------------------------------------------------------------

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


def _shuffled_null(
    open_signal_resolver: Callable[[np.ndarray], StrategyOutput],
    p_to_shuffle: np.ndarray,
    boundaries: pd.DataFrame,
    mn_c: np.ndarray, mn_h: np.ndarray, mn_l: np.ndarray,
    *,
    n: int = N_NULL_BOOTSTRAP, seed: int = SEED,
) -> np.ndarray:
    """Generic shuffled-signal null: shuffle the chosen probability column,
    rerun the strategy, record Sharpe.

    `open_signal_resolver(p_shuffled)` must return a StrategyOutput consistent
    with the strategy under test (size/open_signal under the shuffled p).
    """
    rng = np.random.default_rng(seed)
    nulls = np.empty(n, dtype=float)
    for i in range(n):
        p_shuf = p_to_shuffle.copy()
        rng.shuffle(p_shuf)
        out = open_signal_resolver(p_shuf)
        res = simulate_inventory_aware_sized(
            boundaries, mn_c, mn_h, mn_l,
            out.open_signal, out.size,
            M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        nulls[i] = res.metrics["sharpe"]
    return nulls


# ----------------------------------------------------------------------------
# Sweep helpers
# ----------------------------------------------------------------------------

def _sweep_knob(
    strategy_fn: Callable[..., StrategyOutput],
    pred_df: pd.DataFrame,
    boundaries: pd.DataFrame,
    mn_c: np.ndarray, mn_h: np.ndarray, mn_l: np.ndarray,
    knob_name: str,
    knob_grid: Sequence[float],
    *,
    extra_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Grid-sweep over `knob_name` on the val slice; record metrics per knob."""
    rows = []
    extras = dict(extra_kwargs or {})
    for v in knob_grid:
        out = strategy_fn(pred_df, **{knob_name: v}, **extras)
        res = simulate_inventory_aware_sized(
            boundaries, mn_c, mn_h, mn_l,
            out.open_signal, out.size,
            M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        m = res.metrics
        rows.append({
            knob_name: v,
            "sharpe": m["sharpe"],
            "psr": m["probabilistic_sharpe"],
            "n_trades": m["n_trades"],
            "total_log_return": m["total_log_return"],
            "max_drawdown_log": m["max_drawdown_log"],
            "hit_rate": m["hit_rate"],
        })
    return pd.DataFrame(rows)


def _pick_winning_knob(
    sweep: pd.DataFrame, knob: str, *, min_n_trades: int = 50,
) -> tuple[float, dict]:
    eligible = sweep[sweep["n_trades"] >= min_n_trades]
    if eligible.empty:
        eligible = sweep
    row = eligible.sort_values(["sharpe", "n_trades"], ascending=[False, False]).iloc[0]
    return float(row[knob]), row.to_dict()


# ----------------------------------------------------------------------------
# Strategy spec table — single source of truth for round 012
# ----------------------------------------------------------------------------

def _build_strategy_specs(
    val_unified: pd.DataFrame, baseline_offline_test_rate: float | None,
) -> list[dict]:
    """Return the per-strategy spec list driving the run."""
    stacker = fit_stacker(val_unified)
    print(f"[round 012] stacker: coef_offline={stacker['coef_offline']:.3f}  "
          f"coef_online={stacker['coef_online']:.3f}  "
          f"intercept={stacker['intercept']:.3f}")

    return [
        {"name": "baseline_offline_tau", "fn": baseline_offline_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {},
         "shuffle_col": "p_offline"},
        {"name": "baseline_online_tau", "fn": baseline_online_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {},
         "shuffle_col": "p_online"},
        {"name": "combined_avg_tau", "fn": combined_avg_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {},
         "shuffle_col": "p_offline"},
        {"name": "combined_stacked_tau", "fn": combined_stacked_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {"stacker": stacker},
         "shuffle_col": "p_offline"},
        {"name": "conformal_gate_tau", "fn": conformal_gate_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {"alpha_level": "10"},
         "shuffle_col": "p_offline"},
        {"name": "mondrian_aci_size", "fn": mondrian_aci_size,
         "knob": "k", "grid": K_GRID, "extras": {"alpha_level": "10"},
         "shuffle_col": "p_offline"},
        # `null_random_at_rate` is single-point (target_rate fixed = the
        # offline baseline's test entry rate); kept here for table parity.
        {"name": "null_random_at_rate", "fn": null_random_at_rate,
         "knob": "target_rate",
         "grid": (float(baseline_offline_test_rate or 0.10),),
         "extras": {"seed": SEED},
         "shuffle_col": "p_offline"},
    ]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 012] ===== Phase A round 2 — strategies 4-7 + walk-forward =====")
    t0 = time.time()

    # 1) Load unified val + test prediction parquets from round 011.
    print("[round 012] loading unified val/test predictions …")
    val_unified = pd.read_parquet(PHASE_A_DIR / "val_predictions_unified.parquet")
    test_unified = pd.read_parquet(PHASE_A_DIR / "test_predictions_unified.parquet")
    print(f"  val n={len(val_unified)}  test n={len(test_unified)}")
    needed = ["p_offline", "p_online", "y_true", "regime_id", "in_set_10",
              "q_lo_10", "in_set_05", "q_lo_05", "in_set_20", "q_lo_20"]
    for c in needed:
        if c not in val_unified.columns or c not in test_unified.columns:
            raise RuntimeError(
                f"unified prediction parquets missing '{c}' — re-run round 011."
            )

    # 2) Boundary mapping for val + test (re-derived from feature parquet).
    print("[round 012] mapping boundaries …")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    minute_df = _load_minute()
    val_for_bound = val[["open_time", "close_time"]].reset_index(drop=True)
    val_bound, val_keep, val_mn_c, val_mn_h, val_mn_l = _build_boundaries(
        val_for_bound, minute_df,
    )
    test_for_bound = test_full[["open_time", "close_time"]].reset_index(drop=True)
    test_bound, test_keep, test_mn_c, test_mn_h, test_mn_l = _build_boundaries(
        test_for_bound, minute_df,
    )
    if len(val_bound) != len(val_unified):
        raise RuntimeError(
            f"val unified n={len(val_unified)} but boundaries n={len(val_bound)} "
            "— run round 011 to refresh."
        )
    if len(test_bound) != len(test_unified):
        raise RuntimeError(
            f"test unified n={len(test_unified)} but boundaries n={len(test_bound)} "
            "— run round 011 to refresh."
        )

    # 3) Get round-011's offline baseline rate (used to set null_random_at_rate
    #    target_rate). We re-run the offline tau-grid quickly to compute it.
    qsweep = _sweep_knob(
        baseline_offline_tau, val_unified, val_bound, val_mn_c, val_mn_h, val_mn_l,
        "tau", TAU_GRID,
    )
    tau_off, _ = _pick_winning_knob(qsweep, "tau")
    out_off_test = baseline_offline_tau(test_unified, tau=tau_off)
    test_offline_rate = float(out_off_test.open_signal.mean())
    print(f"  baseline_offline test entry rate at val-tau* {tau_off:.3f}: "
          f"{test_offline_rate:.4f}")

    # 4) Strategy specs.
    specs = _build_strategy_specs(val_unified, test_offline_rate)
    n_trials_total = sum(len(s["grid"]) for s in specs)
    print(f"[round 012] strategies={len(specs)}  total knob-trials={n_trials_total}")

    # 5) Per-strategy val sweep + test eval + walk-forward + bootstrap.
    val_sweeps: dict[str, pd.DataFrame] = {}
    test_results: dict[str, BacktestResult] = {}
    table_rows: list[dict] = []
    walk_rows: list[dict] = []
    all_val_sharpes: list[float] = []

    for spec in specs:
        name = spec["name"]
        fn = spec["fn"]
        knob = spec["knob"]
        grid = spec["grid"]
        extras = spec["extras"]
        shuffle_col = spec["shuffle_col"]

        print(f"\n[round 012] >>> strategy={name}  knob={knob}  grid={list(grid)[:3]}{'...' if len(grid) > 3 else ''}")
        sweep = _sweep_knob(
            fn, val_unified, val_bound, val_mn_c, val_mn_h, val_mn_l,
            knob, grid, extra_kwargs=extras,
        )
        sweep.to_csv(OUTDIR / f"val_sweep_{name}.csv", index=False)
        val_sweeps[name] = sweep
        all_val_sharpes.extend([float(s) for s in sweep["sharpe"]])

        knob_star, row = _pick_winning_knob(sweep, knob)
        print(f"  val {knob}* = {knob_star:.4f}  Sharpe={row['sharpe']:+.3f}  "
              f"n_trades={int(row['n_trades'])}")

        # Test at val-chosen knob.
        out_test = fn(test_unified, **{knob: knob_star}, **extras)
        res = simulate_inventory_aware_sized(
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            out_test.open_signal, out_test.size,
            M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        test_results[name] = res
        observed = float(res.metrics["sharpe"])

        # Bootstrap p_boot via shuffled-signal null on the relevant column.
        # The "resolver" rebuilds the strategy using the shuffled p column;
        # for sized strategies this also pulls the shuffled q-driven size.
        p_test_col = test_unified[shuffle_col].to_numpy(dtype=float)

        def resolver_factory(spec_local, p_col_name):
            def _resolver(p_shuffled):
                pred_df_shuf = test_unified.copy()
                pred_df_shuf[p_col_name] = p_shuffled
                # Recompute confidence_score if mondrian_aci_size; for
                # other strategies the shuffle still flows correctly through
                # the strategy fn's column reads.
                return spec_local["fn"](
                    pred_df_shuf,
                    **{spec_local["knob"]: knob_star},
                    **spec_local["extras"],
                )
            return _resolver

        nulls = _shuffled_null(
            resolver_factory(spec, shuffle_col),
            p_test_col,
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            n=N_NULL_BOOTSTRAP, seed=SEED + abs(hash(name)) % 1000,
        )
        p_boot = float((nulls >= observed).sum() / len(nulls))
        var_within = float(np.var(sweep["sharpe"], ddof=1)) if len(sweep) > 1 else 0.0
        dsr_within, sr_star_within = deflated_sharpe(
            observed_sharpe=observed,
            var_trial_sharpes=var_within,
            n_trials=max(len(grid), 1),
            n_obs=max(int(res.metrics["n_trades"]), 2),
            skew=0.0, kurt=3.0,
        )

        # Walk-forward.
        wf = walk_forward_backtest(
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            out_test.open_signal, out_test.size,
            n_folds=N_FOLDS, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        wf_mean = float(wf["sharpe"].mean())
        wf_std = float(wf["sharpe"].std(ddof=1)) if len(wf) > 1 else 0.0
        for r in wf.to_dict("records"):
            r["strategy"] = name
            walk_rows.append(r)

        print(f"  test Sharpe={observed:+.3f}  PSR={res.metrics['probabilistic_sharpe']:.3f}  "
              f"n={int(res.metrics['n_trades'])}  p_boot={p_boot:.3f}  "
              f"DSR_within={dsr_within:.3f}  WF Sharpe={wf_mean:+.3f}±{wf_std:.3f}")

        table_rows.append({
            "strategy": name,
            "knob": knob,
            "knob_star": knob_star,
            "val_sharpe_at_star": float(row["sharpe"]),
            "val_n_trades": int(row["n_trades"]),
            "test_sharpe": observed,
            "test_psr": float(res.metrics["probabilistic_sharpe"]),
            "test_n_trades": int(res.metrics["n_trades"]),
            "test_total_log_return": float(res.metrics["total_log_return"]),
            "test_max_drawdown_log": float(res.metrics["max_drawdown_log"]),
            "test_cdar_5pct_log": float(res.metrics["cdar_5pct_log"]),
            "test_hit_rate": float(res.metrics["hit_rate"]),
            "test_avg_bars_held": float(res.metrics["avg_bars_held"]),
            "p_boot": p_boot,
            "null_sharpe_mean": float(nulls.mean()),
            "null_sharpe_std": float(nulls.std(ddof=1)) if len(nulls) > 1 else 0.0,
            "deflated_sharpe_within": float(dsr_within),
            "walk_forward_sharpe_mean": wf_mean,
            "walk_forward_sharpe_std": wf_std,
        })

    # 6) Multi-strategy DSR (all knob-trials pooled per §A.4).
    var_total = float(np.var(all_val_sharpes, ddof=1)) if len(all_val_sharpes) > 1 else 0.0
    print(f"\n[round 012] multi-strategy DSR with n_trials={n_trials_total}, "
          f"var_trial={var_total:.3f}")
    for r in table_rows:
        dsr_multi, sr_star_multi = deflated_sharpe(
            observed_sharpe=r["test_sharpe"],
            var_trial_sharpes=var_total,
            n_trials=int(n_trials_total),
            n_obs=max(int(r["test_n_trades"]), 2),
            skew=0.0, kurt=3.0,
        )
        r["deflated_sharpe_multi_strategy"] = float(dsr_multi)
        r["deflated_sr_star_multi"] = float(sr_star_multi)

    table = pd.DataFrame(table_rows)
    csv_path = OUTDIR / "phase_A_partial_v2.csv"
    table.to_csv(csv_path, index=False)
    print(f"\n[round 012] partial table v2 → {csv_path}")
    print(table[
        ["strategy", "knob_star", "val_sharpe_at_star", "test_sharpe",
         "test_n_trades", "p_boot", "deflated_sharpe_within",
         "deflated_sharpe_multi_strategy", "walk_forward_sharpe_mean",
         "walk_forward_sharpe_std"]
    ].to_string(index=False))

    walk_df = pd.DataFrame(walk_rows)
    walk_df.to_csv(OUTDIR / "walk_forward_per_strategy.csv", index=False)

    # 7) Plots.
    colors = {
        "baseline_offline_tau":  "#1B6B33",
        "baseline_online_tau":   "#1B3F6B",
        "combined_avg_tau":      "#7A1B1B",
        "combined_stacked_tau":  "#7A5C00",
        "conformal_gate_tau":    "#6B1B6B",
        "mondrian_aci_size":     "#1B6B6B",
        "null_random_at_rate":   "#888888",
    }

    # -- Plot 1: val sweep, all 7 strategies overlaid (knob axis varies).
    fig, ax = plt.subplots(figsize=(11, 6))
    for spec in specs:
        name = spec["name"]
        sweep = val_sweeps[name]
        knob = spec["knob"]
        if knob == "target_rate" and len(sweep) == 1:
            ax.scatter(sweep[knob], sweep["sharpe"], color=colors[name],
                        s=60, marker="*", label=name, zorder=10)
            continue
        # Map knobs onto a common x-axis: τ/k natural units.
        ax.plot(sweep[knob], sweep["sharpe"], color=colors[name], marker="o",
                markersize=3.5, linewidth=1.4, alpha=0.95,
                label=f"{name} ({knob})")
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xlabel("knob value (τ for {0..6}, k for mondrian_aci_size, target_rate for null)")
    ax.set_ylabel("val Sharpe")
    ax.set_title("VAL knob-sweep — Phase A all 7 strategies")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "val_tau_sweep_round2.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # -- Plot 2: equity overlay on test, val-chosen knob per strategy.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.4, 1]})
    for name, res in test_results.items():
        eq = res.equity.to_numpy(dtype=float)
        bar_idx = np.arange(len(eq))
        running_max = np.maximum.accumulate(eq)
        dd = eq - running_max
        row = next(r for r in table_rows if r["strategy"] == name)
        label = (
            f"{name}: {row['knob']}*={row['knob_star']:.3f}  "
            f"Sharpe={row['test_sharpe']:+.2f}  "
            f"WF={row['walk_forward_sharpe_mean']:+.2f}±{row['walk_forward_sharpe_std']:.2f}  "
            f"DSR_m={row['deflated_sharpe_multi_strategy']:.2f}"
        )
        axes[0].plot(bar_idx, eq, color=colors[name], linewidth=0.95, alpha=0.95,
                      label=label)
        axes[1].fill_between(bar_idx, dd, color=colors[name], alpha=0.18,
                              linewidth=0)
    axes[0].set_title("TEST equity curve overlay — Phase A all 7 strategies (val-chosen knob)")
    axes[0].set_ylabel("cumulative log return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=7.5)
    axes[1].set_xlabel("test decision-bar index")
    axes[1].set_ylabel("drawdown (log)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "equity_overlay_round2.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # -- Plot 3: walk-forward Sharpe bars, one bar per strategy (mean ± std).
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [r["strategy"] for r in table_rows]
    means = [r["walk_forward_sharpe_mean"] for r in table_rows]
    stds = [r["walk_forward_sharpe_std"] for r in table_rows]
    bar_colors = [colors[n] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.85,
           edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel(f"walk-forward Sharpe (mean ± std across {N_FOLDS} folds)")
    ax.set_title(f"Walk-forward Sharpe — {N_FOLDS} contiguous folds of test")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "walk_forward_sharpe_bars.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "012",
        "phase": "A",
        "phase_round": 2,
        "claim": "all 7 §A.2 strategies under unified harness with walk-forward and multi-strategy DSR",
        "alpha": ALPHA,
        "M": M,
        "cost_bps_per_side": COST_BPS,
        "n_strategies": len(specs),
        "n_trials_total": int(n_trials_total),
        "n_walk_forward_folds": N_FOLDS,
        "var_trial_pooled": var_total,
        "n_val": int(len(val_unified)),
        "n_test": int(len(test_unified)),
        "results": table_rows,
        "wall_clock_s": float(time.time() - t0),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8",
    )
    print(f"\n[round 012] saved → {OUTDIR}")
    print(f"  wall clock: {headline['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
