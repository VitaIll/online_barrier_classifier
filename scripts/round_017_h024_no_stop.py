"""Round 017 — H-024 no-stop unified Phase A backtest.

Per round-016 mandate Phase A round 1: re-run round-015's 5 valid strategies
under c_stop=float("inf") (no stop-loss), keeping all other config identical.
Tests whether the symmetric-barrier label/backtest mismatch was the binding
constraint in round-009/round-015's null result.

Reuses round-015's cached unified prediction parquets (no ARF replay needed)
and minute price data. Outputs under RESEARCH/diagrams/phase_A/round_017/.

Architecture (preserved): two-layer; p_online is the system output;
Mondrian-ACI on p_online; no averaging/stacking of (p_offline, p_online).
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
    cscv_pbo,
    deflated_sharpe,
    simulate_inventory_aware_sized,
    walk_forward_backtest,
)
from src.strategies import (  # noqa: E402
    StrategyOutput,
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
C_STOP_NO_STOP = float("inf")  # ← H-024: disable stop-loss
N_NULL_BOOTSTRAP = 200
SEED = 42
TAU_GRID = tuple(round(0.10 + 0.02 * i, 4) for i in range(26))
K_GRID = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
N_FOLDS = 5
CSCV_N_CHUNKS = 16

OUTDIR = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_017"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ARTIFACTS = REPO / "artifacts" / "offline_model"
ONLINE_PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
ROUND015_TABLE = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_015" / "phase_A_table.csv"
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


def _sweep_knob(
    strategy_fn: Callable[..., StrategyOutput],
    pred_df: pd.DataFrame,
    boundaries: pd.DataFrame,
    mn_c: np.ndarray, mn_h: np.ndarray, mn_l: np.ndarray,
    knob_name: str,
    knob_grid: Sequence[float],
    *,
    extra_kwargs: dict | None = None,
    c_stop: float = C_STOP_NO_STOP,
) -> pd.DataFrame:
    rows = []
    extras = dict(extra_kwargs or {})
    for v in knob_grid:
        out = strategy_fn(pred_df, **{knob_name: v}, **extras)
        res = simulate_inventory_aware_sized(
            boundaries, mn_c, mn_h, mn_l, out.open_signal, out.size,
            M=M, phi=ALPHA, c_stop=c_stop, cost_bps=COST_BPS,
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
            "n_tp": m["n_tp"],
            "n_sl": m["n_sl"],
            "n_timeout": m["n_timeout"],
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


def _shuffled_null(
    spec: dict,
    p_to_shuffle: np.ndarray,
    test_unified: pd.DataFrame,
    boundaries: pd.DataFrame,
    mn_c: np.ndarray, mn_h: np.ndarray, mn_l: np.ndarray,
    knob_star: float,
    shuffle_col: str,
    *,
    n: int = N_NULL_BOOTSTRAP, seed: int = SEED,
    c_stop: float = C_STOP_NO_STOP,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nulls = np.empty(n, dtype=float)
    for i in range(n):
        p_shuf = p_to_shuffle.copy()
        rng.shuffle(p_shuf)
        df_shuf = test_unified.copy()
        df_shuf[shuffle_col] = p_shuf
        out = spec["fn"](df_shuf, **{spec["knob"]: knob_star}, **spec["extras"])
        res = simulate_inventory_aware_sized(
            boundaries, mn_c, mn_h, mn_l, out.open_signal, out.size,
            M=M, phi=ALPHA, c_stop=c_stop, cost_bps=COST_BPS,
        )
        nulls[i] = res.metrics["sharpe"]
    return nulls


def _strategy_returns(
    spec: dict,
    pred_df: pd.DataFrame,
    boundaries: pd.DataFrame,
    mn_c: np.ndarray, mn_h: np.ndarray, mn_l: np.ndarray,
    knob_value: float,
    *, c_stop: float = C_STOP_NO_STOP,
) -> np.ndarray:
    out = spec["fn"](pred_df, **{spec["knob"]: knob_value}, **spec["extras"])
    res = simulate_inventory_aware_sized(
        boundaries, mn_c, mn_h, mn_l, out.open_signal, out.size,
        M=M, phi=ALPHA, c_stop=c_stop, cost_bps=COST_BPS,
    )
    eq = res.equity.to_numpy(dtype=float)
    return np.diff(eq, prepend=0.0) if len(eq) else np.zeros(0)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 017] ===== H-024 no-stop unified Phase A (c_stop=inf) =====")
    t0 = time.time()

    # 1) Load splits + minute data.
    print("[round 017] loading features + minute data ...")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    print(f"  split: train={len(train):,}  val={len(val):,}  test={len(test_full):,}")
    minute_df = _load_minute()

    # 2) Load cached unified prediction parquets from round-015.
    val_unified = pd.read_parquet(PHASE_A_DIR / "val_predictions_unified.parquet")
    test_unified = pd.read_parquet(PHASE_A_DIR / "test_predictions_unified.parquet")
    print(f"  loaded val_unified ({len(val_unified)}) + test_unified ({len(test_unified)})")

    # 3) Boundary mapping.
    val_for_bound = val[["open_time", "close_time"]].reset_index(drop=True)
    val_bound, val_keep, val_mn_c, val_mn_h, val_mn_l = _build_boundaries(
        val_for_bound, minute_df,
    )
    print(f"  val bars kept: {int(val_keep.sum())}")

    test_for_bound = test_full[["open_time", "close_time"]].reset_index(drop=True)
    test_bound, test_keep, test_mn_c, test_mn_h, test_mn_l = _build_boundaries(
        test_for_bound, minute_df,
    )
    print(f"  test bars kept: {int(test_keep.sum())}")

    # 4) Five strategies under c_stop=inf.
    qsweep = _sweep_knob(
        baseline_offline_tau, val_unified, val_bound, val_mn_c, val_mn_h, val_mn_l,
        "tau", TAU_GRID,
    )
    tau_off, _ = _pick_winning_knob(qsweep, "tau")
    out_off_test = baseline_offline_tau(test_unified, tau=tau_off)
    test_offline_rate = float(out_off_test.open_signal.mean())
    print(f"  baseline_offline test rate at val-tau* {tau_off:.3f}: {test_offline_rate:.4f}")

    specs: list[dict] = [
        {"name": "baseline_offline_tau", "fn": baseline_offline_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {},
         "shuffle_col": "p_offline"},
        {"name": "baseline_online_tau", "fn": baseline_online_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {},
         "shuffle_col": "p_online"},
        {"name": "conformal_gate_tau", "fn": conformal_gate_tau,
         "knob": "tau", "grid": TAU_GRID, "extras": {"alpha_level": "10"},
         "shuffle_col": "p_online"},
        {"name": "mondrian_aci_size", "fn": mondrian_aci_size,
         "knob": "k", "grid": K_GRID, "extras": {"alpha_level": "10"},
         "shuffle_col": "p_online"},
        {"name": "null_random_at_rate", "fn": null_random_at_rate,
         "knob": "target_rate", "grid": (test_offline_rate,),
         "extras": {"seed": SEED}, "shuffle_col": "p_online"},
    ]
    n_trials_total = sum(len(s["grid"]) for s in specs)
    print(f"[round 017] strategies={len(specs)}  total knob-trials={n_trials_total}")

    val_sweeps: dict[str, pd.DataFrame] = {}
    test_results: dict[str, BacktestResult] = {}
    table_rows: list[dict] = []
    walk_rows: list[dict] = []
    all_val_sharpes: list[float] = []

    for spec in specs:
        name = spec["name"]
        knob = spec["knob"]
        grid = spec["grid"]
        print(f"\n[round 017] >>> strategy={name}  knob={knob}")
        sweep = _sweep_knob(
            spec["fn"], val_unified, val_bound, val_mn_c, val_mn_h, val_mn_l,
            knob, grid, extra_kwargs=spec["extras"],
        )
        sweep.to_csv(OUTDIR / f"val_sweep_{name}.csv", index=False)
        val_sweeps[name] = sweep
        all_val_sharpes.extend(float(s) for s in sweep["sharpe"])

        knob_star, row = _pick_winning_knob(sweep, knob)
        print(f"  val {knob}* = {knob_star:.4f}  Sharpe={row['sharpe']:+.3f}  "
              f"n_trades={int(row['n_trades'])}")

        out_test = spec["fn"](test_unified, **{knob: knob_star}, **spec["extras"])
        res = simulate_inventory_aware_sized(
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            out_test.open_signal, out_test.size,
            M=M, phi=ALPHA, c_stop=C_STOP_NO_STOP, cost_bps=COST_BPS,
        )
        test_results[name] = res
        observed = float(res.metrics["sharpe"])

        nulls = _shuffled_null(
            spec, test_unified[spec["shuffle_col"]].to_numpy(dtype=float),
            test_unified, test_bound, test_mn_c, test_mn_h, test_mn_l,
            knob_star, spec["shuffle_col"],
            n=N_NULL_BOOTSTRAP, seed=SEED + abs(hash(name)) % 1000,
        )
        p_boot = float((nulls >= observed).sum() / len(nulls))

        var_within = float(np.var(sweep["sharpe"], ddof=1)) if len(sweep) > 1 else 0.0
        if len(grid) > 1:
            dsr_within, _ = deflated_sharpe(
                observed_sharpe=observed,
                var_trial_sharpes=var_within,
                n_trials=len(grid),
                n_obs=max(int(res.metrics["n_trades"]), 2),
                skew=0.0, kurt=3.0,
            )
        else:
            dsr_within = float("nan")

        wf = walk_forward_backtest(
            test_bound, test_mn_c, test_mn_h, test_mn_l,
            out_test.open_signal, out_test.size,
            n_folds=N_FOLDS, M=M, phi=ALPHA, c_stop=C_STOP_NO_STOP, cost_bps=COST_BPS,
        )
        wf_mean = float(wf["sharpe"].mean())
        wf_std = float(wf["sharpe"].std(ddof=1)) if len(wf) > 1 else 0.0
        for r in wf.to_dict("records"):
            r["strategy"] = name
            walk_rows.append(r)

        print(f"  test Sharpe={observed:+.3f}  PSR={res.metrics['probabilistic_sharpe']:.3f}  "
              f"n={int(res.metrics['n_trades'])}  p_boot={p_boot:.3f}  "
              f"DSR_within={dsr_within if not np.isnan(dsr_within) else 'n/a'}  "
              f"WF Sharpe={wf_mean:+.3f}+-{wf_std:.3f}")

        table_rows.append({
            "strategy": name,
            "knob": knob,
            "knob_star": knob_star,
            "val_sharpe_at_star": float(row["sharpe"]),
            "val_n_trades": int(row["n_trades"]),
            "test_sharpe": observed,
            "test_psr": float(res.metrics["probabilistic_sharpe"]),
            "test_n_trades": int(res.metrics["n_trades"]),
            "test_n_tp": int(res.metrics["n_tp"]),
            "test_n_sl": int(res.metrics["n_sl"]),
            "test_n_timeout": int(res.metrics["n_timeout"]),
            "test_total_log_return": float(res.metrics["total_log_return"]),
            "test_max_drawdown_log": float(res.metrics["max_drawdown_log"]),
            "test_cdar_5pct_log": float(res.metrics["cdar_5pct_log"]),
            "test_hit_rate": float(res.metrics["hit_rate"]),
            "test_avg_bars_held": float(res.metrics["avg_bars_held"]),
            "p_boot": p_boot,
            "null_sharpe_mean": float(nulls.mean()),
            "null_sharpe_std": float(nulls.std(ddof=1)) if len(nulls) > 1 else 0.0,
            "deflated_sharpe_within": dsr_within,
            "walk_forward_sharpe_mean": wf_mean,
            "walk_forward_sharpe_std": wf_std,
        })

    var_total = float(np.var(all_val_sharpes, ddof=1)) if len(all_val_sharpes) > 1 else 0.0
    print(f"\n[round 017] multi-strategy DSR: n_trials={n_trials_total}, "
          f"var_trial_pooled={var_total:.3f}")
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

    print("\n[round 017] CSCV per-knob PBO ...")
    cscv_rows = []
    for spec in specs:
        name = spec["name"]
        if len(spec["grid"]) < 2:
            cscv_rows.append({
                "strategy": name, "n_knobs": int(len(spec["grid"])),
                "pbo": float("nan"),
            })
            continue
        rmat = np.vstack([
            _strategy_returns(spec, test_unified, test_bound,
                                test_mn_c, test_mn_h, test_mn_l, v)
            for v in spec["grid"]
        ])
        out = cscv_pbo(rmat, n_chunks=CSCV_N_CHUNKS)
        cscv_rows.append({
            "strategy": name,
            "n_knobs": int(rmat.shape[0]),
            "n_periods": int(rmat.shape[1]),
            "n_chunks": int(out["n_chunks"]),
            "n_combinations": int(out["n_combinations"]),
            "pbo": float(out["pbo"]),
            "median_logit": float(out["median_logit"]),
            "is_best_modal_knob": float(spec["grid"][int(out["is_best_strategy_modal_index"])]),
        })
        print(f"  {name}: PBO={out['pbo']:.3f}  median_logit={out['median_logit']:+.3f}")
    pd.DataFrame(cscv_rows).to_csv(OUTDIR / "cscv_per_strategy.csv", index=False)
    pbo_lookup = {r["strategy"]: r["pbo"] for r in cscv_rows}
    for r in table_rows:
        r["within_strategy_pbo"] = float(pbo_lookup.get(r["strategy"], float("nan")))

    print("\n[round 017] cross-strategy CSCV ...")
    cross_rows = [
        _strategy_returns(spec, test_unified, test_bound,
                            test_mn_c, test_mn_h, test_mn_l,
                            float(next(r["knob_star"] for r in table_rows
                                        if r["strategy"] == spec["name"])))
        for spec in specs
    ]
    R_cross = np.vstack(cross_rows)
    cross_pbo = cscv_pbo(R_cross, n_chunks=CSCV_N_CHUNKS)
    print(f"  cross-strategy PBO={cross_pbo['pbo']:.3f}  "
          f"modal IS-best={specs[int(cross_pbo['is_best_strategy_modal_index'])]['name']}")
    cross_summary = {
        "pbo": float(cross_pbo["pbo"]),
        "n_combinations": int(cross_pbo["n_combinations"]),
        "modal_is_best_strategy": specs[
            int(cross_pbo["is_best_strategy_modal_index"])
        ]["name"],
        "is_best_counts": [
            {"strategy": specs[i]["name"], "count": int(c)}
            for i, c in enumerate(cross_pbo["is_best_strategy_counts"])
        ],
    }
    (OUTDIR / "cscv_cross_strategy.json").write_text(
        json.dumps(cross_summary, indent=2), encoding="utf-8",
    )

    table = pd.DataFrame(table_rows)
    table.to_csv(OUTDIR / "phase_A_table.csv", index=False)
    pd.DataFrame(walk_rows).to_csv(OUTDIR / "walk_forward_per_strategy.csv", index=False)
    print("\n[round 017] final 5-row Phase A table (NO-STOP):")
    print(table[
        ["strategy", "knob_star", "val_sharpe_at_star", "test_sharpe",
         "test_n_trades", "test_n_tp", "test_n_timeout",
         "p_boot", "deflated_sharpe_multi_strategy",
         "walk_forward_sharpe_mean", "walk_forward_sharpe_std",
         "within_strategy_pbo"]
    ].to_string(index=False))

    # Diff vs round-015 (with c_stop=alpha)
    diff_rows = []
    if ROUND015_TABLE.exists():
        r015 = pd.read_csv(ROUND015_TABLE)
        for _, row017 in table.iterrows():
            r015row = r015[r015["strategy"] == row017["strategy"]]
            if len(r015row) == 0:
                continue
            r015row = r015row.iloc[0]
            diff_rows.append({
                "strategy": row017["strategy"],
                "r015_knob_star": float(r015row["knob_star"]),
                "r017_knob_star": float(row017["knob_star"]),
                "r015_test_sharpe": float(r015row["test_sharpe"]),
                "r017_test_sharpe": float(row017["test_sharpe"]),
                "delta_sharpe": float(row017["test_sharpe"] - r015row["test_sharpe"]),
                "r015_n_trades": int(r015row["test_n_trades"]),
                "r017_n_trades": int(row017["test_n_trades"]),
                "r015_dsr": float(r015row["deflated_sharpe_multi_strategy"]),
                "r017_dsr": float(row017["deflated_sharpe_multi_strategy"]),
            })
        pd.DataFrame(diff_rows).to_csv(OUTDIR / "diff_vs_round_015.csv", index=False)
        print("\n[round 017] diff vs round-015 (c_stop=alpha):")
        print(pd.DataFrame(diff_rows).to_string(index=False))

    colors = {
        "baseline_offline_tau": "#1B6B33",
        "baseline_online_tau":  "#1B3F6B",
        "conformal_gate_tau":   "#6B1B6B",
        "mondrian_aci_size":    "#1B6B6B",
        "null_random_at_rate":  "#888888",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for spec in specs:
        name = spec["name"]
        sweep = val_sweeps[name]
        knob = spec["knob"]
        if knob == "target_rate" and len(sweep) == 1:
            ax.scatter(sweep[knob], sweep["sharpe"], color=colors[name],
                        s=80, marker="*", label=name, zorder=10)
            continue
        ax.plot(sweep[knob], sweep["sharpe"], color=colors[name], marker="o",
                markersize=3.5, linewidth=1.4, alpha=0.95,
                label=f"{name} ({knob})")
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xlabel("knob")
    ax.set_ylabel("val Sharpe")
    ax.set_title("VAL knob-sweep — round-017 H-024 NO-STOP (c_stop=inf)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "val_knob_sweep.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

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
            f"WF={row['walk_forward_sharpe_mean']:+.2f}+-{row['walk_forward_sharpe_std']:.2f}  "
            f"DSR_m={row['deflated_sharpe_multi_strategy']:.2f}"
        )
        axes[0].plot(bar_idx, eq, color=colors[name], linewidth=0.95, alpha=0.95,
                      label=label)
        axes[1].fill_between(bar_idx, dd, color=colors[name], alpha=0.20,
                              linewidth=0)
    axes[0].set_title("TEST equity overlay — round-017 H-024 NO-STOP (c_stop=inf)")
    axes[0].set_ylabel("cumulative log return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("test decision-bar index")
    axes[1].set_ylabel("drawdown (log)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "equity_overlay.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: comparison bars vs round-015
    if diff_rows:
        fig, ax = plt.subplots(figsize=(11, 5))
        names_d = [d["strategy"] for d in diff_rows]
        r015_sharpes = [d["r015_test_sharpe"] for d in diff_rows]
        r017_sharpes = [d["r017_test_sharpe"] for d in diff_rows]
        x = np.arange(len(names_d))
        w = 0.4
        ax.bar(x - w/2, r015_sharpes, w, color="#666", alpha=0.85,
               edgecolor="black", linewidth=0.4, label="round-015 (c_stop=alpha)")
        ax.bar(x + w/2, r017_sharpes, w, color="#1B6B33", alpha=0.85,
               edgecolor="black", linewidth=0.4, label="round-017 (c_stop=inf)")
        ax.axhline(0.0, color="gray", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names_d, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("test Sharpe")
        ax.set_title("H-024 no-stop vs round-015 stop=alpha")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTDIR / "diff_vs_round_015_bars.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    headline = {
        "round_id": "017",
        "phase": "A",
        "phase_round": "h024_no_stop",
        "claim": "H-024 no-stop unified Phase A run; c_stop=inf, all else identical to round-015",
        "alpha": ALPHA, "M": M, "cost_bps_per_side": COST_BPS,
        "c_stop": "inf",
        "n_strategies": int(len(specs)),
        "n_trials_total": int(n_trials_total),
        "n_walk_forward_folds": N_FOLDS,
        "var_trial_pooled": var_total,
        "n_val": int(len(val_unified)),
        "n_test": int(len(test_unified)),
        "best_test_sharpe_strategy": str(
            table.sort_values("test_sharpe", ascending=False).iloc[0]["strategy"]
        ),
        "best_test_sharpe": float(table["test_sharpe"].max()),
        "best_dsr_multi": float(table["deflated_sharpe_multi_strategy"].max()),
        "cross_strategy_pbo": float(cross_pbo["pbo"]),
        "cross_strategy_modal_is_best": cross_summary["modal_is_best_strategy"],
        "phase_A_table": table.to_dict(orient="records"),
        "diff_vs_round_015": diff_rows,
        "wall_clock_s": float(time.time() - t0),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 017] best test Sharpe = "
          f"{headline['best_test_sharpe']:+.3f} ({headline['best_test_sharpe_strategy']})  "
          f"best DSR_m = {headline['best_dsr_multi']:.3f}  "
          f"cross PBO = {headline['cross_strategy_pbo']:.3f}  "
          f"wall {headline['wall_clock_s']:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
