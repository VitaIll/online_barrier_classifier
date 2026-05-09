"""Phase A round 1 — baselines 1-3 against the unified harness (MANDATE §A).

Strategies covered this round:
    1. baseline_offline_tau   — open = p_offline > τ; size = 1
    2. baseline_online_tau    — open = p_online  > τ; size = 1
    3. combined_avg_tau       — open = 0.5·p_off + 0.5·p_on > τ; size = 1

For each: fit the τ on val Sharpe (grid {0.10..0.60, step 0.02}), apply to test
in `simulate_inventory_aware` with the round-001 settings (φ = c_stop = α =
0.0041113, M=20, cost = 1bp/side). Bootstrap p-value via shuffled signal.
Persist the partial Phase-A table — strategies 4-7 land in subsequent rounds.

Outputs (under `RESEARCH/diagrams/phase_A/round_011/`):
    val_predictions_unified.parquet  — val rows with regime + p_offline + p_online
    test_predictions_unified.parquet — test rows likewise
    regime_cuts.json                 — frozen tercile edges (fit on train+val)
    val_tau_sweep_baselines.csv      — val Sharpe per strategy, per τ
    val_tau_sweep_baselines.png      — visual
    phase_A_partial_v1.csv           — first 3 rows of §A.6 table
    equity_overlay_baselines.png     — overlay of test equity at val-chosen τ
    headline.json                    — round summary

This script is reusable: rounds 012/013 will extend `phase_A_partial_v1.csv`
to the full §A.6 table without re-doing rounds 1-3.

Round-011 contract:
- Uses the persisted offline+online predictions (no retrain).
- Reads val predictions from RESEARCH/diagrams/round_009/val_predictions.parquet
  (n=9,445; produced by round 009 from artifacts/offline_model/model.cbm).
- Re-runs the online ARFClassifier prequentially over the val slice to populate
  `p_online` for val (round 009 only persisted p_offline). Caches the resulting
  parquet so subsequent rounds reuse it.
- Reuses the test predictions parquet (artifacts/online_eval/predictions.parquet).
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
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
)
from src.inference import (  # noqa: E402
    fit_regime_cuts,
    predict,
    warm_q_init_by_regime,
)
from src.strategies import (  # noqa: E402
    baseline_offline_tau,
    baseline_online_tau,
    combined_avg_tau,
)
from src.utils import chronological_split, load_config  # noqa: E402

# ----------------------------------------------------------------------------
# Constants — preserved from round-001/009 so results stay comparable.
# ----------------------------------------------------------------------------
ALPHA = 0.0041113
M = 20
COST_BPS = 1.0
N_NULL_BOOTSTRAP = 200
SEED = 42
TAU_GRID = tuple(round(0.10 + 0.02 * i, 4) for i in range(26))  # 0.10..0.60 step 0.02
ALPHAS_CONFORMAL = (0.05, 0.10, 0.20)
GAMMA_ACI = 0.01

OUTDIR = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_011"
ARTIFACTS = REPO / "artifacts" / "offline_model"
ONLINE_PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
ROUND9_VAL = REPO / "RESEARCH" / "diagrams" / "round_009" / "val_predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"

REGIME_FEATURE = "parkinson_var_rolling_mean_24"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _load_minute() -> pd.DataFrame:
    return pd.read_parquet(
        MIN1, columns=["open_time", "high", "low", "close", "segment_id"]
    )


def _build_boundaries(feat_slice: pd.DataFrame, minute_df: pd.DataFrame):
    """Round-001 boundary mapping reused on an arbitrary contiguous bar slice."""
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
        boundaries,
        keep,
        mn["close"].to_numpy(),
        mn["high"].to_numpy(),
        mn["low"].to_numpy(),
    )


def _build_or_load_val_predictions(
    val_df: pd.DataFrame,
    train_plus_val_for_warmup: pd.DataFrame,
    selected_features: list[str],
    online_config: dict,
) -> pd.DataFrame:
    """Build val p_online via prequential ARF over (warmup + val).

    Round 009 only persisted val p_offline; we replay the prequential ARF on
    (last `WARMUP_BARS` of train) → val so that p_online is the prediction
    the streaming layer would have made entering val. Cached to
    `artifacts/phase_A/val_predictions_with_online.parquet`.
    """
    cache = PHASE_A_DIR / "val_predictions_with_online.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        out = pd.read_parquet(cache)
        if len(out) == len(val_df):
            return out
        # else: stale, rebuild

    print(f"[round 011] building val p_online (cache miss) …")
    from river.forest import ARFClassifier

    # Replay-from-train warmup like the notebook (avoids cold-start ARF).
    WARMUP_BARS = 96
    warmup_df = train_plus_val_for_warmup.iloc[-WARMUP_BARS:].reset_index(drop=True)
    stream_df = pd.concat(
        [
            warmup_df.assign(_is_warmup=True),
            val_df.assign(_is_warmup=False),
        ],
        ignore_index=True,
    )

    # We need p_offline for the ARF input; load the persisted offline model.
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(ARTIFACTS / "model.cbm"))
    model_feats = list(model.feature_names_)

    online_model = ARFClassifier(
        n_models=int(online_config["n_models"]),
        max_features=online_config["max_features"],
        lambda_value=int(online_config["lambda_value"]),
        seed=int(online_config["seed"]),
    )

    label_buffer: deque = deque()
    p_off_all = []
    p_on_all = []
    is_warmup_all = []

    # Pre-compute p_offline batchwise (faster than row-by-row).
    p_off_all_batch = model.predict_proba(stream_df[model_feats])[:, 1]

    for i in range(len(stream_df)):
        row = stream_df.iloc[i]
        is_warmup = bool(row["_is_warmup"])
        current_segment = int(row["segment_id"])

        p_off = float(p_off_all_batch[i])

        z = {f: float(row[f]) for f in selected_features}
        z["p_offline"] = p_off

        p_online_dict = online_model.predict_proba_one(z)
        p_online = p_online_dict.get(1, 0.5) if p_online_dict else 0.5

        # Train when previous bar's label is known.
        if len(label_buffer) > 0:
            buffered = label_buffer.popleft()
            if buffered["segment_id"] == current_segment:
                # Use the bar's pre-computed `label` column (matches the
                # notebook's online_eval semantics — labels come from the
                # feature parquet).
                y_delayed = buffered["label"]
            else:
                y_delayed = None
            if y_delayed is not None and pd.notna(y_delayed):
                online_model.learn_one(buffered["z"], int(y_delayed))

        label_buffer.append({
            "z": z.copy(),
            "ref_close": float(row["close"]),
            "segment_id": current_segment,
            "label": row["label"],
        })

        p_off_all.append(p_off)
        p_on_all.append(p_online)
        is_warmup_all.append(is_warmup)

    arr_off = np.asarray(p_off_all, dtype=float)
    arr_on = np.asarray(p_on_all, dtype=float)
    is_warm = np.asarray(is_warmup_all, dtype=bool)
    keep = ~is_warm

    out = pd.DataFrame({
        "open_time": stream_df.loc[keep, "open_time"].to_numpy(),
        "y_true": stream_df.loc[keep, "label"].astype(int).to_numpy(),
        "p_offline": arr_off[keep],
        "p_online": arr_on[keep],
    })
    out.to_parquet(cache, index=False)
    print(f"[round 011] cached → {cache} (n={len(out)})")
    return out


def _shuffled_null(p, tau, boundaries, mn_c, mn_h, mn_l, *, n: int = N_NULL_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    nulls = np.empty(n, dtype=float)
    for i in range(n):
        p_shuf = p.copy()
        rng.shuffle(p_shuf)
        res = simulate_inventory_aware(
            boundaries, mn_c, mn_h, mn_l, p_shuf,
            tau_open=tau, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        nulls[i] = res.metrics["sharpe"]
    return nulls


def _val_sweep_for_strategy(
    strategy_fn: Callable,
    pred_df: pd.DataFrame,
    boundaries: pd.DataFrame,
    mn_c: np.ndarray,
    mn_h: np.ndarray,
    mn_l: np.ndarray,
    tau_grid: Sequence[float],
    *,
    strategy_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Sweep τ on val for one strategy.

    Builds open_signal at each τ via the strategy, encodes it as p ∈ {0, 1},
    then runs the harness with `tau_open=0.5` so that the encoded signal
    exactly fires when the strategy says fire. Size=1 per strategy spec.
    """
    rows = []
    extras = dict(strategy_kwargs or {})
    for tau in tau_grid:
        out = strategy_fn(pred_df, tau=float(tau), **extras)
        p_signal = out.open_signal.astype(float)
        res = simulate_inventory_aware(
            boundaries, mn_c, mn_h, mn_l, p_signal,
            tau_open=0.5, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        m = res.metrics
        rows.append({
            "tau": tau,
            "sharpe": m["sharpe"],
            "psr": m["probabilistic_sharpe"],
            "n_trades": m["n_trades"],
            "total_log_return": m["total_log_return"],
            "max_drawdown_log": m["max_drawdown_log"],
            "hit_rate": m["hit_rate"],
        })
    return pd.DataFrame(rows)


def _pick_tau_star(sweep: pd.DataFrame, *, min_n_trades: int = 50) -> tuple[float, dict]:
    eligible = sweep[sweep["n_trades"] >= min_n_trades]
    if eligible.empty:
        eligible = sweep
    row = eligible.sort_values(["sharpe", "n_trades"], ascending=[False, False]).iloc[0]
    return float(row["tau"]), row.to_dict()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PHASE_A_DIR.mkdir(parents=True, exist_ok=True)

    print("[round 011] ===== Phase A round 1 — baselines 1-3 =====")
    t0 = time.time()

    # 1) Splits + minute data + offline model + selected features.
    print("[round 011] loading features + minute data …")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    print(f"  split: train={len(train):,}  val={len(val):,}  test={len(test_full):,}")

    minute_df = _load_minute()

    selected_features = json.loads(
        (ARTIFACTS / "selected_features.json").read_text()
    )["features"]
    model_config = load_config(REPO / "config" / "model.yaml")
    online_config = model_config["online"]

    # 2) Persisted predictions: val p_offline (round 009) + test (online_eval).
    if not ROUND9_VAL.exists():
        raise FileNotFoundError(
            f"missing round 009 val predictions at {ROUND9_VAL}; run round 009 first."
        )

    # 3) Build val predictions table with p_online — caches.
    train_plus_val = pd.concat([train, val], ignore_index=True)
    val_pred_unified = _build_or_load_val_predictions(
        val_df=val.reset_index(drop=True),
        train_plus_val_for_warmup=train,
        selected_features=selected_features,
        online_config=online_config,
    )

    # 4) Apply gap-window keep-mask to val + test predictions.
    val_for_bound = val[["open_time", "close_time"]].reset_index(drop=True)
    val_bound, val_keep, val_mn_c, val_mn_h, val_mn_l = _build_boundaries(
        val_for_bound, minute_df,
    )
    val_pred_kept = val_pred_unified.iloc[val_keep].reset_index(drop=True)
    val_regime_kept = val.loc[val_keep, REGIME_FEATURE].to_numpy()
    print(f"  val bars kept (gap-window): {len(val_pred_kept)} / {len(val_pred_unified)}")

    test_for_bound = test_full[["open_time", "close_time"]].reset_index(drop=True)
    test_bound, test_keep, test_mn_c, test_mn_h, test_mn_l = _build_boundaries(
        test_for_bound, minute_df,
    )
    test_pred_full = pd.read_parquet(ONLINE_PRED)
    if len(test_pred_full) != len(test_full):
        raise RuntimeError(
            f"test pred mismatch: {len(test_pred_full)} vs split {len(test_full)}"
        )
    test_pred_kept = test_pred_full.iloc[test_keep].reset_index(drop=True)
    test_regime_kept = test_full.loc[test_keep, REGIME_FEATURE].to_numpy()
    print(f"  test bars kept (gap-window): {len(test_pred_kept)} / {len(test_pred_full)}")

    # 5) Frozen regime cuts on TRAIN ONLY (no leakage).
    print("[round 011] fitting regime cuts on TRAIN only …")
    train_regime_values = train[REGIME_FEATURE].to_numpy()
    regime_cuts = fit_regime_cuts(
        train_regime_values, n_regimes=3, feature_name=REGIME_FEATURE,
    )
    print(f"  regime edges (parkinson_var_rolling_mean_24): {regime_cuts.edges}")

    # 6) Warm Mondrian-ACI per-regime q on val (using val p_offline + y_true).
    val_regime_ids = regime_cuts.assign(val_regime_kept)
    q_init_per_alpha: dict[float, dict] = {}
    for a in ALPHAS_CONFORMAL:
        q_init_per_alpha[a] = warm_q_init_by_regime(
            p_cal=val_pred_kept["p_offline"].to_numpy(),
            y_cal=val_pred_kept["y_true"].to_numpy().astype(int),
            regime_cal=val_regime_ids,
            alpha=float(a),
            min_per_regime=50,
            fallback=0.5,
        )

    # 7) Build the unified val + test prediction tables via predict().
    print("[round 011] building unified prediction tables …")
    val_unified = predict(
        val_pred_kept, regime_values=val_regime_kept,
        regime_cuts=regime_cuts, alphas=ALPHAS_CONFORMAL, gamma=GAMMA_ACI,
        q_init_by_regime_per_alpha=q_init_per_alpha,
    )
    test_unified = predict(
        test_pred_kept, regime_values=test_regime_kept,
        regime_cuts=regime_cuts, alphas=ALPHAS_CONFORMAL, gamma=GAMMA_ACI,
        q_init_by_regime_per_alpha=q_init_per_alpha,
    )

    # Persist regime cuts + unified tables.
    (PHASE_A_DIR / "regime_cuts.json").write_text(
        json.dumps({
            "feature": regime_cuts.feature,
            "edges": list(regime_cuts.edges),
            "labels": list(regime_cuts.labels),
            "fit_on": "train",
            "fit_n": int(len(train)),
        }, indent=2),
        encoding="utf-8",
    )
    val_unified.to_parquet(PHASE_A_DIR / "val_predictions_unified.parquet", index=False)
    test_unified.to_parquet(PHASE_A_DIR / "test_predictions_unified.parquet", index=False)
    print(f"  persisted unified tables → {PHASE_A_DIR}")

    # 8) Strategy roster for THIS round.
    strategies: list[tuple[str, Callable, dict]] = [
        ("baseline_offline_tau", baseline_offline_tau, {}),
        ("baseline_online_tau",  baseline_online_tau,  {}),
        ("combined_avg_tau",     combined_avg_tau,     {}),
    ]

    val_sweeps: dict[str, pd.DataFrame] = {}
    headline_rows: list[dict] = []
    test_results: dict[str, BacktestResult] = {}

    for name, fn, extras in strategies:
        print(f"\n[round 011] >>> strategy={name}")
        sweep = _val_sweep_for_strategy(
            fn, val_unified, val_bound, val_mn_c, val_mn_h, val_mn_l,
            TAU_GRID, strategy_kwargs=extras,
        )
        sweep.to_csv(OUTDIR / f"val_sweep_{name}.csv", index=False)
        val_sweeps[name] = sweep
        tau_star, row = _pick_tau_star(sweep)
        print(f"  val tau* = {tau_star:.3f}  Sharpe={row['sharpe']:+.3f}  "
              f"n_trades={int(row['n_trades'])}")

        # Test at val-chosen τ.
        out_test = fn(test_unified, tau=tau_star, **extras)
        p_signal_test = out_test.open_signal.astype(float)
        res = simulate_inventory_aware(
            test_bound, test_mn_c, test_mn_h, test_mn_l, p_signal_test,
            tau_open=0.5, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        observed = float(res.metrics["sharpe"])
        nulls = _shuffled_null(
            p_signal_test, 0.5, test_bound, test_mn_c, test_mn_h, test_mn_l,
            n=N_NULL_BOOTSTRAP,
        )
        p_boot = float((nulls >= observed).sum() / len(nulls))
        var_trial = float(np.var(sweep["sharpe"], ddof=1)) if len(sweep) > 1 else 0.0
        dsr_within, sr_star_within = deflated_sharpe(
            observed_sharpe=observed,
            var_trial_sharpes=var_trial,
            n_trials=len(TAU_GRID),
            n_obs=max(int(res.metrics["n_trades"]), 2),
            skew=0.0, kurt=3.0,
        )
        # NOTE: round 013 will recompute the cross-strategy DSR with the full
        # n_trials = len(strategies) * len(τ-grid) deflation.
        print(f"  test Sharpe={observed:+.3f}  PSR={res.metrics['probabilistic_sharpe']:.3f}  "
              f"n_trades={int(res.metrics['n_trades'])}  p_boot={p_boot:.3f}  "
              f"DSR(within-strat)={dsr_within:.3f}")

        test_results[name] = res
        headline_rows.append({
            "strategy": name,
            "val_tau_star": tau_star,
            "val_sharpe_at_tau": float(row["sharpe"]),
            "val_n_trades": int(row["n_trades"]),
            "test_sharpe": observed,
            "test_psr": float(res.metrics["probabilistic_sharpe"]),
            "test_n_trades": int(res.metrics["n_trades"]),
            "test_total_log_return": float(res.metrics["total_log_return"]),
            "test_max_drawdown_log": float(res.metrics["max_drawdown_log"]),
            "test_hit_rate": float(res.metrics["hit_rate"]),
            "test_avg_bars_held": float(res.metrics["avg_bars_held"]),
            "p_boot": p_boot,
            "null_sharpe_mean": float(nulls.mean()),
            "null_sharpe_std": float(nulls.std(ddof=1)) if len(nulls) > 1 else 0.0,
            "deflated_sharpe_within_strategy": float(dsr_within),
            "deflated_sr_star_within_strategy": float(sr_star_within),
        })

    # 9) Persist partial Phase-A table + plots.
    table = pd.DataFrame(headline_rows)
    csv_path = OUTDIR / "phase_A_partial_v1.csv"
    table.to_csv(csv_path, index=False)
    print(f"\n[round 011] partial table → {csv_path}")
    print(table.to_string(index=False))

    # ---- Plot 1: val Sharpe vs τ for the three strategies ----
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "baseline_offline_tau": "#1B6B33",
        "baseline_online_tau":  "#1B3F6B",
        "combined_avg_tau":     "#7A1B1B",
    }
    for name, sweep in val_sweeps.items():
        ax.plot(sweep["tau"], sweep["sharpe"], color=colors[name], marker="o",
                markersize=3.5, linewidth=1.4, label=name)
        row = next(r for r in headline_rows if r["strategy"] == name)
        ax.axvline(row["val_tau_star"], color=colors[name], linestyle="--",
                   linewidth=0.8, alpha=0.75)
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xlabel("tau_open")
    ax.set_ylabel("val Sharpe")
    ax.set_title(
        "VAL τ-sweep — Phase A baselines 1-3  (val-chosen τ shown as dashed lines)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "val_tau_sweep_baselines.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: equity overlay on test, val-chosen τ per strategy ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})
    for name, res in test_results.items():
        eq = res.equity.to_numpy(dtype=float)
        bar_idx = np.arange(len(eq))
        running_max = np.maximum.accumulate(eq)
        dd = eq - running_max
        row = next(r for r in headline_rows if r["strategy"] == name)
        label = (
            f"{name}: τ*={row['val_tau_star']:.3f}  "
            f"Sharpe={row['test_sharpe']:+.3f}  "
            f"n={row['test_n_trades']}  "
            f"p_boot={row['p_boot']:.2f}"
        )
        axes[0].plot(bar_idx, eq, color=colors[name], linewidth=1.0, alpha=0.95,
                      label=label)
        axes[1].fill_between(bar_idx, dd, color=colors[name], alpha=0.35,
                              linewidth=0)
    axes[0].set_title(
        "TEST equity curve overlay — Phase A baselines 1-3 at val-chosen τ"
    )
    axes[0].set_ylabel("cumulative log return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("test decision-bar index")
    axes[1].set_ylabel("drawdown (log)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "equity_overlay_baselines.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "011",
        "phase": "A",
        "phase_round": 1,
        "claim": "baselines 1-3 (offline τ, online τ, combined-avg τ) under unified harness, val-chosen τ",
        "alpha": ALPHA,
        "M": M,
        "cost_bps_per_side": COST_BPS,
        "tau_grid_size": len(TAU_GRID),
        "regime_feature": REGIME_FEATURE,
        "regime_edges": list(regime_cuts.edges),
        "n_val": int(len(val_unified)),
        "n_test": int(len(test_unified)),
        "strategies_evaluated": [r["strategy"] for r in headline_rows],
        "results": headline_rows,
        "wall_clock_s": float(time.time() - t0),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8",
    )
    print(f"\n[round 011] saved → {OUTDIR}")
    print(f"  wall clock: {headline['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
