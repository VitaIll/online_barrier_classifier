"""Phase A round 015 — corrective two-layer Phase A.

Supersedes rounds 011/012/013 (which incorrectly framed `p_offline` and
`p_online` as parallel signals to be averaged or stacked). The system is
TWO-LAYER:

    offline CatBoost  ──>  p_offline
                              │
                              ▼  (p_offline + selected_features)
                    online ARFClassifier  ──>  p_online    ← system output
                              │
                              ▼
                    streaming Mondrian-ACI on p_online  ──> q_t, in_set_α

`p_online` is the system's actual output. The conformal layer sits on top
of `p_online`. Strategies that "combine" `p_offline` and `p_online` (avg,
stacking) are architecturally invalid and have been deleted.

This round runs the five legitimate strategies under the unified harness:

  1. baseline_offline_tau   (sanity floor — offline alone, before correction)
  2. baseline_online_tau    (system's actual headline — corrected output)
  3. conformal_gate_tau     ((p_online > τ) AND in_set_10 == 1, on p_online)
  4. mondrian_aci_size      (sized via Mondrian-ACI confidence on p_online)
  5. null_random_at_rate    (no-skill comparator)

For each: val tau-grid sweep on Sharpe (or k-grid for sized), pick by val
Sharpe with n_trades ≥ 50, evaluate at val-chosen knob on test, run shuffled-
signal bootstrap null (n=200), compute walk-forward (5 folds), within-
strategy + multi-strategy DSR. CSCV PBO across the per-strategy knob grids.
The whole package — once. No avg, no stacking, no offline-driven q.

Outputs (under `RESEARCH/diagrams/phase_A/round_015/`):
    val_sweep_<strategy>.csv          — per-strategy val sweep
    val_knob_sweep.png                — visual
    equity_overlay.png                — test equity at val-chosen knob
    walk_forward_per_strategy.csv     — per-fold Sharpe table
    walk_forward_sharpe_bars.png
    cscv_per_strategy.csv             — per-knob-grid PBO
    cscv_cross_strategy.json          — cross-strategy PBO
    pbo_panel.png                     — PBO bars + cross-strategy logits
    phase_A_table.csv                 — final 5-row table
    headline.json                     — round summary
And the corrected unified prediction parquets:
    artifacts/phase_A/val_predictions_unified.parquet
    artifacts/phase_A/test_predictions_unified.parquet
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
from src.inference import (  # noqa: E402
    fit_regime_cuts,
    predict,
    warm_q_init_by_regime,
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

# ----------------------------------------------------------------------------
# Constants — preserved from R009 / R011 / R012 so backtest is comparable.
# ----------------------------------------------------------------------------
ALPHA = 0.0041113
M = 20
COST_BPS = 1.0
N_NULL_BOOTSTRAP = 200
SEED = 42
TAU_GRID = tuple(round(0.10 + 0.02 * i, 4) for i in range(26))  # 0.10..0.60 step 0.02
K_GRID = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
ALPHAS_CONFORMAL = (0.05, 0.10, 0.20)
GAMMA_ACI = 0.01
N_FOLDS = 5
CSCV_N_CHUNKS = 16

OUTDIR = REPO / "RESEARCH" / "diagrams" / "phase_A" / "round_015"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ARTIFACTS = REPO / "artifacts" / "offline_model"
ONLINE_PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"
ROUND9_VAL = REPO / "RESEARCH" / "diagrams" / "round_009" / "val_predictions.parquet"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"
PHASE_A_REPORT = REPO / "phase_A_backtest_report.md"

REGIME_FEATURE = "parkinson_var_rolling_mean_24"


# ----------------------------------------------------------------------------
# Boundary mapping (reused from round-009)
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


def _build_or_load_val_predictions(
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    selected_features: list[str],
    online_config: dict,
) -> pd.DataFrame:
    """Build val (p_offline, p_online) by replaying the prequential ARF over
    (last 96 train bars + val). Cache to artifacts/phase_A/."""
    cache = PHASE_A_DIR / "val_predictions_with_online.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        out = pd.read_parquet(cache)
        if len(out) == len(val_df):
            return out

    print("[round 015] building val (p_offline, p_online) prequentially …")
    from collections import deque

    from catboost import CatBoostClassifier
    from river.forest import ARFClassifier

    WARMUP_BARS = 96
    warmup_df = train_df.iloc[-WARMUP_BARS:].reset_index(drop=True)
    stream_df = pd.concat(
        [
            warmup_df.assign(_is_warmup=True),
            val_df.assign(_is_warmup=False),
        ],
        ignore_index=True,
    )
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
    p_off_batch = model.predict_proba(stream_df[model_feats])[:, 1]
    p_off_all, p_on_all, is_warm = [], [], []

    for i in range(len(stream_df)):
        row = stream_df.iloc[i]
        is_w = bool(row["_is_warmup"])
        seg_id = int(row["segment_id"])
        p_off = float(p_off_batch[i])
        z = {f: float(row[f]) for f in selected_features}
        z["p_offline"] = p_off
        p_on_dict = online_model.predict_proba_one(z)
        p_on = p_on_dict.get(1, 0.5) if p_on_dict else 0.5

        if len(label_buffer) > 0:
            buffered = label_buffer.popleft()
            if buffered["segment_id"] == seg_id:
                y_delayed = buffered["label"]
            else:
                y_delayed = None
            if y_delayed is not None and pd.notna(y_delayed):
                online_model.learn_one(buffered["z"], int(y_delayed))
        label_buffer.append({
            "z": z.copy(),
            "ref_close": float(row["close"]),
            "segment_id": seg_id,
            "label": row["label"],
        })
        p_off_all.append(p_off)
        p_on_all.append(p_on)
        is_warm.append(is_w)

    arr_off = np.asarray(p_off_all, dtype=float)
    arr_on = np.asarray(p_on_all, dtype=float)
    keep = ~np.asarray(is_warm, dtype=bool)
    out = pd.DataFrame({
        "open_time": stream_df.loc[keep, "open_time"].to_numpy(),
        "y_true": stream_df.loc[keep, "label"].astype(int).to_numpy(),
        "p_offline": arr_off[keep],
        "p_online": arr_on[keep],
    })
    out.to_parquet(cache, index=False)
    print(f"  cached → {cache} (n={len(out)})")
    return out


# ----------------------------------------------------------------------------
# Sweep + bootstrap helpers
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
    rows = []
    extras = dict(extra_kwargs or {})
    for v in knob_grid:
        out = strategy_fn(pred_df, **{knob_name: v}, **extras)
        res = simulate_inventory_aware_sized(
            boundaries, mn_c, mn_h, mn_l, out.open_signal, out.size,
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
            M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        nulls[i] = res.metrics["sharpe"]
    return nulls


def _strategy_returns(
    spec: dict,
    pred_df: pd.DataFrame,
    boundaries: pd.DataFrame,
    mn_c: np.ndarray, mn_h: np.ndarray, mn_l: np.ndarray,
    knob_value: float,
) -> np.ndarray:
    out = spec["fn"](pred_df, **{spec["knob"]: knob_value}, **spec["extras"])
    res = simulate_inventory_aware_sized(
        boundaries, mn_c, mn_h, mn_l, out.open_signal, out.size,
        M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
    )
    eq = res.equity.to_numpy(dtype=float)
    return np.diff(eq, prepend=0.0) if len(eq) else np.zeros(0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PHASE_A_DIR.mkdir(parents=True, exist_ok=True)
    print("[round 015] ===== Phase A corrective round — two-layer architecture =====")
    t0 = time.time()

    # 1) Splits + minute data + offline model.
    print("[round 015] loading features + minute data …")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    print(f"  split: train={len(train):,}  val={len(val):,}  test={len(test_full):,}")
    minute_df = _load_minute()

    selected_features = json.loads(
        (ARTIFACTS / "selected_features.json").read_text()
    )["features"]
    from src.utils import load_config
    model_config = load_config(REPO / "config" / "model.yaml")
    online_config = model_config["online"]

    # 2) Val predictions (p_offline + p_online) via prequential ARF replay.
    val_pred = _build_or_load_val_predictions(
        val_df=val.reset_index(drop=True),
        train_df=train,
        selected_features=selected_features,
        online_config=online_config,
    )

    # 3) Boundary mapping.
    val_for_bound = val[["open_time", "close_time"]].reset_index(drop=True)
    val_bound, val_keep, val_mn_c, val_mn_h, val_mn_l = _build_boundaries(
        val_for_bound, minute_df,
    )
    val_pred_kept = val_pred.iloc[val_keep].reset_index(drop=True)
    val_regime_kept = val.loc[val_keep, REGIME_FEATURE].to_numpy()
    print(f"  val bars kept: {len(val_pred_kept)} / {len(val_pred)}")

    test_for_bound = test_full[["open_time", "close_time"]].reset_index(drop=True)
    test_bound, test_keep, test_mn_c, test_mn_h, test_mn_l = _build_boundaries(
        test_for_bound, minute_df,
    )
    test_pred_full = pd.read_parquet(ONLINE_PRED)
    test_pred_kept = test_pred_full.iloc[test_keep].reset_index(drop=True)
    test_regime_kept = test_full.loc[test_keep, REGIME_FEATURE].to_numpy()
    print(f"  test bars kept: {len(test_pred_kept)} / {len(test_pred_full)}")

    # 4) Frozen regime cuts on TRAIN only.
    print("[round 015] fitting regime cuts on TRAIN only …")
    regime_cuts = fit_regime_cuts(
        train[REGIME_FEATURE].to_numpy(), n_regimes=3, feature_name=REGIME_FEATURE,
    )
    print(f"  regime edges: {regime_cuts.edges}")

    # 5) Warm Mondrian-ACI per-regime q on val using p_online (not p_offline).
    val_regime_ids = regime_cuts.assign(val_regime_kept)
    q_init_per_alpha: dict[float, dict] = {}
    for a in ALPHAS_CONFORMAL:
        q_init_per_alpha[a] = warm_q_init_by_regime(
            p_cal=val_pred_kept["p_online"].to_numpy(),  # ← p_online, NOT p_offline
            y_cal=val_pred_kept["y_true"].to_numpy().astype(int),
            regime_cal=val_regime_ids,
            alpha=float(a),
            min_per_regime=50,
            fallback=0.5,
        )

    # 6) Build unified val + test prediction tables.
    print("[round 015] building unified prediction tables (Mondrian-ACI on p_online) …")
    val_unified = predict(
        val_pred_kept, regime_values=val_regime_kept, regime_cuts=regime_cuts,
        alphas=ALPHAS_CONFORMAL, gamma=GAMMA_ACI,
        q_init_by_regime_per_alpha=q_init_per_alpha,
    )
    test_unified = predict(
        test_pred_kept, regime_values=test_regime_kept, regime_cuts=regime_cuts,
        alphas=ALPHAS_CONFORMAL, gamma=GAMMA_ACI,
        q_init_by_regime_per_alpha=q_init_per_alpha,
    )

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

    # 7) Five strategies under the corrected architecture.
    # Build offline rate first to match null_random_at_rate to the baseline.
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
    print(f"[round 015] strategies={len(specs)}  total knob-trials={n_trials_total}")

    val_sweeps: dict[str, pd.DataFrame] = {}
    test_results: dict[str, BacktestResult] = {}
    table_rows: list[dict] = []
    walk_rows: list[dict] = []
    all_val_sharpes: list[float] = []

    for spec in specs:
        name = spec["name"]
        knob = spec["knob"]
        grid = spec["grid"]
        print(f"\n[round 015] >>> strategy={name}  knob={knob}")
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
            M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
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
            n_folds=N_FOLDS, M=M, phi=ALPHA, c_stop=ALPHA, cost_bps=COST_BPS,
        )
        wf_mean = float(wf["sharpe"].mean())
        wf_std = float(wf["sharpe"].std(ddof=1)) if len(wf) > 1 else 0.0
        for r in wf.to_dict("records"):
            r["strategy"] = name
            walk_rows.append(r)

        print(f"  test Sharpe={observed:+.3f}  PSR={res.metrics['probabilistic_sharpe']:.3f}  "
              f"n={int(res.metrics['n_trades'])}  p_boot={p_boot:.3f}  "
              f"DSR_within={dsr_within if not np.isnan(dsr_within) else 'n/a'}  "
              f"WF Sharpe={wf_mean:+.3f}±{wf_std:.3f}")

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
            "deflated_sharpe_within": dsr_within,
            "walk_forward_sharpe_mean": wf_mean,
            "walk_forward_sharpe_std": wf_std,
        })

    # 8) Multi-strategy DSR (Bailey-LdP across pooled trials).
    var_total = float(np.var(all_val_sharpes, ddof=1)) if len(all_val_sharpes) > 1 else 0.0
    print(f"\n[round 015] multi-strategy DSR: n_trials={n_trials_total}, "
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

    # 9) Per-strategy CSCV PBO over knob grid.
    print("\n[round 015] CSCV per-knob PBO …")
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

    # 10) Cross-strategy CSCV (5 strategies at val-chosen knob).
    print("\n[round 015] cross-strategy CSCV …")
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

    # 11) Persist final table + plots.
    table = pd.DataFrame(table_rows)
    table.to_csv(OUTDIR / "phase_A_table.csv", index=False)
    pd.DataFrame(walk_rows).to_csv(OUTDIR / "walk_forward_per_strategy.csv", index=False)
    print("\n[round 015] final 5-row Phase A table:")
    print(table[
        ["strategy", "knob_star", "val_sharpe_at_star", "test_sharpe",
         "test_n_trades", "p_boot", "deflated_sharpe_multi_strategy",
         "walk_forward_sharpe_mean", "walk_forward_sharpe_std",
         "within_strategy_pbo"]
    ].to_string(index=False))

    colors = {
        "baseline_offline_tau": "#1B6B33",
        "baseline_online_tau":  "#1B3F6B",
        "conformal_gate_tau":   "#6B1B6B",
        "mondrian_aci_size":    "#1B6B6B",
        "null_random_at_rate":  "#888888",
    }

    # -- Plot 1: val sweep (knob axis varies).
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
    ax.set_xlabel("knob (τ for {0,1,2}, k for mondrian_aci_size, target_rate for null)")
    ax.set_ylabel("val Sharpe")
    ax.set_title("VAL knob-sweep — round-015 corrected Phase A (5 valid strategies)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "val_knob_sweep.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # -- Plot 2: equity overlay.
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
        axes[1].fill_between(bar_idx, dd, color=colors[name], alpha=0.20,
                              linewidth=0)
    axes[0].set_title("TEST equity overlay — corrected Phase A (5 valid strategies)")
    axes[0].set_ylabel("cumulative log return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("test decision-bar index")
    axes[1].set_ylabel("drawdown (log)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "equity_overlay.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # -- Plot 3: walk-forward bars + per-strategy PBO bars + cross-strategy logits.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    names = [r["strategy"] for r in table_rows]
    means = [r["walk_forward_sharpe_mean"] for r in table_rows]
    stds = [r["walk_forward_sharpe_std"] for r in table_rows]
    bar_colors = [colors[n] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.85,
            edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(f"walk-forward Sharpe (mean ± std, {N_FOLDS} folds)")
    ax.set_title("Walk-forward stability")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    pbo_df = pd.DataFrame(cscv_rows).dropna(subset=["pbo"])
    bar_colors_pbo = [colors[n] for n in pbo_df["strategy"]]
    ax.bar(np.arange(len(pbo_df)), pbo_df["pbo"], color=bar_colors_pbo,
            alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0,
                label="PBO=0.5 (overfit gate)")
    ax.set_xticks(np.arange(len(pbo_df)))
    ax.set_xticklabels(pbo_df["strategy"], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("within-strategy PBO")
    ax.set_ylim(0, 1)
    ax.set_title("Per-strategy CSCV PBO")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    logits = cross_pbo["logits"]
    ax.hist(logits, bins=40, color="#3b7dd8", alpha=0.85, edgecolor="white",
            linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.0,
                label=f"λ=0 (PBO={cross_pbo['pbo']:.2f})")
    ax.set_xlabel("logit ω")
    ax.set_ylabel("count")
    ax.set_title(
        f"Cross-strategy CSCV ({cross_pbo['n_combinations']:,} combinations)\n"
        f"modal IS-best = {cross_summary['modal_is_best_strategy']}"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "pbo_panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "015",
        "phase": "A",
        "phase_round": "corrective",
        "claim": "Phase A under the correct two-layer architecture; 5 valid strategies, sibling-combiners deleted",
        "alpha": ALPHA, "M": M, "cost_bps_per_side": COST_BPS,
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
        "per_strategy_pbo": pbo_lookup,
        "results": table_rows,
        "wall_clock_s": float(time.time() - t0),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8",
    )
    (PHASE_A_DIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8",
    )

    # 12) Phase A backtest report (markdown).
    survivors = table[
        (table["deflated_sharpe_multi_strategy"] > 0.95)
        & (table["within_strategy_pbo"].fillna(1.0) < 0.5)
    ]
    if survivors.empty:
        survivor_paragraph = (
            "**No strategy survives the deflated bar (DSR > 0.95) AND the "
            "within-strategy PBO < 0.5 simultaneously.** The strongest test "
            f"Sharpe is `{headline['best_test_sharpe_strategy']}` at "
            f"{headline['best_test_sharpe']:+.3f} (multi-strategy DSR="
            f"{headline['best_dsr_multi']:.3f}). The two-layer system as "
            "currently fit does not produce tradable alpha at this label / "
            "cost / barrier-strategy combination. Phase B (production "
            "refactor) and Phase D (feature research, especially the "
            "rolling-retrain harness H-310) are the next moves."
        )
    else:
        names = ", ".join(f"`{s}`" for s in survivors["strategy"])
        survivor_paragraph = (
            f"**Phase A survivors at the deflated bar (DSR > 0.95, PBO < 0.5)**: "
            f"{names}. See the table for details."
        )

    md_lines: list[str] = [
        "# Phase A backtest report (round 015 — corrected two-layer architecture)",
        "",
        "Honest, deflated comparison of the **five legitimate** trading "
        "strategies on top of the `online_barrier_classifier` two-layer stack.",
        "",
        "**Architecture** (the correction round 015 enforces): the offline CatBoost "
        "emits `p_offline`, which is consumed (with `selected_features`) by the "
        "streaming online ARFClassifier, which emits `p_online` — the system's "
        "actual output. The streaming Mondrian-ACI conformal layer sits on top of "
        "`p_online` and produces the per-regime `q_t` and `in_set_α` indicators. "
        "Strategies that average or stack `(p_offline, p_online)` are "
        "architecturally invalid and have been deleted (this round supersedes "
        "rounds 011–013).",
        "",
        f"**Generated**: round 015.  ",
        f"**Test slice**: n={len(test_unified):,} decision boundaries.  ",
        f"**Costs**: 1 bp/side (round-trip 2 bp).  ",
        f"**Barriers**: φ = c_stop = α = {ALPHA}.  ",
        f"**τ-grid**: 26 points {{0.10..0.60 step 0.02}}; **k-grid** "
        f"(mondrian_aci_size): {list(K_GRID)}; **target_rate** "
        f"(null_random_at_rate): single fixed point at the offline baseline's "
        f"empirical test rate.",
        "",
        "## 0. Headline",
        "",
        survivor_paragraph,
        "",
        f"Best test Sharpe: `{headline['best_test_sharpe_strategy']}` at "
        f"**{headline['best_test_sharpe']:+.3f}** (multi-strategy DSR="
        f"{headline['best_dsr_multi']:.3f}).  ",
        f"Cross-strategy CSCV PBO = **{cross_pbo['pbo']:.3f}**.  ",
        f"Modal IS-best across {cross_pbo['n_combinations']:,} CSCV "
        f"combinations: **`{cross_summary['modal_is_best_strategy']}`**.",
        "",
        "## 1. Strategy comparison",
        "",
    ]
    cols = [
        "strategy", "knob", "knob_star", "val_sharpe_at_star",
        "test_sharpe", "test_psr", "test_n_trades",
        "test_total_log_return", "test_max_drawdown_log",
        "test_cdar_5pct_log", "test_hit_rate", "test_avg_bars_held",
        "p_boot", "deflated_sharpe_multi_strategy",
        "walk_forward_sharpe_mean", "walk_forward_sharpe_std",
        "within_strategy_pbo",
    ]
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in table.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("—")
            elif c in ("test_n_trades", "val_n_trades"):
                cells.append(f"{int(v)}")
            elif c == "knob_star" and r["knob"] == "tau":
                cells.append(f"τ={v:.3f}")
            elif c == "knob_star" and r["knob"] == "k":
                cells.append(f"k={v:.1f}")
            elif c == "knob_star" and r["knob"] == "target_rate":
                cells.append(f"r={v:.3f}")
            else:
                cells.append(f"{v:+.3f}" if isinstance(v, float) else str(v))
        md_lines.append("| " + " | ".join(cells) + " |")
    md_lines += [
        "",
        "## 2. CSCV PBO",
        "",
        "Per-strategy PBO (16 chunks, C(16,8) = 12,870 combinations):",
        "",
        "| strategy | n_knobs | PBO | median logit | modal IS-best knob |",
        "|---|---|---|---|---|",
    ]
    for r in cscv_rows:
        md_lines.append(
            f"| {r['strategy']} | {r['n_knobs']} | "
            f"{r.get('pbo', float('nan')):.3f} | "
            f"{r.get('median_logit', float('nan')):+.3f} | "
            f"{r.get('is_best_modal_knob', float('nan')):.3f} |"
        )
    md_lines += [
        "",
        f"Cross-strategy CSCV: **PBO = {cross_pbo['pbo']:.3f}** (modal IS-best: "
        f"`{cross_summary['modal_is_best_strategy']}`).",
        "",
        "## 3. Figures",
        "",
        f"![equity overlay]({OUTDIR.relative_to(REPO).as_posix()}/equity_overlay.png)",
        "",
        f"![PBO panel]({OUTDIR.relative_to(REPO).as_posix()}/pbo_panel.png)",
        "",
        f"![val knob sweep]({OUTDIR.relative_to(REPO).as_posix()}/val_knob_sweep.png)",
        "",
        "## 4. Reproduction",
        "",
        "```",
        "python scripts/phase_A_round_015.py",
        "```",
        "",
        "Idempotent given the persisted artefacts in `artifacts/phase_A/`. "
        "Wall ~5 min (ARF replay over val) on the first run; ~1 min on a "
        "warm cache.",
        "",
        "## 5. What changed vs the deleted rounds 011–013",
        "",
        "1. The two combined-signal strategies (`combined_avg_tau`, "
        "`combined_stacked_tau`) have been deleted from `src/strategies.py`. "
        "They averaged or stacked `p_offline` and `p_online`, which violates "
        "the two-layer architecture.",
        "2. `src/inference.py::predict()` now drives the Mondrian-ACI "
        "conformal stream off `p_online`, not `p_offline`. The `q_lo_α` and "
        "`in_set_α` columns are now legitimate per the round-008 contract.",
        "3. `conformal_gate_tau` and `mondrian_aci_size` now consume "
        "`p_online` (not `p_offline`). `mondrian_aci_size`'s confidence "
        "score is the per-regime margin of `p_online` over `(1 − q_lo_α)`.",
        "4. The Phase A table has 5 rows, not 7. The two deleted rows "
        "are not part of any future deflation.",
    ]
    PHASE_A_REPORT.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[round 015] wrote {PHASE_A_REPORT}")
    print(f"[round 015] wall clock: {(time.time() - t0):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
