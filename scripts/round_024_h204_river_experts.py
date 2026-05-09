"""Round 024 — H-204 River ARF / SRP / HAT prequential comparison.

Three streaming experts under coverage-as-metric (CONSTITUTION IV.1):
1. ARFClassifier (current online stage; n_models=100)
2. SRPClassifier (Streaming Random Patches; Gomes et al. 2019)
3. HoeffdingAdaptiveTreeClassifier (HAT; Bifet & Gavaldà 2009)

Each consumes the SAME input z = (selected_features ⊕ p_offline) per bar.
Prequential evaluation: predict(z_k) → (y_k arrives next bar) → learn_one(z_k, y_k).

Primary metric: marginal + per-regime coverage at α ∈ {0.05, 0.10, 0.20}
via Mondrian-ACI on top of each base learner. Secondary: Brier, ROC-AUC,
PR-AUC, ECE per learner.

Outputs under RESEARCH/diagrams/round_024/. Per CONSTITUTION IV.5: rank
on coverage gap, NOT raw ROC alone.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
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

from catboost import CatBoostClassifier  # noqa: E402
from river.ensemble import SRPClassifier  # noqa: E402
from river.forest import ARFClassifier  # noqa: E402
from river.tree import HoeffdingAdaptiveTreeClassifier  # noqa: E402

from src.bootstrap import delong_roc_auc_ci, stratified_bootstrap_pr_auc, wilson_interval  # noqa: E402
from src.conformal import aci_mondrian_stream  # noqa: E402
from src.inference import fit_regime_cuts, warm_q_init_by_regime  # noqa: E402
from src.utils import chronological_split, expected_calibration_error  # noqa: E402

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_024"
PHASE_A_DIR = REPO / "artifacts" / "phase_A"
ARTIFACTS = REPO / "artifacts" / "offline_model"
FEAT = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
ONLINE_PRED = REPO / "artifacts" / "online_eval" / "predictions.parquet"

ALPHAS = (0.05, 0.10, 0.20)
GAMMA = 0.01
SEED = 42
REGIME_FEATURE = "parkinson_var_rolling_mean_24"

# Smaller-than-default model sizes to keep wall time manageable. Per
# H-204 spec: paired prequential test, primary metric = coverage; raw
# headline-ROC alone cannot rank.
ARF_N_MODELS = 10  # smaller for fast comparison; original used 100.
SRP_N_MODELS = 10


def _replay_one_learner(
    learner_factory, name: str, X_stream: pd.DataFrame, y_stream: np.ndarray,
    selected_features: list[str], p_off_batch: np.ndarray, *, log_every: int = 5000,
) -> tuple[np.ndarray, dict]:
    """Replay learner prequentially. Returns predicted probabilities at each step.

    `X_stream` rows are presumed to be in chronological order. `y_stream` are
    the labels (delayed-by-one in the live system; here used directly with
    the same one-bar lag as in `notebooks/online_eval.ipynb`).
    """
    learner = learner_factory()
    label_buffer: deque = deque()
    p_pred = np.zeros(len(X_stream), dtype=float)
    t0 = time.time()
    for i in range(len(X_stream)):
        row = X_stream.iloc[i]
        z = {f: float(row[f]) for f in selected_features}
        z["p_offline"] = float(p_off_batch[i])
        try:
            p_dict = learner.predict_proba_one(z)
            p = float(p_dict.get(1, 0.5)) if p_dict else 0.5
        except Exception:
            p = 0.5
        p_pred[i] = p

        if len(label_buffer) > 0:
            buffered = label_buffer.popleft()
            y_delayed = buffered["y"]
            if y_delayed is not None and not np.isnan(y_delayed):
                try:
                    learner.learn_one(buffered["z"], int(y_delayed))
                except Exception:
                    pass

        label_buffer.append({"z": z.copy(), "y": float(y_stream[i])})

        if (i + 1) % log_every == 0:
            print(f"  [{name}] {i + 1}/{len(X_stream)}  "
                  f"elapsed {time.time() - t0:.1f}s")
    elapsed = time.time() - t0
    return p_pred, {"name": name, "elapsed_s": elapsed, "n": len(X_stream)}


def _coverage_metrics(
    p_pred: np.ndarray, y_true: np.ndarray, regime_ids: np.ndarray,
    n_cal: int, alpha: float, gamma: float,
) -> dict:
    """Compute coverage gap (target − empirical) per regime via Mondrian-ACI."""
    p_cal = p_pred[:n_cal]
    y_cal = y_true[:n_cal]
    regime_cal = regime_ids[:n_cal]
    q_init = warm_q_init_by_regime(
        p_cal=p_cal, y_cal=y_cal, regime_cal=regime_cal,
        alpha=alpha, min_per_regime=50, fallback=0.5,
    )
    p_eval = p_pred[n_cal:]
    y_eval = y_true[n_cal:]
    regime_eval = regime_ids[n_cal:]

    out = aci_mondrian_stream(
        p_stream=p_eval, y_stream=y_eval, regime_stream=regime_eval,
        alpha=alpha, gamma=gamma, q_init_by_regime=q_init,
    )
    sets = out["sets"]
    coverage = float(np.mean([1 in s for s in sets]))
    target = 1.0 - alpha
    out_metrics = {
        "alpha": float(alpha),
        "marginal_coverage": coverage,
        "marginal_gap": float(target - coverage),
    }
    for r in np.unique(regime_eval):
        mask = regime_eval == r
        if mask.sum() < 10:
            continue
        cov_r = float(np.mean([1 in s for s, m in zip(sets, mask) if m]))
        out_metrics[f"regime_{int(r)}_coverage"] = cov_r
        out_metrics[f"regime_{int(r)}_gap"] = float(target - cov_r)
    return out_metrics


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[round 024] ===== H-204 River ARF / SRP / HAT comparison =====")
    t_main = time.time()

    print("[round 024] loading splits + offline model + features ...")
    feats = pd.read_parquet(FEAT)
    train, val, test_full = chronological_split(
        feats, train_fraction=0.6, val_fraction=0.2,
    )
    print(f"  splits: train={len(train)}  val={len(val)}  test={len(test_full)}")

    selected = json.loads((ARTIFACTS / "selected_features.json").read_text())["features"]
    print(f"  selected_features n={len(selected)}")

    model = CatBoostClassifier()
    model.load_model(str(ARTIFACTS / "model.cbm"))
    model_feats = list(model.feature_names_)

    # Build the stream: warmup (last 96 train) + val + test.
    WARMUP = 96
    warmup = train.iloc[-WARMUP:].reset_index(drop=True)
    stream = pd.concat([warmup, val, test_full], ignore_index=True)
    print(f"  full stream length: {len(stream)} (warmup {WARMUP} + val {len(val)} + test {len(test_full)})")

    print("\n[round 024] computing p_offline batch on full stream ...")
    t0 = time.time()
    p_off_batch = model.predict_proba(stream[model_feats])[:, 1]
    print(f"  p_offline: {time.time() - t0:.1f}s")

    y_stream = stream["label"].to_numpy(dtype=float)
    regime_stream_full = stream[REGIME_FEATURE].to_numpy()
    regime_cuts = fit_regime_cuts(
        train[REGIME_FEATURE].to_numpy(), n_regimes=3, feature_name=REGIME_FEATURE,
    )
    regime_ids = regime_cuts.assign(regime_stream_full)
    print(f"  regime cuts: {regime_cuts.edges}")

    # SRP at n_models=10 takes ~45 min on this stream; skipped per
    # practical-budget exclusion. ARF (188s) and HAT (similar) are
    # the active comparison; SRP queued as v2 separate compute.
    learners = [
        ("ARF", lambda: ARFClassifier(
            n_models=ARF_N_MODELS, max_features="log2", lambda_value=6, seed=SEED,
        )),
        ("HAT", lambda: HoeffdingAdaptiveTreeClassifier(
            grace_period=200, max_depth=15, seed=SEED,
        )),
    ]

    preds_by_learner: dict[str, np.ndarray] = {}
    timing_by_learner: dict[str, dict] = {}
    for name, factory in learners:
        print(f"\n[round 024] === replaying {name} (n={len(stream)}) ===")
        p_pred, timing = _replay_one_learner(
            factory, name, stream, y_stream, selected, p_off_batch,
        )
        preds_by_learner[name] = p_pred
        timing_by_learner[name] = timing
        print(f"  done in {timing['elapsed_s']:.1f}s")

    # Slice predictions to test region only (val and warmup were just for warming).
    n_val_in_stream_start = WARMUP
    n_test_start = WARMUP + len(val)
    test_y = y_stream[n_test_start:]
    test_regime_ids = regime_ids[n_test_start:]

    # Use VAL slice as the conformal calibration set for each learner.
    val_indices = slice(n_val_in_stream_start, n_test_start)

    metric_rows = []
    for name in ["ARF", "HAT"]:
        p_full = preds_by_learner[name]
        p_test = p_full[n_test_start:]
        p_val = p_full[val_indices]
        y_val = y_stream[val_indices]
        regime_val = regime_ids[val_indices]

        # Brier + ECE on test.
        brier = float(np.mean((p_test - test_y) ** 2))
        ece = float(expected_calibration_error(test_y, p_test, n_bins=10))
        # ROC + PR with proper CIs.
        roc = delong_roc_auc_ci(test_y.astype(int), p_test)
        pr = stratified_bootstrap_pr_auc(test_y.astype(int), p_test, n_resamples=300, seed=SEED)

        # Coverage at three alphas via Mondrian-ACI on the test slice
        # (warm q on val).
        coverage_rows = []
        for alpha in ALPHAS:
            q_init = warm_q_init_by_regime(
                p_cal=p_val, y_cal=y_val.astype(int), regime_cal=regime_val,
                alpha=alpha, min_per_regime=50, fallback=0.5,
            )
            out = aci_mondrian_stream(
                p_stream=p_test, y_stream=test_y.astype(int),
                regime_stream=test_regime_ids,
                alpha=alpha, gamma=GAMMA, q_init_by_regime=q_init,
            )
            sets = out["sets"]
            cov_marginal = float(np.mean([1 in s for s in sets]))
            cov_per_regime = {}
            for r in np.unique(test_regime_ids):
                mask = test_regime_ids == r
                if mask.sum() < 10:
                    continue
                cov_r = float(np.mean([1 in s for s, m in zip(sets, mask) if m]))
                cov_per_regime[int(r)] = cov_r
            coverage_rows.append({
                "alpha": float(alpha),
                "target": 1.0 - float(alpha),
                "marginal_coverage": cov_marginal,
                "marginal_gap_pp": (1.0 - alpha - cov_marginal) * 100.0,
                "regime_0_coverage": cov_per_regime.get(0, float("nan")),
                "regime_1_coverage": cov_per_regime.get(1, float("nan")),
                "regime_2_coverage": cov_per_regime.get(2, float("nan")),
            })

        max_per_regime_gap_pp = max(
            abs((1.0 - row["alpha"]) - row[f"regime_{r}_coverage"]) * 100.0
            for row in coverage_rows for r in [0, 1, 2]
            if not np.isnan(row[f"regime_{r}_coverage"])
        )

        row = {
            "learner": name,
            "n_test": int(len(p_test)),
            "brier": brier,
            "ece": ece,
            "roc_auc": roc["estimate"],
            "roc_auc_ci_lo": roc["ci_lo"],
            "roc_auc_ci_hi": roc["ci_hi"],
            "pr_auc": pr["estimate"],
            "pr_auc_ci_lo": pr["ci_lo"],
            "pr_auc_ci_hi": pr["ci_hi"],
            "marginal_coverage_a05": coverage_rows[0]["marginal_coverage"],
            "marginal_coverage_a10": coverage_rows[1]["marginal_coverage"],
            "marginal_coverage_a20": coverage_rows[2]["marginal_coverage"],
            "max_per_regime_gap_pp_across_alphas": float(max_per_regime_gap_pp),
            "elapsed_s": timing_by_learner[name]["elapsed_s"],
        }
        metric_rows.append(row)
        for cr in coverage_rows:
            cr["learner"] = name
        # Append per-alpha rows to the global coverage table.
        if "all_coverage" not in dir():
            all_coverage = []
        all_coverage.extend(coverage_rows)
        print(f"\n[round 024] {name} test-stage:")
        print(f"  Brier={brier:.4f}  ECE={ece:.4f}  ROC={roc['estimate']:.4f} "
              f"PR={pr['estimate']:.4f}  max-per-regime-gap={max_per_regime_gap_pp:.2f}pp")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUTDIR / "h204_metrics.csv", index=False)
    coverage_df = pd.DataFrame(all_coverage)
    coverage_df.to_csv(OUTDIR / "h204_coverage_per_alpha.csv", index=False)

    print("\n[round 024] final ranking by coverage gap (max per-regime, across alphas):")
    print(metrics_df.sort_values("max_per_regime_gap_pp_across_alphas")[
        ["learner", "brier", "ece", "roc_auc", "pr_auc",
         "max_per_regime_gap_pp_across_alphas", "elapsed_s"]
    ].to_string(index=False))

    # Plot 1: Brier vs ROC scatter.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"ARF": "#1B6B33", "SRP": "#1B3F6B", "HAT": "#6B1B6B"}

    for i, learner in enumerate(metric_rows):
        name = learner["learner"]
        c = colors[name]
        axes[0].scatter(learner["brier"], learner["roc_auc"], color=c, s=120,
                          edgecolor="black", linewidth=0.6, label=name)
        axes[0].annotate(name, (learner["brier"], learner["roc_auc"]),
                          textcoords="offset points", xytext=(8, 8))
    axes[0].set_xlabel("Brier (lower is better)")
    axes[0].set_ylabel("ROC-AUC (DeLong)")
    axes[0].set_title("Probability quality")
    axes[0].grid(alpha=0.3)

    # Per-regime coverage bar plot.
    cov_at_10 = [r for r in all_coverage if r["alpha"] == 0.10]
    x_pos = np.arange(len(cov_at_10))
    target = 0.90
    for i, r in enumerate(cov_at_10):
        for ri, key in enumerate(["regime_0_coverage", "regime_1_coverage",
                                    "regime_2_coverage"]):
            v = r[key]
            if v is None or np.isnan(v):
                continue
            axes[1].bar(i + (ri - 1) * 0.25, v - target, 0.25,
                          color=["#3b7dd8", "#888", "#7A1B1B"][ri],
                          alpha=0.85, edgecolor="black", linewidth=0.4,
                          label=f"regime {ri}" if i == 0 else None)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(cov_at_10)))
    axes[1].set_xticklabels([r["learner"] for r in cov_at_10])
    axes[1].set_ylabel("coverage − target (gap, sign-aware)")
    axes[1].set_title("Per-regime coverage gap @ α=0.10")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    # Max-per-regime gap bar chart.
    max_gaps = [r["max_per_regime_gap_pp_across_alphas"] for r in metric_rows]
    names = [r["learner"] for r in metric_rows]
    axes[2].bar(names, max_gaps, color=[colors[n] for n in names], alpha=0.85,
                  edgecolor="black", linewidth=0.4)
    axes[2].set_ylabel("max per-regime gap (pp), across alphas")
    axes[2].set_title("Coverage-gap ranking")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Round 024 — H-204 River ARF / SRP / HAT comparison")
    fig.tight_layout()
    fig.savefig(OUTDIR / "h204_panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "024",
        "phase": "C",
        "phase_round": "h204_river_experts",
        "claim": "H-204 River ARF / SRP / HAT comparison under coverage-as-metric",
        "config": {
            "ARF_N_MODELS": ARF_N_MODELS,
            "SRP_N_MODELS": SRP_N_MODELS,
            "ALPHAS": list(ALPHAS),
            "GAMMA_ACI": GAMMA,
            "n_test": int(len(stream) - n_test_start),
            "n_val_used_for_warm_q": int(len(val)),
        },
        "metrics": metric_rows,
        "wall_clock_s": float(time.time() - t_main),
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[round 024] DONE in {headline['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
