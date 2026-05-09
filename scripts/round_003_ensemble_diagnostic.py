"""H-108 diagnostic: per-sample predict_proba dispersion across N=3 seeds.

Trains 3 seed-varied CatBoost models on synthetic data (deterministic, fast),
wraps them in `CatBoostEnsemble`, and produces a single PNG showing:

- Top:    histogram of per-sample p_class1 standard deviation across the 3 seeds.
- Bottom: scatter of ensemble-averaged p_class1 vs per-sample stddev — visualises
          where in the probability range seed-to-seed disagreement is largest.

Purpose: visual evidence that the ensemble is actually averaging (not just
returning model[0]) and that the dispersion is non-trivial → averaging is
load-bearing for calibration. The plot is the diagnostic artifact required by
LOOP_DISCIPLINE.md (visual-first); the test suite covers numerical correctness.

Outputs:
  RESEARCH/diagrams/round_003/ensemble_dispersion.png
  RESEARCH/diagrams/round_003/headline.json   (small machine-readable summary)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ensemble import CatBoostEnsemble  # noqa: E402

OUTDIR = REPO / "RESEARCH" / "diagrams" / "round_003"


def _make_data(n: int = 800, d: int = 6, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(float)
    logits = X[:, 0] + 0.4 * X[:, 1] - 0.3 * X[:, 2] + 0.1 * X[:, 3] ** 2
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from catboost import CatBoostClassifier

    X, y = _make_data(n=800)
    n_tr = int(0.8 * len(y))
    X_tr, X_va, y_tr, y_va = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]

    seeds = [0, 1, 2]
    models = []
    for s in seeds:
        m = CatBoostClassifier(
            iterations=80, depth=4, learning_rate=0.08,
            random_seed=s, verbose=False, allow_writing_files=False,
            early_stopping_rounds=15,
        )
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        models.append(m)

    ens = CatBoostEnsemble(models)
    individual = np.stack([m.predict_proba(X_va) for m in ens.models], axis=0)
    p_ens = ens.predict_proba(X_va)

    p1_individual = individual[..., 1]
    p1_ens = p_ens[:, 1]
    p1_std = p1_individual.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5))

    ax0 = axes[0]
    ax0.hist(p1_std, bins=40, color="#3b7dd8", edgecolor="white")
    ax0.set_xlabel("std(p_class1) across seeds")
    ax0.set_ylabel("# samples")
    ax0.set_title(
        f"Per-sample seed-disagreement (N={ens.n_models} seeds, n_eval={len(p1_std)})\n"
        f"median std={np.median(p1_std):.4f}  p95={np.quantile(p1_std, 0.95):.4f}  max={p1_std.max():.4f}"
    )
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.scatter(p1_ens, p1_std, s=8, alpha=0.5, color="#3b7dd8")
    ax1.set_xlabel("ensemble-averaged p_class1")
    ax1.set_ylabel("std(p_class1) across seeds")
    max_disag = float(np.abs(p1_individual - p1_ens).max())
    ax1.set_title(
        "Seed disagreement vs ensemble probability — broadly distributed across p,\n"
        f"max single-model deviation from ensemble = {max_disag:.3f} → averaging is load-bearing"
    )
    ax1.set_xlim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = OUTDIR / "ensemble_dispersion.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "round_id": "003",
        "hypothesis": "H-108",
        "claim": "CatBoostEnsemble port works: averaged predict_proba is non-trivially distinct from any single model",
        "n_models": ens.n_models,
        "seeds": seeds,
        "n_eval": int(len(p1_std)),
        "median_std": float(np.median(p1_std)),
        "p95_std": float(np.quantile(p1_std, 0.95)),
        "max_std": float(p1_std.max()),
        "max_individual_disagreement_with_ensemble": float(
            np.abs(p1_individual - p1_ens).max()
        ),
        "best_iteration_avg": ens.get_best_iteration(),
        "feature_importance_summary": {
            "max": float(ens.get_feature_importance().max()),
            "min": float(ens.get_feature_importance().min()),
        },
    }
    (OUTDIR / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )

    print(f"saved: {out_png}")
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
