"""H-201 round-002 driver: split-conformal coverage baseline.

Reads the persisted predictions from `artifacts/online_eval/predictions.parquet`
+ regime feature `parkinson_var_rolling_mean_24` from
`data/model_data/BTCUSDT/bars_20m_features.parquet` and produces:

- `RESEARCH/diagrams/round_002/coverage_metrics.csv`  (all (predictor, alpha,
  mode, regime) combinations)
- `RESEARCH/diagrams/round_002/coverage_summary.png`  (multi-panel)
- `RESEARCH/diagrams/round_002/headline.json`         (decision-relevant scalars)

Pure measurement; no model retrain. The diagnostic is deterministic given the
fixed predictions parquet — invoking this script twice yields byte-identical
CSV (seed_noise_band = 0).

Modes evaluated per (predictor, alpha):
- `naive_threshold` — set = {y : p̂(y|x) ≥ α}, no calibration.
- `lac_marginal` — split-conformal LAC with global q_hat (`fit_conformal`).
- `lac_mondrian` — split-conformal LAC with per-vol-tercile q_hat
  (`fit_conformal(... regime_cal=tercile)`).

Predictors: `p_offline`, `p_online`, `p_const = 0.0971` (sanity).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.conformal import (  # noqa: E402
    ConformalCalibrator,
    coverage_by_regime,
    empirical_coverage,
    fit_conformal,
    predict_set,
    set_size_distribution,
)

ALPHAS = (0.05, 0.10, 0.20)
PREDICTORS = ("p_offline", "p_online")  # p_final ≡ p_online here; skip duplicate
REGIME_FEATURE = "parkinson_var_rolling_mean_24"
N_TERCILES = 3
CAL_FRAC = 0.30  # chronological calibration window = first 30% of test slice

OUT_DIR = REPO / "RESEARCH" / "diagrams" / "round_002"


def load_predictions_with_regime() -> pd.DataFrame:
    preds = pd.read_parquet(REPO / "artifacts" / "online_eval" / "predictions.parquet")
    feats = pd.read_parquet(
        REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
    )
    n_test = len(preds)
    test_start = len(feats) - n_test
    feats_test = feats.iloc[test_start : test_start + n_test].reset_index(drop=True)
    # Sanity: labels must agree on alignment.
    assert (feats_test["label"].values == preds["y_true"].values).all(), (
        "predictions parquet does not align with bars_20m_features tail"
    )
    df = preds.copy()
    df[REGIME_FEATURE] = feats_test[REGIME_FEATURE].values
    # Drop the (very few) rows where the regime feature is NaN — the rolling
    # mean has burn-in for the first windows; on the tail of a long series this
    # should be empty in practice, but guard anyway.
    n_nan = int(df[REGIME_FEATURE].isna().sum())
    if n_nan > 0:
        df = df.dropna(subset=[REGIME_FEATURE]).reset_index(drop=True)
    df["test_idx"] = np.arange(len(df))
    return df


def make_terciles(values: np.ndarray) -> np.ndarray:
    """Fixed-label vol terciles from a 1-D series; labels are 'low'/'med'/'high'."""
    cats = pd.qcut(values, N_TERCILES, labels=["low", "med", "high"])
    return np.asarray(cats.astype(str))


def naive_threshold_sets(p_test: np.ndarray, alpha: float) -> np.ndarray:
    """set = {y : p̂(y|x) ≥ α}; the H-201 'as if it were a conformal predictor'
    baseline (no held-out calibration)."""
    p = np.asarray(p_test, dtype=float)
    sets = np.zeros((len(p), 2), dtype=bool)
    sets[:, 0] = (1.0 - p) >= alpha
    sets[:, 1] = p >= alpha
    return sets


@dataclass
class CoverageRow:
    predictor: str
    alpha: float
    mode: str
    regime: str  # "ALL" for marginal; tercile label for per-regime
    n: int
    coverage: float
    nominal: float
    gap: float
    frac_empty: float
    frac_singleton: float
    frac_full: float


def evaluate_one(
    predictor: str,
    p_full: np.ndarray,
    y_full: np.ndarray,
    regime_full: np.ndarray,
    cal_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    p_cal, y_cal, r_cal = p_full[cal_idx], y_full[cal_idx], regime_full[cal_idx]
    p_eval, y_eval, r_eval = p_full[eval_idx], y_full[eval_idx], regime_full[eval_idx]

    for alpha in ALPHAS:
        nominal = 1.0 - alpha

        # 1) Naive threshold (no calibration).
        sets_naive = naive_threshold_sets(p_eval, alpha)
        cov_naive = empirical_coverage(sets_naive, y_eval)
        sd_naive = set_size_distribution(sets_naive)
        rows.append(CoverageRow(
            predictor, alpha, "naive_threshold", "ALL",
            len(y_eval), cov_naive, nominal, nominal - cov_naive,
            sd_naive["empty"], sd_naive["singleton_0"] + sd_naive["singleton_1"],
            sd_naive["full"],
        ))
        for r in ("low", "med", "high"):
            mask = r_eval == r
            if mask.sum() == 0:
                continue
            cov_r = empirical_coverage(sets_naive[mask], y_eval[mask])
            sizes = sets_naive[mask].sum(axis=1)
            rows.append(CoverageRow(
                predictor, alpha, "naive_threshold", r,
                int(mask.sum()), cov_r, nominal, nominal - cov_r,
                float((sizes == 0).mean()),
                float((sizes == 1).mean()),
                float((sizes == 2).mean()),
            ))

        # 2) Marginal LAC (global q_hat).
        cal_marg = fit_conformal(p_cal, y_cal, alpha)
        sets_marg = predict_set(cal_marg, p_eval)
        cov_marg = empirical_coverage(sets_marg, y_eval)
        sd_marg = set_size_distribution(sets_marg)
        rows.append(CoverageRow(
            predictor, alpha, "lac_marginal", "ALL",
            len(y_eval), cov_marg, nominal, nominal - cov_marg,
            sd_marg["empty"], sd_marg["singleton_0"] + sd_marg["singleton_1"],
            sd_marg["full"],
        ))
        # Per-regime breakdown of MARGINAL LAC (same q_hat, just stratified eval).
        per_regime_marg = coverage_by_regime(sets_marg, y_eval, r_eval)
        for _, prr in per_regime_marg.iterrows():
            rows.append(CoverageRow(
                predictor, alpha, "lac_marginal", prr["regime"],
                int(prr["n"]), float(prr["coverage"]), nominal,
                nominal - float(prr["coverage"]),
                float(prr["frac_empty"]),
                float(prr["frac_singleton"]),
                float(prr["frac_full"]),
            ))

        # 3) Mondrian LAC (per-tercile q_hat).
        cal_mond = fit_conformal(p_cal, y_cal, alpha, regime_cal=r_cal)
        sets_mond = predict_set(cal_mond, p_eval, regime_test=r_eval)
        cov_mond = empirical_coverage(sets_mond, y_eval)
        sd_mond = set_size_distribution(sets_mond)
        rows.append(CoverageRow(
            predictor, alpha, "lac_mondrian", "ALL",
            len(y_eval), cov_mond, nominal, nominal - cov_mond,
            sd_mond["empty"], sd_mond["singleton_0"] + sd_mond["singleton_1"],
            sd_mond["full"],
        ))
        per_regime_mond = coverage_by_regime(sets_mond, y_eval, r_eval)
        for _, prr in per_regime_mond.iterrows():
            rows.append(CoverageRow(
                predictor, alpha, "lac_mondrian", prr["regime"],
                int(prr["n"]), float(prr["coverage"]), nominal,
                nominal - float(prr["coverage"]),
                float(prr["frac_empty"]),
                float(prr["frac_singleton"]),
                float(prr["frac_full"]),
            ))

    return rows


def evaluate_constant_predictor(
    base_rate: float,
    y_full: np.ndarray,
    regime_full: np.ndarray,
    eval_idx: np.ndarray,
) -> list[CoverageRow]:
    """Sanity check: p ≡ base_rate. At alpha=0.05, set is always {0,1}; at
    alpha=0.20, set is always {0} (positive class never reached → cov_for_y=1
    is 0, cov_for_y=0 is 1, marginal cov = base_rate of class 0)."""
    p_eval = np.full_like(y_full[eval_idx], base_rate, dtype=float)
    y_eval = y_full[eval_idx]
    r_eval = regime_full[eval_idx]
    rows: list[CoverageRow] = []
    for alpha in ALPHAS:
        nominal = 1.0 - alpha
        sets = naive_threshold_sets(p_eval, alpha)
        cov = empirical_coverage(sets, y_eval)
        sd = set_size_distribution(sets)
        rows.append(CoverageRow(
            "p_const", alpha, "naive_threshold", "ALL",
            len(y_eval), cov, nominal, nominal - cov,
            sd["empty"], sd["singleton_0"] + sd["singleton_1"], sd["full"],
        ))
    return rows


def plot_summary(metrics: pd.DataFrame, headline: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel (a) marginal coverage by mode + alpha + predictor.
    ax = axes[0, 0]
    marg = metrics[metrics["regime"] == "ALL"].copy()
    width = 0.10
    alphas = sorted(marg["alpha"].unique())
    modes = ["naive_threshold", "lac_marginal", "lac_mondrian"]
    preds = list(PREDICTORS)
    x = np.arange(len(alphas))
    for j, predictor in enumerate(preds):
        for k, mode in enumerate(modes):
            sub = marg[(marg["predictor"] == predictor) & (marg["mode"] == mode)]
            sub = sub.sort_values("alpha")
            offset = (j * len(modes) + k - (len(preds) * len(modes) - 1) / 2) * width
            ax.bar(
                x + offset, sub["coverage"].values, width=width,
                label=f"{predictor[2:]}/{mode.split('_')[-1]}",
            )
    for k, alpha in enumerate(alphas):
        ax.hlines(1 - alpha, k - 0.5, k + 0.5, colors="k", linestyles="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("marginal empirical coverage")
    ax.set_title("(a) Marginal coverage vs nominal (1-α dashed)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel (b) per-regime coverage gap (1-α − cov) at alpha=0.10, Mondrian.
    ax = axes[0, 1]
    gap_data = metrics[
        (metrics["mode"] == "lac_mondrian")
        & (metrics["alpha"] == 0.10)
        & (metrics["regime"].isin(["low", "med", "high"]))
    ].copy()
    pivot = gap_data.pivot(index="regime", columns="predictor", values="gap").reindex(
        ["low", "med", "high"]
    )
    pivot.plot(kind="bar", ax=ax, color=["#4477aa", "#aa4422"])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(0.05, color="r", linestyle="--", label="5% gap threshold (BACKLOG)")
    ax.axhline(-0.05, color="r", linestyle="--")
    ax.set_ylabel("coverage gap = (1-α) − empirical")
    ax.set_xlabel("parkinson_var_rolling_mean_24 tercile")
    ax.set_title("(b) Per-regime gap at alpha=0.10 (Mondrian LAC)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel (c) per-regime coverage gap at alpha=0.10, naive_threshold (uncalibrated).
    ax = axes[1, 0]
    gap_naive = metrics[
        (metrics["mode"] == "naive_threshold")
        & (metrics["alpha"] == 0.10)
        & (metrics["regime"].isin(["low", "med", "high"]))
    ].copy()
    pivot_naive = gap_naive.pivot(
        index="regime", columns="predictor", values="gap"
    ).reindex(["low", "med", "high"])
    pivot_naive.plot(kind="bar", ax=ax, color=["#4477aa", "#aa4422"])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(0.05, color="r", linestyle="--")
    ax.axhline(-0.05, color="r", linestyle="--")
    ax.set_ylabel("coverage gap = (1-α) − empirical")
    ax.set_xlabel("parkinson_var_rolling_mean_24 tercile")
    ax.set_title("(c) Per-regime gap at alpha=0.10 (naive p ≥ α — uncalibrated)")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel (d) tightness: mean set size by mode at alpha=0.10.
    ax = axes[1, 1]
    tight = metrics[
        (metrics["regime"] == "ALL") & (metrics["alpha"] == 0.10)
    ].copy()
    tight["mean_set_size"] = (
        2.0 * tight["frac_full"]
        + 1.0 * tight["frac_singleton"]
        + 0.0 * tight["frac_empty"]
    )
    pivot_t = tight.pivot(index="mode", columns="predictor", values="mean_set_size")
    pivot_t = pivot_t.reindex(modes)
    pivot_t.plot(kind="bar", ax=ax, color=["#4477aa", "#aa4422"])
    ax.set_ylabel("mean prediction-set size (binary; max=2)")
    ax.set_xlabel("mode")
    ax.set_xticklabels(pivot_t.index, rotation=15)
    ax.set_title("(d) Tightness: mean |C(x)| at alpha=0.10")
    ax.grid(True, axis="y", alpha=0.3)

    title = (
        f"H-201 coverage baseline — n_eval={headline['n_eval']}, "
        f"n_cal={headline['n_cal']}, "
        f"max |gap| (Mondrian, alpha=0.10) = {headline['max_abs_gap_mondrian_a010']:+.3f}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / "coverage_summary.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions_with_regime()
    n = len(df)
    n_cal = int(round(CAL_FRAC * n))
    cal_idx = np.arange(n_cal)
    eval_idx = np.arange(n_cal, n)

    regime = make_terciles(df[REGIME_FEATURE].values)
    y_full = df["y_true"].values.astype(int)
    base_rate = float(y_full.mean())

    all_rows: list[CoverageRow] = []
    for predictor in PREDICTORS:
        p_full = df[predictor].values.astype(float)
        all_rows.extend(
            evaluate_one(predictor, p_full, y_full, regime, cal_idx, eval_idx)
        )
    all_rows.extend(
        evaluate_constant_predictor(base_rate, y_full, regime, eval_idx)
    )

    metrics = pd.DataFrame([r.__dict__ for r in all_rows])
    csv_path = OUT_DIR / "coverage_metrics.csv"
    metrics.to_csv(csv_path, index=False, float_format="%.6f")

    # Headline scalars.
    mond_a010_per_regime = metrics[
        (metrics["mode"] == "lac_mondrian")
        & (metrics["alpha"] == 0.10)
        & (metrics["regime"].isin(["low", "med", "high"]))
    ]
    max_abs_gap_mondrian_a010 = float(mond_a010_per_regime["gap"].abs().max())
    naive_a010_per_regime = metrics[
        (metrics["mode"] == "naive_threshold")
        & (metrics["alpha"] == 0.10)
        & (metrics["regime"].isin(["low", "med", "high"]))
    ]
    max_abs_gap_naive_a010 = float(naive_a010_per_regime["gap"].abs().max())

    # Constant-predictor sanity bounds (THEORIST):
    # at alpha=0.05, p_const sets are full → cov ≈ 1.0 (allow [0.999, 1.000])
    # at alpha=0.20, p_const sets are {0} → cov = P(y=0) on the EVAL slice.
    # Reference must be the eval-slice base rate (NOT the global base rate),
    # since the diagnostic is computed on the eval slice.
    eval_base_rate = float(y_full[eval_idx].mean())
    cp_a005 = metrics[(metrics["predictor"] == "p_const") & (metrics["alpha"] == 0.05)]
    cp_a020 = metrics[(metrics["predictor"] == "p_const") & (metrics["alpha"] == 0.20)]
    sanity_a005 = float(cp_a005["coverage"].iloc[0])
    sanity_a020 = float(cp_a020["coverage"].iloc[0])
    sanity_a005_pass = 0.999 <= sanity_a005 <= 1.000
    sanity_a020_pass = abs(sanity_a020 - (1.0 - eval_base_rate)) < 0.0005

    headline = {
        "round": 2,
        "hypothesis": "H-201",
        "n_total": int(n),
        "n_cal": int(n_cal),
        "n_eval": int(n - n_cal),
        "base_rate": base_rate,
        "eval_base_rate": eval_base_rate,
        "regime_feature": REGIME_FEATURE,
        "alphas": list(ALPHAS),
        "predictors": list(PREDICTORS) + ["p_const"],
        "max_abs_gap_mondrian_a010": max_abs_gap_mondrian_a010,
        "max_abs_gap_naive_a010": max_abs_gap_naive_a010,
        "sanity_a005_coverage_p_const": sanity_a005,
        "sanity_a005_pass": bool(sanity_a005_pass),
        "sanity_a020_coverage_p_const": sanity_a020,
        "sanity_a020_pass": bool(sanity_a020_pass),
        "marginal_coverage_a010": {
            predictor: {
                mode: float(
                    metrics[
                        (metrics["predictor"] == predictor)
                        & (metrics["alpha"] == 0.10)
                        & (metrics["mode"] == mode)
                        & (metrics["regime"] == "ALL")
                    ]["coverage"].iloc[0]
                )
                for mode in ("naive_threshold", "lac_marginal", "lac_mondrian")
            }
            for predictor in PREDICTORS
        },
    }

    json_path = OUT_DIR / "headline.json"
    json_path.write_text(json.dumps(headline, indent=2), encoding="utf-8")

    plot_path = plot_summary(metrics, headline)

    # Stdout summary for the round agent.
    print(f"[round-002] CSV:    {csv_path.relative_to(REPO)}")
    print(f"[round-002] JSON:   {json_path.relative_to(REPO)}")
    print(f"[round-002] PNG:    {plot_path.relative_to(REPO)}")
    print(f"[round-002] n_cal={n_cal}  n_eval={n - n_cal}  base_rate={base_rate:.4f}")
    print(
        f"[round-002] max |gap| (Mondrian, alpha=0.10) = {max_abs_gap_mondrian_a010:+.4f}"
    )
    print(
        f"[round-002] max |gap| (naive,    alpha=0.10) = {max_abs_gap_naive_a010:+.4f}"
    )
    print(
        f"[round-002] sanity p_const alpha=0.05 cov={sanity_a005:.4f} "
        f"(expected ~1.000, pass={sanity_a005_pass})"
    )
    print(
        f"[round-002] sanity p_const alpha=0.20 cov={sanity_a020:.4f} "
        f"(expected ~{1.0 - eval_base_rate:.4f} = 1-eval_base_rate, "
        f"pass={sanity_a020_pass})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
