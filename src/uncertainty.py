"""Uncertainty Quantification helpers for the barrier classifier.

This module wraps CatBoost's virtual-ensemble feature (Malinin, Prokhorenkova
& Ustimenko, ICLR 2021), which works with any model trained under SGLB
(`langevin=True`). The boosting trajectory is treated as an MCMC chain over
function space; `virtual_ensembles_predict` slices the trained tree sequence
into K disjoint sub-ensembles, each producing a posterior-style prediction.

The key separation (after Depeweg et al. 2018, surfaced in Malinin et al. 2021):
- **`data_uncertainty`** = average per-member predictive variance (aleatoric).
- **`total_uncertainty`** = variance of the per-member means (epistemic + aleatoric).
- **σ_epistemic = sqrt(total - data)** is what we use for trade selection:
  it flags points the model is *unsure about* (model-leverage), not points
  that are intrinsically noisy (which we still want to bet on at p ≈ 0.5
  if calibrated).

Provides:
- `predict_with_virtual_ensemble`: per-virtual-member predictions (raw).
- `predict_with_decomposed_uq`: aleatoric/epistemic split via TotalUncertainty.
- `predictive_intervals`: gaussian or empirical two-sided intervals.
- `risk_coverage_curve` + `area_under_risk_coverage`: selective-prediction quality.
- `random_baseline_band`: bootstrap baseline for the risk-coverage curve.
- `cantelli_decision_rule`: P(y=1) - k·σ_epistemic > τ_open trade gating.
- `coverage_calibration` / `uq_quality_summary`: diagnostics.

References (local corpus per RESEARCH/literature/INDEX.md):
- Malinin, Prokhorenkova, Ustimenko (ICLR 2021) — primary mechanism (gap in
  local index; flagged for download).
- Manokhin (2024), Practical Conformal Prediction, Ch. 2 pp. 16-17 — aleatoric
  vs epistemic framing (`Downloads\Valeri Manokhin - Practical Guide to
  Applied Conformal Prediction in Python (2024).pdf`).
- Lekeufack et al., Conformal Decision Theory pp. 1-7 — confidence-band
  decision rule, λ as conservatism scalar (`Downloads\conformal_decision_theory.pdf`).
- Geifman & El-Yaniv (2017) — risk-coverage curve / AURC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


# -----------------------------------------------------------------------------
# Virtual ensemble prediction
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class UQPrediction:
    """Container for per-sample predictions with uncertainty.

    Attributes
    ----------
    p_mean : np.ndarray, shape (n_samples,)
        Mean predicted probability of class 1 across virtual ensembles.
    p_std : np.ndarray, shape (n_samples,)
        Standard deviation across virtual ensembles. Higher = more uncertain.
    p_samples : np.ndarray, shape (n_virtual, n_samples)
        Raw per-virtual-model probabilities (None if not retained).
    """

    p_mean: np.ndarray
    p_std: np.ndarray
    p_samples: np.ndarray | None


def predict_with_virtual_ensemble(
    model,
    X,
    *,
    n_virtual: int = 10,
    return_samples: bool = False,
) -> UQPrediction:
    """Run CatBoost virtual-ensemble prediction.

    Parameters
    ----------
    model : CatBoostClassifier or CatBoostEnsemble
        Trained model. For ensembles (`utils.CatBoostEnsemble`), virtual
        ensembles are produced per base model and pooled.
    X : array-like or pd.DataFrame
        Feature matrix.
    n_virtual : int, default 10
        Number of virtual-ensemble draws per base model. CatBoost requires
        `n_virtual <= iteration / 100` typically; safer small values: 5..20.
    return_samples : bool, default False
        If True, retain the full sample tensor (memory-heavy on large X).

    Returns
    -------
    UQPrediction
    """
    if hasattr(model, "models"):  # CatBoostEnsemble
        all_samples = []
        for m in model.models:
            all_samples.append(_virtual_ensemble_single(m, X, n_virtual))
        # Stack across base models and virtual draws.
        samples = np.concatenate(all_samples, axis=0)
    else:
        samples = _virtual_ensemble_single(model, X, n_virtual)

    p_mean = samples.mean(axis=0)
    p_std = samples.std(axis=0, ddof=0)

    return UQPrediction(
        p_mean=p_mean,
        p_std=p_std,
        p_samples=samples if return_samples else None,
    )


def _virtual_ensemble_single(model, X, n_virtual: int) -> np.ndarray:
    """CatBoost virtual-ensemble prediction for a single CatBoostClassifier.

    Returns array of shape (n_virtual, n_samples) of P(y=1).

    Falls back to `n_virtual` repeated copies of the standard `predict_proba`
    if virtual ensembles are not supported (for instance, when the model was
    trained without langevin). Falls back signal: zero std.
    """
    try:
        # `prediction_type="TotalUncertainty"` returns shape (n_samples, 3 or 4)
        # for binary classifiers: [mean, knowledge_uncertainty, data_uncertainty,
        # ...]. We still need per-sample draws to compute selective metrics
        # robustly. So instead use "VirtEnsembles" which returns per-virtual
        # predictions.
        per_virtual = model.virtual_ensembles_predict(
            X,
            prediction_type="Probability",
            virtual_ensembles_count=n_virtual,
            thread_count=-1,
        )
        # Returns shape (n_samples, n_virtual, 2) for binary classifier.
        if per_virtual.ndim == 3:
            samples = per_virtual[:, :, 1].T  # (n_virtual, n_samples)
        elif per_virtual.ndim == 2:
            # Fallback shape (n_samples, n_virtual)
            samples = per_virtual.T
        else:
            raise ValueError(f"Unexpected shape from virtual_ensembles_predict: {per_virtual.shape}")
        return samples
    except (AttributeError, Exception):
        # Fallback: virtual ensembles not supported (model not trained with
        # langevin). Return n_virtual copies of the deterministic prediction;
        # std will be zero, signalling that no UQ is available.
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            p1 = proba[:, 1]
        else:
            p1 = proba
        return np.tile(p1, (n_virtual, 1))


# -----------------------------------------------------------------------------
# Predictive intervals
# -----------------------------------------------------------------------------

def predictive_intervals(
    pred: UQPrediction,
    *,
    alpha: float = 0.1,
    method: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-sided predictive intervals for P(y=1).

    Parameters
    ----------
    pred : UQPrediction
        Output of `predict_with_virtual_ensemble`.
    alpha : float, default 0.1
        Coverage = 1 - alpha. Default = 90%.
    method : {"gaussian", "empirical"}
        - "gaussian" uses (p_mean ± z_{1-α/2} · p_std), clipped to [0, 1].
        - "empirical" uses per-sample quantiles of `p_samples` (requires
          `return_samples=True` in `predict_with_virtual_ensemble`).

    Returns
    -------
    (lower, upper) : Tuple of arrays, each shape (n_samples,)
    """
    if method == "gaussian":
        from scipy.stats import norm

        z = norm.ppf(1.0 - alpha / 2.0)
        lower = np.clip(pred.p_mean - z * pred.p_std, 0.0, 1.0)
        upper = np.clip(pred.p_mean + z * pred.p_std, 0.0, 1.0)
        return lower, upper

    if method == "empirical":
        if pred.p_samples is None:
            raise ValueError("'empirical' method requires return_samples=True")
        lower = np.quantile(pred.p_samples, alpha / 2.0, axis=0)
        upper = np.quantile(pred.p_samples, 1.0 - alpha / 2.0, axis=0)
        return lower, upper

    raise ValueError(f"Unknown method: {method!r}")


# -----------------------------------------------------------------------------
# UQ diagnostics
# -----------------------------------------------------------------------------

def selective_prediction_curve(
    y_true: np.ndarray,
    pred: UQPrediction,
    *,
    metric: str = "brier",
    n_quantiles: int = 20,
) -> pd.DataFrame:
    """Evaluate metric on the most-confident k-fraction of predictions.

    A useful UQ→trading bridge: if we only act on predictions where p_std
    is below the q-th quantile, do we get better PnL? This produces the
    underlying calibration for that decision.

    Parameters
    ----------
    y_true : np.ndarray
    pred : UQPrediction
    metric : {"brier", "roc_auc", "pr_auc"}
    n_quantiles : int, default 20
        Number of points on the curve (covers fractions 1/n .. 1).

    Returns
    -------
    pd.DataFrame with columns [confidence_fraction, n_samples, metric_value]
    sorted by confidence_fraction ascending.
    """
    metric_fn = {
        "brier": brier_score_loss,
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
    }[metric]

    # Sort by uncertainty ascending; lowest uncertainty first.
    order = np.argsort(pred.p_std)
    y_sorted = y_true[order]
    p_sorted = pred.p_mean[order]

    rows = []
    for i in range(1, n_quantiles + 1):
        frac = i / n_quantiles
        n = max(2, int(round(frac * len(y_sorted))))
        ys = y_sorted[:n]
        ps = p_sorted[:n]
        if len(np.unique(ys)) < 2 and metric in {"roc_auc", "pr_auc"}:
            value = float("nan")
        else:
            value = float(metric_fn(ys, ps))
        rows.append({"confidence_fraction": frac, "n_samples": n, metric: value})
    return pd.DataFrame(rows)


def coverage_calibration(
    y_true: np.ndarray,
    pred: UQPrediction,
    *,
    alphas: np.ndarray | None = None,
) -> pd.DataFrame:
    """Empirical vs nominal coverage of predictive intervals at multiple alpha.

    Tests whether nominal (1-α) intervals contain the realized class probability.
    Since y is binary, we use a slightly weakened test: at each α, count the
    fraction of samples where `p_lower ≤ y_true ≤ p_upper`.

    Returns
    -------
    pd.DataFrame with columns [alpha, nominal_coverage, empirical_coverage,
    mean_width].
    """
    if alphas is None:
        alphas = np.array([0.05, 0.1, 0.2, 0.3, 0.5])

    rows = []
    for alpha in alphas:
        lo, hi = predictive_intervals(pred, alpha=alpha, method="gaussian")
        empirical = float(((y_true >= lo) & (y_true <= hi)).mean())
        rows.append(
            {
                "alpha": float(alpha),
                "nominal_coverage": 1.0 - float(alpha),
                "empirical_coverage": empirical,
                "mean_width": float((hi - lo).mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class DecomposedUQ:
    """Aleatoric/epistemic decomposition from CatBoost SGLB virtual ensembles.

    Attributes
    ----------
    p_mean : (n_samples,)  Mean predicted P(y=1) across virtual members.
    sigma_total : (n_samples,)  sqrt(total_uncertainty), all in probability space.
    sigma_data : (n_samples,)  sqrt(data_uncertainty); aleatoric component.
    sigma_epistemic : (n_samples,)  sqrt(max(total - data, 0)); model-leverage.
    n_virtual : int  Number of virtual ensembles requested.
    """

    p_mean: np.ndarray
    sigma_total: np.ndarray
    sigma_data: np.ndarray
    sigma_epistemic: np.ndarray
    n_virtual: int


def predict_with_decomposed_uq(model, X, *, n_virtual: int = 10) -> DecomposedUQ:
    """Run CatBoost virtual ensembles with the TotalUncertainty decomposition.

    For binary classification, CatBoost returns shape (n_samples, 3) under
    `prediction_type="TotalUncertainty"`: columns are
    [mean, knowledge_uncertainty, data_uncertainty]
    where `knowledge_uncertainty + data_uncertainty = total_uncertainty` in
    probability space (per Malinin et al. 2021 + CatBoost docs).

    For ensembles (`utils.CatBoostEnsemble`), we compute per-base-model
    decompositions and pool: mean is the mean of base means, total is the
    sum of base totals plus the variance of base means, data is the mean
    of base datas. This is the standard ensemble-of-ensembles formula.
    """
    if hasattr(model, "models"):
        base_means = []
        base_data = []
        base_total = []
        for m in model.models:
            mean_, total_, data_ = _td_predict_single(m, X, n_virtual)
            base_means.append(mean_)
            base_data.append(data_)
            base_total.append(total_)
        base_means = np.stack(base_means, axis=0)
        base_data = np.stack(base_data, axis=0)
        base_total = np.stack(base_total, axis=0)

        p_mean = base_means.mean(axis=0)
        sigma_data_sq = base_data.mean(axis=0)
        # total = E[total_per_base] + Var[mean_per_base]
        sigma_total_sq = base_total.mean(axis=0) + base_means.var(axis=0, ddof=0)
    else:
        mean_, total_, data_ = _td_predict_single(model, X, n_virtual)
        p_mean = mean_
        sigma_data_sq = data_
        sigma_total_sq = total_

    sigma_data = np.sqrt(np.maximum(sigma_data_sq, 0.0))
    sigma_total = np.sqrt(np.maximum(sigma_total_sq, 0.0))
    sigma_epistemic_sq = np.maximum(sigma_total_sq - sigma_data_sq, 0.0)
    sigma_epistemic = np.sqrt(sigma_epistemic_sq)

    return DecomposedUQ(
        p_mean=p_mean,
        sigma_total=sigma_total,
        sigma_data=sigma_data,
        sigma_epistemic=sigma_epistemic,
        n_virtual=int(n_virtual),
    )


def _td_predict_single(model, X, n_virtual: int):
    """Wrap virtual_ensembles_predict(prediction_type='TotalUncertainty') for one model.

    Returns (mean, total_uncertainty, data_uncertainty), each shape (n_samples,)
    for binary classification.
    """
    try:
        out = model.virtual_ensembles_predict(
            X,
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=n_virtual,
            thread_count=-1,
        )
    except (AttributeError, Exception):
        # Fallback: model wasn't trained with langevin -> no UQ. Return zeros.
        proba = model.predict_proba(X)
        p1 = proba[:, 1] if proba.ndim == 2 else proba
        zero = np.zeros_like(p1)
        return p1, zero, zero

    # CatBoost output shapes for binary classification with TotalUncertainty
    # depend on the version. Two known layouts:
    #   shape (n, 2) = [mean_proba, total_uncertainty]  (newer)
    #   shape (n, 3) = [mean_proba, knowledge, data]    (older)
    # For Bernoulli outputs, aleatoric data_uncertainty = mean*(1-mean) by
    # construction (variance of a Bernoulli with mean = p); epistemic knowledge
    # uncertainty is the residual = total - data. We use this identity to
    # decompose the (n, 2) layout.
    out = np.asarray(out)
    if out.ndim != 2:
        raise ValueError(f"Unexpected TotalUncertainty ndim {out.ndim} (expected 2D).")
    if out.shape[1] == 2:
        mean_ = out[:, 0]
        total = out[:, 1]
        data_ = mean_ * (1.0 - mean_)  # Bernoulli aleatoric
        return mean_, total, data_
    if out.shape[1] >= 3:
        mean_ = out[:, 0]
        knowledge = out[:, 1]
        data_ = out[:, 2]
        total = knowledge + data_
        return mean_, total, data_
    raise ValueError(f"Unexpected TotalUncertainty shape {out.shape}.")


# -----------------------------------------------------------------------------
# Risk-coverage curve and AURC
# -----------------------------------------------------------------------------

def risk_coverage_curve(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    sigma: np.ndarray,
    *,
    metric: str = "brier",
    coverage_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Selective-prediction risk-coverage curve (Geifman & El-Yaniv 2017).

    Sort by `sigma` ascending; for each coverage `c`, evaluate `metric` on
    the most-confident `c` fraction. Returns a DataFrame ready for plotting.

    `metric` ∈ {"brier", "log_loss", "roc_auc", "pr_auc"}.
    """
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    metric_fn = {
        "brier": brier_score_loss,
        "log_loss": log_loss,
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
    }[metric]

    if coverage_grid is None:
        coverage_grid = np.linspace(0.05, 1.0, 20)

    order = np.argsort(sigma)
    y_sorted = np.asarray(y_true)[order]
    p_sorted = np.asarray(p_mean)[order]

    rows = []
    for c in coverage_grid:
        n = max(2, int(round(float(c) * len(y_sorted))))
        ys = y_sorted[:n]
        ps = p_sorted[:n]
        if metric in {"roc_auc", "pr_auc"} and len(np.unique(ys)) < 2:
            value = float("nan")
        else:
            value = float(metric_fn(ys, ps))
        rows.append(
            {"coverage": float(c), "n_samples": int(n), metric: value}
        )
    return pd.DataFrame(rows)


def area_under_risk_coverage(curve: pd.DataFrame, metric: str = "brier") -> float:
    """Trapezoidal AURC over the coverage grid; lower is better for risk metrics."""
    c = curve["coverage"].to_numpy()
    v = curve[metric].to_numpy()
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return float("nan")
    return float(np.trapz(v[finite], c[finite]))


def random_baseline_band(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    *,
    n_reps: int = 30,
    metric: str = "brier",
    coverage_grid: np.ndarray | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Risk-coverage curve under random selection, repeated for confidence band.

    Returns a DataFrame with columns [coverage, mean, std, q05, q95, n_reps].
    The UQ source is "useful" iff its risk-coverage curve sits below the
    `mean - std` band over a contiguous low-coverage region.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if coverage_grid is None:
        coverage_grid = np.linspace(0.05, 1.0, 20)

    by_coverage: dict[float, list[float]] = {float(c): [] for c in coverage_grid}
    for _ in range(n_reps):
        order = rng.permutation(n)
        # Random sigma proxy = position in this permutation, so smaller = "kept"
        sigma_random = np.argsort(order).astype(float)
        curve = risk_coverage_curve(
            y_true, p_mean, sigma_random, metric=metric, coverage_grid=coverage_grid
        )
        for c, v in zip(curve["coverage"], curve[metric]):
            by_coverage[float(c)].append(float(v))

    rows = []
    for c in coverage_grid:
        vs = np.asarray(by_coverage[float(c)])
        finite = vs[np.isfinite(vs)]
        if finite.size == 0:
            rows.append({"coverage": float(c), "mean": float("nan"), "std": float("nan"),
                          "q05": float("nan"), "q95": float("nan"), "n_reps": 0})
            continue
        rows.append({
            "coverage": float(c),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "q05": float(np.quantile(finite, 0.05)),
            "q95": float(np.quantile(finite, 0.95)),
            "n_reps": int(len(finite)),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Decision rule
# -----------------------------------------------------------------------------

def cantelli_decision_rule(
    pred: DecomposedUQ,
    *,
    tau_open: float,
    k: float = 1.0,
) -> np.ndarray:
    """One-sided Cantelli/Chebyshev confidence-band decision rule.

    Trade if `p_mean - k * sigma_epistemic > tau_open`. Distribution-free:
    P(p_mean - k*sigma < true_p) <= 1/(1+k^2). Default k=1.0 corresponds to
    requiring the lower-50% Cantelli bound on `P(y=1)` to clear `tau_open`;
    k in [0.5, 2.0] is the recommended tuning range (RESEARCH/literature/
    INDEX.md citation Manokhin / Lekeufack et al.).

    Returns a boolean array of trade decisions.
    """
    return (pred.p_mean - k * pred.sigma_epistemic) > tau_open


def uq_quality_summary(y_true: np.ndarray, pred: UQPrediction) -> dict:
    """Summary scalar diagnostics that the LEDGER can record.

    Returns dict with:
    - `mean_p_std`: average uncertainty across samples
    - `p_std_correlation_with_error`: Spearman correlation between p_std and
      the squared error |p_mean - y|^2. >0 means high uncertainty correlates
      with high error (UQ is signal); ≈0 means UQ is noise.
    - `selective_brier_top10`: Brier on the 10% most confident predictions.
    """
    from scipy.stats import spearmanr

    err = (pred.p_mean - y_true) ** 2
    if np.std(pred.p_std) < 1e-12:
        rho = float("nan")
    else:
        rho, _ = spearmanr(pred.p_std, err)
    rho = float(rho) if rho == rho else float("nan")  # NaN guard

    curve = selective_prediction_curve(y_true, pred, metric="brier", n_quantiles=10)
    selective_brier_top10 = float(curve.iloc[0]["brier"])

    return {
        "mean_p_std": float(pred.p_std.mean()),
        "p_std_correlation_with_error": rho,
        "selective_brier_top10": selective_brier_top10,
        "overall_brier": float(brier_score_loss(y_true, pred.p_mean)),
    }
