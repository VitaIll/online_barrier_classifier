"""Phase A unified prediction surface.

`predict()` consumes an already-loaded `predictions_df` (one row per decision
boundary, with at minimum `p_offline`, `p_online`, `y_true`) plus the **frozen**
regime cuts and a Mondrian-ACI configuration, and returns the prediction
surface defined in MANDATE §A.1 — `p_offline`, `p_online`, per-regime conformal
thresholds at α ∈ {0.05, 0.10, 0.20}, `in_set_*` indicators, plus optional
`sigma_epistemic` / `confidence_score`.

This is the **prediction layer** every Phase-A strategy reads from. It is pure
(no I/O), deterministic given inputs, and testable. Reuses
`src.conformal.aci_mondrian_stream` for the prequential q-trajectory; reuses
`src.utils.calibrate_alpha` semantics to validate that the regime cuts were
fit on the appropriate slice (caller responsibility).

Round-011 scope: prediction surface populated for p_offline + p_online +
regime + per-α q + in_set; sigma_epistemic / confidence_score are optional
(emitted as NaN when virtual-ensemble is unavailable on the persisted
single-CatBoost model). Phase C revisits sigma_epistemic when the
3-seed CatBoost ensemble is materialised on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.conformal import aci_mondrian_stream


DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)
DEFAULT_REGIME_FEATURE: str = "parkinson_var_rolling_mean_24"


@dataclass(frozen=True)
class RegimeCuts:
    """Frozen tercile cuts on a volatility proxy.

    `edges` are the inner cut points (length n_regimes-1) so that
    `regime_id = sum(value > e for e in edges)` produces ids in
    `0..n_regimes-1`. Persisted by the caller to ensure val/test are
    binned with the same boundaries fit on train (or train+cal) only.
    """

    feature: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]

    def assign(self, values: np.ndarray) -> np.ndarray:
        """Return the integer regime id (0-indexed) for each value."""
        v = np.asarray(values, dtype=float)
        ids = np.zeros(len(v), dtype=int)
        for e in self.edges:
            ids += (v > float(e)).astype(int)
        return ids

    def assign_labels(self, values: np.ndarray) -> np.ndarray:
        """Return the human label ('low'/'med'/'high') for each value."""
        return np.asarray(self.labels, dtype=object)[self.assign(values)]


def fit_regime_cuts(
    values: np.ndarray,
    *,
    feature_name: str = DEFAULT_REGIME_FEATURE,
    n_regimes: int = 3,
    labels: Sequence[str] | None = None,
) -> RegimeCuts:
    """Fit equal-frequency cuts on `values` and return a frozen `RegimeCuts`.

    Used once on the train-or-train+cal slice; the resulting object is then
    serialized and applied to val/test without refitting.
    """
    if labels is None:
        if n_regimes == 3:
            labels = ("low", "med", "high")
        else:
            labels = tuple(f"r{i}" for i in range(n_regimes))
    if len(labels) != n_regimes:
        raise ValueError(f"labels has {len(labels)} entries; expected {n_regimes}.")
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("Cannot fit regime cuts on empty array.")
    quantiles = np.linspace(0.0, 1.0, n_regimes + 1)[1:-1]
    edges = tuple(float(q) for q in np.quantile(v, quantiles))
    return RegimeCuts(feature=feature_name, edges=edges, labels=tuple(labels))


def warm_q_init_by_regime(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    regime_cal: np.ndarray,
    alpha: float,
    *,
    min_per_regime: int = 50,
    fallback: float = 0.5,
) -> dict[Any, float]:
    """Warm-start the per-regime q for Mondrian-ACI.

    Per-regime LAC quantile of the calibration scores: the same starting
    point used by the round-008 Mondrian-ACI run. Regimes with fewer than
    `min_per_regime` calibration samples fall back to `fallback`.
    """
    from src.conformal import lac_score, finite_sample_quantile

    p = np.asarray(p_cal, dtype=float)
    y = np.asarray(y_cal).astype(int)
    regimes = np.asarray(regime_cal)
    if not (p.shape == y.shape == regimes.shape):
        raise ValueError(
            f"p_cal/y_cal/regime_cal shape mismatch: {p.shape}/{y.shape}/{regimes.shape}"
        )
    scores = lac_score(p, y)
    out: dict[Any, float] = {}
    for r in np.unique(regimes):
        mask = regimes == r
        if int(mask.sum()) >= min_per_regime:
            out[r] = float(finite_sample_quantile(scores[mask], alpha))
        else:
            out[r] = float(fallback)
    return out


def predict(
    predictions_df: pd.DataFrame,
    *,
    regime_values: np.ndarray,
    regime_cuts: RegimeCuts,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    gamma: float = 0.01,
    q_init_by_regime_per_alpha: Mapping[float, Mapping[Any, float]] | None = None,
    sigma_epistemic: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build the unified Phase-A prediction surface.

    Parameters
    ----------
    predictions_df : DataFrame
        Must contain at least `p_offline`, `p_online`, `y_true`. Extra
        columns are preserved.
    regime_values : (n,) array
        Values of `regime_cuts.feature` aligned row-wise with `predictions_df`.
    regime_cuts : RegimeCuts
        Frozen cut points (fit on train or train+cal only — caller's
        responsibility).
    alphas : sequence of float
        Conformal miscoverage levels. Defaults to (0.05, 0.10, 0.20).
    gamma : float
        Mondrian-ACI learning rate (round-008 default = 0.01).
    q_init_by_regime_per_alpha : optional mapping `α -> {regime_id: q}`.
        Per-α warm-start q. If absent for a given α, that α's q starts at 0.5
        for every regime (plain ACI default).
    sigma_epistemic : optional (n,) array
        CatBoost virtual-ensemble σ for each row; if None, NaN is emitted.

    Returns
    -------
    DataFrame with the original columns plus:
        - `regime_id` (int) and `regime_label` (str)
        - `q_lo_05`, `q_lo_10`, `q_lo_20` (or whichever αs are passed)
        - `in_set_05`, `in_set_10`, `in_set_20` (whether {1} ∈ set at α)
        - `sigma_epistemic` (NaN if not supplied)
        - `confidence_score` = `max(0, p_offline - q_lo_10)` (default α=0.10
          for the gating rule; reflects how much the offline prediction
          clears the conformal threshold for class 1).
    """
    out = predictions_df.copy()
    n = len(out)
    if len(regime_values) != n:
        raise ValueError(
            f"regime_values length {len(regime_values)} != predictions_df length {n}"
        )

    regime_ids = regime_cuts.assign(regime_values)
    regime_labels = regime_cuts.assign_labels(regime_values)
    out["regime_id"] = regime_ids
    out["regime_label"] = regime_labels

    p_offline = out["p_offline"].to_numpy(dtype=float)
    p_online = out["p_online"].to_numpy(dtype=float)
    y_true = out["y_true"].to_numpy(dtype=int)

    overrides = dict(q_init_by_regime_per_alpha or {})

    for alpha in alphas:
        # The conformal threshold is computed on `p_offline` (the offline
        # prediction is the conformal score's basis; the LAC score is built
        # from p_offline. The mandate's `q_lo_*` is the q_t the conformal
        # layer uses against `p_offline` to decide whether {1} is in the
        # singleton set. We could equally construct q on p_online; for
        # Phase-A we follow the round-008 convention which used p_offline
        # for the offline branch and p_online for the online branch — but
        # the §A.1 contract calls out the "active regime's q at each row",
        # which is defined by the score at that row. We therefore store
        # one q per α per row driven by p_offline; strategies that gate on
        # p_online use the same q via the singleton {1} check below.
        q_init_for_alpha: dict[Any, float] = {}
        for r in np.unique(regime_ids):
            q_init_for_alpha[int(r)] = float(overrides.get(alpha, {}).get(r, 0.5))

        stream = aci_mondrian_stream(
            p_stream=p_offline,
            y_stream=y_true,
            regime_stream=regime_ids,
            alpha=float(alpha),
            gamma=float(gamma),
            q_init_by_regime=q_init_for_alpha,
        )
        q_history = stream["q_history"]
        # `in_set_* = 1` iff p_offline at row t >= 1 - q_t (singleton {1} OR full).
        in_set = (p_offline >= 1.0 - q_history).astype(int)

        suffix = f"{int(round(alpha * 100)):02d}"
        out[f"q_lo_{suffix}"] = q_history
        out[f"in_set_{suffix}"] = in_set

    if sigma_epistemic is not None:
        if len(sigma_epistemic) != n:
            raise ValueError(
                f"sigma_epistemic length {len(sigma_epistemic)} != n {n}"
            )
        out["sigma_epistemic"] = np.asarray(sigma_epistemic, dtype=float)
    else:
        out["sigma_epistemic"] = np.full(n, np.nan, dtype=float)

    # Default confidence_score: how much the offline probability clears the
    # α=0.10 conformal threshold (mandate §A.1).
    if "q_lo_10" in out.columns:
        out["confidence_score"] = np.maximum(
            0.0, p_offline - (1.0 - out["q_lo_10"].to_numpy())
        )
    else:
        out["confidence_score"] = np.full(n, np.nan, dtype=float)

    return out
