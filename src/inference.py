"""Phase A unified prediction surface — two-layer architecture.

The pipeline is hierarchical:

    offline CatBoost  ──>  p_offline
                              │  (p_offline + selected_features)
                              ▼
                    online ARFClassifier  ──>  p_online    ← system output
                              │
                              ▼
                    streaming Mondrian-ACI on p_online  ──> per-regime q_t,
                                                             in_set_α, etc.

`predict()` consumes an already-loaded `predictions_df` (one row per decision
boundary, with `p_offline`, `p_online`, `y_true`) plus the **frozen** regime
cuts and a Mondrian-ACI configuration, and emits the unified prediction
surface that downstream strategies read from.

Per round-008 (Mondrian-ACI on p_online), the conformal stream MUST be driven
by `p_online` — the calibrated/corrected output the system actually emits.
Driving it off `p_offline` (as round-011's predict() incorrectly did) treats
the offline layer as the system output and ignores the architecture.

Round-015 fixes that: `p_stream=p_online` everywhere, and `confidence_score`
is computed against `p_online` too. The offline-driven conformal columns are
NOT emitted; downstream strategies must consume `p_online` for any conformal
gating.
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
    `0..n_regimes-1`. Caller persists this object so val/test are binned
    with the same boundaries fit on train (or train+cal) only.
    """

    feature: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]

    def assign(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        ids = np.zeros(len(v), dtype=int)
        for e in self.edges:
            ids += (v > float(e)).astype(int)
        return ids

    def assign_labels(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.labels, dtype=object)[self.assign(values)]


def fit_regime_cuts(
    values: np.ndarray,
    *,
    feature_name: str = DEFAULT_REGIME_FEATURE,
    n_regimes: int = 3,
    labels: Sequence[str] | None = None,
) -> RegimeCuts:
    """Fit equal-frequency cuts on `values`."""
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

    `p_cal` should be `p_online` (the streaming-layer output) — the
    Mondrian-ACI conformal layer always sits on top of `p_online` per
    the two-layer architecture.

    Per-regime LAC quantile of the calibration scores. Regimes with fewer
    than `min_per_regime` calibration samples fall back to `fallback`.
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
    """Build the unified Phase-A prediction surface — Mondrian-ACI on p_online.

    Parameters
    ----------
    predictions_df : DataFrame
        Must contain `p_offline`, `p_online`, `y_true`. The offline column is
        retained for the sanity-floor strategy; the conformal layer is built
        from `p_online` only.
    regime_values : (n,) array
        Values of `regime_cuts.feature` aligned row-wise with `predictions_df`.
    regime_cuts : RegimeCuts
        Frozen cut points fit on train (or train+cal) only.
    alphas : sequence of float
        Conformal miscoverage levels. Defaults to (0.05, 0.10, 0.20).
    gamma : float
        Mondrian-ACI learning rate (round-008 default = 0.01).
    q_init_by_regime_per_alpha : optional mapping `α -> {regime_id: q}`
        Warm-start q from a per-regime LAC quantile on a held-out cal set
        of `p_online`. If absent for a given α, q starts at 0.5.
    sigma_epistemic : optional (n,) array
        CatBoost virtual-ensemble σ for each row; if None, NaN is emitted.

    Returns
    -------
    DataFrame with the original columns plus:
        - `regime_id` (int) and `regime_label` (str)
        - `q_lo_*` columns at each α (one per request)
        - `in_set_*` indicators (1 if {1} is in the conformal set at α
          using `p_online` as the score)
        - `sigma_epistemic` (NaN if not supplied)
        - `confidence_score` = max(0, p_online − (1 − q_lo_10)) — the
          per-regime confidence margin AGAINST p_online at α=0.10.
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

    if "p_online" not in out.columns:
        raise KeyError(
            "predict() requires `p_online` in predictions_df — the conformal "
            "layer is driven by the online layer's output (two-layer "
            "architecture; round-015 contract)."
        )
    p_online = out["p_online"].to_numpy(dtype=float)
    y_true = out["y_true"].to_numpy(dtype=int)

    overrides = dict(q_init_by_regime_per_alpha or {})

    for alpha in alphas:
        # The conformal stream is driven by p_online — the two-layer
        # architecture's output. `in_set_α = 1` iff p_online ≥ 1 - q_t at
        # the active regime's threshold.
        q_init_for_alpha: dict[Any, float] = {}
        for r in np.unique(regime_ids):
            q_init_for_alpha[int(r)] = float(overrides.get(alpha, {}).get(r, 0.5))

        stream = aci_mondrian_stream(
            p_stream=p_online,
            y_stream=y_true,
            regime_stream=regime_ids,
            alpha=float(alpha),
            gamma=float(gamma),
            q_init_by_regime=q_init_for_alpha,
        )
        q_history = stream["q_history"]
        in_set = (p_online >= 1.0 - q_history).astype(int)

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

    if "q_lo_10" in out.columns:
        out["confidence_score"] = np.maximum(
            0.0, p_online - (1.0 - out["q_lo_10"].to_numpy())
        )
    else:
        out["confidence_score"] = np.full(n, np.nan, dtype=float)

    return out
