"""Phase A strategies (MANDATE §A.2).

Each strategy is a function `strategy(predictions_df, **knobs) -> StrategyOutput`
producing an `open_signal` boolean array and a `size` array in `[0, 1]`. The
backtest harness consumes (open_signal, size) — see `simulate_inventory_aware`
for the unit-position case and `simulate_inventory_aware_sized` (round-012) for
the size-aware variant.

Strategies catalogued here:
1. baseline_offline_tau   — open = p_offline > τ; size = 1.
2. baseline_online_tau    — open = p_online  > τ; size = 1.
3. combined_avg_tau       — open = 0.5·p_off + 0.5·p_on > τ; size = 1.
4. combined_stacked_tau   — logistic on (p_off, p_on) fit on val; open at τ.
5. conformal_gate_tau     — open = (p_offline > τ) & in_set_10 == 1; size = 1.
6. mondrian_aci_size      — open = singleton {1} at active regime;
                            size = clip(k · max(0, p - q), 0, 1).
7. null_random_at_rate    — random entries matched to a target trade rate.

Strategies 1-3 are implemented in this round (Phase A round 1). Strategies
4-7 land in subsequent rounds without changing this module's public API:
new entries are appended to `STRATEGIES`. Tests verify the registry contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


__all__ = [
    "StrategyOutput",
    "baseline_offline_tau",
    "baseline_online_tau",
    "combined_avg_tau",
    "combined_stacked_tau",
    "conformal_gate_tau",
    "mondrian_aci_size",
    "null_random_at_rate",
    "fit_stacker",
    "STRATEGIES",
]


@dataclass(frozen=True)
class StrategyOutput:
    """Per-row open/size arrays.

    `open_signal` ∈ {0, 1} per decision boundary; `size` ∈ [0, 1] per
    boundary, applied multiplicatively to the unit-position when the
    sized backtest harness is used.
    """

    open_signal: np.ndarray
    size: np.ndarray

    def __post_init__(self):
        if self.open_signal.shape != self.size.shape:
            raise ValueError(
                f"open_signal {self.open_signal.shape} vs size {self.size.shape}"
            )
        if self.open_signal.dtype != bool:
            object.__setattr__(self, "open_signal", self.open_signal.astype(bool))
        if not np.all((self.size >= 0.0) & (self.size <= 1.0)):
            raise ValueError("size must lie in [0, 1] elementwise.")


def _require(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"strategy requires columns {missing}; got {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# 1. baseline_offline_tau
# ---------------------------------------------------------------------------

def baseline_offline_tau(predictions_df: pd.DataFrame, *, tau: float) -> StrategyOutput:
    """Open long when `p_offline > tau`; size = 1."""
    _require(predictions_df, ["p_offline"])
    p = predictions_df["p_offline"].to_numpy(dtype=float)
    open_signal = p > float(tau)
    size = np.ones(len(p), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 2. baseline_online_tau
# ---------------------------------------------------------------------------

def baseline_online_tau(predictions_df: pd.DataFrame, *, tau: float) -> StrategyOutput:
    """Open long when `p_online > tau`; size = 1."""
    _require(predictions_df, ["p_online"])
    p = predictions_df["p_online"].to_numpy(dtype=float)
    open_signal = p > float(tau)
    size = np.ones(len(p), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 3. combined_avg_tau
# ---------------------------------------------------------------------------

def combined_avg_tau(
    predictions_df: pd.DataFrame,
    *,
    tau: float,
    w_offline: float = 0.5,
    w_online: float = 0.5,
) -> StrategyOutput:
    """Open long when `w_off·p_off + w_on·p_on > tau`; size = 1.

    Default weights are equal; weights need not sum to 1 (caller can use 0.6/0.4
    etc.) but the comparison is to `tau` directly.
    """
    _require(predictions_df, ["p_offline", "p_online"])
    p_off = predictions_df["p_offline"].to_numpy(dtype=float)
    p_on = predictions_df["p_online"].to_numpy(dtype=float)
    p_combined = w_offline * p_off + w_online * p_on
    open_signal = p_combined > float(tau)
    size = np.ones(len(p_combined), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 4. combined_stacked_tau (logistic regression on (p_offline, p_online) on val)
# ---------------------------------------------------------------------------

def fit_stacker(val_df: pd.DataFrame, *, C: float = 1.0, max_iter: int = 1000) -> dict:
    """Fit a logistic regression of y_true on (p_offline, p_online) on val.

    Returns a parameter dict consumed by `combined_stacked_tau`. The fit is
    light (2-feature logistic) but is the first place the strategy *learns*
    how to combine offline + online signals from labeled data.
    """
    from sklearn.linear_model import LogisticRegression

    _require(val_df, ["p_offline", "p_online", "y_true"])
    X = val_df[["p_offline", "p_online"]].to_numpy(dtype=float)
    y = val_df["y_true"].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "fit_stacker needs both classes in val; got "
            f"{int(y.sum())} positives in {len(y)} samples."
        )
    clf = LogisticRegression(C=float(C), fit_intercept=True, max_iter=int(max_iter))
    clf.fit(X, y)
    return {
        "coef_offline": float(clf.coef_[0, 0]),
        "coef_online": float(clf.coef_[0, 1]),
        "intercept": float(clf.intercept_[0]),
    }


def _stacker_proba(
    p_offline: np.ndarray, p_online: np.ndarray, stacker: dict
) -> np.ndarray:
    """Apply the fitted logistic combination: σ(b + w0·p_off + w1·p_on)."""
    z = (
        float(stacker["intercept"])
        + float(stacker["coef_offline"]) * np.asarray(p_offline, dtype=float)
        + float(stacker["coef_online"]) * np.asarray(p_online, dtype=float)
    )
    # Numerically stable sigmoid.
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def combined_stacked_tau(
    predictions_df: pd.DataFrame, *, tau: float, stacker: dict,
) -> StrategyOutput:
    """Open when σ(stacker(p_off, p_on)) > τ; size = 1.

    `stacker` is the dict returned by `fit_stacker(val_df)` — fitted ONCE on
    the validation slice, then frozen.
    """
    _require(predictions_df, ["p_offline", "p_online"])
    p_off = predictions_df["p_offline"].to_numpy(dtype=float)
    p_on = predictions_df["p_online"].to_numpy(dtype=float)
    p_stack = _stacker_proba(p_off, p_on, stacker)
    open_signal = p_stack > float(tau)
    size = np.ones(len(p_stack), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 5. conformal_gate_tau
# ---------------------------------------------------------------------------

def conformal_gate_tau(
    predictions_df: pd.DataFrame, *, tau: float, alpha_level: str = "10",
) -> StrategyOutput:
    """Open when (p_offline > τ) AND (in_set_{α} == 1); size = 1.

    `alpha_level` selects which `in_set_*` column gates the trade. Default is
    α=0.10 ("90% conformal coverage" — round-008's headline level). Other
    valid values: "05" (tighter, more abstention), "20" (looser).
    """
    col = f"in_set_{alpha_level}"
    _require(predictions_df, ["p_offline", col])
    p_off = predictions_df["p_offline"].to_numpy(dtype=float)
    in_set = predictions_df[col].to_numpy().astype(int)
    open_signal = (p_off > float(tau)) & (in_set == 1)
    size = np.ones(len(p_off), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 6. mondrian_aci_size
# ---------------------------------------------------------------------------

def mondrian_aci_size(
    predictions_df: pd.DataFrame, *, k: float = 5.0, alpha_level: str = "10",
) -> StrategyOutput:
    """Open when `in_set_{α}` says {1} is in the set; size by confidence score.

    Size = clip(k · max(0, p_offline - (1 - q_lo_α)), 0, 1). The mandate's
    `k` is the val-chosen sizing scale; default k=5 corresponds to "fully
    sized when p_offline exceeds the conformal threshold by 0.20".
    """
    col_in = f"in_set_{alpha_level}"
    col_q = f"q_lo_{alpha_level}"
    _require(predictions_df, ["p_offline", col_in, col_q])
    p_off = predictions_df["p_offline"].to_numpy(dtype=float)
    q = predictions_df[col_q].to_numpy(dtype=float)
    in_set = predictions_df[col_in].to_numpy().astype(int)

    open_signal = in_set == 1
    raw_excess = np.maximum(0.0, p_off - (1.0 - q))
    size = np.clip(float(k) * raw_excess, 0.0, 1.0)
    # When open_signal is False, set size to 0 to keep contract clean.
    size = np.where(open_signal, size, 0.0)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 7. null_random_at_rate
# ---------------------------------------------------------------------------

def null_random_at_rate(
    predictions_df: pd.DataFrame, *, target_rate: float, seed: int = 42,
) -> StrategyOutput:
    """Random entries with target rate; size = 1.

    Used as the bootstrap null for `p_boot` against a fixed strategy: rerun
    multiple seeds, compare the strategy's Sharpe to the null distribution.
    `target_rate` should match the strategy's empirical entry rate so the
    null has the same trade rate.
    """
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError(f"target_rate must be in [0, 1]; got {target_rate}")
    n = len(predictions_df)
    rng = np.random.default_rng(int(seed))
    open_signal = rng.uniform(0.0, 1.0, size=n) < float(target_rate)
    size = np.ones(n, dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Strategies in the order they are listed in MANDATE §A.2. Subsequent rounds
# extend this dict by appending entries; the iteration order is preserved.

StrategyFn = Callable[..., StrategyOutput]

STRATEGIES: dict[str, StrategyFn] = {
    "baseline_offline_tau": baseline_offline_tau,
    "baseline_online_tau": baseline_online_tau,
    "combined_avg_tau": combined_avg_tau,
    "combined_stacked_tau": combined_stacked_tau,
    "conformal_gate_tau": conformal_gate_tau,
    "mondrian_aci_size": mondrian_aci_size,
    "null_random_at_rate": null_random_at_rate,
}
