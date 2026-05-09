"""Phase A strategies — two-layer architecture.

The system architecture is HIERARCHICAL, not parallel:

    offline CatBoost  ──>  p_offline
                              │  (p_offline + selected_features)
                              ▼
                    online ARFClassifier  ──>  p_online   ◄── system output
                              │
                              ▼
                    streaming Mondrian-ACI on p_online  ──>  q_t per regime,
                                                              in_set_α, etc.

`p_online` is the *corrected* / *calibrated* probability the streaming layer
emits AFTER consuming `p_offline` as one of its inputs. The conformal layer
(round-007 ACI / round-008 Mondrian-ACI) then sits on top of `p_online`.

**Therefore**: `p_offline` and `p_online` are **NOT siblings**. Averaging them
or stacking a logistic regression on `(p_offline, p_online)` is architecturally
backwards (it regresses the child against itself + its parent). Such combiners
are explicitly forbidden under this module's contract; round-015 deletes them.

Each strategy is `strategy(predictions_df, **knobs) -> StrategyOutput`
producing an `open_signal` boolean array and a `size` array in `[0, 1]`. The
backtest harness consumes (open_signal, size) — see `simulate_inventory_aware`
for the unit-position case and `simulate_inventory_aware_sized` for the
size-aware variant.

Valid strategies catalogued here:
1. ``baseline_offline_tau``  — open = p_offline > τ; size = 1.  *Sanity floor:
                                 the offline layer alone, before the online
                                 correction is applied.*
2. ``baseline_online_tau``   — open = p_online  > τ; size = 1.  *The system's
                                 actual output; the headline number.*
3. ``conformal_gate_tau``    — open = (p_online > τ) & (in_set_α == 1); size=1.
                                 *Conformal layer gates entries on top of the
                                 calibrated p_online — round-008's Mondrian-ACI
                                 confidence interval.*
4. ``mondrian_aci_size``     — open = (in_set_α == 1); size = clip(k · max(0,
                                 p_online − (1 − q_lo_α)), 0, 1).  *Sized
                                 entries by per-regime confidence on p_online.*
5. ``null_random_at_rate``   — random entries matched to a target trade rate.
                                 *No-skill comparator for bootstrap p_boot.*

The σ_epistemic abstention card (H-302) is the natural extension once the
3-seed CatBoost ensemble lands — it adds a confidence band to the *p_online*
output, not to p_offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


__all__ = [
    "StrategyOutput",
    "baseline_offline_tau",
    "baseline_online_tau",
    "conformal_gate_tau",
    "mondrian_aci_size",
    "null_random_at_rate",
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
# 1. baseline_offline_tau — sanity floor
# ---------------------------------------------------------------------------

def baseline_offline_tau(predictions_df: pd.DataFrame, *, tau: float) -> StrategyOutput:
    """Open long when `p_offline > tau`; size = 1.

    The offline layer alone — used only as a sanity floor against the
    online layer's headline. Architecturally this is the *input* to the
    online layer, not a competing predictor.
    """
    _require(predictions_df, ["p_offline"])
    p = predictions_df["p_offline"].to_numpy(dtype=float)
    open_signal = p > float(tau)
    size = np.ones(len(p), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 2. baseline_online_tau — system output
# ---------------------------------------------------------------------------

def baseline_online_tau(predictions_df: pd.DataFrame, *, tau: float) -> StrategyOutput:
    """Open long when `p_online > tau`; size = 1.

    `p_online` is the corrected, regime-calibrated probability the streaming
    online layer emits after consuming `(features + p_offline)`. This is the
    headline strategy — the two-layer system's actual output, before any
    conformal-layer gating.
    """
    _require(predictions_df, ["p_online"])
    p = predictions_df["p_online"].to_numpy(dtype=float)
    open_signal = p > float(tau)
    size = np.ones(len(p), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 3. conformal_gate_tau — conformal layer on p_online
# ---------------------------------------------------------------------------

def conformal_gate_tau(
    predictions_df: pd.DataFrame, *, tau: float, alpha_level: str = "10",
) -> StrategyOutput:
    """Open when (p_online > τ) AND (in_set_{α} == 1); size = 1.

    The third architectural layer: Mondrian-ACI's per-regime conformal set on
    top of `p_online`. `in_set_{α} == 1` ⇒ the prediction set at confidence
    1−α contains class 1, i.e., the conformal layer endorses the singleton-
    or-full {1} call. `alpha_level` selects which `in_set_*` column gates the
    trade. Default is α=0.10 (round-008's headline level).
    """
    col = f"in_set_{alpha_level}"
    _require(predictions_df, ["p_online", col])
    p_on = predictions_df["p_online"].to_numpy(dtype=float)
    in_set = predictions_df[col].to_numpy().astype(int)
    open_signal = (p_on > float(tau)) & (in_set == 1)
    size = np.ones(len(p_on), dtype=float)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 4. mondrian_aci_size — sized via confidence on p_online
# ---------------------------------------------------------------------------

def mondrian_aci_size(
    predictions_df: pd.DataFrame, *, k: float = 5.0, alpha_level: str = "10",
) -> StrategyOutput:
    """Open when `in_set_{α} == 1`; size by Mondrian-ACI confidence on p_online.

    Size = clip(k · max(0, p_online − (1 − q_lo_α)), 0, 1). The conformal
    threshold against `p_online` is `(1 − q_lo_α)`; the excess `p_online − (1 −
    q_lo_α)` is the per-regime confidence margin. `k` is the val-chosen
    sizing scale; default k=5 ⇒ fully sized when `p_online` exceeds the
    conformal threshold by 0.20.

    Distinct from the offline-only sized variant: q_lo_α here is computed
    *from p_online* (round-008 contract), not from p_offline.
    """
    col_in = f"in_set_{alpha_level}"
    col_q = f"q_lo_{alpha_level}"
    _require(predictions_df, ["p_online", col_in, col_q])
    p_on = predictions_df["p_online"].to_numpy(dtype=float)
    q = predictions_df[col_q].to_numpy(dtype=float)
    in_set = predictions_df[col_in].to_numpy().astype(int)

    open_signal = in_set == 1
    raw_excess = np.maximum(0.0, p_on - (1.0 - q))
    size = np.clip(float(k) * raw_excess, 0.0, 1.0)
    size = np.where(open_signal, size, 0.0)
    return StrategyOutput(open_signal=open_signal, size=size)


# ---------------------------------------------------------------------------
# 5. null_random_at_rate — no-skill comparator
# ---------------------------------------------------------------------------

def null_random_at_rate(
    predictions_df: pd.DataFrame, *, target_rate: float, seed: int = 42,
) -> StrategyOutput:
    """Random entries with `target_rate`; size = 1.

    Used as the bootstrap null for `p_boot`. `target_rate` should match the
    strategy-under-test's empirical entry rate so the null has matched
    trade frequency.
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
# Five legitimate strategies under the two-layer architecture. Sibling-style
# combiners of (p_offline, p_online) are deliberately NOT in this registry;
# they violate the architectural contract.

StrategyFn = Callable[..., StrategyOutput]

STRATEGIES: dict[str, StrategyFn] = {
    "baseline_offline_tau": baseline_offline_tau,
    "baseline_online_tau": baseline_online_tau,
    "conformal_gate_tau": conformal_gate_tau,
    "mondrian_aci_size": mondrian_aci_size,
    "null_random_at_rate": null_random_at_rate,
}
