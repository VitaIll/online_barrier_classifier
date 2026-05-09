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
}
