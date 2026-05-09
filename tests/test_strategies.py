"""Tests for `src.strategies` (Phase A round 1 strategies 1-3).

Pin:
- StrategyOutput rejects shape mismatch and out-of-range size.
- baseline_offline_tau / baseline_online_tau / combined_avg_tau emit boolean
  open arrays with size=1 and respect the τ threshold.
- combined_avg_tau with default weights = average; weights are honored.
- Required columns checked with helpful KeyError.
- STRATEGIES registry enumerates the three strategies in declaration order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.strategies import (  # noqa: E402
    STRATEGIES,
    StrategyOutput,
    baseline_offline_tau,
    baseline_online_tau,
    combined_avg_tau,
)


# ---------------------------------------------------------------------------
# StrategyOutput contract
# ---------------------------------------------------------------------------

def test_strategy_output_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        StrategyOutput(
            open_signal=np.array([True, False]),
            size=np.array([1.0, 1.0, 1.0]),
        )


def test_strategy_output_rejects_size_out_of_range():
    with pytest.raises(ValueError):
        StrategyOutput(
            open_signal=np.array([True]),
            size=np.array([1.5]),
        )
    with pytest.raises(ValueError):
        StrategyOutput(
            open_signal=np.array([True]),
            size=np.array([-0.1]),
        )


def test_strategy_output_coerces_open_signal_to_bool():
    out = StrategyOutput(
        open_signal=np.array([1, 0, 1]),
        size=np.array([1.0, 1.0, 1.0]),
    )
    assert out.open_signal.dtype == bool


# ---------------------------------------------------------------------------
# Required-columns errors
# ---------------------------------------------------------------------------

def test_baseline_offline_requires_p_offline():
    df = pd.DataFrame({"p_online": [0.1, 0.2]})
    with pytest.raises(KeyError):
        baseline_offline_tau(df, tau=0.3)


def test_baseline_online_requires_p_online():
    df = pd.DataFrame({"p_offline": [0.1, 0.2]})
    with pytest.raises(KeyError):
        baseline_online_tau(df, tau=0.3)


def test_combined_avg_requires_both():
    df = pd.DataFrame({"p_offline": [0.1, 0.2]})
    with pytest.raises(KeyError):
        combined_avg_tau(df, tau=0.3)


# ---------------------------------------------------------------------------
# baseline_offline_tau
# ---------------------------------------------------------------------------

def test_baseline_offline_open_threshold_strict():
    df = pd.DataFrame({"p_offline": [0.10, 0.30, 0.50, 0.70]})
    out = baseline_offline_tau(df, tau=0.30)
    np.testing.assert_array_equal(out.open_signal, [False, False, True, True])
    np.testing.assert_array_equal(out.size, [1.0, 1.0, 1.0, 1.0])


def test_baseline_offline_size_unit_position():
    df = pd.DataFrame({"p_offline": np.linspace(0.0, 1.0, 10)})
    out = baseline_offline_tau(df, tau=0.5)
    assert out.size.shape == (10,)
    assert (out.size == 1.0).all()


# ---------------------------------------------------------------------------
# baseline_online_tau
# ---------------------------------------------------------------------------

def test_baseline_online_threshold_strict():
    df = pd.DataFrame({"p_online": [0.10, 0.30, 0.50, 0.70]})
    out = baseline_online_tau(df, tau=0.30)
    np.testing.assert_array_equal(out.open_signal, [False, False, True, True])


# ---------------------------------------------------------------------------
# combined_avg_tau
# ---------------------------------------------------------------------------

def test_combined_avg_default_equal_weights():
    df = pd.DataFrame({
        "p_offline": [0.10, 0.50, 0.80, 0.20],
        "p_online":  [0.30, 0.20, 0.40, 0.50],
    })
    out = combined_avg_tau(df, tau=0.30)  # avg = [0.20, 0.35, 0.60, 0.35]
    np.testing.assert_array_equal(out.open_signal, [False, True, True, True])


def test_combined_avg_custom_weights():
    df = pd.DataFrame({
        "p_offline": [0.40, 0.40],
        "p_online":  [0.10, 0.10],
    })
    # 0.8·0.4 + 0.2·0.1 = 0.34 → above τ=0.30
    out = combined_avg_tau(df, tau=0.30, w_offline=0.8, w_online=0.2)
    np.testing.assert_array_equal(out.open_signal, [True, True])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_phase_A_round_1_strategies():
    expected_round_1 = ["baseline_offline_tau", "baseline_online_tau",
                         "combined_avg_tau"]
    for name in expected_round_1:
        assert name in STRATEGIES, f"missing {name}"
        assert callable(STRATEGIES[name])


def test_registry_strategies_return_strategy_output():
    df = pd.DataFrame({
        "p_offline": [0.1, 0.5, 0.7],
        "p_online":  [0.2, 0.4, 0.6],
    })
    for name, fn in STRATEGIES.items():
        out = fn(df, tau=0.3)
        assert isinstance(out, StrategyOutput)
        assert out.open_signal.shape == (3,)
        assert out.size.shape == (3,)
