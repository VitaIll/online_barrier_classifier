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
    combined_stacked_tau,
    conformal_gate_tau,
    fit_stacker,
    mondrian_aci_size,
    null_random_at_rate,
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

def test_registry_contains_all_phase_A_strategies():
    expected = [
        "baseline_offline_tau", "baseline_online_tau", "combined_avg_tau",
        "combined_stacked_tau", "conformal_gate_tau", "mondrian_aci_size",
        "null_random_at_rate",
    ]
    for name in expected:
        assert name in STRATEGIES, f"missing {name}"
        assert callable(STRATEGIES[name])


# ---------------------------------------------------------------------------
# fit_stacker / combined_stacked_tau
# ---------------------------------------------------------------------------

def test_fit_stacker_returns_required_keys():
    rng = np.random.default_rng(0)
    n = 1000
    p_off = rng.beta(2, 8, n)
    p_on = np.clip(p_off + rng.normal(0, 0.05, n), 0, 1)
    y = (rng.uniform(0, 1, n) < p_off).astype(int)
    df = pd.DataFrame({"p_offline": p_off, "p_online": p_on, "y_true": y})
    stacker = fit_stacker(df)
    for k in ("coef_offline", "coef_online", "intercept"):
        assert k in stacker
        assert np.isfinite(stacker[k])


def test_fit_stacker_rejects_single_class():
    df = pd.DataFrame({
        "p_offline": [0.1, 0.2, 0.3],
        "p_online":  [0.1, 0.2, 0.3],
        "y_true": [0, 0, 0],
    })
    with pytest.raises(ValueError):
        fit_stacker(df)


def test_combined_stacked_threshold_emits_open_array():
    df = pd.DataFrame({
        "p_offline": [0.1, 0.5, 0.9],
        "p_online":  [0.1, 0.5, 0.9],
    })
    # Coefs that produce p_stack ≈ p_offline at intercept=0 with online
    # coefficient zero.
    stacker = {"coef_offline": 100.0, "coef_online": 0.0, "intercept": -50.0}
    out = combined_stacked_tau(df, tau=0.5, stacker=stacker)
    # σ(100·p - 50): crosses 0.5 at p=0.5; below for p=0.1, just above for p=0.9.
    assert not out.open_signal[0]
    assert out.open_signal[2]


def test_combined_stacked_size_unit():
    df = pd.DataFrame({"p_offline": [0.5], "p_online": [0.5]})
    stacker = {"coef_offline": 1.0, "coef_online": 1.0, "intercept": 0.0}
    out = combined_stacked_tau(df, tau=0.5, stacker=stacker)
    assert out.size[0] == 1.0


# ---------------------------------------------------------------------------
# conformal_gate_tau
# ---------------------------------------------------------------------------

def test_conformal_gate_requires_in_set_column():
    df = pd.DataFrame({"p_offline": [0.1, 0.5]})
    with pytest.raises(KeyError):
        conformal_gate_tau(df, tau=0.3)


def test_conformal_gate_combines_threshold_and_set():
    df = pd.DataFrame({
        "p_offline": [0.20, 0.50, 0.50, 0.80],
        "in_set_10": [1,    0,    1,    0],
    })
    out = conformal_gate_tau(df, tau=0.30)
    # τ=0.30: pass[0]=False (p<τ); pass[1]=False (in_set=0);
    # pass[2]=True (p>τ AND in_set=1); pass[3]=False (in_set=0).
    np.testing.assert_array_equal(out.open_signal, [False, False, True, False])


def test_conformal_gate_alpha_05():
    df = pd.DataFrame({
        "p_offline": [0.5, 0.5],
        "in_set_05": [1, 0],
        "in_set_10": [1, 1],
    })
    out = conformal_gate_tau(df, tau=0.3, alpha_level="05")
    np.testing.assert_array_equal(out.open_signal, [True, False])


# ---------------------------------------------------------------------------
# mondrian_aci_size
# ---------------------------------------------------------------------------

def test_mondrian_aci_size_open_when_in_set():
    df = pd.DataFrame({
        "p_offline": [0.10, 0.30, 0.50, 0.90],
        "in_set_10": [0,    1,    1,    0],
        "q_lo_10":   [0.10, 0.50, 0.50, 0.10],
    })
    out = mondrian_aci_size(df, k=5.0)
    np.testing.assert_array_equal(out.open_signal, [False, True, True, False])


def test_mondrian_aci_size_zero_when_not_in_set():
    df = pd.DataFrame({
        "p_offline": [0.99, 0.20],
        "in_set_10": [0, 1],
        "q_lo_10":   [0.50, 0.50],
    })
    out = mondrian_aci_size(df, k=5.0)
    assert out.size[0] == 0.0
    assert out.open_signal[0] is np.False_ or not out.open_signal[0]


def test_mondrian_aci_size_clipped_to_1():
    """Very high k or excess produces size=1 (clipped)."""
    df = pd.DataFrame({
        "p_offline": [0.99],
        "in_set_10": [1],
        "q_lo_10":   [0.99],
    })
    out = mondrian_aci_size(df, k=100.0)
    assert out.size[0] == 1.0


def test_mondrian_aci_size_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "p_offline": rng.uniform(0, 1, n),
        "in_set_10": rng.integers(0, 2, n),
        "q_lo_10":   rng.uniform(0, 1, n),
    })
    out = mondrian_aci_size(df, k=3.0)
    assert (out.size >= 0).all() and (out.size <= 1).all()


# ---------------------------------------------------------------------------
# null_random_at_rate
# ---------------------------------------------------------------------------

def test_null_random_rate_in_unit_interval():
    df = pd.DataFrame({"p_offline": np.zeros(2_000)})
    out = null_random_at_rate(df, target_rate=0.10, seed=42)
    assert (out.size == 1.0).all()
    rate = out.open_signal.mean()
    # 2σ ≈ 0.013 at n=2000 / p=0.1; gives ~0.087..0.113.
    assert 0.07 < rate < 0.13


def test_null_random_rate_rejects_invalid():
    df = pd.DataFrame({"p_offline": [0.1]})
    with pytest.raises(ValueError):
        null_random_at_rate(df, target_rate=-0.1)
    with pytest.raises(ValueError):
        null_random_at_rate(df, target_rate=1.1)


def test_null_random_seed_deterministic():
    df = pd.DataFrame({"p_offline": np.zeros(100)})
    a = null_random_at_rate(df, target_rate=0.2, seed=7).open_signal
    b = null_random_at_rate(df, target_rate=0.2, seed=7).open_signal
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Registry smoke (all 7 strategies)
# ---------------------------------------------------------------------------

def test_registry_strategies_return_strategy_output_minimal():
    """Each strategy returns a StrategyOutput on a 3-row df with all required cols."""
    df = pd.DataFrame({
        "p_offline": [0.1, 0.5, 0.7],
        "p_online":  [0.2, 0.4, 0.6],
        "y_true":    [0, 1, 1],
        "in_set_10": [0, 1, 1],
        "q_lo_10":   [0.5, 0.5, 0.5],
    })
    stacker = {"coef_offline": 1.0, "coef_online": 1.0, "intercept": 0.0}
    expected_kwargs = {
        "baseline_offline_tau": {"tau": 0.3},
        "baseline_online_tau":  {"tau": 0.3},
        "combined_avg_tau":     {"tau": 0.3},
        "combined_stacked_tau": {"tau": 0.5, "stacker": stacker},
        "conformal_gate_tau":   {"tau": 0.3},
        "mondrian_aci_size":    {"k": 5.0},
        "null_random_at_rate":  {"target_rate": 0.5},
    }
    for name, fn in STRATEGIES.items():
        out = fn(df, **expected_kwargs[name])
        assert isinstance(out, StrategyOutput)
        assert out.open_signal.shape == (3,)
        assert out.size.shape == (3,)
