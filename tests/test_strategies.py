"""Tests for `src.strategies` — five legitimate strategies under the
two-layer architecture.

Round-015 deletes `combined_avg_tau` / `combined_stacked_tau` / `fit_stacker`
because averaging or stacking `(p_offline, p_online)` is architecturally
backwards: `p_online` is downstream of `p_offline`, not a sibling. These
tests verify the corrected registry.

Pin:
- StrategyOutput rejects shape mismatch and out-of-range size.
- baseline_offline_tau / baseline_online_tau respect the τ threshold.
- conformal_gate_tau combines `p_online > τ` with `in_set_α == 1`.
- mondrian_aci_size sizes by `clip(k · max(0, p_online − (1 − q_lo_α)), 0, 1)`.
- null_random_at_rate is deterministic under fixed seed.
- STRATEGIES registry contains exactly the five legitimate strategies.
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
    conformal_gate_tau,
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


def test_conformal_gate_requires_p_online_and_in_set():
    df = pd.DataFrame({"p_offline": [0.1, 0.5]})
    with pytest.raises(KeyError):
        conformal_gate_tau(df, tau=0.3)
    df = pd.DataFrame({"p_online": [0.1, 0.5]})
    with pytest.raises(KeyError):
        conformal_gate_tau(df, tau=0.3)


def test_mondrian_aci_size_requires_p_online_and_q_columns():
    df = pd.DataFrame({"p_offline": [0.1, 0.5]})
    with pytest.raises(KeyError):
        mondrian_aci_size(df)


# ---------------------------------------------------------------------------
# baseline_offline_tau (sanity floor)
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
# baseline_online_tau (system output)
# ---------------------------------------------------------------------------

def test_baseline_online_threshold_strict():
    df = pd.DataFrame({"p_online": [0.10, 0.30, 0.50, 0.70]})
    out = baseline_online_tau(df, tau=0.30)
    np.testing.assert_array_equal(out.open_signal, [False, False, True, True])


# ---------------------------------------------------------------------------
# conformal_gate_tau — gates on p_online (NOT p_offline)
# ---------------------------------------------------------------------------

def test_conformal_gate_combines_threshold_and_set_on_p_online():
    df = pd.DataFrame({
        "p_online":  [0.20, 0.50, 0.50, 0.80],
        "in_set_10": [1,    0,    1,    0],
    })
    out = conformal_gate_tau(df, tau=0.30)
    # τ=0.30: pass[0]=False (p_on<τ); pass[1]=False (in_set=0);
    # pass[2]=True (p_on>τ AND in_set=1); pass[3]=False (in_set=0).
    np.testing.assert_array_equal(out.open_signal, [False, False, True, False])


def test_conformal_gate_alpha_05():
    df = pd.DataFrame({
        "p_online": [0.5, 0.5],
        "in_set_05": [1, 0],
        "in_set_10": [1, 1],
    })
    out = conformal_gate_tau(df, tau=0.3, alpha_level="05")
    np.testing.assert_array_equal(out.open_signal, [True, False])


def test_conformal_gate_does_not_consult_p_offline():
    """Even with p_offline absent, the strategy runs on p_online + in_set."""
    df = pd.DataFrame({
        "p_online": [0.50],
        "in_set_10": [1],
    })
    out = conformal_gate_tau(df, tau=0.30)
    assert out.open_signal[0]


# ---------------------------------------------------------------------------
# mondrian_aci_size — sizes on p_online's confidence margin
# ---------------------------------------------------------------------------

def test_mondrian_aci_size_open_when_in_set():
    df = pd.DataFrame({
        "p_online":  [0.10, 0.30, 0.50, 0.90],
        "in_set_10": [0,    1,    1,    0],
        "q_lo_10":   [0.10, 0.50, 0.50, 0.10],
    })
    out = mondrian_aci_size(df, k=5.0)
    np.testing.assert_array_equal(out.open_signal, [False, True, True, False])


def test_mondrian_aci_size_zero_when_not_in_set():
    df = pd.DataFrame({
        "p_online":  [0.99, 0.20],
        "in_set_10": [0, 1],
        "q_lo_10":   [0.50, 0.50],
    })
    out = mondrian_aci_size(df, k=5.0)
    assert out.size[0] == 0.0
    assert not out.open_signal[0]


def test_mondrian_aci_size_clipped_to_1():
    df = pd.DataFrame({
        "p_online": [0.99],
        "in_set_10": [1],
        "q_lo_10":   [0.99],
    })
    out = mondrian_aci_size(df, k=100.0)
    assert out.size[0] == 1.0


def test_mondrian_aci_size_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "p_online": rng.uniform(0, 1, n),
        "in_set_10": rng.integers(0, 2, n),
        "q_lo_10":   rng.uniform(0, 1, n),
    })
    out = mondrian_aci_size(df, k=3.0)
    assert (out.size >= 0).all() and (out.size <= 1).all()


# ---------------------------------------------------------------------------
# null_random_at_rate — no-skill comparator
# ---------------------------------------------------------------------------

def test_null_random_rate_in_unit_interval():
    df = pd.DataFrame({"p_online": np.zeros(2_000)})
    out = null_random_at_rate(df, target_rate=0.10, seed=42)
    assert (out.size == 1.0).all()
    rate = out.open_signal.mean()
    assert 0.07 < rate < 0.13


def test_null_random_rate_rejects_invalid():
    df = pd.DataFrame({"p_online": [0.1]})
    with pytest.raises(ValueError):
        null_random_at_rate(df, target_rate=-0.1)
    with pytest.raises(ValueError):
        null_random_at_rate(df, target_rate=1.1)


def test_null_random_seed_deterministic():
    df = pd.DataFrame({"p_online": np.zeros(100)})
    a = null_random_at_rate(df, target_rate=0.2, seed=7).open_signal
    b = null_random_at_rate(df, target_rate=0.2, seed=7).open_signal
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Registry — exactly five legitimate strategies (no sibling combiners)
# ---------------------------------------------------------------------------

def test_registry_contains_exactly_five_strategies():
    expected = {
        "baseline_offline_tau", "baseline_online_tau", "conformal_gate_tau",
        "mondrian_aci_size", "null_random_at_rate",
    }
    assert set(STRATEGIES.keys()) == expected


def test_registry_rejects_sibling_combiners():
    """Round-015 contract: combiners that average or stack p_offline and
    p_online are architecturally invalid and MUST NOT appear in the registry."""
    forbidden = {"combined_avg_tau", "combined_stacked_tau", "fit_stacker"}
    assert not (set(STRATEGIES.keys()) & forbidden)


def test_registry_strategies_return_strategy_output_minimal():
    df = pd.DataFrame({
        "p_offline": [0.1, 0.5, 0.7],
        "p_online":  [0.2, 0.4, 0.6],
        "y_true":    [0, 1, 1],
        "in_set_10": [0, 1, 1],
        "q_lo_10":   [0.5, 0.5, 0.5],
    })
    expected_kwargs = {
        "baseline_offline_tau": {"tau": 0.3},
        "baseline_online_tau":  {"tau": 0.3},
        "conformal_gate_tau":   {"tau": 0.3},
        "mondrian_aci_size":    {"k": 5.0},
        "null_random_at_rate":  {"target_rate": 0.5},
    }
    for name, fn in STRATEGIES.items():
        out = fn(df, **expected_kwargs[name])
        assert isinstance(out, StrategyOutput)
        assert out.open_signal.shape == (3,)
        assert out.size.shape == (3,)
