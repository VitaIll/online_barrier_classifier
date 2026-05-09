"""Property-based tests using hypothesis (H-004).

These tests randomise inputs to falsify invariants the loop relies on. They
complement the example-based tests in test_causality.py / test_weights.py /
test_splits.py: examples test specific cases; properties test universal claims.

Slow tests are kept under @settings(max_examples=...) caps so the test suite
stays under the round budget.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import backtest, utils


# ----------------------------------------------------------------------------
# Property: barrier-distance weight is monotone non-decreasing in d_k for negatives
# ----------------------------------------------------------------------------

@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=20, max_value=600),
    phi=st.floats(min_value=1e-4, max_value=0.05),
    spread=st.floats(min_value=1e-4, max_value=0.1),
    w_max=st.floats(min_value=1.5, max_value=20.0),
    q_tail=st.floats(min_value=0.005, max_value=0.2),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_barrier_distance_weight_monotone_in_d_k(n, phi, spread, w_max, q_tail, seed):
    """For negatives (m_k < phi), w_dist must be non-decreasing in d_k = phi - m_k.

    This is the load-bearing property of risk-aware weighting (deep losses
    upweighted more than near-misses). Anchor of CONSTITUTION I.4.
    """
    rng = np.random.default_rng(seed)
    # Symmetric m_k around phi to ensure both classes present.
    m_k = phi + rng.uniform(-spread, spread, size=n)
    w_dist, info = utils.compute_barrier_distance_weight(
        m_k, phi=phi, w_max=w_max, q_tail=q_tail, enabled=True
    )
    neg = m_k < phi
    if neg.sum() < 2:
        return  # nothing to check
    # Sort negatives by ascending d_k = phi - m_k (= descending m_k).
    order = np.argsort(-(m_k[neg]))
    sorted_w = w_dist[neg][order]
    # Allow tiny numerical slack for the soft-cap exp.
    diffs = np.diff(sorted_w)
    assert (diffs >= -1e-9).all(), (
        f"non-monotone weights at idx={int(np.argmin(diffs))}, "
        f"min_diff={diffs.min():.3e}"
    )


# ----------------------------------------------------------------------------
# Property: barrier-distance weight respects the soft cap
# ----------------------------------------------------------------------------

@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(min_value=10, max_value=400),
    phi=st.floats(min_value=1e-4, max_value=0.02),
    spread=st.floats(min_value=1e-4, max_value=0.05),
    w_max=st.floats(min_value=1.5, max_value=10.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_weight_cap_holds(n, phi, spread, w_max, seed):
    rng = np.random.default_rng(seed)
    m_k = phi + rng.uniform(-spread, spread, size=n)
    w_dist, _ = utils.compute_barrier_distance_weight(
        m_k, phi=phi, w_max=w_max, q_tail=0.01, enabled=True
    )
    assert (w_dist >= 1.0 - 1e-9).all(), "weight floor < 1 violated"
    assert (w_dist <= w_max + 1e-9).all(), f"soft cap {w_max} violated, max={w_dist.max():.3f}"


# ----------------------------------------------------------------------------
# Property: time-discount weights in (0, 1] and monotone non-decreasing in chronological order
# ----------------------------------------------------------------------------

@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(min_value=10, max_value=2000),
    r=st.floats(min_value=0.0, max_value=1.0),
    delta=st.floats(min_value=0.5, max_value=0.99999),
)
def test_property_time_discount_in_unit_interval(n, r, delta):
    w, info = utils.compute_time_discount_weight(N=n, r=r, delta=delta, enabled=True)
    assert (w > 0).all(), "time-discount weights must be strictly positive"
    assert (w <= 1.0 + 1e-12).all(), "time-discount weights must be <= 1"
    # Monotone non-decreasing along the rank axis (oldest=0, newest=n-1 by default).
    diffs = np.diff(w)
    assert (diffs >= -1e-12).all(), "time-discount must be non-decreasing in chronological order"


# ----------------------------------------------------------------------------
# Property: chronological_split_with_embargo respects ordering + embargo
# ----------------------------------------------------------------------------

@settings(max_examples=80, deadline=None)
@given(
    n=st.integers(min_value=2_000, max_value=50_000),
    train_frac=st.floats(min_value=0.4, max_value=0.85),
    val_frac=st.floats(min_value=0.05, max_value=0.3),
    embargo_k=st.integers(min_value=0, max_value=200),
)
def test_property_split_chronological_and_embargoed(n, train_frac, val_frac, embargo_k):
    # Skip parameter combos that leave too little for val/test.
    leftover = 1.0 - train_frac - val_frac
    if leftover * n - embargo_k < 50 or val_frac * n - embargo_k < 50:
        return
    df = pd.DataFrame({"k": np.arange(n)})
    train, val, test = utils.chronological_split_with_embargo(
        df, train_frac=train_frac, val_frac=val_frac, embargo_k=embargo_k
    )
    assert train["k"].max() < val["k"].min(), "train must precede val"
    assert val["k"].max() < test["k"].min(), "val must precede test"
    assert val["k"].min() - train["k"].max() >= embargo_k, "embargo train->val violated"
    assert test["k"].min() - val["k"].max() >= embargo_k, "embargo val->test violated"


# ----------------------------------------------------------------------------
# Property: simulate_inventory_aware respects round-trip cost identity
# ----------------------------------------------------------------------------

def _flat_path(n_minutes: int, base: float = 100.0):
    a = np.full(n_minutes, base)
    return a.copy(), a.copy(), a.copy()


@settings(max_examples=30, deadline=None)
@given(
    n_bound=st.integers(min_value=20, max_value=80),
    cost_bps=st.floats(min_value=0.0, max_value=20.0),
    M=st.integers(min_value=5, max_value=15),
)
def test_property_round_trip_cost_identity_on_flat_path(n_bound, cost_bps, M):
    """On a perfectly flat price path, the only PnL is `-2 * cost_bps * 1e-4` per trade."""
    boundaries = pd.DataFrame({
        "k": np.arange(n_bound, dtype=int),
        "ts": pd.date_range("2024-01-01", periods=n_bound, freq=f"{M}min", tz="UTC"),
    })
    close, high, low = _flat_path(n_bound * M + M)
    p = np.zeros(n_bound)
    if n_bound >= 3:
        p[1] = 0.99  # exactly one trade
    res = backtest.simulate_inventory_aware(
        boundaries, close, high, low, p, tau_open=0.5, M=M, cost_bps=cost_bps
    )
    assert res.metrics["n_trades"] == (1 if n_bound >= 3 else 0)
    if res.metrics["n_trades"] == 1:
        t = res.trades.iloc[0]
        assert math.isclose(t["pnl_log_gross"], 0.0, abs_tol=1e-9)
        assert math.isclose(t["pnl_log_net"], -2.0 * cost_bps * 1e-4, abs_tol=1e-9)


# ----------------------------------------------------------------------------
# Property: get_imputation_value is total over the documented patterns
# ----------------------------------------------------------------------------

DOC_PATTERNS = [
    "ret__lag1__f__w0",
    "ret__rsi__f__w14",
    "vol__semivar_ratio__f__w60",
    "vol__bpv_ratio__f__w120",
    "vol__ratio__f__ws10__wl60",
    "vol__rs__f__w240",
    "logp__pos__f__w240",
    "logp__dd__f__w480",
    "logp__z__f__w240",
    "tb_ratio__inst__f__w0",
    "pentropy_norm__inst__f__w60__m3__tau1",
    "hit__prev__h__w0",
    "hit__rate__h__w24",
    "hit__since__h__w0",
    "logvol__mean__f__w60",
    "block__close_to_high__h__w0",
    "barrier__z_tight__f__w240",
    "barrier__emax_ratio__f__w240",
    "ret__posfrac__f__w120",
    "ret__q90__f__w240",
]


@pytest.mark.parametrize("name", DOC_PATTERNS)
def test_property_get_imputation_value_finite_and_documented(name):
    v = utils.get_imputation_value(name, p_hit_prior=0.5, cap_h_blocks=144)
    assert v is not None
    assert isinstance(v, (int, float))
    assert math.isfinite(float(v)), f"imputation for {name} is not finite"


# ----------------------------------------------------------------------------
# Property: deflated Sharpe is in [0, 1] for any sane input
# ----------------------------------------------------------------------------

@settings(max_examples=80, deadline=None)
@given(
    sr=st.floats(min_value=-5.0, max_value=5.0),
    var=st.floats(min_value=1e-6, max_value=10.0),
    n_trials=st.integers(min_value=2, max_value=10_000),
    n_obs=st.integers(min_value=5, max_value=100_000),
    skew=st.floats(min_value=-5.0, max_value=5.0),
    kurt=st.floats(min_value=1.5, max_value=20.0),
)
def test_property_deflated_sharpe_in_unit_interval(sr, var, n_trials, n_obs, skew, kurt):
    dsr, sr_star = backtest.deflated_sharpe(sr, var, n_trials, n_obs, skew, kurt)
    assert 0.0 <= dsr <= 1.0, f"DSR out of [0, 1]: {dsr}"
    assert math.isfinite(sr_star)
