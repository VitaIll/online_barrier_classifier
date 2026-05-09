"""Tests for src/conformal.py (split-conformal LAC + Mondrian).

The non-negotiable property is *marginal coverage*: when the calibrator is
fit on a held-out calibration set drawn from the same distribution as test,
the empirical coverage on test must be >= 1 - α with high probability under
sample size.

We use synthetic data with a known generating process so we can stress the
correctness without depending on the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import conformal


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def test_lac_score_matches_definition():
    p1 = np.array([0.7, 0.3, 0.9, 0.1])
    y = np.array([1, 0, 1, 0])
    expected = np.array([0.3, 0.3, 0.1, 0.1])  # 1-p for y=1, p for y=0
    np.testing.assert_allclose(conformal.lac_score(p1, y), expected)


def test_finite_sample_quantile_uses_plus_one_correction():
    scores = np.linspace(0, 1, 100)
    # alpha=0.1 -> q_level = ceil(101 * 0.9) / 100 = 91/100 = 0.91
    q = conformal.finite_sample_quantile(scores, 0.1)
    # Should be greater than naive 0.9 quantile.
    assert q > np.quantile(scores, 0.9, method="higher") - 1e-9
    assert 0.89 <= q <= 1.0


def test_finite_sample_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal.finite_sample_quantile(np.array([0.1, 0.2]), alpha=0.0)
    with pytest.raises(ValueError):
        conformal.finite_sample_quantile(np.array([0.1, 0.2]), alpha=1.0)


# ---------------------------------------------------------------------------
# Marginal coverage on synthetic data (the load-bearing property)
# ---------------------------------------------------------------------------

def _make_synthetic_calibrated(n: int, base_rate: float, noise: float, seed: int):
    """Generate (p_proba, y) where p_proba is a calibrated noisy estimator of P(y=1)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base_rate).astype(int)
    # p_proba ~ Beta-like around base rate, noisy near truth
    p_proba = np.clip(base_rate + (y - base_rate) * (1 - noise) + rng.normal(0, noise, n), 0.01, 0.99)
    return p_proba, y


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    base_rate=st.floats(min_value=0.05, max_value=0.5),
    alpha=st.floats(min_value=0.05, max_value=0.3),
    noise=st.floats(min_value=0.05, max_value=0.4),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_marginal_coverage_meets_target(base_rate, alpha, noise, seed):
    """Marginal coverage on test must be >= 1 - alpha (within sampling slack)."""
    n_cal, n_test = 2000, 2000
    p_cal, y_cal = _make_synthetic_calibrated(n_cal, base_rate, noise, seed)
    p_test, y_test = _make_synthetic_calibrated(n_test, base_rate, noise, seed + 1)

    cal = conformal.fit_conformal(p_cal, y_cal, alpha)
    sets = conformal.predict_set(cal, p_test)
    cov = conformal.empirical_coverage(sets, y_test)

    # Allow 2 sampling-sigma slack: std of binomial(n_test, 1-alpha) ~ sqrt((1-alpha)*alpha/n_test).
    sigma = np.sqrt(max(alpha * (1 - alpha) / n_test, 1e-9))
    assert cov >= (1 - alpha) - 3 * sigma, (
        f"Coverage {cov:.4f} below target {1-alpha:.4f} - 3σ ({3*sigma:.4f})"
    )


# ---------------------------------------------------------------------------
# Mondrian calibrator: per-regime coverage holds
# ---------------------------------------------------------------------------

def test_mondrian_per_regime_coverage():
    rng = np.random.default_rng(42)
    n_cal, n_test = 3000, 3000
    # Three regimes with different miscalibration.
    regime_cal = rng.integers(0, 3, size=n_cal)
    regime_test = rng.integers(0, 3, size=n_test)

    def gen(regime, n, seed_offset=0):
        loc_rng = np.random.default_rng(42 + regime + seed_offset)
        base = 0.1 + 0.2 * regime
        noise = 0.05 + 0.15 * regime
        y = (loc_rng.random(n) < base).astype(int)
        p = np.clip(base + (y - base) * (1 - noise) + loc_rng.normal(0, noise, n), 0.01, 0.99)
        return p, y

    p_cal = np.zeros(n_cal); y_cal = np.zeros(n_cal, dtype=int)
    for r in range(3):
        m = regime_cal == r
        p_cal[m], y_cal[m] = gen(r, m.sum(), 0)

    p_test = np.zeros(n_test); y_test = np.zeros(n_test, dtype=int)
    for r in range(3):
        m = regime_test == r
        p_test[m], y_test[m] = gen(r, m.sum(), 100)

    alpha = 0.1
    cal = conformal.fit_conformal(p_cal, y_cal, alpha, regime_cal=regime_cal)
    sets = conformal.predict_set(cal, p_test, regime_test=regime_test)
    cov = conformal.empirical_coverage(sets, y_test)
    assert cov >= 1 - alpha - 0.05, f"global cov {cov:.4f} below 1-alpha-slack"

    # Per-regime
    df = conformal.coverage_by_regime(sets, y_test, regime_test)
    for _, row in df.iterrows():
        # Per-regime coverage tolerance is wider (smaller n per stratum).
        assert row["coverage"] >= 1 - alpha - 0.08, (
            f"regime {row['regime']} cov {row['coverage']:.4f} "
            f"below target with n={row['n']}"
        )


# ---------------------------------------------------------------------------
# Set-size distribution sanity
# ---------------------------------------------------------------------------

def test_set_size_distribution_sums_to_one():
    rng = np.random.default_rng(0)
    p_cal = rng.uniform(0, 1, 1000)
    y_cal = rng.integers(0, 2, 1000)
    p_test = rng.uniform(0, 1, 1000)
    cal = conformal.fit_conformal(p_cal, y_cal, alpha=0.1)
    sets = conformal.predict_set(cal, p_test)
    dist = conformal.set_size_distribution(sets)
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9, f"set sizes do not sum to 1: {total}"


# ---------------------------------------------------------------------------
# Integration with UQ: conformal_trade_signal
# ---------------------------------------------------------------------------

def test_conformal_trade_signal_requires_both_gates():
    rng = np.random.default_rng(0)
    n = 100
    pred_sets = np.zeros((n, 2), dtype=bool)
    pred_sets[:, 1] = True  # singleton {1} for all
    p_mean = np.full(n, 0.6)
    sigma_eps = np.full(n, 0.1)

    # Tau_open = 0.55, k = 1.0 -> p - k*sigma = 0.5 < 0.55 -> NO trade.
    out = conformal.conformal_trade_signal(pred_sets, p_mean, sigma_eps,
                                            tau_open=0.55, k=1.0)
    assert not out.any()

    # Tau_open = 0.45 -> 0.5 > 0.45 -> trade.
    out = conformal.conformal_trade_signal(pred_sets, p_mean, sigma_eps,
                                            tau_open=0.45, k=1.0)
    assert out.all()

    # Sub-singleton sets -> no trade regardless of Cantelli.
    pred_sets[:, 0] = True  # full set {0,1}
    out = conformal.conformal_trade_signal(pred_sets, p_mean, sigma_eps,
                                            tau_open=0.45, k=1.0)
    assert not out.any()
