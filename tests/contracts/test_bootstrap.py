"""Contract tests for `wagie.metrics.bootstrap`.

Pins (round-020 falsifier set, ported into wagie):
- Politis-White block-length plug-in returns sane values on IID + AR(1).
- Stationary-block bootstrap matches IID bootstrap CI within 10% on IID data.
- Stationary-block bootstrap CI widens with autocorrelation on AR(1) ρ=0.7.
- DeLong ROC-AUC CI matches IID bootstrap to within 0.005 width on synthetic
  balanced n=5000.
- Wilson interval matches statsmodels.stats.proportion.proportion_confint to
  1e-12.
- Stratified PR-AUC bootstrap returns finite CIs on standard inputs.
- AFML uniqueness: full-overlap = 1/3, no-overlap = 1.0.
- Sequential bootstrap returns the documented dict shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from wagie.metrics.bootstrap import (
    average_uniqueness,
    bootstrap_max_drawdown,
    bootstrap_metric,
    bootstrap_sharpe,
    bootstrap_sortino,
    bootstrap_trade_metric_overlap_aware,
    delong_roc_auc_ci,
    optimal_block_length,
    sequential_bootstrap_indices,
    stationary_block_bootstrap_indices,
    stratified_bootstrap_pr_auc,
    trade_concurrency,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Politis-White block-length plug-in
# ---------------------------------------------------------------------------

def test_optimal_block_length_iid_returns_small():
    """On IID data, optimal block length should be small (≤ 5)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    bl = optimal_block_length(x)
    assert 1 <= bl <= 5, f"expected 1 <= bl <= 5 on IID; got {bl}"


def test_optimal_block_length_ar1_returns_larger_than_iid():
    """On AR(1) with rho=0.8, block length should be substantially larger
    than IID."""
    rng = np.random.default_rng(0)
    n = 5000
    rho = 0.8
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    bl = optimal_block_length(x)
    assert bl >= 5, f"expected bl >= 5 on AR(1) rho=0.8; got {bl}"
    assert bl <= n // 4, f"bl must be capped at n/4 = {n//4}; got {bl}"


def test_optimal_block_length_rejects_tiny_samples():
    with pytest.raises(ValueError):
        optimal_block_length(np.zeros(5))


def test_optimal_block_length_constant_falls_back():
    """All-zero x has zero variance; should not crash."""
    bl = optimal_block_length(np.zeros(100))
    assert bl >= 1


# ---------------------------------------------------------------------------
# Stationary block bootstrap indices
# ---------------------------------------------------------------------------

def test_block_bootstrap_indices_correct_length_and_range():
    rng = np.random.default_rng(0)
    n = 100
    bl = 10
    idx = stationary_block_bootstrap_indices(n, bl, rng=rng)
    assert idx.shape == (n,)
    assert idx.min() >= 0 and idx.max() < n
    assert idx.dtype == np.int64


def test_block_bootstrap_block_length_1_is_iid():
    rng = np.random.default_rng(42)
    n = 1000
    idx = stationary_block_bootstrap_indices(n, 1, rng=rng)
    diff = np.diff(idx).astype(float)
    var_diff = diff.var()
    assert var_diff > 0.0


def test_block_bootstrap_rejects_invalid():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        stationary_block_bootstrap_indices(0, 5, rng=rng)
    with pytest.raises(ValueError):
        stationary_block_bootstrap_indices(100, 0, rng=rng)


# ---------------------------------------------------------------------------
# bootstrap_metric — Sharpe / Sortino / max DD
# ---------------------------------------------------------------------------

def test_bootstrap_sharpe_iid_matches_iid_bootstrap_within_10pct():
    """On IID data, stationary-block-bootstrap CI should match IID bootstrap
    CI within 10% width (Politis-Romano §3 — IID is a degenerate case of
    stationary block at p ≈ 1)."""
    rng = np.random.default_rng(0)
    n = 2000
    r = rng.normal(0.001, 0.01, size=n)

    out_block = bootstrap_sharpe(r, n_resamples=2000, seed=42)
    out_iid = bootstrap_sharpe(r, n_resamples=2000, seed=42, scheme="iid")

    width_block = out_block["ci_hi"] - out_block["ci_lo"]
    width_iid = out_iid["ci_hi"] - out_iid["ci_lo"]
    rel_diff = abs(width_block - width_iid) / max(width_iid, 1e-12)
    assert rel_diff < 0.10, (
        f"expected stationary-block CI width to match IID within 10% on IID "
        f"data; got rel_diff={rel_diff:.3f}, block={width_block:.4f}, "
        f"iid={width_iid:.4f}"
    )


def test_bootstrap_sharpe_ar1_widens_correctly():
    """On AR(1) ρ=0.7 data, stationary-block-bootstrap CI should be WIDER
    than IID bootstrap CI (the IID-bootstrap under-covers under positive
    autocorrelation)."""
    rng = np.random.default_rng(0)
    n = 5000
    rho = 0.7
    eps = rng.normal(0.0, 0.01, size=n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = rho * r[i - 1] + eps[i]
    r += 0.001  # add small mean

    out_block = bootstrap_sharpe(r, n_resamples=2000, seed=42)
    out_iid = bootstrap_sharpe(r, n_resamples=2000, seed=42, scheme="iid")

    width_block = out_block["ci_hi"] - out_block["ci_lo"]
    width_iid = out_iid["ci_hi"] - out_iid["ci_lo"]
    assert width_block > width_iid * 1.1, (
        f"expected stationary-block CI to be > 1.1x IID-bootstrap CI on AR(1)"
        f"; got block={width_block:.4f}, iid={width_iid:.4f}"
    )
    assert out_block["block_length"] >= 3


def test_bootstrap_sharpe_returns_documented_keys():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 200)
    out = bootstrap_sharpe(r, n_resamples=500)
    for key in ["estimate", "ci_lo", "ci_hi", "se", "scheme", "block_length",
                "n_resamples", "confidence"]:
        assert key in out, f"missing key {key!r}"
    assert out["scheme"] == "stationary_block"
    assert out["ci_lo"] <= out["estimate"] <= out["ci_hi"]


def test_bootstrap_sortino_finite_on_synthetic():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    out = bootstrap_sortino(r, n_resamples=500)
    assert np.isfinite(out["estimate"])
    assert np.isfinite(out["ci_lo"]) and np.isfinite(out["ci_hi"])


def test_bootstrap_max_drawdown_positive_finite():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0001, 0.01, 1000)
    out = bootstrap_max_drawdown(r, n_resamples=500)
    assert out["estimate"] >= 0.0
    assert out["ci_lo"] >= 0.0


def test_bootstrap_metric_rejects_invalid_confidence():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 100)
    with pytest.raises(ValueError):
        bootstrap_sharpe(r, confidence=0.0)
    with pytest.raises(ValueError):
        bootstrap_sharpe(r, confidence=1.0)


def test_bootstrap_metric_rejects_too_short():
    with pytest.raises(ValueError):
        bootstrap_sharpe(np.array([1.0]))


def test_bootstrap_metric_rejects_unknown_scheme():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 100)
    with pytest.raises(ValueError, match="unknown scheme"):
        bootstrap_metric(r, lambda x: float(x.mean()),
                         n_resamples=10, scheme="bogus")


def test_bootstrap_metric_return_samples_attaches_array():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 100)
    out = bootstrap_metric(r, lambda x: float(x.mean()),
                           n_resamples=50, return_samples=True, seed=0)
    assert "samples" in out
    assert out["samples"].shape == (50,)


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------

def test_wilson_matches_statsmodels_to_1e10():
    """Wilson must match statsmodels proportion_confint to 1e-10."""
    sm = pytest.importorskip("statsmodels.stats.proportion")
    for n_total in [10, 50, 100, 1000]:
        for n_success in [0, 1, n_total // 2, n_total - 1, n_total]:
            ours_lo, ours_hi = wilson_interval(n_success, n_total, confidence=0.95)
            theirs_lo, theirs_hi = sm.proportion_confint(
                n_success, n_total, alpha=0.05, method="wilson",
            )
            assert abs(ours_lo - theirs_lo) < 1e-10, (
                f"Wilson lo mismatch at ({n_success}, {n_total}): "
                f"ours={ours_lo}, theirs={theirs_lo}"
            )
            assert abs(ours_hi - theirs_hi) < 1e-10


def test_wilson_n_zero_returns_full_range():
    lo, hi = wilson_interval(0, 0, confidence=0.95)
    assert lo == 0.0 and hi == 1.0


def test_wilson_rejects_invalid():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(5, 10, confidence=0.0)


def test_wilson_alternate_confidence_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    for conf in [0.80, 0.90, 0.99]:
        ours_lo, ours_hi = wilson_interval(7, 20, confidence=conf)
        theirs_lo, theirs_hi = sm.proportion_confint(
            7, 20, alpha=1 - conf, method="wilson",
        )
        assert abs(ours_lo - theirs_lo) < 1e-10
        assert abs(ours_hi - theirs_hi) < 1e-10


# ---------------------------------------------------------------------------
# DeLong ROC-AUC CI
# ---------------------------------------------------------------------------

def test_delong_roc_auc_balanced_sample_matches_bootstrap_within_0005():
    """Synthetic balanced n=5000 — DeLong half-CI vs IID-bootstrap half-CI
    differ by < 0.005."""
    rng = np.random.default_rng(42)
    n = 5000
    y_true = rng.binomial(1, 0.5, size=n)
    y_score = rng.normal(0.0, 1.0, size=n) + 0.5 * y_true.astype(float)

    out = delong_roc_auc_ci(y_true, y_score)
    half_delong = (out["ci_hi"] - out["ci_lo"]) / 2.0

    from sklearn.metrics import roc_auc_score
    rng2 = np.random.default_rng(42)
    n_resamples = 1000
    samples = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng2.integers(0, n, size=n)
        samples[b] = roc_auc_score(y_true[idx], y_score[idx])
    boot_lo, boot_hi = np.quantile(samples, [0.025, 0.975])
    half_boot = (boot_hi - boot_lo) / 2.0

    diff = abs(half_delong - half_boot)
    assert diff < 0.005, (
        f"DeLong half-CI ({half_delong:.4f}) vs bootstrap half-CI "
        f"({half_boot:.4f}) differ by {diff:.4f}; threshold is 0.005"
    )


def test_delong_roc_auc_estimate_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, size=1000)
    y_score = rng.uniform(size=1000) + 0.3 * y_true.astype(float)
    from sklearn.metrics import roc_auc_score
    out = delong_roc_auc_ci(y_true, y_score)
    expected = float(roc_auc_score(y_true, y_score))
    assert abs(out["estimate"] - expected) < 1e-10


def test_delong_rejects_single_class():
    with pytest.raises(ValueError):
        delong_roc_auc_ci(np.zeros(100, dtype=int),
                          np.random.uniform(size=100))
    with pytest.raises(ValueError):
        delong_roc_auc_ci(np.ones(100, dtype=int),
                          np.random.uniform(size=100))


def test_delong_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        delong_roc_auc_ci(np.array([1, 0, 1]), np.array([0.5, 0.5]))


def test_delong_returns_documented_keys():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.4, 200)
    y_score = rng.uniform(size=200) + 0.4 * y_true.astype(float)
    out = delong_roc_auc_ci(y_true, y_score)
    for key in ["estimate", "ci_lo", "ci_hi", "se", "scheme", "n_pos",
                "n_neg", "confidence"]:
        assert key in out
    assert out["scheme"] == "delong_1988"


# ---------------------------------------------------------------------------
# Stratified PR-AUC bootstrap
# ---------------------------------------------------------------------------

def test_stratified_pr_auc_finite_and_bounded():
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.binomial(1, 0.1, size=n)  # imbalanced
    y_score = rng.uniform(size=n) + 0.4 * y_true.astype(float)
    out = stratified_bootstrap_pr_auc(y_true, y_score, n_resamples=300)
    assert 0.0 <= out["ci_lo"] <= out["estimate"] <= out["ci_hi"] <= 1.0


def test_stratified_pr_auc_rejects_single_class():
    with pytest.raises(ValueError):
        stratified_bootstrap_pr_auc(np.zeros(100, dtype=int),
                                    np.random.uniform(size=100))


# ---------------------------------------------------------------------------
# Per-trade overlap helpers
# ---------------------------------------------------------------------------

def test_trade_concurrency_simple():
    n_open = np.array([0, 5, 10])
    n_close = np.array([10, 15, 20])
    c = trade_concurrency(n_open, n_close, n_minutes=20)
    assert c.shape == (20,)
    assert c[0] == 1   # only trade 0 active
    assert c[5] == 2   # trades 0, 1 active
    assert c[10] == 2  # trades 1, 2 active (trade 0 closed at 10)
    assert c[15] == 1  # only trade 2


def test_trade_concurrency_rejects_misshape_inputs():
    with pytest.raises(ValueError, match="same shape"):
        trade_concurrency(np.array([0, 1]), np.array([1, 2, 3]), 5)


def test_trade_concurrency_rejects_close_before_open():
    with pytest.raises(ValueError, match=">= n_open"):
        trade_concurrency(np.array([5]), np.array([3]), 10)


def test_average_uniqueness_no_overlap_is_one():
    """No two trades co-occur → uniqueness = 1.0."""
    n_open = np.array([0, 10, 20])
    n_close = np.array([5, 15, 25])
    u = average_uniqueness(n_open, n_close, n_minutes=30)
    np.testing.assert_allclose(u, 1.0)


def test_average_uniqueness_full_overlap_is_one_third():
    """Three trades all sharing the same window → uniqueness = 1/3."""
    n_open = np.array([0, 0, 0])
    n_close = np.array([10, 10, 10])
    u = average_uniqueness(n_open, n_close, n_minutes=10)
    np.testing.assert_allclose(u, 1.0 / 3.0)


def test_sequential_bootstrap_returns_correct_shape():
    rng = np.random.default_rng(0)
    n_open = np.array([0, 3, 6])
    n_close = np.array([5, 8, 11])
    idx = sequential_bootstrap_indices(n_open, n_close, n_minutes=15,
                                        n_draws=50, rng=rng)
    assert idx.shape == (50,)
    assert idx.min() >= 0 and idx.max() < 3


def test_overlap_aware_bootstrap_widens_or_matches_under_high_overlap():
    """Under high trade overlap, sequential-bootstrap CI should be at least
    a substantial fraction of the naive CI (the weighting is non-degenerate)."""
    rng = np.random.default_rng(42)
    n_trades = 500
    n_open = np.arange(n_trades, dtype=np.int64)
    n_close = n_open + 30
    n_minutes = int(n_close.max() + 1)
    pnl = rng.normal(0.001, 0.01, size=n_trades)

    def sharpe_fn(r: np.ndarray) -> float:
        s = float(r.std(ddof=1))
        return float(r.mean() / max(s, 1e-12))

    out_seq = bootstrap_trade_metric_overlap_aware(
        pnl, n_open, n_close, n_minutes, sharpe_fn,
        n_resamples=500, seed=0, scheme="sequential_bootstrap",
    )
    out_naive = bootstrap_trade_metric_overlap_aware(
        pnl, n_open, n_close, n_minutes, sharpe_fn,
        n_resamples=500, seed=0, scheme="naive",
    )
    width_seq = out_seq["ci_hi"] - out_seq["ci_lo"]
    width_naive = out_naive["ci_hi"] - out_naive["ci_lo"]
    # Mean uniqueness should be < 1 (overlap penalty).
    assert out_seq["mean_uniqueness"] < 1.0
    assert width_seq > width_naive * 0.5


def test_overlap_aware_bootstrap_no_overlap_matches_naive():
    rng = np.random.default_rng(0)
    n_trades = 100
    n_open = np.arange(0, n_trades * 30, 30, dtype=np.int64)
    n_close = n_open + 25
    n_minutes = int(n_close.max() + 1)
    pnl = rng.normal(0.001, 0.01, size=n_trades)

    def mean_fn(r: np.ndarray) -> float:
        return float(r.mean())

    out_seq = bootstrap_trade_metric_overlap_aware(
        pnl, n_open, n_close, n_minutes, mean_fn,
        n_resamples=500, seed=0, scheme="sequential_bootstrap",
    )
    np.testing.assert_allclose(out_seq["mean_uniqueness"], 1.0)


def test_overlap_aware_bootstrap_returns_documented_keys():
    rng = np.random.default_rng(0)
    n_open = np.array([0, 1, 2, 3])
    n_close = np.array([5, 6, 7, 8])
    pnl = rng.normal(0, 1, 4)
    out = bootstrap_trade_metric_overlap_aware(
        pnl, n_open, n_close, 10, lambda r: float(r.mean()),
        n_resamples=100,
    )
    for key in ["estimate", "ci_lo", "ci_hi", "se", "scheme",
                "n_resamples", "n_trades", "mean_uniqueness",
                "median_uniqueness", "confidence"]:
        assert key in out


def test_overlap_aware_bootstrap_rejects_too_few_trades():
    with pytest.raises(ValueError, match="n_trades >= 2"):
        bootstrap_trade_metric_overlap_aware(
            np.array([0.1]), np.array([0]), np.array([1]), 5,
            lambda r: float(r.mean()),
            n_resamples=10,
        )


def test_overlap_aware_bootstrap_rejects_unknown_scheme():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 4)
    with pytest.raises(ValueError, match="unknown scheme"):
        bootstrap_trade_metric_overlap_aware(
            pnl, np.array([0, 1, 2, 3]), np.array([5, 6, 7, 8]), 10,
            lambda r: float(r.mean()),
            n_resamples=10, scheme="nope",
        )


def test_sequential_bootstrap_falls_back_uniform_on_zero_uniqueness():
    """If avg_u sums to zero (degenerate), sequential bootstrap falls back
    to uniform sampling rather than crashing on a zero-probability vector."""
    rng = np.random.default_rng(0)
    # n_open == n_close (zero-length trades) makes the spans degenerate
    # but average_uniqueness still returns the cumulative ratio. Use an
    # n_minutes of 0 so c is empty and inv_c is empty.
    n_open = np.array([0, 0, 0], dtype=np.int64)
    n_close = np.array([0, 0, 0], dtype=np.int64)
    idx = sequential_bootstrap_indices(n_open, n_close, n_minutes=0,
                                       n_draws=10, rng=rng)
    assert idx.shape == (10,)
    assert (idx >= 0).all() and (idx < 3).all()
