"""Tests for the H-106 calibration + threshold metric helpers in src/utils.py.

These pin the metric semantics this project relies on (CONSTITUTION IV
primary metric for offline-stage rounds is BSS / log-loss / ECE, not raw
ranking). We test invariants that must hold regardless of the underlying
distribution:

- ECE = 0 on perfectly calibrated predictions (within numerical noise).
- ECE = |mean_p - base_rate| on a constant predictor (one bin lights up).
- compute_all_metrics returns the documented keys, all finite.
- calibration_by_regime returns a dict keyed by regime label, with
  per-regime n_samples summing to N.
- threshold_analysis is monotone in trade-rate (higher threshold → fewer
  trades) and matches manual computation at a sample threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils import (
    DEFAULT_REGIME_LABELS,
    calibration_by_regime,
    compute_all_metrics,
    expected_calibration_error,
    threshold_analysis,
)


# -----------------------------------------------------------------------------
# expected_calibration_error
# -----------------------------------------------------------------------------

def test_ece_zero_on_perfect_calibration():
    """If p_pred ≈ p_true within each bin, ECE → 0.

    Build a dataset where bin midpoints exactly match observed accuracy.
    """
    rng = np.random.default_rng(0)
    n_per_bin = 1000
    n_bins = 10
    y_true_parts = []
    p_parts = []
    for i in range(n_bins):
        bin_mid = (i + 0.5) / n_bins
        # Accuracy in this bin equals bin_mid → ECE contribution = 0.
        y_bin = (rng.uniform(size=n_per_bin) < bin_mid).astype(int)
        p_bin = np.full(n_per_bin, bin_mid, dtype=float)
        y_true_parts.append(y_bin)
        p_parts.append(p_bin)
    y_true = np.concatenate(y_true_parts)
    p = np.concatenate(p_parts)
    ece = expected_calibration_error(y_true, p, n_bins=n_bins)
    # Sampling noise → small but bounded; with n_per_bin=1000 expect < 0.02.
    assert ece < 0.02, f"ece={ece}"


def test_ece_constant_predictor_equals_abs_gap():
    """A constant predictor lights up exactly one bin: ECE = |p - base_rate|.

    Not exactly though — bin_weight = 1.0 since the whole population is
    in one bin, and bin_accuracy = base_rate. So ECE = |p - base_rate|.
    """
    rng = np.random.default_rng(1)
    n = 10000
    base_rate = 0.13
    y = (rng.uniform(size=n) < base_rate).astype(int)
    constant_p = 0.30  # picks bin [0.3, 0.4) only
    p = np.full(n, constant_p, dtype=float)
    ece = expected_calibration_error(y, p, n_bins=10)
    expected = abs(constant_p - y.mean())
    assert ece == pytest.approx(expected, abs=1e-12)


def test_ece_empty_input_returns_zero():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_ece_p1_includes_in_last_bin():
    """p=1.0 must fall into the last bin, not be silently dropped.

    The function uses `<= hi` for the last bin and `< hi` for earlier ones;
    if that branch were missed, p=1.0 would be ignored and ECE would change.
    """
    y = np.array([1, 1, 0, 1])
    p = np.array([1.0, 1.0, 1.0, 1.0])
    # base_rate = 0.75; constant p=1.0 → ECE = 1.0 - 0.75 = 0.25.
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.25, abs=1e-12)


# -----------------------------------------------------------------------------
# compute_all_metrics
# -----------------------------------------------------------------------------

def test_compute_all_metrics_returns_documented_keys_and_finite():
    rng = np.random.default_rng(2)
    n = 5000
    y = (rng.uniform(size=n) < 0.10).astype(int)
    # A weakly informative predictor.
    p = np.clip(0.10 + 0.15 * (y - 0.5) + 0.05 * rng.normal(size=n), 0.001, 0.999)
    metrics = compute_all_metrics(y, p)
    assert set(metrics.keys()) == {"roc_auc", "pr_auc", "log_loss", "brier_score", "ece"}
    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} = {v} is not finite"
    # Sanity: roc_auc > 0.5 since p is positively correlated with y.
    assert metrics["roc_auc"] > 0.55


# -----------------------------------------------------------------------------
# calibration_by_regime
# -----------------------------------------------------------------------------

def test_calibration_by_regime_default_labels_keyed_low_med_high():
    rng = np.random.default_rng(3)
    n = 9000
    y = (rng.uniform(size=n) < 0.10).astype(int)
    p = np.clip(0.10 + 0.05 * rng.normal(size=n), 0.001, 0.999)
    vol = rng.exponential(scale=1.0, size=n)
    out = calibration_by_regime(y, p, vol, n_bins=10, n_regimes=3)
    assert set(out.keys()) == set(DEFAULT_REGIME_LABELS)
    for regime, stats in out.items():
        assert set(stats.keys()) == {"n_samples", "base_rate", "ece", "brier", "mean_predicted"}
        assert stats["n_samples"] >= 50
        assert 0.0 <= stats["ece"] <= 1.0
        assert 0.0 <= stats["base_rate"] <= 1.0


def test_calibration_by_regime_n_samples_sum_to_total():
    rng = np.random.default_rng(4)
    n = 6000
    y = (rng.uniform(size=n) < 0.20).astype(int)
    p = np.full(n, 0.5)
    vol = rng.normal(size=n)
    out = calibration_by_regime(y, p, vol, n_regimes=3)
    total = sum(s["n_samples"] for s in out.values())
    assert total == n


def test_calibration_by_regime_drops_undersized_buckets():
    """If a bucket has fewer than `min_samples_per_regime`, it must NOT appear."""
    n = 100
    y = np.zeros(n, dtype=int)
    p = np.zeros(n)
    # qcut on a wildly skewed signal → roughly equal buckets numerically,
    # but request 5 regimes on n=100 → buckets ~20 each, below the default 50.
    vol = np.arange(n)
    out = calibration_by_regime(
        y, p, vol, n_regimes=5, labels=("a", "b", "c", "d", "e"),
        min_samples_per_regime=50,
    )
    assert out == {}


def test_calibration_by_regime_rejects_label_count_mismatch():
    rng = np.random.default_rng(5)
    n = 1000
    y = np.zeros(n, dtype=int)
    p = np.zeros(n)
    vol = rng.normal(size=n)
    with pytest.raises(ValueError, match="labels length"):
        calibration_by_regime(y, p, vol, n_regimes=3, labels=("a", "b"))


# -----------------------------------------------------------------------------
# threshold_analysis
# -----------------------------------------------------------------------------

def test_threshold_analysis_n_trades_monotone_decreasing():
    rng = np.random.default_rng(6)
    n = 2000
    y = (rng.uniform(size=n) < 0.15).astype(int)
    p = rng.uniform(size=n)
    df = threshold_analysis(y, p)
    # Higher threshold → fewer trades.
    diffs = np.diff(df["n_trades"].to_numpy())
    assert (diffs <= 0).all(), f"n_trades must be monotone non-increasing"


def test_threshold_analysis_at_zero_takes_all():
    rng = np.random.default_rng(7)
    n = 500
    y = (rng.uniform(size=n) < 0.20).astype(int)
    p = rng.uniform(size=n)
    df = threshold_analysis(y, p, thresholds=np.array([0.0]))
    assert df["n_trades"].iloc[0] == n
    assert df["trade_rate"].iloc[0] == 1.0
    assert df["precision"].iloc[0] == pytest.approx(y.mean(), abs=1e-12)


def test_threshold_analysis_precision_recall_match_manual():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    p = np.array([0.9, 0.1, 0.7, 0.4, 0.8, 0.6, 0.3, 0.55])
    df = threshold_analysis(y, p, thresholds=np.array([0.55]))
    pred = p >= 0.55
    n_trades = int(pred.sum())
    tp = int(((y == 1) & pred).sum())
    expected_precision = tp / n_trades
    expected_recall = tp / int((y == 1).sum())
    assert df["precision"].iloc[0] == pytest.approx(expected_precision, abs=1e-12)
    assert df["recall"].iloc[0] == pytest.approx(expected_recall, abs=1e-12)


def test_threshold_analysis_precision_nan_when_no_trades():
    y = np.array([0, 1, 0])
    p = np.array([0.1, 0.2, 0.05])
    df = threshold_analysis(y, p, thresholds=np.array([0.99]))
    assert df["n_trades"].iloc[0] == 0
    assert np.isnan(df["precision"].iloc[0])
