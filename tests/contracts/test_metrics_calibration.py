"""Contract tests for `wagie.metrics.calibration`.

Covers:
  - Empty-input branches for brier_score / ECE / reliability_bins.
  - Streaming Brier ≡ batch Brier on identical inputs.
  - Perfectly-calibrated synthetic dataset → ECE close to 0.
  - Pathologically miscalibrated → ECE near 1.
  - Reliability bins are size-weighted and skip empty bins.
  - n_bins parametrize.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wagie.metrics import (
    Brier,
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


# ---------- empty input -----------------------------------------------------


def test_brier_empty() -> None:
    assert brier_score([], []) == 0.0


def test_ece_empty() -> None:
    assert expected_calibration_error([], [], n_bins=10) == 0.0


def test_reliability_bins_empty() -> None:
    assert reliability_bins([], []) == []


def test_streaming_brier_initial_get_is_zero() -> None:
    b = Brier()
    assert b.get() == 0.0


# ---------- streaming ≡ batch ----------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_streaming_brier_equals_batch(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 200
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)

    streaming = Brier()
    for yi, pi in zip(y, p):
        streaming.update(int(yi), float(pi))
    batch = brier_score(y, p)
    assert streaming.get() == pytest.approx(batch, abs=1e-12)


def test_streaming_brier_chains_returns_self() -> None:
    b = Brier()
    out = b.update(1, 0.7).update(0, 0.2)
    assert out is b


# ---------- known-value Brier ----------------------------------------------


def test_brier_constant_half_known_value() -> None:
    """Predict 0.5 always → Brier = 0.25 regardless of label distribution."""
    y = [0, 1, 0, 1, 1]
    p = [0.5] * 5
    assert brier_score(y, p) == pytest.approx(0.25)


@pytest.mark.parametrize("p,expected", [(0.0, 0.0), (1.0, 1.0), (0.5, 0.25)])
def test_brier_all_zero_labels(p: float, expected: float) -> None:
    y = [0] * 10
    pred = [p] * 10
    assert brier_score(y, pred) == pytest.approx(expected)


# ---------- ECE: synthetic well-calibrated case ----------------------------


def test_ece_perfect_calibration_close_to_zero() -> None:
    """y_i sampled Bernoulli(p_i): with N large, ECE → 0 (Monte-Carlo small)."""
    rng = np.random.default_rng(123)
    n = 20_000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.02


def test_ece_pathological_is_near_one() -> None:
    """y always 0 but predict 1 always → ECE = 1.0 exactly."""
    y = [0] * 100
    p = [1.0] * 100
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(1.0)


def test_ece_perfect_predictions_zero() -> None:
    """y == p ∈ {0,1} → all bins have zero |gap| → ECE = 0."""
    y = [0, 0, 1, 1, 0, 1, 0, 1] * 5
    p = [float(yi) for yi in y]
    assert expected_calibration_error(y, p, n_bins=5) == pytest.approx(0.0)


@pytest.mark.parametrize("n_bins", [2, 5, 10, 20])
def test_ece_bin_count_does_not_exceed_one(n_bins: int) -> None:
    rng = np.random.default_rng(0)
    n = 500
    p = rng.uniform(0.0, 1.0, n)
    y = (p > 0.5).astype(int)
    ece = expected_calibration_error(y, p, n_bins=n_bins)
    assert 0.0 <= ece <= 1.0


# ---------- ECE: hand-constructed weighted bins ----------------------------


def test_ece_weighted_by_bin_size() -> None:
    """Bin A: 80% mass, gap 0.0; Bin B: 20% mass, gap 0.5 → ECE = 0.1."""
    # Construct two distinct prob clusters; quantiles will split them cleanly
    # for n_bins=2.
    p = np.concatenate([np.full(80, 0.10), np.full(20, 0.80)])
    # Bin A perfectly calibrated, bin B off by 0.5 (predict 0.8, observe 0.3).
    y = np.concatenate([
        # 80 samples with 10% positive rate (~ p=0.10)
        np.array([1] * 8 + [0] * 72),
        # 20 samples with 30% positive rate (predict 0.80, gap = 0.5)
        np.array([1] * 6 + [0] * 14),
    ])
    ece = expected_calibration_error(y, p, n_bins=2)
    # Bin A: |0.10 - 0.10| * 0.80 = 0.00
    # Bin B: |0.80 - 0.30| * 0.20 = 0.10
    assert ece == pytest.approx(0.10, abs=1e-9)


# ---------- reliability_bins -----------------------------------------------


def test_reliability_bins_skips_empty_and_returns_dataclass() -> None:
    """Constant predictions → only one populated bin."""
    p = [0.3] * 10
    y = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    bins = reliability_bins(y, p, n_bins=10)
    assert len(bins) == 1
    b = bins[0]
    assert isinstance(b, ReliabilityBin)
    assert b.p_mean == pytest.approx(0.3)
    assert b.y_mean == pytest.approx(0.4)
    assert b.n == 10


def test_reliability_bins_total_n_equals_input() -> None:
    rng = np.random.default_rng(2024)
    n = 1000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)
    bins = reliability_bins(y, p, n_bins=10)
    assert sum(b.n for b in bins) == n


@pytest.mark.parametrize("n_bins", [1, 5, 10])
def test_reliability_bins_n_bins_param(n_bins: int) -> None:
    rng = np.random.default_rng(7)
    n = 300
    p = rng.uniform(0.0, 1.0, n)
    y = (p > 0.5).astype(int)
    bins = reliability_bins(y, p, n_bins=n_bins)
    assert all(b.n > 0 for b in bins)
    assert len(bins) <= n_bins
