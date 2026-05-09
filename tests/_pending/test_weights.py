"""Sample-weighting invariants (Section W).

The barrier-distance weight upweights deep losses, not near-misses, so the
weight on negatives must be monotone non-decreasing in d_k = max(0, phi-m_k).
The time-discount weight must lie in (0, 1] and be non-decreasing in
chronological order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils


def test_barrier_distance_weight_is_monotone_in_d_k(m_k_vector: np.ndarray):
    """For negatives (m_k < phi), w_dist must be non-decreasing in d_k = phi - m_k.

    This is the load-bearing property of the risk-aware weighting: deeper
    losses receive at least as much weight as shallower losses. The user
    corrected an earlier description that had this inverted.
    """
    phi = utils.PHI
    w_dist, info = utils.compute_barrier_distance_weight(
        m_k_vector,
        phi=phi,
        w_max=5.0,
        q_tail=0.01,
        enabled=True,
    )

    neg_mask = m_k_vector < phi
    neg_m = m_k_vector[neg_mask]
    neg_w = w_dist[neg_mask]

    # Sort by d_k ascending = m_k descending; weights along that order must
    # be non-decreasing.
    order = np.argsort(-neg_m)  # m_k descending  →  d_k ascending
    sorted_w = neg_w[order]
    assert np.all(np.diff(sorted_w) >= -1e-12), (
        "barrier-distance weight on negatives is not monotone in d_k"
    )


def test_barrier_distance_weight_capped_at_w_max(m_k_vector: np.ndarray):
    w_max = 5.0
    w_dist, _info = utils.compute_barrier_distance_weight(
        m_k_vector,
        phi=utils.PHI,
        w_max=w_max,
        q_tail=0.01,
        enabled=True,
    )
    assert w_dist.max() <= w_max + 1e-12, "soft cap violated"


def test_barrier_distance_weight_disabled_is_unit():
    """When enabled=False, every weight must be exactly 1."""
    m_k = np.linspace(-0.05, 0.05, 200)
    w, _ = utils.compute_barrier_distance_weight(
        m_k, phi=utils.PHI, enabled=False
    )
    assert np.all(w == 1.0)


def test_time_discount_in_zero_one():
    """Time discount weights must be in (0, 1] and non-decreasing in time order."""
    N = 200
    k_index = np.arange(N)
    w, info = utils.compute_time_discount_weight(
        N=N, r=0.5, delta=0.99, k_index=k_index, enabled=True
    )
    assert (w > 0).all(), "time-discount weights must be strictly positive"
    assert (w <= 1.0 + 1e-12).all(), "time-discount weights must be <= 1"
    # Most recent should be 1.0; oldest should be < 1.0
    most_recent_idx = int(np.argmax(k_index))
    oldest_idx = int(np.argmin(k_index))
    assert w[most_recent_idx] == pytest.approx(1.0)
    assert w[oldest_idx] < 1.0


def test_combined_weights_are_positive(m_k_vector):
    w_combined, w_dist, w_time, info = utils.compute_training_weights(
        m_k=m_k_vector,
        phi=utils.PHI,
        use_dist=True,
        use_time=True,
        w_max=5.0,
        q_tail=0.01,
        r=0.3,
        delta=0.99996,
        normalize=False,
    )
    assert (w_combined > 0).all(), "combined weights must be strictly positive"
    assert np.allclose(w_combined, w_dist * w_time)


def test_effective_n_below_actual(m_k_vector):
    """Kish's effective sample size cannot exceed the actual sample size."""
    _, _, _, info = utils.compute_training_weights(
        m_k=m_k_vector,
        phi=utils.PHI,
        use_dist=True,
        use_time=True,
        w_max=5.0,
        q_tail=0.01,
        normalize=False,
    )
    n = len(m_k_vector)
    assert info["combined"]["effective_n"] <= n + 1e-9
    assert info["combined"]["effective_n"] > 0
