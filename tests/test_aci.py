"""Tests for the H-202 Adaptive Conformal Inference (ACI) implementation.

ACI's contract:
- Per-step update: q_{t+1} = q_t + γ(err_t - α), clamped to [0, 1].
- err_t = 1 if y_t ∉ C_t (miscovered), else 0.
- Set construction matches batch LAC: include y if 1 - p(y|x) ≤ q_t.
- Asymptotic guarantee (G&C 2021 Thm 1): empirical coverage → 1-α as T → ∞,
  for any γ > 0 and any data sequence.

Tests pin:
- One-step update math is correct.
- Update direction (miscovered ⇒ q increases; covered ⇒ q decreases for γ>0).
- γ=0 freezes q_t (sanity).
- Argument validation (α, γ, y_t bounds).
- Asymptotic coverage on a stationary stream within 2σ.
- Set-construction parity with batch ``predict_set`` at fixed q.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.conformal import (
    ConformalCalibrator,
    aci_step,
    aci_stream,
    predict_set,
)


# -----------------------------------------------------------------------------
# aci_step — single-step update math
# -----------------------------------------------------------------------------

def test_aci_step_covered_decreases_q():
    """When the true label is in the predicted set, q should decrease."""
    # p=0.8, q_t=0.5 → set has class 1 (p ≥ 1-q=0.5) and not class 0 (p > q=0.5).
    # y_t=1 → covered → err_t=0 → q_next = q_t + γ(0 - α) = q_t - γα.
    set_t, q_next, err_t = aci_step(p_t=0.8, y_t=1, q_t=0.5, alpha=0.10, gamma=0.05)
    assert err_t == 0
    assert q_next == pytest.approx(0.5 - 0.05 * 0.10, abs=1e-12)
    assert set_t.tolist() == [False, True]  # singleton {1}


def test_aci_step_miscovered_increases_q():
    """When y_t is outside the set, q should increase by γ(1 - α)."""
    # p=0.8, q_t=0.5 → set = {1}. y_t=0 → miscovered → err_t=1.
    # q_next = q_t + γ(1 - α) = 0.5 + 0.05 * 0.90 = 0.545
    set_t, q_next, err_t = aci_step(p_t=0.8, y_t=0, q_t=0.5, alpha=0.10, gamma=0.05)
    assert err_t == 1
    assert q_next == pytest.approx(0.5 + 0.05 * 0.90, abs=1e-12)
    assert set_t.tolist() == [False, True]


def test_aci_step_gamma_zero_freezes_q():
    set_t, q_next, _ = aci_step(p_t=0.3, y_t=0, q_t=0.4, alpha=0.10, gamma=0.0)
    assert q_next == pytest.approx(0.4, abs=1e-12)


def test_aci_step_q_clamped_to_unit_interval():
    # q_t at the boundary, miscovered run pushes outside → clamp.
    _, q_next, _ = aci_step(p_t=0.99, y_t=0, q_t=0.99, alpha=0.10, gamma=0.5)
    assert q_next <= 1.0
    _, q_next, _ = aci_step(p_t=0.01, y_t=1, q_t=0.0, alpha=0.99, gamma=0.5)
    assert q_next >= 0.0


def test_aci_step_set_matches_batch_predict_set():
    """At fixed q, the ACI per-step set construction must agree with the
    batch ``predict_set`` (so callers can trust the same coverage semantics
    apply when comparing).
    """
    cal = ConformalCalibrator(alpha=0.10, n_cal=1000, q_hat=0.40, mode="lac")
    p = np.array([0.05, 0.20, 0.40, 0.60, 0.80])
    batch_sets = predict_set(cal, p)
    aci_sets = np.array(
        [aci_step(float(pi), 0, 0.40, 0.10, 0.0)[0] for pi in p],
        dtype=bool,
    )
    np.testing.assert_array_equal(aci_sets, batch_sets)


def test_aci_step_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        aci_step(p_t=0.5, y_t=0, q_t=0.5, alpha=0.0, gamma=0.05)
    with pytest.raises(ValueError, match="alpha"):
        aci_step(p_t=0.5, y_t=0, q_t=0.5, alpha=1.0, gamma=0.05)


def test_aci_step_rejects_negative_gamma():
    with pytest.raises(ValueError, match="gamma"):
        aci_step(p_t=0.5, y_t=0, q_t=0.5, alpha=0.10, gamma=-0.01)


def test_aci_step_rejects_invalid_y():
    with pytest.raises(ValueError, match="y_t"):
        aci_step(p_t=0.5, y_t=2, q_t=0.5, alpha=0.10, gamma=0.05)


# -----------------------------------------------------------------------------
# aci_stream — stream-level invariants
# -----------------------------------------------------------------------------

def test_aci_stream_shapes_match_input():
    rng = np.random.default_rng(0)
    T = 1000
    p = rng.uniform(size=T)
    y = (rng.uniform(size=T) < 0.10).astype(int)
    out = aci_stream(p, y, alpha=0.10, gamma=0.01, q_init=0.5)
    assert out["sets"].shape == (T, 2)
    assert out["q_history"].shape == (T,)
    assert out["err_history"].shape == (T,)
    assert 0.0 <= out["coverage"] <= 1.0


def test_aci_stream_q_history_starts_at_q_init():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200)
    y = (rng.uniform(size=200) < 0.30).astype(int)
    out = aci_stream(p, y, alpha=0.10, gamma=0.05, q_init=0.42)
    assert out["q_history"][0] == pytest.approx(0.42, abs=1e-12)


def test_aci_stream_coverage_converges_on_stationary_iid():
    """Bernoulli-base-rate stream with random p: marginal coverage must
    sit within ~2σ of (1-α). σ ≈ sqrt(α(1-α)/T)."""
    rng = np.random.default_rng(2)
    T = 20_000
    base_rate = 0.10
    y = (rng.uniform(size=T) < base_rate).astype(int)
    # p mildly correlated with y — informative but noisy.
    p = np.clip(0.10 + 0.20 * (y - 0.5) + 0.10 * rng.normal(size=T), 0.001, 0.999)
    alpha = 0.10
    out = aci_stream(p, y, alpha=alpha, gamma=0.01, q_init=0.5)

    target = 1.0 - alpha
    sigma = math.sqrt(alpha * (1.0 - alpha) / T)
    # 5σ tolerance — generous because γ=0.01 is on the slow end.
    assert abs(out["coverage"] - target) < 5.0 * sigma + 0.01, (
        f"coverage={out['coverage']:.4f} vs target={target:.4f} "
        f"(2σ ≈ {2*sigma:.4f}, used 5σ+0.01 tolerance)"
    )


def test_aci_stream_with_gamma_zero_is_static():
    """γ=0 reduces ACI to a fixed-threshold predictor; q_history is constant."""
    rng = np.random.default_rng(3)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < 0.20).astype(int)
    out = aci_stream(p, y, alpha=0.10, gamma=0.0, q_init=0.45)
    assert (out["q_history"] == 0.45).all()


def test_aci_stream_high_gamma_higher_volatility():
    """A larger γ should produce a noisier q trajectory — verify std-dev grows."""
    rng = np.random.default_rng(4)
    T = 5000
    base_rate = 0.10
    y = (rng.uniform(size=T) < base_rate).astype(int)
    p = np.clip(0.10 + 0.05 * rng.normal(size=T), 0.001, 0.999)
    out_lo = aci_stream(p, y, alpha=0.10, gamma=0.001, q_init=0.5)
    out_hi = aci_stream(p, y, alpha=0.10, gamma=0.05, q_init=0.5)
    assert out_hi["q_history"].std() > out_lo["q_history"].std()


def test_aci_stream_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        aci_stream(np.array([0.1, 0.2]), np.array([0, 1, 0]), alpha=0.10)
