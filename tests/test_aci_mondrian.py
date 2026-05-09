"""Tests for the H-203 Mondrian-ACI implementation in src/conformal.py.

Mondrian-ACI's contract:
- Each regime label has its own q_t; the per-step update touches ONLY the
  active regime's q. Inactive regimes' q values are unchanged.
- With one regime label everywhere, Mondrian-ACI is bit-equivalent to plain
  aci_stream (this is the load-bearing reduction test).
- Per-regime asymptotic coverage: each regime's empirical coverage on its
  own substream → 1-α (G&C 2021 applied per regime, since each is a
  self-contained ACI process).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.conformal import (
    aci_mondrian_step,
    aci_mondrian_stream,
    aci_step,
    aci_stream,
)


# -----------------------------------------------------------------------------
# aci_mondrian_step
# -----------------------------------------------------------------------------

def test_mondrian_step_only_updates_active_regime():
    q_in = {"low": 0.40, "med": 0.50, "high": 0.60}
    set_t, q_out, err_t = aci_mondrian_step(
        p_t=0.8, y_t=1, regime_t="med",
        q_t_by_regime=q_in, alpha=0.10, gamma=0.05,
    )
    # med was active → its q changed
    assert q_out["med"] != q_in["med"]
    # low and high untouched
    assert q_out["low"] == q_in["low"]
    assert q_out["high"] == q_in["high"]
    # set construction matches plain aci_step at q=0.50
    set_plain, q_plain, err_plain = aci_step(0.8, 1, 0.50, 0.10, 0.05)
    np.testing.assert_array_equal(set_t, set_plain)
    assert q_out["med"] == pytest.approx(q_plain, abs=1e-12)
    assert err_t == err_plain


def test_mondrian_step_returns_independent_dict():
    """Updating must NOT mutate the caller's q_t_by_regime mapping."""
    q_in = {"a": 0.30}
    snapshot = dict(q_in)
    aci_mondrian_step(0.5, 0, "a", q_in, alpha=0.10, gamma=0.05)
    assert q_in == snapshot, "input q_t_by_regime was mutated"


def test_mondrian_step_rejects_unknown_regime():
    with pytest.raises(KeyError, match="not in q_t_by_regime"):
        aci_mondrian_step(0.5, 0, "novel", {"a": 0.5}, alpha=0.10, gamma=0.05)


# -----------------------------------------------------------------------------
# aci_mondrian_stream — reduction to plain ACI under single regime
# -----------------------------------------------------------------------------

def test_mondrian_stream_single_regime_equivalent_to_plain_aci():
    """With only one regime, Mondrian-ACI must produce identical sets / q /
    coverage to plain aci_stream. Bit-exact equivalence under same q_init.
    """
    rng = np.random.default_rng(0)
    T = 1000
    p = rng.uniform(size=T)
    y = (rng.uniform(size=T) < 0.20).astype(int)
    regimes = np.full(T, "only", dtype=object)

    plain = aci_stream(p, y, alpha=0.10, gamma=0.02, q_init=0.55)
    mondrian = aci_mondrian_stream(p, y, regimes, alpha=0.10, gamma=0.02, q_init=0.55)

    np.testing.assert_array_equal(plain["sets"], mondrian["sets"])
    np.testing.assert_allclose(plain["q_history"], mondrian["q_history"], rtol=0, atol=0)
    np.testing.assert_array_equal(plain["err_history"], mondrian["err_history"])
    assert plain["coverage"] == mondrian["coverage"]


# -----------------------------------------------------------------------------
# aci_mondrian_stream — per-regime semantics
# -----------------------------------------------------------------------------

def test_mondrian_stream_per_regime_q_history_lengths_sum_to_T():
    rng = np.random.default_rng(1)
    T = 600
    p = rng.uniform(size=T)
    y = (rng.uniform(size=T) < 0.15).astype(int)
    regimes = rng.choice(["a", "b", "c"], size=T)
    out = aci_mondrian_stream(p, y, regimes, alpha=0.10, gamma=0.02, q_init=0.5)
    total = sum(len(v) for v in out["q_history_by_regime"].values())
    assert total == T


def test_mondrian_stream_per_regime_coverage_converges():
    """Each regime's substream should hit 1-α independently."""
    rng = np.random.default_rng(2)
    T = 30_000
    base_rate = 0.10
    y = (rng.uniform(size=T) < base_rate).astype(int)
    p = np.clip(0.10 + 0.20 * (y - 0.5) + 0.10 * rng.normal(size=T), 0.001, 0.999)
    regimes = rng.choice(["A", "B", "C"], size=T)
    alpha = 0.10
    out = aci_mondrian_stream(p, y, regimes, alpha=alpha, gamma=0.01, q_init=0.5)

    target = 1.0 - alpha
    for regime in ("A", "B", "C"):
        mask = regimes == regime
        n_r = int(mask.sum())
        sub_y = y[mask]
        sub_sets = out["sets"][mask]
        # coverage on this regime's substream
        cov_r = float(np.mean([sub_sets[i, sub_y[i]] for i in range(n_r)]))
        sigma_r = math.sqrt(alpha * (1.0 - alpha) / n_r)
        # 6σ tolerance: γ=0.01 with substream ~10k is on the slow side.
        assert abs(cov_r - target) < 6.0 * sigma_r + 0.01, (
            f"regime {regime}: coverage={cov_r:.4f}, target={target:.4f}, "
            f"6σ+0.01 = {6*sigma_r + 0.01:.4f}"
        )


def test_mondrian_stream_q_init_by_regime_overrides_default():
    rng = np.random.default_rng(3)
    T = 100
    p = rng.uniform(size=T)
    y = np.zeros(T, dtype=int)
    regimes = np.array(["a"] * 50 + ["b"] * 50)
    out = aci_mondrian_stream(
        p, y, regimes, alpha=0.10, gamma=0.0,  # frozen q
        q_init=0.5, q_init_by_regime={"a": 0.30, "b": 0.70},
    )
    # γ=0 means q never moves; first step in regime 'a' starts at 0.30.
    a_traj = out["q_history_by_regime"]["a"]
    b_traj = out["q_history_by_regime"]["b"]
    assert (a_traj == 0.30).all()
    assert (b_traj == 0.70).all()


def test_mondrian_stream_rejects_shape_mismatch():
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1, 0])
    bad_regimes = np.array(["a", "b"])  # wrong length
    with pytest.raises(ValueError, match="same shape"):
        aci_mondrian_stream(p, y, bad_regimes, alpha=0.10)


def test_mondrian_stream_handles_integer_regime_labels():
    rng = np.random.default_rng(4)
    T = 200
    p = rng.uniform(size=T)
    y = (rng.uniform(size=T) < 0.20).astype(int)
    regimes = rng.integers(0, 3, size=T)  # int regime labels
    out = aci_mondrian_stream(p, y, regimes, alpha=0.10, gamma=0.02, q_init=0.5)
    # All distinct integer regimes seen should appear in q_history_by_regime.
    seen = set(int(r) for r in np.unique(regimes))
    assert seen.issubset(set(int(k) for k in out["q_history_by_regime"]))
