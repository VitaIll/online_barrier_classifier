"""Tests for the round-002 coverage diagnostic glue.

The math primitives (`fit_conformal`, `predict_set`, `empirical_coverage`,
`coverage_by_regime`) are already covered by `tests/test_conformal.py`. These
tests cover the round-002 driver-script invariants:

1. `naive_threshold_sets` matches the formal definition `set = {y : p̂(y|x) ≥ α}`
   on hand-checked inputs.
2. The constant-predictor `p ≡ base_rate` sanity check returns the expected
   coverage at α=0.05 (≈1.0) and α=0.20 (≈ 1 − base_rate). This is the
   harness-falsifier the THEORIST asked for.
3. The diagnostic is deterministic: invoking `evaluate_one` twice on the same
   inputs returns byte-identical metric rows. Required by HYPOTHESIS.md
   (`seed_noise_band = 0.0`).
4. `make_terciles` partitions a strictly-increasing series into three roughly
   equal-sized contiguous blocks (sanity for the regime cuts).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_round_module():
    name = "round_002_coverage_baseline"
    spec = importlib.util.spec_from_file_location(
        name,
        REPO / "scripts" / "round_002_coverage_baseline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def round_mod():
    return _load_round_module()


# ----------------------------------------------------------------------------
# (1) naive_threshold_sets matches the formal definition
# ----------------------------------------------------------------------------

def test_naive_threshold_sets_match_definition(round_mod):
    # Use values strictly off the threshold to avoid float-eq edge cases
    # (e.g., 1.0 - 0.9 != 0.1 exactly in IEEE 754).
    p = np.array([0.02, 0.15, 0.50, 0.85, 0.98])
    alpha = 0.10
    sets = round_mod.naive_threshold_sets(p, alpha)
    # Class-1 in set iff p >= 0.10 — true for p∈{0.15, 0.50, 0.85, 0.98}.
    np.testing.assert_array_equal(
        sets[:, 1], np.array([False, True, True, True, True])
    )
    # Class-0 in set iff (1-p) >= 0.10 — true for p∈{0.02, 0.15, 0.50, 0.85}.
    np.testing.assert_array_equal(
        sets[:, 0], np.array([True, True, True, True, False])
    )


# ----------------------------------------------------------------------------
# (2) Constant-predictor sanity (THEORIST-asked harness falsifier)
# ----------------------------------------------------------------------------

def test_constant_predictor_alpha_005_full_set(round_mod):
    """At α=0.05 with p ≡ base_rate=0.0971, both 0.0971 ≥ 0.05 AND
    1−0.0971=0.9029 ≥ 0.05 — set is always {0,1} → coverage = 1.0 exactly."""
    rng = np.random.default_rng(42)
    n = 5000
    base_rate = 0.0971
    y = (rng.random(n) < base_rate).astype(int)
    p = np.full(n, base_rate)
    sets = round_mod.naive_threshold_sets(p, alpha=0.05)
    cov = float(np.mean([sets[i, y[i]] for i in range(n)]))
    assert cov == pytest.approx(1.0, abs=1e-12)


def test_constant_predictor_alpha_020_drops_positive_class(round_mod):
    """At α=0.20 with p ≡ 0.0971, p < 0.20 so class-1 is never in set; but
    1-p = 0.9029 ≥ 0.20 so class-0 is always in set. Coverage =
    P(y_true == 0) = 1 - base_rate."""
    rng = np.random.default_rng(7)
    n = 10000
    base_rate = 0.0971
    y = (rng.random(n) < base_rate).astype(int)
    p = np.full(n, base_rate)
    sets = round_mod.naive_threshold_sets(p, alpha=0.20)
    cov = float(np.mean([sets[i, y[i]] for i in range(n)]))
    expected = 1.0 - float(y.mean())
    assert cov == pytest.approx(expected, abs=1e-12)


# ----------------------------------------------------------------------------
# (3) Determinism — seed_noise_band = 0
# ----------------------------------------------------------------------------

def test_evaluate_one_is_deterministic(round_mod):
    """Two invocations on the same arrays return identical metric rows."""
    rng = np.random.default_rng(2026)
    n = 1500
    p = np.clip(rng.beta(1.5, 8.0, n), 1e-3, 1 - 1e-3)
    y = (rng.random(n) < p).astype(int)
    regime = np.array(["low", "med", "high"])[(np.arange(n) % 3)]
    cal_idx = np.arange(450)
    eval_idx = np.arange(450, n)

    rows_a = round_mod.evaluate_one("p_offline", p, y, regime, cal_idx, eval_idx)
    rows_b = round_mod.evaluate_one("p_offline", p, y, regime, cal_idx, eval_idx)
    assert len(rows_a) == len(rows_b)
    for a, b in zip(rows_a, rows_b):
        assert a == b


# ----------------------------------------------------------------------------
# (4) make_terciles partitions sensibly
# ----------------------------------------------------------------------------

def test_make_terciles_balanced_on_strictly_increasing(round_mod):
    n = 900
    values = np.arange(n, dtype=float)  # strictly increasing
    cats = round_mod.make_terciles(values)
    counts = {label: int((cats == label).sum()) for label in ("low", "med", "high")}
    # qcut should produce three roughly equal buckets.
    for label, c in counts.items():
        assert abs(c - n / 3) <= 1, f"tercile {label} has {c} items"
    # Strictly increasing → 'low' is the first block, 'high' the last.
    assert (cats[: n // 3] == "low").all()
    assert (cats[2 * n // 3 :] == "high").all()


# ----------------------------------------------------------------------------
# (5) Marginal coverage on well-calibrated synthetic data ≈ nominal
# ----------------------------------------------------------------------------

def test_marginal_coverage_on_calibrated_synth(round_mod):
    """If p_i is the true Bernoulli parameter, split-conformal LAC should give
    near-nominal marginal coverage on a held-out eval set (within ~2σ)."""
    rng = np.random.default_rng(1234)
    n = 6000
    p = np.clip(rng.beta(2.0, 6.0, n), 1e-3, 1 - 1e-3)
    y = (rng.random(n) < p).astype(int)
    regime = np.array(["low", "med", "high"])[(np.arange(n) // (n // 3))]
    cal_idx = np.arange(2000)
    eval_idx = np.arange(2000, n)
    rows = round_mod.evaluate_one("p_calib", p, y, regime, cal_idx, eval_idx)
    df = {(r.alpha, r.mode, r.regime): r.coverage for r in rows}
    # Marginal LAC at α=0.10 should be ≥ 1-α - 2σ; with n_eval=4000, σ ≈
    # sqrt(0.9*0.1/4000) ≈ 0.0047 — allow 3σ slack.
    cov_marg_010 = df[(0.10, "lac_marginal", "ALL")]
    assert cov_marg_010 >= 0.90 - 3 * 0.0047
    assert cov_marg_010 <= 0.90 + 3 * 0.0047
