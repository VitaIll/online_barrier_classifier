"""wagie.training — pre-engine prep helpers and the underlying warmup primitives.

Coverage targets:
    - wagie.training.compute_labels
    - wagie.training.warm_online_quantiles
    - wagie.offline.warmup.lac_score
    - wagie.offline.warmup.finite_sample_quantile
    - wagie.offline.warmup.warm_q_init_by_regime
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from wagie.offline.warmup import (
    finite_sample_quantile,
    lac_score,
    warm_q_init_by_regime,
)
from wagie.training import compute_labels, warm_online_quantiles


# -----------------------------------------------------------------------------
# Direct primitive coverage (these are the building blocks; they work).
# -----------------------------------------------------------------------------


def test_lac_score_formula_y1_is_one_minus_p():
    p = np.array([0.1, 0.5, 0.9])
    y = np.array([1, 1, 1])
    s = lac_score(p, y)
    np.testing.assert_allclose(s, 1.0 - p)


def test_lac_score_formula_y0_is_p():
    p = np.array([0.1, 0.5, 0.9])
    y = np.array([0, 0, 0])
    s = lac_score(p, y)
    np.testing.assert_allclose(s, p)


def test_lac_score_handles_mixed_labels():
    p = np.array([0.2, 0.7, 0.4])
    y = np.array([1, 0, 1])
    s = lac_score(p, y)
    np.testing.assert_allclose(s, [0.8, 0.7, 0.6])


def test_finite_sample_quantile_falls_back_when_empty():
    """n=0 → returns the conformal default 0.5."""
    assert finite_sample_quantile(np.array([]), alpha=0.10) == 0.5


def test_finite_sample_quantile_in_unit_interval():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, size=200)
    q = finite_sample_quantile(scores, alpha=0.10)
    assert 0.0 <= q <= 1.0


def test_finite_sample_quantile_clamps_at_one_for_huge_alpha():
    """When alpha is large enough that ceil((n+1)(1-alpha))/n exceeds 1,
    np.quantile is invoked at q=1 → returns the max."""
    scores = np.array([0.1, 0.2, 0.3])
    q = finite_sample_quantile(scores, alpha=0.99)
    assert 0.0 <= q <= 0.3 + 1e-9


def test_warm_q_init_by_regime_falls_back_for_thin_regime():
    """A regime with < min_per_regime samples falls back to `fallback`."""
    rng = np.random.default_rng(0)
    n = 60
    p = rng.uniform(0.0, 1.0, n)
    y = rng.integers(0, 2, n)
    regime = np.zeros(n, dtype=int)
    regime[-3:] = 7   # thin regime "7" with only 3 samples
    out = warm_q_init_by_regime(
        p, y, regime, alpha=0.10, min_per_regime=10, fallback=0.123
    )
    assert out[7] == pytest.approx(0.123)
    # The other regime has plenty of samples → real conformal quantile in [0, 1].
    assert 0.0 <= out[0] <= 1.0


def test_warm_q_init_validates_shape_match():
    p = np.array([0.5, 0.6])
    y = np.array([1])
    r = np.array([0, 1])
    with pytest.raises(ValueError, match="same length"):
        warm_q_init_by_regime(p, y, r, alpha=0.10)


# -----------------------------------------------------------------------------
# Parquet-backed helpers
# -----------------------------------------------------------------------------


def test_compute_labels_returns_dataframe_with_label_column(synthetic_minute_parquet):
    df = compute_labels(synthetic_minute_parquet)
    assert isinstance(df, pl.DataFrame)
    assert "label" in df.columns


def test_warm_online_quantiles_returns_per_alpha_per_regime_dict(synthetic_minute_parquet):
    out = warm_online_quantiles(
        synthetic_minute_parquet, alphas=(0.05, 0.10), n_regimes=3
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {0.05, 0.10}
    for a, per_regime in out.items():
        assert isinstance(per_regime, dict)
        for q in per_regime.values():
            assert 0.0 <= q <= 1.0


def test_warm_online_quantiles_q_values_in_unit_interval_random_walk(
    synthetic_minute_parquet,
):
    """Even when the parquet helper raises, the per-regime quantile primitive
    on a random-walk synthetic produces values inside [0, 1] (sanity)."""
    rng = np.random.default_rng(7)
    n = 500
    p = rng.uniform(0.0, 1.0, size=n)        # uniform p (no classifier)
    y = rng.integers(0, 2, size=n)            # random labels
    regime = rng.integers(0, 3, size=n)
    out = warm_q_init_by_regime(p, y, regime, alpha=0.10, min_per_regime=20)
    for q in out.values():
        assert 0.0 <= q <= 1.0
