"""Tests for `src.inference` — Phase A unified prediction surface
(round-015 corrected: Mondrian-ACI driven by p_online, the streaming-layer
output of the two-layer architecture).

The tests pin:
- `RegimeCuts.assign` returns ids in [0, n_regimes-1] and matches the equal-
  frequency partition.
- `fit_regime_cuts` produces ~equal-sized buckets on a uniform sample.
- `warm_q_init_by_regime` returns finite q in [0, 1] per regime, falling back
  for under-sampled regimes.
- `predict()` returns a DataFrame with all mandated columns; q histories are
  finite; in_set indicators are 0/1; `confidence_score` is non-negative.
- `predict()` requires `p_online` and uses `p_online` to drive the conformal
  stream (round-015 contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.inference import (  # noqa: E402
    DEFAULT_ALPHAS,
    RegimeCuts,
    fit_regime_cuts,
    predict,
    warm_q_init_by_regime,
)


# ---------------------------------------------------------------------------
# RegimeCuts
# ---------------------------------------------------------------------------

def test_regime_cuts_assign_basic():
    cuts = RegimeCuts(feature="x", edges=(1.0, 2.0), labels=("low", "med", "high"))
    ids = cuts.assign(np.array([0.5, 1.5, 2.5]))
    assert ids.tolist() == [0, 1, 2]
    labels = cuts.assign_labels(np.array([0.5, 1.5, 2.5]))
    assert labels.tolist() == ["low", "med", "high"]


def test_regime_cuts_boundary_values():
    """Values exactly at an edge fall into the *lower* bucket (strict >)."""
    cuts = RegimeCuts(feature="x", edges=(1.0,), labels=("a", "b"))
    assert cuts.assign(np.array([1.0]))[0] == 0
    assert cuts.assign(np.array([1.0 + 1e-9]))[0] == 1


def test_fit_regime_cuts_balanced():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.0, 1.0, size=10_000)
    cuts = fit_regime_cuts(v, n_regimes=3)
    assert len(cuts.edges) == 2
    ids = cuts.assign(v)
    counts = np.bincount(ids, minlength=3)
    # Each tercile should hold ~3,333 ± noise.
    assert (counts > 3000).all() and (counts < 3700).all()


def test_fit_regime_cuts_rejects_label_mismatch():
    with pytest.raises(ValueError):
        fit_regime_cuts(np.arange(10), n_regimes=3, labels=("a", "b"))


def test_fit_regime_cuts_rejects_empty():
    with pytest.raises(ValueError):
        fit_regime_cuts(np.array([]), n_regimes=3)


# ---------------------------------------------------------------------------
# warm_q_init_by_regime
# ---------------------------------------------------------------------------

def test_warm_q_init_per_regime_finite():
    rng = np.random.default_rng(1)
    n = 1_500
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < 0.1).astype(int)
    regimes = rng.integers(0, 3, n)
    q = warm_q_init_by_regime(p, y, regimes, alpha=0.10, min_per_regime=50)
    for r, qr in q.items():
        assert np.isfinite(qr)
        assert 0.0 <= qr <= 1.0


def test_warm_q_init_falls_back_for_undersampled():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 1, 0, 1])
    regimes = np.array([0, 0, 1, 1])  # 2 per regime; below default min_per_regime=50.
    q = warm_q_init_by_regime(p, y, regimes, alpha=0.10, min_per_regime=50,
                                fallback=0.7)
    assert q[0] == pytest.approx(0.7)
    assert q[1] == pytest.approx(0.7)


def test_warm_q_init_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        warm_q_init_by_regime(
            p_cal=np.array([0.1, 0.2]),
            y_cal=np.array([0, 1, 0]),
            regime_cal=np.array([0, 0, 1]),
            alpha=0.1,
        )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def _toy_predictions(n: int = 600, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    p_offline = rng.beta(2.0, 8.0, n)
    p_online = np.clip(p_offline + rng.normal(0.0, 0.05, n), 0.0, 1.0)
    y = (rng.uniform(0.0, 1.0, n) < p_offline).astype(int)
    regime_values = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame({
        "k": np.arange(n),
        "y_true": y,
        "p_offline": p_offline,
        "p_online": p_online,
    })
    return df, regime_values


def test_predict_emits_all_mandated_columns():
    df, regime_values = _toy_predictions()
    cuts = fit_regime_cuts(regime_values, n_regimes=3)
    out = predict(df, regime_values=regime_values, regime_cuts=cuts)
    expected = {
        "p_offline", "p_online", "y_true", "regime_id", "regime_label",
        "q_lo_05", "q_lo_10", "q_lo_20",
        "in_set_05", "in_set_10", "in_set_20",
        "sigma_epistemic", "confidence_score",
    }
    assert expected.issubset(set(out.columns))


def test_predict_q_histories_finite_and_in_unit_interval():
    df, regime_values = _toy_predictions()
    cuts = fit_regime_cuts(regime_values, n_regimes=3)
    out = predict(df, regime_values=regime_values, regime_cuts=cuts)
    for col in ("q_lo_05", "q_lo_10", "q_lo_20"):
        v = out[col].to_numpy()
        assert np.isfinite(v).all(), col
        assert (v >= 0.0).all() and (v <= 1.0).all(), col


def test_predict_in_set_is_zero_or_one():
    df, regime_values = _toy_predictions()
    cuts = fit_regime_cuts(regime_values, n_regimes=3)
    out = predict(df, regime_values=regime_values, regime_cuts=cuts)
    for col in ("in_set_05", "in_set_10", "in_set_20"):
        v = out[col].to_numpy()
        assert set(np.unique(v)).issubset({0, 1}), col


def test_predict_confidence_score_non_negative():
    df, regime_values = _toy_predictions()
    cuts = fit_regime_cuts(regime_values, n_regimes=3)
    out = predict(df, regime_values=regime_values, regime_cuts=cuts)
    cs = out["confidence_score"].to_numpy()
    assert np.isfinite(cs).all()
    assert (cs >= 0.0).all()


def test_predict_regime_shape_mismatch_rejected():
    df, _ = _toy_predictions(n=10)
    cuts = RegimeCuts(feature="x", edges=(1.0, 2.0), labels=("low", "med", "high"))
    with pytest.raises(ValueError):
        predict(df, regime_values=np.zeros(11), regime_cuts=cuts)


def test_predict_sigma_epistemic_passthrough():
    df, regime_values = _toy_predictions(n=120)
    cuts = fit_regime_cuts(regime_values, n_regimes=3)
    sigma = np.linspace(0.0, 0.1, 120)
    out = predict(df, regime_values=regime_values, regime_cuts=cuts,
                   sigma_epistemic=sigma)
    np.testing.assert_array_equal(out["sigma_epistemic"].to_numpy(), sigma)


def test_predict_q_init_overrides_applied():
    df, regime_values = _toy_predictions(n=200)
    cuts = fit_regime_cuts(regime_values, n_regimes=3)
    # If we lock all per-regime q at 1.0 with γ=0, no row should cross
    # 1 - q = 0, so in_set_* = 1 everywhere (singleton {1} in set always).
    overrides = {0.10: {0: 1.0, 1: 1.0, 2: 1.0}}
    out = predict(df, regime_values=regime_values, regime_cuts=cuts,
                   gamma=0.0, q_init_by_regime_per_alpha=overrides)
    np.testing.assert_array_equal(out["q_lo_10"].to_numpy(), 1.0)
    np.testing.assert_array_equal(out["in_set_10"].to_numpy(), 1)


def test_predict_requires_p_online_driven_two_layer_architecture():
    """Round-015 contract: the conformal stream is driven by `p_online` (the
    streaming-layer output), not `p_offline`. Calling predict() without
    `p_online` is an architectural error and must raise."""
    df = pd.DataFrame({
        "p_offline": [0.1, 0.5],
        "y_true":    [0, 1],
    })
    cuts = RegimeCuts(feature="x", edges=(0.5,), labels=("a", "b"))
    with pytest.raises(KeyError, match="p_online"):
        predict(df, regime_values=np.array([0.1, 0.7]), regime_cuts=cuts)


def test_predict_in_set_uses_p_online_not_p_offline():
    """If we set p_online to 1.0 (very confident) and p_offline to 0.0 (no
    signal), `in_set_α` should be 1 because the conformal layer reads the
    online output. Round-015 fixes round-011's offline-driven mistake."""
    n = 50
    df = pd.DataFrame({
        "p_offline": np.zeros(n),
        "p_online":  np.full(n, 0.99),
        "y_true":    np.ones(n, dtype=int),
    })
    regime_values = np.zeros(n)
    cuts = RegimeCuts(feature="x", edges=(), labels=("only",))
    out = predict(df, regime_values=regime_values, regime_cuts=cuts,
                   alphas=(0.10,))
    # With p_online = 0.99 ≥ 1 - q across all rows, in_set_10 should be 1.
    assert (out["in_set_10"].to_numpy() == 1).all()


def test_predict_default_alphas_produce_three_q_columns():
    df, regime_values = _toy_predictions(n=120)
    cuts = fit_regime_cuts(regime_values, n_regimes=3)
    out = predict(df, regime_values=regime_values, regime_cuts=cuts,
                   alphas=DEFAULT_ALPHAS)
    q_cols = sorted(c for c in out.columns if c.startswith("q_lo_"))
    assert q_cols == ["q_lo_05", "q_lo_10", "q_lo_20"]
