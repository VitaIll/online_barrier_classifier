"""Tests for src/ensemble.py::CatBoostEnsemble.

Covers the H-108 contract: averaged probabilistic predictions, averaged feature
importances, save+load roundtrip, ergonomics (len/n_models), and the new
load_ensemble classmethod (which the sibling lacks). Uses tiny synthetic data
so each test runs in <1s; CatBoost is loaded lazily inside fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ensemble import CatBoostEnsemble


@pytest.fixture(scope="module")
def synthetic_xy():
    """Tiny, deterministic, separable-ish dataset that CatBoost can fit fast."""
    rng = np.random.default_rng(0)
    n, d = 200, 4
    X = rng.normal(size=(n, d)).astype(float)
    y = (X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.05 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted_ensemble(synthetic_xy):
    catboost = pytest.importorskip("catboost")
    X, y = synthetic_xy
    # 80/20 split so eval_set is populated (gives best_iteration semantics).
    n_train = int(0.8 * len(y))
    X_tr, X_va, y_tr, y_va = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    seeds = [0, 1, 2]
    models = []
    for s in seeds:
        m = catboost.CatBoostClassifier(
            iterations=30,
            depth=3,
            learning_rate=0.1,
            random_seed=s,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=10,
        )
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        models.append(m)
    return CatBoostEnsemble(models)


def test_init_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one"):
        CatBoostEnsemble([])


def test_len_and_n_models(fitted_ensemble):
    assert len(fitted_ensemble) == 3
    assert fitted_ensemble.n_models == 3


def test_predict_proba_is_average(fitted_ensemble, synthetic_xy):
    X, _ = synthetic_xy
    individual = np.stack([m.predict_proba(X) for m in fitted_ensemble.models], axis=0)
    expected = individual.mean(axis=0)
    actual = fitted_ensemble.predict_proba(X)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_predict_proba_rows_sum_to_one(fitted_ensemble, synthetic_xy):
    X, _ = synthetic_xy
    p = fitted_ensemble.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7)


def test_predict_threshold(fitted_ensemble, synthetic_xy):
    X, _ = synthetic_xy
    p = fitted_ensemble.predict_proba(X)
    yhat_05 = fitted_ensemble.predict(X, threshold=0.5)
    np.testing.assert_array_equal(yhat_05, (p[:, 1] >= 0.5).astype(int))
    yhat_09 = fitted_ensemble.predict(X, threshold=0.9)
    assert (yhat_09 <= yhat_05).all()


def test_feature_importance_is_average(fitted_ensemble):
    individual = np.stack(
        [
            np.asarray(m.get_feature_importance(type="PredictionValuesChange"), dtype=float)
            for m in fitted_ensemble.models
        ],
        axis=0,
    )
    expected = individual.mean(axis=0)
    actual = fitted_ensemble.get_feature_importance()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_get_best_iteration(fitted_ensemble):
    bi = fitted_ensemble.get_best_iteration()
    assert isinstance(bi, int)
    assert 0 <= bi <= 30


def test_save_load_roundtrip(fitted_ensemble, synthetic_xy, tmp_path):
    X, _ = synthetic_xy
    base = tmp_path / "model.cbm"
    fitted_ensemble.save_model(base)

    # All shards exist (n_models of them).
    shards = sorted(tmp_path.glob("model.*.cbm"))
    assert len(shards) == fitted_ensemble.n_models

    reloaded = CatBoostEnsemble.load_ensemble(base, n_models=fitted_ensemble.n_models)
    assert reloaded.n_models == fitted_ensemble.n_models

    p_original = fitted_ensemble.predict_proba(X)
    p_reloaded = reloaded.predict_proba(X)
    np.testing.assert_allclose(p_reloaded, p_original, rtol=1e-6, atol=1e-7)


def test_load_ensemble_rejects_zero_n_models(tmp_path):
    with pytest.raises(ValueError, match="n_models must be >= 1"):
        CatBoostEnsemble.load_ensemble(tmp_path / "x.cbm", n_models=0)


def test_load_ensemble_rejects_missing_shard(tmp_path):
    # Create only shard 0; ask for 3.
    catboost = pytest.importorskip("catboost")
    X = np.random.default_rng(0).normal(size=(50, 2)).astype(float)
    y = (X[:, 0] > 0).astype(int)
    m = catboost.CatBoostClassifier(iterations=10, verbose=False, allow_writing_files=False)
    m.fit(X, y)
    base = tmp_path / "partial.cbm"
    m.save_model(str(base.with_suffix(".0.cbm")))
    with pytest.raises(FileNotFoundError, match="shard 1 missing"):
        CatBoostEnsemble.load_ensemble(base, n_models=3)
