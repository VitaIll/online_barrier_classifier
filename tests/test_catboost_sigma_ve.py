"""Tests for FrozenCatBoostPredictor virtual-ensemble dispersion (sigma_ve).

The predictor must:
    * emit `sigma_ve = 0` for a single-member ensemble (no spread possible)
    * emit `sigma_ve > 0` for non-degenerate multi-member ensembles
    * compute population std (ddof=0), matching numpy.std(ps, ddof=0) to 1e-9
    * keep `p_offline` set as before (no regression on the existing emit)
    * keep `sigma_ve` in [0, 0.5] (max possible std of probabilities in [0,1])

We bypass real CatBoost model loading by injecting fake models exposing the
single attribute the predictor calls: `predict_proba(X) -> ndarray[N, 2]`.
"""

from __future__ import annotations

import numpy as np
import pytest

from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import Price, Quantity
from wagie.core.observation import FeatureMap, Observation
from wagie.core.time import Duration, Timestamp
from wagie.pipeline.catboost_predictor import FrozenCatBoostPredictor


_NS_PER_MIN = 60_000_000_000


class _FakeModel:
    """Minimal stand-in for CatBoostClassifier exposing predict_proba + feature_names_."""

    def __init__(self, p_pos: float, feature_names: list[str]):
        self.p_pos = float(p_pos)
        self.feature_names_ = list(feature_names)

    def predict_proba(self, X):  # noqa: N802 - mirror sklearn API
        return np.array([[1.0 - self.p_pos, self.p_pos]])


def _bar(close: float = 100.0) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(_NS_PER_MIN),
        instrument=DEFAULT_INSTRUMENT,
        open=Price(close), high=Price(close * 1.01), low=Price(close * 0.99),
        close=Price(close), volume=Quantity(1.0),
        duration=Duration.from_minutes(20), segment_id=0,
    )


def _obs_with_features(feats: dict[str, float]) -> Observation:
    return Observation(bar=_bar(), features=FeatureMap(feats))


def _predictor_with_models(model_probs: list[float], feature_cols: list[str]) -> FrozenCatBoostPredictor:
    """Build a predictor with fake in-memory models — bypasses disk I/O."""
    pred = FrozenCatBoostPredictor(
        model_path="unused.cbm",
        feature_columns=feature_cols,
        ensemble_n=len(model_probs),
    )
    pred._models = [_FakeModel(p, feature_cols) for p in model_probs]
    pred._loaded = True
    return pred


# ---- Tests ----------------------------------------------------------------


def test_sigma_ve_zero_for_single_member_ensemble() -> None:
    pred = _predictor_with_models([0.7], ["f1", "f2"])
    obs = pred.transform(_obs_with_features({"f1": 0.5, "f2": -1.0}))
    assert obs.sigma_ve == pytest.approx(0.0, abs=1e-12)
    assert obs.p_offline == pytest.approx(0.7)


def test_sigma_ve_positive_for_disagreeing_ensemble() -> None:
    ps = [0.2, 0.5, 0.8]
    pred = _predictor_with_models(ps, ["f1"])
    obs = pred.transform(_obs_with_features({"f1": 0.0}))
    assert obs.sigma_ve > 0.0
    assert obs.p_offline == pytest.approx(np.mean(ps))


def test_sigma_ve_matches_numpy_population_std() -> None:
    """sigma_ve must equal numpy.std(ps, ddof=0) to 1e-9 for arbitrary ensembles."""
    cases = [
        [0.1, 0.9],
        [0.25, 0.5, 0.75],
        [0.05, 0.15, 0.45, 0.55, 0.95],
        [0.5] * 5,            # zero-spread degenerate
        [0.0, 1.0],           # max spread
    ]
    for ps in cases:
        pred = _predictor_with_models(ps, ["f1"])
        obs = pred.transform(_obs_with_features({"f1": 0.0}))
        expected = float(np.std(np.array(ps), ddof=0))
        assert obs.sigma_ve == pytest.approx(expected, abs=1e-9), (
            f"ps={ps}: got {obs.sigma_ve}, expected {expected}"
        )


def test_sigma_ve_bounded_by_half() -> None:
    """std of values in [0,1] cannot exceed 0.5 (achieved at extremes 0 and 1)."""
    extreme_cases = [
        [0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0] * 3 + [1.0] * 3,
    ]
    for ps in extreme_cases:
        pred = _predictor_with_models(ps, ["f1"])
        obs = pred.transform(_obs_with_features({"f1": 0.0}))
        assert obs.sigma_ve is not None
        assert 0.0 <= obs.sigma_ve <= 0.5 + 1e-9


def test_sigma_ve_zero_when_all_members_agree() -> None:
    pred = _predictor_with_models([0.42, 0.42, 0.42, 0.42], ["f1"])
    obs = pred.transform(_obs_with_features({"f1": 0.0}))
    assert obs.sigma_ve == pytest.approx(0.0, abs=1e-12)
    assert obs.p_offline == pytest.approx(0.42)


def test_transform_sets_both_sigma_ve_and_p_offline_no_regression() -> None:
    """No regression: p_offline still set; new sigma_ve attribute also populated."""
    ps = [0.3, 0.6]
    pred = _predictor_with_models(ps, ["a", "b"])
    obs = pred.transform(_obs_with_features({"a": 1.0, "b": 2.0}))
    # p_offline behaviour preserved.
    assert obs.p_offline is not None
    assert float(obs.p_offline) == pytest.approx(np.mean(ps))
    # New sigma_ve attribute set.
    assert obs.sigma_ve is not None
    assert obs.sigma_ve == pytest.approx(float(np.std(np.array(ps), ddof=0)), abs=1e-9)
