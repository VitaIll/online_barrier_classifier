"""wagie.cv + wagie.cscv — purged CV walker, CSCV PBO, drift-aware wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from wagie.config import WagieConfig
from wagie.cscv import cscv_pbo
from wagie.cv import CrossValResult, DriftAwarePipeline, cross_validation


def test_cscv_pbo_returns_value_in_unit_interval():
    """Random fold returns: PBO must be in [0, 1]; payload keys are stable."""
    rng = np.random.default_rng(0)
    R = rng.normal(0.0, 1.0, size=(8, 256))
    out = cscv_pbo(R, n_chunks=8)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["n_chunks"] == 8
    assert out["n_strategies"] == 8
    assert "median_logit" in out
    assert "is_best_strategy_modal_index" in out


def test_cscv_pbo_validates_input_shape():
    with pytest.raises(ValueError, match="2D"):
        cscv_pbo(np.zeros(8))
    with pytest.raises(ValueError, match=">= 2 strategies"):
        cscv_pbo(np.zeros((1, 64)))
    with pytest.raises(ValueError, match="even"):
        cscv_pbo(np.zeros((4, 64)), n_chunks=3)
    with pytest.raises(ValueError, match="too small"):
        cscv_pbo(np.zeros((4, 4)), n_chunks=8)


def test_cross_val_result_summary_is_non_empty_string():
    r = CrossValResult(
        per_fold=[
            {"fold": 0, "n_trades": 10, "sharpe": 0.5},
            {"fold": 1, "n_trades": 8, "sharpe": -0.2},
        ],
        pbo=0.42,
        n_folds=2,
        state_hashes=["abc", "def"],
    )
    s = r.summary()
    assert isinstance(s, str) and s
    assert "n_folds=2" in s
    assert "PBO" in s
    assert "0.500" in s or "+0.500" in s


def test_drift_aware_pipeline_forwards_transform_and_learn(synthetic_minute_parquet):
    """Composition wrapper forwards transform_one + learn_one to the base."""

    class _FakeStage:
        kind = None

    class _FakePipeline:
        def __init__(self):
            self.transformed: list = []
            self.learned: list = []
            self.stages = []

        def transform_one(self, obs, ctx=None):
            self.transformed.append(obs)
            return obs

        def learn_one(self, obs, label=None):
            self.learned.append((obs, label))

        def state_dict(self):
            return {"base": True}

        def state_hash(self):
            return b"\x00" * 32

    class _FakeDrift:
        def __init__(self):
            self.drift_detected = False
            self.updates = 0

        def update(self, x):
            self.updates += 1

    base = _FakePipeline()
    detector = _FakeDrift()
    wrapper = DriftAwarePipeline(base, detector, recalib_window=10)

    # transform_one passthrough.
    sentinel = object()
    out = wrapper.transform_one(sentinel)
    assert out is sentinel
    assert base.transformed == [sentinel]

    # learn_one with no in_set: no drift updates, no recalibration.
    class _Obs:
        in_set = {}

    wrapper.learn_one(_Obs(), label=1)
    assert wrapper._n_drifts == 0
    assert detector.updates == 0
    assert base.learned and base.learned[0][1] == 1


def test_drift_aware_pipeline_resets_mondrian_calibrator_on_drift():
    """When drift fires and a MondrianACICalibrator is in stages, it is reset."""
    from wagie.pipeline import MondrianACICalibrator

    aci = MondrianACICalibrator()
    reset_called = {"n": 0}
    real_reset = aci.reset

    def _spy_reset():
        reset_called["n"] += 1
        return real_reset()

    aci.reset = _spy_reset  # type: ignore[assignment]

    class _Base:
        stages = [aci]
        def transform_one(self, obs, ctx=None): return obs
        def learn_one(self, obs, label=None): return None
        def state_dict(self): return {}
        def state_hash(self): return b"\x00" * 32

    class _DriftAlways:
        drift_detected = True
        def update(self, x): return None

    class _Obs:
        in_set = {0.10: False}

    wrapper = DriftAwarePipeline(_Base(), _DriftAlways())
    wrapper.learn_one(_Obs(), label=1)
    assert wrapper._n_drifts == 1
    assert reset_called["n"] == 1


def test_drift_aware_pipeline_increments_on_drift(synthetic_minute_parquet):
    class _FakeBase:
        def __init__(self):
            self.stages = []

        def transform_one(self, obs, ctx=None):
            return obs

        def learn_one(self, obs, label=None):
            return None

        def state_dict(self):
            return {}

        def state_hash(self):
            return b"\x00" * 32

    class _DriftAlways:
        drift_detected = True

        def update(self, x):
            return None

    class _Obs:
        in_set = {0.10: False}

    wrapper = DriftAwarePipeline(_FakeBase(), _DriftAlways())
    wrapper.learn_one(_Obs(), label=1)
    assert wrapper._n_drifts == 1
    # state_dict augments with drift counter.
    sd = wrapper.state_dict()
    assert sd["__drift__"]["n_drifts"] == 1
    # state_hash forwards to the base.
    assert wrapper.state_hash() == b"\x00" * 32


@pytest.mark.gating
def test_cross_validation_runs_with_synthetic_parquet(synthetic_minute_parquet):
    """Full cross_validation walk on synthetic data — slow (~30s).

    We use small fold counts so the walk completes quickly. cross_validation
    is allowed to log and skip individual folds; we only require the overall
    walker to return a CrossValResult with a populated summary.
    """
    cfg = WagieConfig.model_validate(
        {
            "data": {"parquet_path": str(synthetic_minute_parquet), "m_minutes": 20},
            "model": {
                "catboost_path": None,
                "aci": {
                    "alphas": [0.05, 0.10, 0.20],
                    "gamma": 0.01,
                    "n_regimes": 3,
                    "q_init": 0.4,
                },
            },
            "strategy": {"kind": "pure_conformal", "alpha": 0.10},
            "runtime": {"warmup_samples": 5},
        }
    )
    # skfolio's CombinatorialPurgedCV requires n_test_folds >= 2 (its
    # constructor raises otherwise); brief asked for n_test_folds=1 but the
    # underlying impl rejects that, so we use the smallest legal value.
    res = cross_validation(
        cfg, n_folds=4, n_test_folds=2, embargo_size=2, purged_size=1
    )
    assert isinstance(res, CrossValResult)
    assert res.n_folds >= 1
    assert 0.0 <= res.pbo <= 1.0
    s = res.summary()
    assert "CrossValResult" in s
    assert "PBO" in s
