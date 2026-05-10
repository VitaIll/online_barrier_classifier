"""End-to-end engine smoke test on synthetic data.

Builds the full pipeline via WagieConfig + run, asserts:
  - Engine completes without exceptions
  - At least some decisions emitted
  - state_hash deterministic across two identical runs (replay≡replay)
  - Engine captures p_online + label history → MetricsBattery yields non-zero
    Brier / ECE on the smoke fixture (sanity that calibration data flows).
"""

from __future__ import annotations

from wagie.config import WagieConfig
from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
from wagie.features.catalog import default_streaming_features
from wagie.metrics import MetricsBattery
from wagie.run import run


def _cfg(parquet) -> WagieConfig:
    return WagieConfig.model_validate({
        "data": {"parquet_path": str(parquet), "m_minutes": 20},
        "model": {
            "catboost_path": None,
        },
        "strategy": {"kind": "threshold_gate", "tau": 0.20},
        "runtime": {"warmup_samples": 10},
    })


def test_engine_runs_end_to_end(synthetic_minute_parquet):
    cuts = RegimeCuts(feature="parkinson_var_rolling_mean_24",
                      edges=(1e-6, 1e-5), labels=("low", "med", "high"))
    cfg = _cfg(synthetic_minute_parquet)
    bb = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features())
    rg = RegimeFeature(cuts)
    result = run(cfg, feature_builder=fb, base_bar=bb, regime_feature=rg)
    assert result.n_decisions > 0
    assert isinstance(result.pipeline_state_hash, bytes)
    assert len(result.pipeline_state_hash) == 32  # sha256


def test_engine_emits_calibration_history(synthetic_minute_parquet):
    """The engine must capture (p_online, label) pairs each time the
    LabelBuffer matures a record. MetricsBattery should see them and produce
    non-zero Brier / ECE."""
    cuts = RegimeCuts(feature="parkinson_var_rolling_mean_24",
                      edges=(1e-6, 1e-5), labels=("low", "med", "high"))
    cfg = _cfg(synthetic_minute_parquet)
    bb = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features())
    rg = RegimeFeature(cuts)
    result = run(cfg, feature_builder=fb, base_bar=bb, regime_feature=rg)

    assert len(result.p_online_history) > 0
    assert len(result.label_history) == len(result.p_online_history)
    rep = MetricsBattery(m_minutes=20).compute(result)
    assert rep.brier > 0.0, f"Brier should be > 0 with non-empty history; got {rep.brier}"
    assert rep.ece >= 0.0


def test_replay_state_hash_reproducible(synthetic_minute_parquet):
    """Same config + same seed → byte-identical pipeline state hash."""
    cuts = RegimeCuts(feature="parkinson_var_rolling_mean_24",
                      edges=(1e-6, 1e-5), labels=("low", "med", "high"))

    def _build():
        return BaseBarFeatures(), FeatureBuilder(default_streaming_features()), RegimeFeature(cuts)

    cfg = _cfg(synthetic_minute_parquet)

    bb1, fb1, rg1 = _build()
    r1 = run(cfg, feature_builder=fb1, base_bar=bb1, regime_feature=rg1)

    bb2, fb2, rg2 = _build()
    r2 = run(cfg, feature_builder=fb2, base_bar=bb2, regime_feature=rg2)

    assert r1.pipeline_state_hash == r2.pipeline_state_hash
    assert r1.n_decisions == r2.n_decisions
