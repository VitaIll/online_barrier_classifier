"""Contract tests for OnlineARFCorrector (river ARF wrapper, online learner)."""

from __future__ import annotations

import math

import pytest

from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.time import Duration, Timestamp
from wagie.pipeline.online_arf import OnlineARFCorrector


def _bar(close: float = 100.0, segment_id: int = 0, ts_ns: int = 1_000_000_000) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(ts_ns),
        instrument=DEFAULT_INSTRUMENT,
        open=Price(close),
        high=Price(close * 1.01),
        low=Price(close * 0.99),
        close=Price(close),
        volume=Quantity(1.0),
        duration=Duration.from_minutes(20),
        segment_id=segment_id,
    )


def _obs(
    p_offline: float | None = 0.55,
    features: dict | None = None,
    close: float = 100.0,
) -> Observation:
    o = Observation(bar=_bar(close=close))
    if features:
        o = o.with_features_dict(features)
    if p_offline is not None:
        o = o.with_p_offline(Probability(p_offline))
    return o


# ---------------------------------------------------------------------------
# Without features, predict_proba returns None or empty → p1 falls to 0.5
# ---------------------------------------------------------------------------

def test_predict_without_training_returns_obs_with_default_p_online():
    """Cold ARF returns empty proba → p1 defaults to 0.5 → with_p_online sets it.

    The src docstring claims 'without features → predict returns None', but the
    code path in transform() falls back to p1=0.5 when proba is empty/missing.
    Document actual behavior.
    """
    arf = OnlineARFCorrector(n_models=3, seed=42, selected_features=["f1"])
    o = _obs(p_offline=0.55, features={"f1": 1.0})
    out = arf.transform(o)
    # Cold: no learn_one yet, so predict_proba_one returns {} → p1=0.5.
    assert out.p_online is not None
    assert math.isclose(float(out.p_online), 0.5)


def test_predict_with_nan_feature_returns_obs_unchanged():
    """If any selected feature is NaN, transform skips ARF and returns obs as-is."""
    arf = OnlineARFCorrector(n_models=3, seed=42, selected_features=["f1"])
    o = _obs(p_offline=0.55, features={"f1": float("nan")})
    out = arf.transform(o)
    assert out.p_online is None  # not modified


# ---------------------------------------------------------------------------
# predict → learn cycle: model accepts (obs, label) and updates
# ---------------------------------------------------------------------------

def test_learn_cycle_updates_model():
    arf = OnlineARFCorrector(n_models=3, seed=42, selected_features=["f1", "f2"])
    o_pos = _obs(p_offline=0.7, features={"f1": 1.0, "f2": 0.5})
    o_neg = _obs(p_offline=0.3, features={"f1": -1.0, "f2": -0.5})
    # Train on a few positive/negative samples
    for _ in range(20):
        arf.transform(o_pos)
        arf.update(o_pos, label=1)
        arf.transform(o_neg)
        arf.update(o_neg, label=0)
    # After training, the ARF should have a nonzero state.
    assert arf._arf is not None
    assert arf.n_seen > 0


def test_update_label_none_is_noop():
    arf = OnlineARFCorrector(n_models=3, seed=42, selected_features=["f1"])
    o = _obs(features={"f1": 0.5})
    arf.update(o, label=None)
    # No model created (update() bails before _ensure_init when label is None)
    assert arf._arf is None


# ---------------------------------------------------------------------------
# Determinism under seed
# ---------------------------------------------------------------------------

def test_same_seed_same_predictions():
    """Two ARFs with identical seed and identical training stream produce identical
    p_online sequences."""
    feats = [{"f1": 1.0, "f2": 0.5}, {"f1": -0.5, "f2": -1.0}, {"f1": 0.2, "f2": 0.0}]
    labels = [1, 0, 1]

    def run() -> list[float]:
        arf = OnlineARFCorrector(n_models=3, seed=123, selected_features=["f1", "f2"])
        preds: list[float] = []
        for f, y in zip(feats * 5, labels * 5):
            o = _obs(p_offline=0.5, features=f)
            out = arf.transform(o)
            preds.append(float(out.p_online) if out.p_online is not None else float("nan"))
            arf.update(o, label=y)
        return preds

    preds1 = run()
    preds2 = run()
    assert preds1 == preds2


# ---------------------------------------------------------------------------
# selected_features mask drops other keys before passing to ARF
# ---------------------------------------------------------------------------

def test_selected_features_mask_includes_only_listed_keys_plus_p_offline():
    arf = OnlineARFCorrector(selected_features=["f1"])
    o = _obs(p_offline=0.55, features={"f1": 1.0, "f2": 99.0, "extra": 7.0})
    z = arf._z(o)
    assert "f1" in z
    assert "p_offline" in z
    assert z["p_offline"] == pytest.approx(0.55)
    # Other keys must be excluded.
    assert "f2" not in z
    assert "extra" not in z
    # Bar OHLCV (close, volume, etc.) also excluded by the mask.
    assert "close" not in z


def test_no_selected_features_passes_full_dict_minus_metadata():
    arf = OnlineARFCorrector(selected_features=None)
    o = _obs(p_offline=0.55, features={"f1": 1.0})
    z = arf._z(o)
    # All bar/feature/p_offline keys present; no underscore-prefixed or "decision".
    assert "f1" in z
    assert "p_offline" in z
    assert "close" in z
    for k in z:
        assert not k.startswith("_")
        assert k != "decision"


def test_z_handles_missing_selected_feature_with_nan():
    arf = OnlineARFCorrector(selected_features=["missing_key"])
    o = _obs(p_offline=0.55, features={"f1": 1.0})
    z = arf._z(o)
    assert math.isnan(z["missing_key"])


# ---------------------------------------------------------------------------
# state_hash, n_seen, reset
# ---------------------------------------------------------------------------

def test_state_hash_deterministic_for_same_state():
    arf1 = OnlineARFCorrector(n_models=3, seed=7)
    arf2 = OnlineARFCorrector(n_models=3, seed=7)
    assert arf1.state_hash() == arf2.state_hash()


def test_state_hash_changes_with_n_seen():
    """state_hash includes n_seen, so transform() should change it.

    NOTE: state_hash does NOT include the underlying ARF tree state — it only
    includes (n_models, lambda_value, seed, max_features, n_seen). So learn_one
    alone (which doesn't move n_seen) would not change the hash; only transform
    does. Documenting this gap.
    """
    arf = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    h0 = arf.state_hash()
    o = _obs(features={"f1": 1.0})
    arf.transform(o)
    h1 = arf.state_hash()
    assert h0 != h1
    # update() alone (no transform) does not affect state_hash, despite mutating ARF.
    arf2 = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    h_pre = arf2.state_hash()
    arf2.update(o, label=1)
    h_post = arf2.state_hash()
    # Documented as an observation: hash unchanged by update alone.
    assert h_pre == h_post


def test_n_seen_increments_on_transform():
    arf = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    o = _obs(features={"f1": 1.0})
    n0 = arf.n_seen
    arf.transform(o)
    arf.transform(o)
    assert arf.n_seen == n0 + 2


def test_state_dict_round_trip():
    arf = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    o = _obs(features={"f1": 1.0})
    for _ in range(5):
        arf.transform(o)
        arf.update(o, label=1)
    sd = arf.state_dict()
    assert sd["n_seen"] == 5
    arf2 = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    arf2.load_state_dict(sd)
    assert arf2.n_seen == 5


def test_reset_clears_arf_and_n_seen():
    arf = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    o = _obs(features={"f1": 1.0})
    arf.transform(o)
    arf.update(o, label=1)
    assert arf._arf is not None
    arf.reset()
    assert arf._arf is None
    assert arf.n_seen == 0


def test_transform_proba_exception_returns_obs():
    """If predict_proba_one raises, transform returns obs unchanged (defensive)."""
    arf = OnlineARFCorrector(n_models=3, seed=42, selected_features=["f1"])
    arf._ensure_init()
    # Stub the ARF to raise on predict_proba_one
    class _Raiser:
        def predict_proba_one(self, _):
            raise RuntimeError("simulated failure")
        def learn_one(self, *_a, **_k):
            return None

    arf._arf = _Raiser()  # type: ignore[assignment]
    o = _obs(p_offline=0.5, features={"f1": 1.0})
    out = arf.transform(o)
    assert out.p_online is None  # unchanged
