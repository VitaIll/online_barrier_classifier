"""Contract tests for sealed Pipeline (state_dict, hash, reset, learn dispatch)."""

from __future__ import annotations

import math

import pytest

from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.core.time import Duration, Timestamp
from wagie.pipeline.label_buffer import LabelBuffer
from wagie.pipeline.mondrian_aci import MondrianACICalibrator
from wagie.pipeline.online_arf import OnlineARFCorrector
from wagie.pipeline.sealed import Pipeline
from wagie.strategy import PureConformalGate, StrategyContext


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


def _obs() -> Observation:
    return (Observation(bar=_bar())
            .with_p_offline(Probability(0.6))
            .with_p_online(Probability(0.55))
            .with_regime(0))


def _pipeline_3stage() -> Pipeline:
    arf = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    aci = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    strat = PureConformalGate(name="strategy", alpha=0.1)
    return Pipeline([arf, aci, strat])


# ---------------------------------------------------------------------------
# state_dict round-trip preserves state_hash
# ---------------------------------------------------------------------------

def test_state_dict_round_trip_preserves_state_hash():
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    # Drive a few transforms + learns to mutate state.
    for _ in range(3):
        pipe.transform_one(obs)
        pipe.learn_one(obs, label=1)
    sd = pipe.state_dict()
    h_orig = pipe.state_hash()

    pipe2 = _pipeline_3stage()
    pipe2.load_state_dict(sd)
    # ACI hash is fully captured; ARF-internals are NOT in state_hash, only
    # (n_seen, seed, n_models, lambda_value, max_features). Since load_state_dict
    # restores n_seen on the corrector, hashes match.
    assert pipe2.state_hash() == h_orig


# ---------------------------------------------------------------------------
# state_hash changes after learn_one
# ---------------------------------------------------------------------------

def test_state_hash_changes_after_learn_one():
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    # First a transform so calibrator has p_online valid + ARF n_seen advances.
    pipe.transform_one(obs)
    h_before = pipe.state_hash()
    # learn_one drives the calibrator's update (q changes).
    pipe.learn_one(obs, label=1)
    h_after = pipe.state_hash()
    assert h_before != h_after


# ---------------------------------------------------------------------------
# reset() restores all stages to fresh state
# ---------------------------------------------------------------------------

def test_reset_restores_all_stages_to_fresh_state():
    pipe = _pipeline_3stage()
    fresh_hash = _pipeline_3stage().state_hash()

    obs = _obs().with_features_dict({"f1": 1.0})
    for _ in range(5):
        pipe.transform_one(obs)
        pipe.learn_one(obs, label=1)
    assert pipe.state_hash() != fresh_hash

    pipe.reset()
    assert pipe.state_hash() == fresh_hash


# ---------------------------------------------------------------------------
# n_stages property
# ---------------------------------------------------------------------------

def test_n_stages_property():
    pipe = _pipeline_3stage()
    assert pipe.n_stages == 3

    arf = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    aci = MondrianACICalibrator(alphas=(0.1,), gamma=0.05, n_regimes=1, q_init=0.5)
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    strat = PureConformalGate(name="strategy", alpha=0.1)
    p4 = Pipeline([arf, aci, lb, strat])
    assert p4.n_stages == 4


# ---------------------------------------------------------------------------
# transform_one walks every stage in order
# ---------------------------------------------------------------------------

def test_transform_one_walks_every_stage_in_order():
    """Each stage's effect is observable in the final Observation:
       - ARF (corrector) wrote p_online,
       - ACI (calibrator) wrote q_lo and in_set,
       - PureConformalGate (strategy) wrote actions.
    """
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    out = pipe.transform_one(obs)

    # Corrector wrote (or kept) p_online
    assert out.p_online is not None
    # Calibrator wrote q_lo & in_set for alpha=0.1
    assert 0.1 in out.q_lo
    assert 0.1 in out.in_set
    # Strategy wrote actions (tuple, possibly empty)
    assert isinstance(out.actions, tuple)


def test_transform_one_default_ctx_when_none_provided():
    """If ctx=None, transform_one constructs a default StrategyContext."""
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    out = pipe.transform_one(obs, ctx=None)
    assert out is not None


def test_transform_one_with_explicit_ctx():
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    ctx = StrategyContext()
    out = pipe.transform_one(obs, ctx=ctx)
    assert out is not None


# ---------------------------------------------------------------------------
# learn_one swallows TypeError and other exceptions
# ---------------------------------------------------------------------------

def test_learn_one_swallows_typeerror_when_update_no_label_kwarg():
    """If a stage's update() doesn't accept `label`, learn_one falls back to
    update(obs) without exception."""

    class StageNoLabel:
        name = "no_label"
        kind = StageKind.FEATURE

        def __init__(self):
            self.calls = 0

        def transform(self, obs):
            return obs

        def update(self, obs):  # no label kwarg
            self.calls += 1

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b"x"

        def reset(self):
            pass

    strat = PureConformalGate(name="strat")
    stage = StageNoLabel()
    pipe = Pipeline([stage, strat])
    obs = _obs()
    # Should not raise
    pipe.learn_one(obs, label=1)
    assert stage.calls == 1


def test_learn_one_swallows_general_exceptions():
    """Misbehaving stage.update() must not break the pipeline."""

    class BoomStage:
        name = "boom"
        kind = StageKind.FEATURE

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            raise RuntimeError("simulated stage failure")

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b"b"

        def reset(self):
            pass

    strat = PureConformalGate(name="strat")
    pipe = Pipeline([BoomStage(), strat])
    pipe.learn_one(_obs(), label=1)  # must not raise


# ---------------------------------------------------------------------------
# Validation of construction
# ---------------------------------------------------------------------------

def test_empty_pipeline_rejected():
    with pytest.raises(ValueError, match="at least one stage"):
        Pipeline([])


def test_stage_without_kind_rejected():
    class NoKind:
        name = "x"
        # missing kind

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b""

        def reset(self):
            pass

    with pytest.raises(TypeError, match="StageKind"):
        Pipeline([NoKind()])


def test_load_state_dict_skips_unknown_stage_names():
    pipe = _pipeline_3stage()
    pipe.load_state_dict({"unknown_stage_name": {"foo": "bar"}})
    # Should not raise; existing stages untouched.


def test_state_hash_recovers_when_stage_state_hash_throws():
    """If a stage's state_hash() raises, Pipeline.state_hash falls back to repr."""

    class FlakyStage:
        name = "flaky"
        kind = StageKind.FEATURE

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            raise RuntimeError("flaky")

        def reset(self):
            pass

    strat = PureConformalGate(name="strat")
    pipe = Pipeline([FlakyStage(), strat])
    h = pipe.state_hash()
    assert isinstance(h, bytes) and len(h) == 32  # sha256 digest


def test_pipeline_must_end_with_strategy():
    """Last stage must be STRATEGY (line 63)."""
    arf = OnlineARFCorrector(n_models=3, seed=7, selected_features=["f1"])
    with pytest.raises(ValueError, match="STRATEGY"):
        Pipeline([arf])


def test_stage_without_name_uses_class_name():
    """Stage with name=None uses type(s).__name__ (line 58)."""

    class NoNameStage:
        kind = StageKind.FEATURE
        name = None  # explicit None triggers fallback

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b""

        def reset(self):
            pass

    strat = PureConformalGate(name="strat")
    pipe = Pipeline([NoNameStage(), strat])  # must not raise
    # The fallback name was used internally; pipeline accepted it.
    assert pipe.n_stages == 2


def test_strategy_stage_without_decide_uses_transform():
    """Strategy stage missing .decide falls back to .transform (line 80)."""

    class TransformOnlyStrategy:
        name = "to_strat"
        kind = StageKind.STRATEGY

        def transform(self, obs):
            return obs.with_actions(())

        def update(self, obs, label=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b""

        def reset(self):
            pass

    pipe = Pipeline([TransformOnlyStrategy()])
    out = pipe.transform_one(_obs())
    assert out.actions == ()


def test_learn_one_inner_exception_swallowed():
    """update(obs, label=…) raises TypeError; update(obs) then raises something
    else — both must be swallowed (lines 94-95)."""

    class DoubleBoom:
        name = "double_boom"
        kind = StageKind.FEATURE

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            # First call (with label kwarg) — raise TypeError to trigger fallback.
            # Second call (without label) — raise Exception to hit the inner handler.
            if label is None:
                raise RuntimeError("inner boom")
            raise TypeError("no label kwarg")

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b""

        def reset(self):
            pass

    strat = PureConformalGate(name="strat")
    pipe = Pipeline([DoubleBoom(), strat])
    pipe.learn_one(_obs(), label=1)  # must not raise


def test_reset_swallows_stage_reset_exceptions():
    class ResetBoom:
        name = "reset_boom"
        kind = StageKind.FEATURE

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b"r"

        def reset(self):
            raise RuntimeError("nope")

    strat = PureConformalGate(name="strat")
    pipe = Pipeline([ResetBoom(), strat])
    pipe.reset()  # must not raise
