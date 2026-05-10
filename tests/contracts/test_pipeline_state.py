"""Contract tests for sealed Pipeline (state_dict, hash, reset, learn dispatch).

Mondrian-ACI is removed from the canonical pipeline; tests now cover the
ARF + LabelBuffer + ThresholdGate stack.

Error policy (audit-fix):
    learn_one used to silently swallow every exception via ``except Exception:
    pass`` — a model that stopped learning was invisible. The new contract:
    only ``(TypeError, AttributeError)`` from a missing ``label=`` kwarg is
    tolerated; every other exception is wrapped in
    :class:`wagie.core.errors.StageError` and re-raised so the engine can
    count it on ``EngineResult.n_stage_errors``.
"""

from __future__ import annotations

import math

import pytest

from wagie.core.errors import StageError
from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.core.time import Duration, Timestamp
from wagie.pipeline import (
    STAGE_KINDS,
    STAGE_REGISTRY,
    build_stage,
    list_stages,
    register_stage,
    stage_kind,
    unregister_stage,
)
from wagie.pipeline.label_buffer import LabelBuffer
from wagie.pipeline.online_arf import OnlineARFCorrector
from wagie.pipeline.sealed import Pipeline
from wagie.strategy import StrategyContext, ThresholdGate


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
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    strat = ThresholdGate(name="strategy", tau=0.5)
    return Pipeline([arf, lb, strat])


# ---------------------------------------------------------------------------
# state_dict round-trip preserves state_hash
# ---------------------------------------------------------------------------

def test_state_dict_round_trip_preserves_state_hash():
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    # Drive a few learns (no transforms — LabelBuffer.transform records
    # observations into a deque whose contents feed state_hash but are not
    # captured by state_dict; that's by design — the buffer is volatile).
    for _ in range(3):
        pipe.learn_one(obs, label=1)
    sd = pipe.state_dict()
    h_orig = pipe.state_hash()

    pipe2 = _pipeline_3stage()
    pipe2.load_state_dict(sd)
    # ARF n_seen is restored, label_buffer params match — hashes match.
    assert pipe2.state_hash() == h_orig


# ---------------------------------------------------------------------------
# state_hash changes after learn_one
# ---------------------------------------------------------------------------

def test_state_hash_changes_after_transform():
    """ARF.transform increments n_seen and the LabelBuffer records the
    observation — both feed the state_hash."""
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    h_before = pipe.state_hash()
    pipe.transform_one(obs)
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
    lb = LabelBuffer(alpha_label=0.001, max_buffer=4)
    strat = ThresholdGate(name="strategy", tau=0.5)
    p3 = Pipeline([arf, lb, strat])
    assert p3.n_stages == 3


# ---------------------------------------------------------------------------
# transform_one walks every stage in order
# ---------------------------------------------------------------------------

def test_transform_one_walks_every_stage_in_order():
    """Each stage's effect is observable in the final Observation:
       - ARF (corrector) wrote p_online,
       - ThresholdGate (strategy) wrote actions.
    """
    pipe = _pipeline_3stage()
    obs = _obs().with_features_dict({"f1": 1.0})
    out = pipe.transform_one(obs)

    # Corrector wrote (or kept) p_online
    assert out.p_online is not None
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

    strat = ThresholdGate(name="strat")
    stage = StageNoLabel()
    pipe = Pipeline([stage, strat])
    obs = _obs()
    # Should not raise
    pipe.learn_one(obs, label=1)
    assert stage.calls == 1


def test_learn_one_propagates_general_exceptions_as_stage_error():
    """Misbehaving stage.update() now raises StageError (visible failure mode).

    The old contract silently swallowed every exception — a model that
    stopped learning was invisible. The new contract: anything that isn't a
    signature mismatch propagates as :class:`StageError` so the engine can
    count it. See module docstring for context.
    """

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

    strat = ThresholdGate(name="strat")
    pipe = Pipeline([BoomStage(), strat])
    with pytest.raises(StageError) as excinfo:
        pipe.learn_one(_obs(), label=1)
    assert excinfo.value.stage_name == "boom"
    assert excinfo.value.op == "update"
    assert isinstance(excinfo.value.original, RuntimeError)
    # Context must include bar timestamp + label for postmortem.
    assert "bar_ts_ns" in excinfo.value.context
    assert excinfo.value.context.get("label") == 1
    # __cause__ chains for traceback completeness.
    assert isinstance(excinfo.value.__cause__, RuntimeError)


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

    strat = ThresholdGate(name="strat")
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

    strat = ThresholdGate(name="strat")
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


def test_learn_one_inner_exception_after_signature_retry_propagates():
    """Stage's update raises TypeError on first call (looks like signature
    mismatch), then a real RuntimeError on the no-label retry. The inner
    error is now wrapped as StageError instead of being silently dropped.
    """

    class DoubleBoom:
        name = "double_boom"
        kind = StageKind.FEATURE

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            # First call (with label kwarg) — raise TypeError so the
            # signature-mismatch retry path triggers.
            # Second call (without label) — raise a real error that must
            # NOT be silently swallowed.
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

    strat = ThresholdGate(name="strat")
    pipe = Pipeline([DoubleBoom(), strat])
    with pytest.raises(StageError) as excinfo:
        pipe.learn_one(_obs(), label=1)
    assert excinfo.value.stage_name == "double_boom"
    assert isinstance(excinfo.value.original, RuntimeError)


def test_learn_one_signature_only_failure_is_silent():
    """A stage whose update() simply lacks the ``label`` kwarg (and raises no
    real error) must not produce a StageError — that's a benign
    signature-mismatch, not a learning failure.
    """
    calls = {"n": 0}

    class StageNoLabel:
        name = "no_label"
        kind = StageKind.FEATURE

        def transform(self, obs):
            return obs

        def update(self, obs):  # no label kwarg
            calls["n"] += 1

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b"x"

        def reset(self):
            pass

    pipe = Pipeline([StageNoLabel(), PureConformalGate(name="strat")])
    pipe.learn_one(_obs(), label=1)  # must not raise
    assert calls["n"] == 1


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

    strat = ThresholdGate(name="strat")
    pipe = Pipeline([ResetBoom(), strat])
    pipe.reset()  # must not raise — log-and-continue


# ---------------------------------------------------------------------------
# StageError carries rich context (audit-fix)
# ---------------------------------------------------------------------------

def test_stage_error_carries_full_context_chain():
    """The StageError raised by learn_one carries: stage_name, op,
    original exception, bar timestamp, instrument, label.
    """

    class CtxStage:
        name = "ctx_stage"
        kind = StageKind.FEATURE

        def transform(self, obs):
            return obs

        def update(self, obs, label=None):
            raise ValueError("ctx test")

        def state_dict(self):
            return {}

        def load_state_dict(self, _):
            pass

        def state_hash(self):
            return b""

        def reset(self):
            pass

    pipe = Pipeline([CtxStage(), PureConformalGate(name="strat")])
    with pytest.raises(StageError) as ei:
        pipe.learn_one(_obs(), label=0)
    err = ei.value
    assert err.stage_name == "ctx_stage"
    assert err.op == "update"
    assert isinstance(err.original, ValueError)
    assert "bar_ts_ns" in err.context
    assert err.context["instrument"]
    assert err.context["label"] == 0
    # WagieError ancestor for catch-all clauses.
    from wagie.core.errors import WagieError
    assert isinstance(err, WagieError)


# ---------------------------------------------------------------------------
# STAGE_REGISTRY round-trip + auto-registration
# ---------------------------------------------------------------------------

def test_stage_registry_auto_registers_shipped_stages():
    """Importing ``wagie.pipeline`` registers the shipped concrete stages so
    third-party callers (and ARCH's build_pipeline) can look them up by name.
    """
    # Shipped stages auto-register; Mondrian-ACI is intentionally NOT registered
    # (sibling agent owns the conformal layer). Just check the canonical core ones.
    for name in ("frozen_catboost", "online_arf", "label_buffer"):
        assert name in STAGE_REGISTRY, (
            f"{name!r} must auto-register at import time; "
            f"registered={sorted(STAGE_REGISTRY)}"
        )
        assert name in STAGE_KINDS

    # StageKinds line up with the actual classes.
    assert stage_kind("frozen_catboost") == StageKind.PREDICTOR
    assert stage_kind("online_arf") == StageKind.CORRECTOR
    assert stage_kind("label_buffer") == StageKind.LABEL_BUFFER


def test_register_stage_round_trip_then_build():
    """register_stage + build_stage give back a usable Stage instance."""

    class ToyPredictor:
        name = "toy_predictor"
        kind = StageKind.PREDICTOR

        def __init__(self, label_value: float = 0.5):
            self.label_value = float(label_value)
            self._n_seen = 0

        def transform(self, obs):
            return obs.with_p_offline(Probability(self.label_value))

        def update(self, obs, label=None):
            self._n_seen += 1

        def state_dict(self):
            return {"n_seen": self._n_seen}

        def load_state_dict(self, s):
            self._n_seen = int(s.get("n_seen", 0))

        def state_hash(self):
            return f"toy:{self._n_seen}".encode()

        def reset(self):
            self._n_seen = 0

    register_stage("toy_predictor_test", ToyPredictor, kind=StageKind.PREDICTOR)
    try:
        assert "toy_predictor_test" in STAGE_REGISTRY
        assert stage_kind("toy_predictor_test") == StageKind.PREDICTOR

        stage = build_stage("toy_predictor_test", label_value=0.7)
        assert isinstance(stage, ToyPredictor)
        assert stage.label_value == 0.7

        # The built stage plays nicely in a real Pipeline.
        pipe = Pipeline([stage, PureConformalGate(name="strat", alpha=0.1)])
        out = pipe.transform_one(_obs())
        assert out.p_offline is not None and abs(float(out.p_offline) - 0.7) < 1e-9
    finally:
        unregister_stage("toy_predictor_test")
        assert "toy_predictor_test" not in STAGE_REGISTRY


def test_register_stage_rejects_duplicate_without_overwrite():
    register_stage("dup_stage", lambda: None, kind=StageKind.FEATURE)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_stage("dup_stage", lambda: None, kind=StageKind.FEATURE)
        # overwrite=True works.
        register_stage("dup_stage", lambda: None, kind=StageKind.FEATURE,
                       overwrite=True)
    finally:
        unregister_stage("dup_stage")


def test_register_stage_rejects_unknown_kind():
    with pytest.raises(TypeError, match="StageKind"):
        register_stage("no_kind", lambda: None)  # no kind attr, no kind kwarg


def test_register_stage_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        register_stage("not_callable", "not a factory",  # type: ignore[arg-type]
                       kind=StageKind.FEATURE)


def test_register_stage_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        register_stage("", lambda: None, kind=StageKind.FEATURE)


def test_build_stage_unknown_name_raises_keyerror():
    with pytest.raises(KeyError, match="unknown stage"):
        build_stage("definitely_not_registered")


def test_list_stages_filters_by_kind():
    names = list_stages(StageKind.PREDICTOR)
    assert "frozen_catboost" in names
    assert "online_arf" not in names  # CORRECTOR, not PREDICTOR


def test_unregister_stage_idempotent():
    unregister_stage("never_registered")  # no-op; must not raise


def test_register_stage_returns_factory_for_chaining():
    """register_stage returns the factory unchanged so callers can chain
    (e.g. assign-and-register in one expression).
    """
    def my_factory():
        return None

    out = register_stage("chain_form_test", my_factory, kind=StageKind.FEATURE)
    try:
        assert out is my_factory
    finally:
        unregister_stage("chain_form_test")
