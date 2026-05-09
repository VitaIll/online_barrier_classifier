"""I8: sealed Pipeline rejects invalid stack ordering and enforces invariants."""

from __future__ import annotations

import pytest

from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.features import BaseBarFeatures
from wagie.pipeline import LabelBuffer, MondrianACICalibrator, OnlineARFCorrector, Pipeline
from wagie.strategy import PureConformalGate


def test_pipeline_requires_strategy_at_end():
    with pytest.raises(ValueError, match="STRATEGY"):
        Pipeline([BaseBarFeatures()])


def test_pipeline_rejects_out_of_order_kinds():
    """Predictor before feature should fail (kind 2 then kind 1 is descending)."""
    arf = OnlineARFCorrector()
    bb = BaseBarFeatures()
    strat = PureConformalGate(name="s")
    with pytest.raises(ValueError, match="monotone"):
        Pipeline([arf, bb, strat])


def test_pipeline_accepts_valid_order():
    bb = BaseBarFeatures()
    arf = OnlineARFCorrector()
    aci = MondrianACICalibrator()
    lb = LabelBuffer()
    strat = PureConformalGate(name="s")
    p = Pipeline([bb, arf, aci, lb, strat])
    assert p.n_stages == 5


def test_pipeline_rejects_duplicate_names():
    a = BaseBarFeatures()
    b = BaseBarFeatures()  # same default name
    strat = PureConformalGate(name="s")
    with pytest.raises(ValueError, match="duplicate"):
        Pipeline([a, b, strat])
