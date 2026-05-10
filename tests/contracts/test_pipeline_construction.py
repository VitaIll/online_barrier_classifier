"""I8: sealed Pipeline rejects invalid stack ordering and enforces invariants.

The Mondrian-ACI calibrator is gone — the canonical pipeline is now:
    BaseBarFeatures | ARF | LabelBuffer | ThresholdGate
"""

from __future__ import annotations

import pytest

from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.features import BaseBarFeatures
from wagie.pipeline import LabelBuffer, OnlineARFCorrector, Pipeline
from wagie.strategy import ThresholdGate


def test_pipeline_requires_strategy_at_end():
    with pytest.raises(ValueError, match="STRATEGY"):
        Pipeline([BaseBarFeatures()])


def test_pipeline_rejects_out_of_order_kinds():
    """Predictor before feature should fail (kind 2 then kind 1 is descending)."""
    arf = OnlineARFCorrector()
    bb = BaseBarFeatures()
    strat = ThresholdGate(name="s")
    with pytest.raises(ValueError, match="monotone"):
        Pipeline([arf, bb, strat])


def test_pipeline_accepts_valid_order_no_calibrator():
    bb = BaseBarFeatures()
    arf = OnlineARFCorrector()
    lb = LabelBuffer()
    strat = ThresholdGate(name="s")
    p = Pipeline([bb, arf, lb, strat])
    assert p.n_stages == 4


def test_pipeline_rejects_duplicate_names():
    a = BaseBarFeatures()
    b = BaseBarFeatures()  # same default name
    strat = ThresholdGate(name="s")
    with pytest.raises(ValueError, match="duplicate"):
        Pipeline([a, b, strat])
