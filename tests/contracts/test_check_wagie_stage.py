"""I13 — every concrete Stage passes the check_wagie_stage battery."""

from __future__ import annotations

import pytest

from wagie.checks import check_wagie_stage
from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
from wagie.features.catalog import default_streaming_features
from wagie.pipeline import LabelBuffer, OnlineARFCorrector
from wagie.strategy import EvCalibratedSize, ThresholdGate


def _stages():
    return [
        BaseBarFeatures(),
        FeatureBuilder(default_streaming_features()[:5]),  # small sample
        RegimeFeature(RegimeCuts(feature="x", edges=(1.0, 2.0), labels=("a", "b", "c"))),
        OnlineARFCorrector(),
        LabelBuffer(),
        ThresholdGate(name="threshold"),
        EvCalibratedSize(name="ev"),
    ]


@pytest.mark.parametrize("stage", _stages(), ids=lambda s: type(s).__name__)
def test_stage_passes_check_battery(stage):
    results = check_wagie_stage(stage)
    failures = {k: v for k, v in results.items() if v != "ok"}
    assert not failures, f"{type(stage).__name__} failed checks: {failures}"
