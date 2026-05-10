"""Pipeline stages.

The sealed Pipeline (in pipeline.sealed) walks a tuple of Stage-protocol
objects, validating monotone StageKind ordering at construction.
"""

from wagie.pipeline.catboost_predictor import FrozenCatBoostPredictor
from wagie.pipeline.label_buffer import LabelBuffer
from wagie.pipeline.online_arf import OnlineARFCorrector
from wagie.pipeline.sealed import Pipeline

__all__ = [
    "Pipeline",
    "FrozenCatBoostPredictor",
    "OnlineARFCorrector",
    "LabelBuffer",
]
