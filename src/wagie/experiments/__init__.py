"""wagie.experiments — the SINGLE experiment protocol.

There is exactly one way to run an experiment:

    spec = ExperimentSpec.from_yaml("experiments/baseline.yaml")
    ExperimentProtocol().run(spec)

Everything else (custom scripts, ad-hoc training loops, bespoke charting) is
strictly forbidden — extend the spec/protocol instead.
"""

from .protocol import ExperimentProtocol
from .result import ExperimentResult
from .spec import (
    ArtifactsSpec, ChartsSpec, CVSpec, ExperimentSpec,
    FeaturesSpec, ReportSpec, TrainingSpec,
)


__all__ = [
    "ExperimentProtocol", "ExperimentResult", "ExperimentSpec",
    "FeaturesSpec", "ChartsSpec", "ReportSpec", "ArtifactsSpec",
    "CVSpec", "TrainingSpec",
]
