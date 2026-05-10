"""Pipeline stages.

The sealed Pipeline (in pipeline.sealed) walks a tuple of Stage-protocol
objects, validating monotone StageKind ordering at construction.

Stage registry
--------------

``build_pipeline`` in :mod:`wagie.run` used to be a hard-coded ladder of
concrete classes — adding a 2nd predictor required editing run.py, the
WagieConfig schema, and pyproject. This module provides a name -> factory
registry so third-party stages can plug in without touching the harness:

    from wagie.pipeline import register_stage, build_stage, STAGE_REGISTRY
    register_stage("my_predictor", MyPredictor)
    stage = build_stage("my_predictor", **kwargs)

The shipped predictors / correctors / calibrators / label-buffers
auto-register at import time. ``STAGE_REGISTRY`` is the inspectable map
keyed by string name; ``STAGE_KINDS`` maps the same name to its
:class:`StageKind` so ARCH's ``build_pipeline`` (or any third-party
ladder) can sort by kind.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from wagie.core.pipeline import Stage, StageKind
from wagie.pipeline.catboost_predictor import FrozenCatBoostPredictor
from wagie.pipeline.label_buffer import LabelBuffer
from wagie.pipeline.online_arf import OnlineARFCorrector
from wagie.pipeline.sealed import Pipeline


# A stage factory: callable(**kwargs) -> Stage instance.
StageFactory = Callable[..., Stage]


# name -> factory
STAGE_REGISTRY: dict[str, StageFactory] = {}
# name -> StageKind (for sorting / validation by callers)
STAGE_KINDS: dict[str, StageKind] = {}


def register_stage(
    name: str,
    factory: StageFactory,
    *,
    kind: Optional[StageKind] = None,
    overwrite: bool = False,
) -> StageFactory:
    """Register a Stage factory under ``name``.

    Args:
        name: registry key used by ``build_stage``.
        factory: callable returning an object satisfying the Stage Protocol.
        kind: explicit StageKind. If None, attempt to read ``factory.kind``
              (works for the shipped concrete classes — they expose ``kind``
              as a class attribute).
        overwrite: if False (default) raise on duplicate registration.

    Returns the factory unchanged so this can be used as a decorator:

        @register_stage("my_predictor", kind=StageKind.PREDICTOR)
        class MyPredictor: ...
    """
    if not name:
        raise ValueError("register_stage: name must be non-empty")
    if not callable(factory):
        raise TypeError(f"register_stage({name!r}): factory must be callable")
    if name in STAGE_REGISTRY and not overwrite:
        raise ValueError(
            f"register_stage({name!r}): already registered; "
            f"pass overwrite=True to replace"
        )
    if kind is None:
        kind = getattr(factory, "kind", None)
    if not isinstance(kind, StageKind):
        raise TypeError(
            f"register_stage({name!r}): could not infer StageKind. "
            f"Pass kind=StageKind.X or expose a `kind` class attribute."
        )
    STAGE_REGISTRY[name] = factory
    STAGE_KINDS[name] = kind
    return factory


def unregister_stage(name: str) -> None:
    """Remove a stage from the registry. No-op if not present.

    Useful in tests and when swapping implementations.
    """
    STAGE_REGISTRY.pop(name, None)
    STAGE_KINDS.pop(name, None)


def build_stage(name: str, **kwargs: Any) -> Stage:
    """Construct a stage by registry name.

    Raises ``KeyError`` if the name is not registered.
    """
    if name not in STAGE_REGISTRY:
        valid = sorted(STAGE_REGISTRY.keys())
        raise KeyError(
            f"build_stage: unknown stage {name!r}; registered: {valid}"
        )
    return STAGE_REGISTRY[name](**kwargs)


def stage_kind(name: str) -> StageKind:
    """Look up the StageKind for a registered name."""
    if name not in STAGE_KINDS:
        raise KeyError(f"stage_kind: unknown stage {name!r}")
    return STAGE_KINDS[name]


def list_stages(kind: Optional[StageKind] = None) -> list[str]:
    """List registered stage names, optionally filtered by kind."""
    if kind is None:
        return sorted(STAGE_REGISTRY)
    return sorted(n for n, k in STAGE_KINDS.items() if k == kind)


# --- Auto-register the shipped concrete stages -------------------------------
# These are the same classes ``wagie.run.build_pipeline`` instantiates today;
# registering them keeps existing call sites unchanged while opening the door
# to plugin predictors. The historic Mondrian-ACI calibrator was removed in
# the conformal-layer simplification — strategies now gate on the calibrated
# ``p_online`` from OnlineARFCorrector directly, so no calibration stage is
# registered.

register_stage(
    "frozen_catboost",
    FrozenCatBoostPredictor,
    kind=StageKind.PREDICTOR,
)
register_stage(
    "online_arf",
    OnlineARFCorrector,
    kind=StageKind.CORRECTOR,
)
register_stage(
    "label_buffer",
    LabelBuffer,
    kind=StageKind.LABEL_BUFFER,
)


__all__ = [
    "Pipeline",
    "FrozenCatBoostPredictor",
    "OnlineARFCorrector",
    "LabelBuffer",
    # Stage registry surface
    "STAGE_REGISTRY",
    "STAGE_KINDS",
    "register_stage",
    "unregister_stage",
    "build_stage",
    "stage_kind",
    "list_stages",
    "StageFactory",
]
