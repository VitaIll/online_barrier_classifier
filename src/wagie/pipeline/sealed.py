"""Sealed Pipeline — validates stage ordering at construction.

Replaces the river.compose.Pipeline-as-everything design with a typed
container that:
    1. Walks an explicit list of Stage objects (each implements wagie.core.Stage)
    2. Validates monotone StageKind ordering at __init__
    3. Operates on Observation (NOT dict)
    4. Exposes state_dict / load_state_dict / state_hash uniformly
    5. Uses pattern-matched dispatch based on StageKind for predict-vs-learn

Error policy (audit-fix):
    Stage failures used to be swallowed by a bare ``except Exception: pass``
    inside ``learn_one`` / ``transform_one`` — a model that stopped learning
    was invisible. The new policy is targeted: ``TypeError`` /
    ``AttributeError`` from a missing ``label=`` kwarg are tolerated (the
    stage just doesn't consume labels); every other exception is wrapped in
    :class:`wagie.core.errors.StageError` and propagated, so the Engine can
    count it on ``EngineResult.n_stage_errors``.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from wagie.core.errors import StageError
from wagie.core.observation import Observation
from wagie.core.pipeline import Stage, StageKind
from wagie.strategy import StrategyContext


logger = logging.getLogger(__name__)


def _accepts_label_kwarg(fn: Callable) -> bool:
    """Best-effort: does ``fn`` accept ``label=`` (or **kwargs)?

    Used so we don't have to catch TypeError just to discover the signature.
    Returns True on inspection failure (caller falls back to try/except).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    params = sig.parameters
    if "label" in params:
        return True
    # **kwargs swallows everything.
    for p in params.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _stage_name(stage: Stage) -> str:
    return getattr(stage, "name", None) or type(stage).__name__


@dataclass
class Pipeline:
    """Sealed, validated pipeline of Stages operating on Observation.

    Construction validates:
        - all stages have a `kind: StageKind`
        - kinds are monotonically non-decreasing
        - last stage has kind == STRATEGY
        - no duplicate names
    """

    stages: tuple[Stage, ...]

    def __init__(self, stages: Iterable[Stage]):
        self.stages = tuple(stages)
        self._validate()

    def _validate(self) -> None:
        if not self.stages:
            raise ValueError("Pipeline requires at least one stage")
        seen_kinds = []
        seen_names = set()
        for s in self.stages:
            kind = getattr(s, "kind", None)
            if not isinstance(kind, StageKind):
                raise TypeError(f"stage {s!r} has no StageKind `kind` attribute")
            if seen_kinds and kind.value < seen_kinds[-1].value:
                raise ValueError(
                    f"stage ordering violates monotone kind: "
                    f"{seen_kinds[-1].name} (kind={seen_kinds[-1].value}) "
                    f"followed by {kind.name} (kind={kind.value})"
                )
            seen_kinds.append(kind)
            name = getattr(s, "name", None)
            if name is None:
                name = type(s).__name__
            if name in seen_names:
                raise ValueError(f"duplicate stage name: {name!r}")
            seen_names.add(name)
        if self.stages[-1].kind != StageKind.STRATEGY:
            raise ValueError(
                f"Pipeline must end with a STRATEGY stage; got {self.stages[-1].kind}"
            )

    # -- transform / learn ---------------------------------------------------

    def transform_one(self, obs: Observation, ctx: Optional[StrategyContext] = None) -> Observation:
        """Walk the stack. Each non-strategy stage transforms; the strategy
        consumes (obs, ctx) and writes Sequence[Action] into obs.

        Error policy (audit-fix): a stage's transform/decide failure is
        wrapped in :class:`StageError` and re-raised. The Engine catches it
        and counts on ``EngineResult.n_stage_errors``. The legacy bare
        ``except Exception: pass`` in the engine is gone.
        """
        if ctx is None:
            ctx = StrategyContext()
        for stage in self.stages:
            stage_name = _stage_name(stage)
            try:
                if stage.kind == StageKind.STRATEGY:
                    if hasattr(stage, "decide"):
                        actions = stage.decide(obs, ctx)
                        obs = obs.with_actions(actions)
                    else:
                        obs = stage.transform(obs)
                else:
                    obs = stage.transform(obs)
            except StageError:
                raise  # already wrapped by an inner pipeline; pass through
            except Exception as exc:
                op = "decide" if (
                    stage.kind == StageKind.STRATEGY and hasattr(stage, "decide")
                ) else "transform"
                raise StageError(
                    stage_name=stage_name,
                    op=op,
                    original=exc,
                    context=self._stage_ctx(obs),
                ) from exc
        return obs

    def learn_one(self, obs: Observation, label: Optional[int] = None) -> None:
        """Drive learning on stages that consume labels (calibrator, online).

        Targeted error policy (replaces silent ``except Exception: pass``):

        - Inspect each stage's ``update`` signature; call with or without
          ``label=`` accordingly. This avoids using TypeError as control flow.
        - If we still get ``TypeError`` / ``AttributeError`` (legacy stage
          with an opaque signature, or a Stage Protocol not implementing
          update), retry without ``label`` then silently move on — these
          really are signature-mismatch noise, not learning failures.
        - **Any other exception is wrapped in StageError and re-raised**
          so the engine can count it (EngineResult.n_stage_errors) and so
          a model that stops learning is no longer invisible.
        """
        for stage in self.stages:
            stage_name = _stage_name(stage)
            update_fn = getattr(stage, "update", None)
            if update_fn is None:
                continue  # not a Stateful stage at all

            try:
                if _accepts_label_kwarg(update_fn):
                    update_fn(obs, label=label)
                else:
                    update_fn(obs)
            except (TypeError, AttributeError) as sig_exc:
                # Possibly a stale signature inspection — retry without label.
                try:
                    update_fn(obs)
                except (TypeError, AttributeError):
                    # Stage genuinely doesn't accept this calling convention.
                    logger.debug(
                        "stage %s.update() does not accept the call: %s",
                        stage_name, sig_exc,
                    )
                except Exception as inner_exc:
                    raise StageError(
                        stage_name=stage_name,
                        op="update",
                        original=inner_exc,
                        context=self._stage_ctx(obs, label=label),
                    ) from inner_exc
            except StageError:
                raise  # already wrapped; pass through
            except Exception as exc:
                raise StageError(
                    stage_name=stage_name,
                    op="update",
                    original=exc,
                    context=self._stage_ctx(obs, label=label),
                ) from exc

    @staticmethod
    def _stage_ctx(obs: Observation, *, label: Optional[int] = None) -> dict:
        """Compact context dict attached to every StageError raised here."""
        ctx: dict = {}
        try:
            ctx["bar_ts_ns"] = int(obs.bar.close_time.ns)
        except Exception:
            pass
        try:
            ctx["instrument"] = str(obs.bar.instrument)
        except Exception:
            pass
        if label is not None:
            ctx["label"] = int(label) if isinstance(label, (int, bool)) else repr(label)
        return ctx

    # -- Stateful --------------------------------------------------------------

    def state_dict(self) -> dict:
        return {s.name: s.state_dict() for s in self.stages}

    def load_state_dict(self, state: dict) -> None:
        for s in self.stages:
            if s.name in state:
                s.load_state_dict(state[s.name])

    def state_hash(self) -> bytes:
        """SHA-256 over (stage_name, stage_state_hash) tuples.

        If a stage's ``state_hash()`` raises, fall back to ``repr(stage)`` —
        deterministic but coarser. The exception is logged (not silenced) so
        the operator can see the degradation.
        """
        h = hashlib.sha256()
        h.update(b"Pipeline|")
        for s in self.stages:
            name = _stage_name(s)
            h.update(name.encode())
            h.update(b"=")
            try:
                h.update(s.state_hash())
            except Exception as exc:
                logger.warning(
                    "stage %s.state_hash() raised %s: %s; falling back to repr",
                    name, type(exc).__name__, exc,
                )
                h.update(repr(s).encode())
            h.update(b"|")
        return h.digest()

    def reset(self) -> None:
        """Best-effort reset of every stage. A failing stage is logged (not
        silenced) and the loop continues — reset must not bubble per-stage
        failures because callers (CV folds) need a fresh pipeline regardless.
        """
        for s in self.stages:
            try:
                s.reset()
            except Exception as exc:
                logger.warning(
                    "stage %s.reset() raised %s: %s",
                    _stage_name(s), type(exc).__name__, exc,
                )

    @property
    def n_stages(self) -> int:
        return len(self.stages)


__all__ = ["Pipeline"]
