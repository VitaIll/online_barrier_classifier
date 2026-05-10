"""wagie.core.errors — typed exceptions with rich context.

The wagie SW-architecture audit found that the engine and pipeline silently
swallowed every exception via bare ``except Exception: pass``. A model that
stops learning was invisible. This module replaces that with typed,
context-bearing exceptions:

    StageError(stage_name, op, original_exc)
        Wraps any failure inside a Pipeline stage's transform/update method.
    PipelineError
        Pipeline-level structural / construction failure.
    EngineError
        Engine-level invariant break (e.g. clock regression, label leakage).

These are raised, NOT swallowed. The engine bumps ``EngineResult.n_stage_errors``
so failures are visible in metrics.json — the original silent failure mode is
gone.

Exception chaining (``raise ... from original_exc``) preserves the full
traceback for postmortem.
"""

from __future__ import annotations

from typing import Any, Optional


class WagieError(Exception):
    """Root of the wagie exception tree. Catch this to catch any wagie failure."""


class StageError(WagieError):
    """A Pipeline stage failed in transform() or update().

    Carries:
        stage_name : the failing stage's `name` attribute (or class name)
        op         : "transform" | "update" | "reset" | "state_hash"
        original   : the underlying exception (also chained via ``__cause__``)
        context    : optional dict of bar.ts, n_seen, label, etc.

    Convention: callers raise via ``raise StageError(...) from original_exc``
    so the traceback shows both the wrapper and the root cause.
    """

    __slots__ = ("stage_name", "op", "original", "context")

    def __init__(
        self,
        stage_name: str,
        op: str,
        original: BaseException,
        context: Optional[dict[str, Any]] = None,
    ):
        self.stage_name = str(stage_name)
        self.op = str(op)
        self.original = original
        self.context = dict(context or {})
        ctx_str = ""
        if self.context:
            kvs = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            ctx_str = f" [{kvs}]"
        super().__init__(
            f"stage={self.stage_name!r} op={self.op}: "
            f"{type(original).__name__}: {original}{ctx_str}"
        )


class PipelineError(WagieError):
    """Pipeline-level structural failure (construction, ordering, dispatch).

    Distinct from StageError: PipelineError is the pipeline's own logic, while
    StageError wraps a stage's runtime failure.
    """


class EngineError(WagieError):
    """Engine-level invariant break (clock regression, label leakage, etc.).

    Raised by the engine when a constitutional invariant is violated.
    """


__all__ = [
    "WagieError",
    "StageError",
    "PipelineError",
    "EngineError",
]
