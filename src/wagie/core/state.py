"""State and causality.

CausalState: marker dataclass asserting "state is a function of past events only."
Stateful: Protocol every Pipeline stage and Broker satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from wagie.core.time import Timestamp


@dataclass(frozen=True, slots=True)
class CausalState:
    """Marker: this state is derived from events with ts_init <= as_of.

    Concrete states subclass this and add their domain-specific fields.
    """

    as_of: Timestamp


@runtime_checkable
class Stateful(Protocol):
    """The contract every Stage and Broker satisfies.

    Replaces ad-hoc state_dict / state_hash / reset methods scattered across
    wagie. A class is Stateful iff it implements all five methods.
    """

    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
    def state_hash(self) -> bytes: ...
    def reset(self) -> None: ...

    @property
    def n_seen(self) -> int: ...


__all__ = ["CausalState", "Stateful"]
