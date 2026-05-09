"""Stream typing helpers — forward reference to avoid circular imports."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")
StreamLike = Iterable
