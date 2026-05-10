"""Section emitters — one module per report section.

A ``SectionEmitter`` writes its own figures and tables under the report
root, builds an HTML fragment, and returns a ``SectionRecord`` so the
renderer can list its files (for orphan cleanup) and inline the
fragment under ``<section id="{name}">`` in ``index.html``.

Adding a new section:
    1. Subclass :class:`SectionEmitter` in a new module under this package.
    2. Set ``name``, ``title``, ``order`` class attributes.
    3. Implement ``emit(ctx) -> SectionRecord``.
    4. Register via :func:`register` (or use the ``@register_emitter`` decorator).

Run order is dictated by ``order``: the *Overview* section sits at 0,
*Calibration* leads at 10 (per repo policy — calibration first), then
trading at 20, operational at 30, coverage/drift at 40, spec at 90.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Type

from ..manifest import RunMeta, SectionRecord


@dataclass
class EmitterContext:
    """Everything a section needs to do its work.

    Every emitter writes ONLY into ``figs_dir / self.name`` and
    ``tables_dir / self.name`` (creating those subdirs as needed). Files
    are recorded in the returned ``SectionRecord.files`` as POSIX paths
    relative to ``report_root``. The renderer will wipe any file under
    ``figs/`` or ``tables/`` that isn't in the new manifest.
    """

    report_root: Path
    figs_dir: Path                       # report_root / "figs"
    tables_dir: Path                     # report_root / "tables"
    metrics: dict[str, Any]
    spec_dict: dict[str, Any]
    run_meta: RunMeta
    engine_result: Any = None            # Optional; None on rebuild if not pickled
    cv_result: Any = None                # Optional; populated in CV mode
    use_plotly: bool = True
    plotly_div_prefix: str = "plotly"    # used to build unique div ids per chart

    def section_figs_dir(self, name: str) -> Path:
        d = self.figs_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def section_tables_dir(self, name: str) -> Path:
        d = self.tables_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, path: Path) -> str:
        """Path relative to ``report_root``, POSIX-style."""
        try:
            return Path(path).resolve().relative_to(
                self.report_root.resolve()
            ).as_posix()
        except ValueError:
            return Path(path).as_posix()


class SectionEmitter:
    """Abstract base for a report section emitter."""

    name: str = ""           # short slug — used as section id and folder name
    title: str = ""          # human-readable, shown in TOC and <h2>
    order: int = 100         # smaller = earlier; ties broken by name
    requires_engine_result: bool = False

    def emit(self, ctx: EmitterContext) -> SectionRecord:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience helpers usable from subclasses.
    # ------------------------------------------------------------------

    @staticmethod
    def escape(text: Any) -> str:
        return _html.escape("" if text is None else str(text), quote=True)

    @staticmethod
    def fmt_float(value: Any, spec: str = ".4f", default: str = "n/a") -> str:
        if isinstance(value, (int, float)) and value == value:  # NaN-safe
            try:
                return format(float(value), spec)
            except (TypeError, ValueError):
                return default
        return default

    @staticmethod
    def fmt_int(value: Any, default: str = "0") -> str:
        try:
            if value is None:
                return default
            return str(int(value))
        except (TypeError, ValueError):
            return default

    def empty_record(self, *, message: str = "no data", files: Optional[list[str]] = None
                     ) -> SectionRecord:
        return SectionRecord(
            name=self.name, title=self.title, order=self.order,
            html_fragment=(
                f"<div class=\"empty\">"
                f"<p><em>{self.escape(message)}</em></p></div>"
            ),
            files=list(files or []),
        )


# =============================================================================
# Registry
# =============================================================================

_REGISTRY: dict[str, Type[SectionEmitter]] = {}


def register(emitter_cls: Type[SectionEmitter]) -> Type[SectionEmitter]:
    """Register a section emitter class. Idempotent."""
    if not getattr(emitter_cls, "name", ""):
        raise ValueError(f"emitter {emitter_cls!r} has no `name`")
    _REGISTRY[emitter_cls.name] = emitter_cls
    return emitter_cls


# Decorator alias
register_emitter = register


def get_registry() -> dict[str, Type[SectionEmitter]]:
    """Return a copy of the live registry."""
    return dict(_REGISTRY)


def all_emitters() -> list[Type[SectionEmitter]]:
    """All registered emitter classes, sorted by ``order`` then ``name``."""
    return sorted(_REGISTRY.values(), key=lambda c: (c.order, c.name))


def get_emitter(name: str) -> Optional[Type[SectionEmitter]]:
    return _REGISTRY.get(name)


def reset_registry() -> None:
    """Test/teardown only — clears the registry."""
    _REGISTRY.clear()


# =============================================================================
# Auto-registration of the built-in emitters.
#
# The submodules import this module; importing them here triggers the
# decorator-side-effect of registering each emitter. New emitters added
# under ``wagie.reporting.sections.<name>`` should add themselves to this
# import block (or be picked up via the registry decorator at import time).
# =============================================================================

def _autoload_builtin_sections() -> None:
    """Import built-in section modules so their @register decorators fire."""
    # Import order doesn't matter — registry sorts by `order` attr.
    from importlib import import_module
    for mod in (
        "wagie.reporting.sections.overview",
        "wagie.reporting.sections.calibration",
        "wagie.reporting.sections.controller",   # adaptive-threshold controller (order=15)
        "wagie.reporting.sections.trading",
        "wagie.reporting.sections.inventory",     # inventory + hold-age (order=25)
        "wagie.reporting.sections.operational",
        "wagie.reporting.sections.drift",         # rolling Brier drift monitor (order=35)
        "wagie.reporting.sections.coverage",
        "wagie.reporting.sections.spec",
    ):
        try:
            import_module(mod)
        except ImportError:
            # Module not yet present (during foundation boot); skip silently.
            # Emitter agents will land them.
            pass


__all__ = [
    "EmitterContext", "SectionEmitter",
    "register", "register_emitter", "get_registry", "all_emitters",
    "get_emitter", "reset_registry", "_autoload_builtin_sections",
]
