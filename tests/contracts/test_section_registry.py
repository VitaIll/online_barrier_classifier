"""Contract tests for ``wagie.reporting.sections`` registry.

Pin down:
  - all built-in emitters auto-load on first ``_autoload_builtin_sections()`` call
  - each built-in has a unique ``name``, non-empty ``title``,
    sensible ``order`` (calibration < trading; overview is 0)
  - ``register(cls)`` is idempotent for the same class
  - ``register(cls)`` raises ValueError when ``name`` is empty
"""

from __future__ import annotations

import pytest

from wagie.reporting.sections import (
    EmitterContext,
    SectionEmitter,
    _autoload_builtin_sections,
    all_emitters,
    get_emitter,
    get_registry,
    register,
    reset_registry,
)


@pytest.fixture
def restore_registry():
    """Snapshot the registry, run the test, then restore it.

    Used by tests that call ``register(...)`` so they don't leak emitters
    into other tests in the session (pytest-randomly is in play).
    """
    snap = get_registry()
    yield
    reset_registry()
    for name, cls in snap.items():
        # Re-register without going through validation again
        from wagie.reporting.sections import _REGISTRY  # noqa: PLC0415
        _REGISTRY[name] = cls


# -----------------------------------------------------------------
# Auto-load: ALL built-in emitters appear on first call
# -----------------------------------------------------------------

EXPECTED_BUILTIN_NAMES = {
    "overview", "calibration", "trading",
    "operational", "coverage", "spec",
}


def test_autoload_registers_all_builtin_sections() -> None:
    _autoload_builtin_sections()
    names = {cls.name for cls in all_emitters()}
    # Every built-in should be in the registry after autoload.
    missing = EXPECTED_BUILTIN_NAMES - names
    assert not missing, f"missing built-in emitters: {missing}"


def test_autoload_is_idempotent() -> None:
    _autoload_builtin_sections()
    a = {cls.name for cls in all_emitters()}
    _autoload_builtin_sections()
    b = {cls.name for cls in all_emitters()}
    assert a == b


# -----------------------------------------------------------------
# Each built-in: unique name, non-empty title, sensible order
# -----------------------------------------------------------------

def test_builtin_names_are_unique() -> None:
    _autoload_builtin_sections()
    names = [cls.name for cls in all_emitters()]
    assert len(names) == len(set(names)), \
        f"duplicate emitter names: {names}"


def test_builtin_titles_are_nonempty() -> None:
    _autoload_builtin_sections()
    for cls in all_emitters():
        assert cls.title, f"emitter {cls.name!r} has empty title"
        assert isinstance(cls.title, str)


def test_overview_order_is_zero() -> None:
    """Per repo policy ('lead with charts'), Overview sits at 0."""
    _autoload_builtin_sections()
    cls = get_emitter("overview")
    assert cls is not None
    assert cls.order == 0


def test_calibration_order_strictly_less_than_trading() -> None:
    """Per repo policy ('calibration leads, not trading'), calibration must
    sort before trading."""
    _autoload_builtin_sections()
    cal = get_emitter("calibration")
    tr = get_emitter("trading")
    assert cal is not None and tr is not None
    assert cal.order < tr.order, \
        f"calibration order {cal.order} must be < trading order {tr.order}"


def test_spec_section_is_last_among_builtins() -> None:
    """Provenance / Spec should sit at the back among the built-ins.

    We deliberately filter to known built-ins so dynamically-registered test
    emitters from other tests in the same session don't interfere.
    """
    _autoload_builtin_sections()
    sec = get_emitter("spec")
    assert sec is not None
    others = [
        cls.order for cls in all_emitters()
        if cls.name in EXPECTED_BUILTIN_NAMES and cls.name != "spec"
    ]
    assert all(sec.order >= o for o in others), \
        f"spec order {sec.order} should be >= every other built-in's order"


def test_all_emitters_returned_in_order_then_name(monkeypatch) -> None:
    """all_emitters() sorts by (order, name)."""
    _autoload_builtin_sections()
    seq = list(all_emitters())
    keys = [(c.order, c.name) for c in seq]
    assert keys == sorted(keys), f"all_emitters not order-sorted: {keys}"


# -----------------------------------------------------------------
# register(cls) idempotency + validation
# -----------------------------------------------------------------

def test_register_is_idempotent_for_same_class(restore_registry) -> None:
    class MyEm(SectionEmitter):
        name = "test_my_em_idempotent"
        title = "MyEm"
        order = 200

    register(MyEm)
    snap1 = get_registry()
    register(MyEm)
    snap2 = get_registry()
    assert snap1 == snap2


def test_register_returns_class_for_decorator_use(restore_registry) -> None:
    """register can be used as a decorator and returns the class."""
    class MyEm(SectionEmitter):
        name = "test_my_em_decorator"
        title = "MyEm"
        order = 201

    rv = register(MyEm)
    assert rv is MyEm


def test_register_raises_value_error_when_name_is_empty(restore_registry) -> None:
    class Bad(SectionEmitter):
        name = ""              # empty name should be rejected
        title = "Bad"
        order = 999

    with pytest.raises(ValueError):
        register(Bad)


def test_register_raises_value_error_when_name_attr_missing(
    restore_registry,
) -> None:
    class Bad(SectionEmitter):
        # no `name` override → inherited "" from base class
        title = "Bad"
        order = 999

    with pytest.raises(ValueError):
        register(Bad)


# -----------------------------------------------------------------
# get_emitter contract
# -----------------------------------------------------------------

def test_get_emitter_returns_none_for_unknown_name() -> None:
    assert get_emitter("does_not_exist_xyz") is None


def test_get_emitter_returns_class_for_known_name() -> None:
    _autoload_builtin_sections()
    cls = get_emitter("calibration")
    assert cls is not None
    assert issubclass(cls, SectionEmitter)
    assert cls.name == "calibration"


# -----------------------------------------------------------------
# EmitterContext smoke
# -----------------------------------------------------------------

def test_emitter_context_has_required_fields(tmp_path) -> None:
    """Sanity: EmitterContext is a dataclass with the documented fields."""
    from wagie.reporting.manifest import RunMeta

    ctx = EmitterContext(
        report_root=tmp_path,
        figs_dir=tmp_path / "figs",
        tables_dir=tmp_path / "tables",
        metrics={}, spec_dict={},
        run_meta=RunMeta(run_id="x", spec_name="x", spec_hash="x"),
    )
    # rel() helper produces POSIX path
    p = tmp_path / "figs" / "calibration" / "rel.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    rel = ctx.rel(p)
    assert "/" in rel
    assert "\\" not in rel
    assert rel.endswith("figs/calibration/rel.png")


def test_section_figs_dir_creates_directory(tmp_path) -> None:
    from wagie.reporting.manifest import RunMeta

    ctx = EmitterContext(
        report_root=tmp_path,
        figs_dir=tmp_path / "figs",
        tables_dir=tmp_path / "tables",
        metrics={}, spec_dict={},
        run_meta=RunMeta(run_id="x", spec_name="x", spec_hash="x"),
    )
    out = ctx.section_figs_dir("foo")
    assert out.is_dir()
    assert out == tmp_path / "figs" / "foo"
