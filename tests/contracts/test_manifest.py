"""Contract tests for ``wagie.reporting.manifest``.

Pin down the manifest schema:
  - ``ReportManifest.upsert(record)`` replaces existing entries by name
  - ``ReportManifest.ordered()`` sorts by (order, name)
  - ``ReportManifest.to_json()`` round-trips losslessly through ``from_json()``
  - ``ReportManifest.read()`` returns None for missing / malformed files
  - ``ReportManifest.all_listed_files()`` returns the union across sections,
    normalized to forward slashes
  - ``RunMeta.from_dict({})`` doesn't crash and returns sensible defaults
"""

from __future__ import annotations

from pathlib import Path

from wagie.reporting.manifest import (
    SCHEMA_VERSION,
    ReportManifest,
    RunMeta,
    SectionRecord,
)


# -----------------------------------------------------------------
# upsert: replaces existing entries
# -----------------------------------------------------------------

def test_upsert_replaces_existing_entry_by_name() -> None:
    m = ReportManifest()
    first = SectionRecord(
        name="calibration", title="Calibration",
        order=10, html_fragment="<p>old</p>",
        files=["figs/calibration/old.png"],
    )
    second = SectionRecord(
        name="calibration", title="Calibration",
        order=10, html_fragment="<p>new</p>",
        files=["figs/calibration/new.png", "figs/calibration/extra.png"],
    )
    m.upsert(first)
    m.upsert(second)

    assert len(m.sections) == 1
    rec = m.sections["calibration"]
    assert rec.html_fragment == "<p>new</p>"
    assert rec.files == ["figs/calibration/new.png", "figs/calibration/extra.png"]


def test_upsert_keeps_distinct_named_entries() -> None:
    m = ReportManifest()
    m.upsert(SectionRecord(name="a", title="A", order=1, html_fragment=""))
    m.upsert(SectionRecord(name="b", title="B", order=2, html_fragment=""))
    assert set(m.sections.keys()) == {"a", "b"}


# -----------------------------------------------------------------
# ordered(): sorts by (order, name)
# -----------------------------------------------------------------

def test_ordered_sorts_by_order_then_name() -> None:
    m = ReportManifest()
    # Insert deliberately out of order
    recs = [
        SectionRecord(name="z_late", title="zlate", order=20, html_fragment=""),
        SectionRecord(name="a_first", title="afirst", order=0, html_fragment=""),
        SectionRecord(name="b_mid", title="bmid", order=10, html_fragment=""),
        SectionRecord(name="m_mid", title="mmid", order=10, html_fragment=""),
        SectionRecord(name="a_mid", title="amid", order=10, html_fragment=""),
    ]
    for r in recs:
        m.upsert(r)
    seq = [r.name for r in m.ordered()]
    # order=0 first, then the three order=10 records sorted alphabetically,
    # then order=20.
    assert seq == ["a_first", "a_mid", "b_mid", "m_mid", "z_late"]


def test_ordered_with_empty_manifest_is_empty_list() -> None:
    assert ReportManifest().ordered() == []


# -----------------------------------------------------------------
# to_json / from_json round-trip
# -----------------------------------------------------------------

def test_to_from_json_round_trips_losslessly() -> None:
    m = ReportManifest(
        generated_at_utc="2026-05-10T12:00:00Z",
        run_meta=RunMeta(
            run_id="run-xyz", spec_name="smoke", spec_hash="deadbeef",
            mode="backtest", accepted=True, blocked_reasons=[],
            state_hash="abc123", use_plotly=True,
            git_sha="abcdef0", note="hello",
        ),
    )
    m.upsert(SectionRecord(
        name="overview", title="Overview", order=0,
        html_fragment="<div>hi</div>",
        files=["figs/overview/eq.png"],
        metadata={"uses_plotly": True},
    ))
    m.upsert(SectionRecord(
        name="calibration", title="Calibration", order=10,
        html_fragment="<p>cal</p>",
        files=["figs/calibration/rel.png", "tables/calibration/per_regime.json"],
        metadata={"uses_plotly": False, "n_predictions": 42},
    ))

    text = m.to_json()
    rt = ReportManifest.from_json(text)

    assert rt.schema_version == SCHEMA_VERSION
    assert rt.generated_at_utc == m.generated_at_utc
    assert rt.run_meta.run_id == "run-xyz"
    assert rt.run_meta.spec_name == "smoke"
    assert rt.run_meta.spec_hash == "deadbeef"
    assert rt.run_meta.mode == "backtest"
    assert rt.run_meta.accepted is True
    assert rt.run_meta.use_plotly is True
    assert rt.run_meta.git_sha == "abcdef0"
    assert rt.run_meta.note == "hello"

    assert set(rt.sections.keys()) == {"overview", "calibration"}
    cal = rt.sections["calibration"]
    assert cal.title == "Calibration"
    assert cal.order == 10
    assert cal.files == [
        "figs/calibration/rel.png",
        "tables/calibration/per_regime.json",
    ]
    assert cal.metadata == {"uses_plotly": False, "n_predictions": 42}


def test_to_json_includes_schema_version() -> None:
    text = ReportManifest().to_json()
    assert f'"schema_version": {SCHEMA_VERSION}' in text


# -----------------------------------------------------------------
# read(): missing or malformed → None
# -----------------------------------------------------------------

def test_read_missing_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.json"
    assert ReportManifest.read(p) is None


def test_read_invalid_json_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    p.write_text("{this is not valid json", encoding="utf-8")
    assert ReportManifest.read(p) is None


def test_read_empty_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    p.write_text("", encoding="utf-8")
    assert ReportManifest.read(p) is None


def test_read_valid_file_returns_manifest(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    m = ReportManifest(
        generated_at_utc="2026-05-10T00:00:00Z",
        run_meta=RunMeta(run_id="x", spec_name="x", spec_hash="x"),
    )
    m.upsert(SectionRecord(name="a", title="A", order=0, html_fragment="<p>a</p>"))
    p.write_text(m.to_json(), encoding="utf-8")

    rt = ReportManifest.read(p)
    assert rt is not None
    assert "a" in rt.sections


# -----------------------------------------------------------------
# all_listed_files: union, normalized to forward slashes
# -----------------------------------------------------------------

def test_all_listed_files_returns_union_across_sections() -> None:
    m = ReportManifest()
    m.upsert(SectionRecord(
        name="overview", title="Overview", order=0,
        html_fragment="", files=["figs/overview/a.png"],
    ))
    m.upsert(SectionRecord(
        name="calibration", title="Calibration", order=10,
        html_fragment="", files=[
            "figs/calibration/rel.png",
            "tables/calibration/per_regime.json",
        ],
    ))
    m.upsert(SectionRecord(
        name="trading", title="Trading", order=20,
        html_fragment="", files=["figs/trading/equity.png"],
    ))
    listed = m.all_listed_files()
    assert listed == {
        "figs/overview/a.png",
        "figs/calibration/rel.png",
        "tables/calibration/per_regime.json",
        "figs/trading/equity.png",
    }


def test_all_listed_files_normalizes_backslashes_to_forward() -> None:
    """Windows-style backslashes from a section emitter should be normalized
    to POSIX forward slashes — orphan cleanup compares POSIX paths."""
    m = ReportManifest()
    m.upsert(SectionRecord(
        name="overview", title="Overview", order=0,
        html_fragment="",
        files=[r"figs\overview\eq.png", r"figs\overview\sub\d.png"],
    ))
    listed = m.all_listed_files()
    # Forward-slashed only; no backslashes survive.
    assert listed == {"figs/overview/eq.png", "figs/overview/sub/d.png"}
    for p in listed:
        assert "\\" not in p


def test_all_listed_files_empty_when_no_sections() -> None:
    assert ReportManifest().all_listed_files() == set()


def test_all_listed_files_handles_section_without_files() -> None:
    m = ReportManifest()
    m.upsert(SectionRecord(
        name="empty", title="Empty", order=0,
        html_fragment="<p>no files</p>", files=[],
    ))
    assert m.all_listed_files() == set()


# -----------------------------------------------------------------
# RunMeta.from_dict({}) doesn't crash
# -----------------------------------------------------------------

def test_run_meta_from_empty_dict_returns_sensible_defaults() -> None:
    r = RunMeta.from_dict({})
    # Strings default to ""
    assert r.run_id == ""
    assert r.spec_name == ""
    assert r.spec_hash == ""
    assert r.git_sha == ""
    assert r.note == ""
    assert r.state_hash == ""
    # Mode defaults to "backtest"
    assert r.mode == "backtest"
    # Booleans default
    assert r.accepted is True
    assert r.use_plotly is True
    # Lists default to empty
    assert r.blocked_reasons == []


def test_run_meta_from_dict_round_trip_via_to_dict() -> None:
    r1 = RunMeta(
        run_id="r1", spec_name="s", spec_hash="h",
        mode="cv", accepted=False, blocked_reasons=["foo", "bar"],
        state_hash="abcd", use_plotly=False, git_sha="g", note="note!",
    )
    r2 = RunMeta.from_dict(r1.to_dict())
    assert r2.run_id == r1.run_id
    assert r2.spec_name == r1.spec_name
    assert r2.spec_hash == r1.spec_hash
    assert r2.mode == r1.mode
    assert r2.accepted == r1.accepted
    assert r2.blocked_reasons == r1.blocked_reasons
    assert r2.state_hash == r1.state_hash
    assert r2.use_plotly == r1.use_plotly
    assert r2.git_sha == r1.git_sha
    assert r2.note == r1.note


def test_run_meta_from_dict_coerces_types() -> None:
    """Robustness: from_dict should coerce ints to str and tuples to lists."""
    r = RunMeta.from_dict({
        "run_id": 12345,                     # int → str
        "blocked_reasons": ("a", "b"),       # tuple → list
        "use_plotly": 0,                     # 0 → False
        "accepted": 1,                       # 1 → True
    })
    assert r.run_id == "12345"
    assert r.blocked_reasons == ["a", "b"]
    assert r.use_plotly is False
    assert r.accepted is True


# -----------------------------------------------------------------
# SectionRecord round-trip
# -----------------------------------------------------------------

def test_section_record_round_trip_via_to_from_dict() -> None:
    rec = SectionRecord(
        name="trading", title="Trading", order=20,
        html_fragment="<p>x</p>",
        files=["figs/trading/equity.png"],
        metadata={"uses_plotly": True},
    )
    rt = SectionRecord.from_dict(rec.to_dict())
    assert rt.name == rec.name
    assert rt.title == rec.title
    assert rt.order == rec.order
    assert rt.html_fragment == rec.html_fragment
    assert rt.files == rec.files
    assert rt.metadata == rec.metadata


# -----------------------------------------------------------------
# write/read round-trip on disk
# -----------------------------------------------------------------

def test_manifest_write_and_read_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "manifest.json"
    m = ReportManifest(
        generated_at_utc="2026-05-10T00:00:00Z",
        run_meta=RunMeta(run_id="rid", spec_name="s", spec_hash="h"),
    )
    m.upsert(SectionRecord(
        name="overview", title="Overview", order=0,
        html_fragment="<p>x</p>",
        files=["figs/overview/a.png"],
    ))
    written = m.write(p)
    assert written == p
    assert p.is_file()

    rt = ReportManifest.read(p)
    assert rt is not None
    assert rt.run_meta.run_id == "rid"
    assert "overview" in rt.sections
