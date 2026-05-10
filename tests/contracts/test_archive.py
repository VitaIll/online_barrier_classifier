"""Contract tests for ``wagie.reporting.archive``.

Pin down archive rotation semantics:
  - ``archive_current`` zips every payload file EXCEPT ``_archive/`` itself
  - ``archive_current`` returns None on empty / nonexistent dir
  - Repeated calls produce unique filenames (no collisions)
  - ``prune_archive`` keeps the newest N archives by lex name
  - The "last 10" rule: write 12 → prune to 10 → assert oldest 2 are dropped
  - ``wipe_payload`` deletes everything except ``_archive/``
  - ``list_archives`` returns archives sorted by name (newest last)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from wagie.reporting.archive import (
    ARCHIVE_DIRNAME,
    DEFAULT_KEEP,
    archive_current,
    list_archives,
    prune_archive,
    wipe_payload,
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _populate(report_root: Path) -> None:
    """Lay down a small report-shape directory with files, subdirs."""
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "index.html").write_text("<html></html>", encoding="utf-8")
    (report_root / "manifest.json").write_text("{}", encoding="utf-8")
    figs = report_root / "figs" / "calibration"
    figs.mkdir(parents=True, exist_ok=True)
    (figs / "rel.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    state = report_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "pipeline_state_hash.txt").write_text("deadbeef", encoding="utf-8")


# -----------------------------------------------------------------
# archive_current — basic shape
# -----------------------------------------------------------------

def test_archive_current_creates_zip_under_archive_dir(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    _populate(rr)
    archive_path = archive_current(rr, timestamp="2026-05-10T12:00:00Z",
                                   spec_hash="abcd1234")
    assert archive_path is not None
    assert archive_path.is_file()
    assert archive_path.suffix == ".zip"
    assert archive_path.parent == rr / ARCHIVE_DIRNAME
    # Archive sits in the right dir
    assert archive_path.parent.name == ARCHIVE_DIRNAME


def test_archive_zip_contains_every_payload_file_except_archive(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    _populate(rr)
    # Pre-existing _archive/ entry: must NOT be archived.
    (rr / ARCHIVE_DIRNAME).mkdir(exist_ok=True)
    (rr / ARCHIVE_DIRNAME / "old.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    archive_path = archive_current(rr, timestamp="2026-05-10T12:00:00Z",
                                   spec_hash="abcd1234")
    assert archive_path is not None

    with zipfile.ZipFile(archive_path) as zf:
        names = sorted(zf.namelist())
    expected_files = sorted([
        "figs/calibration/rel.png",
        "index.html",
        "manifest.json",
        "state/pipeline_state_hash.txt",
    ])
    assert names == expected_files
    # _archive/ files must be absent:
    assert not any(ARCHIVE_DIRNAME in n for n in names)


def test_archive_returns_none_when_report_dir_missing(tmp_path: Path) -> None:
    rr = tmp_path / "no_such_report"
    assert archive_current(rr, timestamp="2026-05-10T12:00:00Z") is None


def test_archive_returns_none_when_report_dir_empty(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    # Empty directory — nothing to archive.
    assert archive_current(rr, timestamp="2026-05-10T12:00:00Z") is None


def test_archive_returns_none_when_only_archive_dir_present(tmp_path: Path) -> None:
    """If only the _archive/ dir exists with no other payload, return None."""
    rr = tmp_path / "report"
    rr.mkdir()
    (rr / ARCHIVE_DIRNAME).mkdir()
    (rr / ARCHIVE_DIRNAME / "x.zip").write_bytes(b"x")
    # Nothing else under the report root → no payload to archive.
    assert archive_current(rr, timestamp="2026-05-10T12:00:00Z") is None


# -----------------------------------------------------------------
# archive_current — repeated calls produce unique filenames
# -----------------------------------------------------------------

def test_archive_repeated_calls_produce_unique_filenames(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    _populate(rr)
    a1 = archive_current(rr, timestamp="2026-05-10T12:00:00Z",
                         spec_hash="abcd1234")
    # Re-populate (archive_current does NOT wipe; a real renderer call wipes
    # itself before archiving the next one — but here we just want to test
    # that the archive function tolerates duplicate timestamps gracefully).
    _populate(rr)
    a2 = archive_current(rr, timestamp="2026-05-10T12:00:00Z",
                         spec_hash="abcd1234")
    _populate(rr)
    a3 = archive_current(rr, timestamp="2026-05-10T12:00:00Z",
                         spec_hash="abcd1234")
    assert a1 is not None and a2 is not None and a3 is not None
    assert a1.name != a2.name != a3.name
    assert a1.is_file() and a2.is_file() and a3.is_file()


def test_archive_distinct_timestamps_produce_sortable_names(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    _populate(rr)
    a1 = archive_current(rr, timestamp="2026-01-01T00:00:00Z", spec_hash="aa")
    _populate(rr)
    a2 = archive_current(rr, timestamp="2026-06-01T00:00:00Z", spec_hash="bb")
    _populate(rr)
    a3 = archive_current(rr, timestamp="2026-12-31T23:59:59Z", spec_hash="cc")
    assert a1 is not None and a2 is not None and a3 is not None
    # Names sort lex == chronological.
    assert sorted([a1.name, a2.name, a3.name]) == [a1.name, a2.name, a3.name]


# -----------------------------------------------------------------
# prune_archive — keeps newest N
# -----------------------------------------------------------------

def _drop_archives(rr: Path, names: list[str]) -> None:
    """Drop placeholder zip files named ``names`` into the archive dir."""
    arch = rr / ARCHIVE_DIRNAME
    arch.mkdir(parents=True, exist_ok=True)
    for n in names:
        (arch / n).write_bytes(b"PK\x05\x06" + b"\x00" * 18)


def test_prune_archive_keeps_newest_N_by_lex_name(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    names = [
        "20260101T000000Z_aaaa.zip",
        "20260201T000000Z_bbbb.zip",
        "20260301T000000Z_cccc.zip",
        "20260401T000000Z_dddd.zip",
        "20260501T000000Z_eeee.zip",
    ]
    _drop_archives(rr, names)

    removed = prune_archive(rr, max_keep=2)
    # Should remove the 3 oldest by name
    assert sorted(p.name for p in removed) == names[:3]

    remaining = sorted(p.name for p in (rr / ARCHIVE_DIRNAME).iterdir())
    assert remaining == names[-2:]


def test_prune_archive_no_op_when_under_keep(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    _drop_archives(rr, ["20260101T000000Z_aa.zip", "20260201T000000Z_bb.zip"])
    removed = prune_archive(rr, max_keep=10)
    assert removed == []
    assert len(list((rr / ARCHIVE_DIRNAME).iterdir())) == 2


def test_prune_archive_max_keep_zero_drops_everything(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    names = ["20260101T_aa.zip", "20260201T_bb.zip"]
    _drop_archives(rr, names)
    removed = prune_archive(rr, max_keep=0)
    assert sorted(p.name for p in removed) == sorted(names)
    assert list((rr / ARCHIVE_DIRNAME).iterdir()) == []


def test_prune_archive_negative_max_keep_treated_as_zero(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    _drop_archives(rr, ["20260101T_aa.zip"])
    removed = prune_archive(rr, max_keep=-5)
    assert len(removed) == 1


def test_prune_archive_missing_archive_dir_returns_empty(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    assert prune_archive(rr, max_keep=10) == []


# -----------------------------------------------------------------
# THE LAST 10 RULE — write 12, prune to 10, assert exactly which 10 remain
# -----------------------------------------------------------------

def test_last_10_rule_write_12_keep_10_drop_oldest_2(tmp_path: Path) -> None:
    """Strictly assert the 'last 10' rule the user asked for."""
    rr = tmp_path / "report"
    rr.mkdir()
    # 12 archives in chronological (== lex) order
    names = [f"20260101T{i:06d}Z_xxxx.zip" for i in range(12)]
    _drop_archives(rr, names)
    assert len(list((rr / ARCHIVE_DIRNAME).iterdir())) == 12

    removed = prune_archive(rr, max_keep=10)
    # The two oldest are the ones removed
    assert sorted(p.name for p in removed) == sorted(names[:2])
    remaining = sorted(p.name for p in (rr / ARCHIVE_DIRNAME).iterdir())
    assert remaining == sorted(names[2:])
    # And there are exactly 10
    assert len(remaining) == 10


def test_archive_current_with_max_keep_runs_inline_prune(tmp_path: Path) -> None:
    """archive_current calls prune_archive at the end — verify max_keep flows through."""
    rr = tmp_path / "report"
    _populate(rr)
    # Pre-stuff the archive dir so the first call needs to prune.
    _drop_archives(rr, [f"20260101T{i:06d}Z_xxxx.zip" for i in range(5)])
    archive_current(rr, timestamp="2026-05-10T00:00:00Z",
                    spec_hash="abcd1234", max_keep=3)
    archives = list_archives(rr)
    assert len(archives) == 3


# -----------------------------------------------------------------
# wipe_payload
# -----------------------------------------------------------------

def test_wipe_payload_deletes_everything_except_archive(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    _populate(rr)
    _drop_archives(rr, ["20260101T_aa.zip"])
    # Sanity precondition
    assert (rr / "index.html").is_file()
    assert (rr / "figs" / "calibration" / "rel.png").is_file()
    assert (rr / ARCHIVE_DIRNAME / "20260101T_aa.zip").is_file()

    wipe_payload(rr)

    # Archive preserved
    assert (rr / ARCHIVE_DIRNAME).is_dir()
    assert (rr / ARCHIVE_DIRNAME / "20260101T_aa.zip").is_file()
    # Everything else deleted
    assert not (rr / "index.html").exists()
    assert not (rr / "figs").exists()
    assert not (rr / "manifest.json").exists()
    assert not (rr / "state").exists()
    # Top-level directory still exists (we only wiped its CONTENTS)
    assert rr.is_dir()


def test_wipe_payload_missing_dir_no_op(tmp_path: Path) -> None:
    rr = tmp_path / "no_such_dir"
    # Should not raise
    wipe_payload(rr)
    assert not rr.exists()


def test_wipe_payload_with_preserve_archive_false_drops_archive_too(
    tmp_path: Path,
) -> None:
    rr = tmp_path / "report"
    _populate(rr)
    _drop_archives(rr, ["20260101T_aa.zip"])
    wipe_payload(rr, preserve_archive=False)
    assert not (rr / ARCHIVE_DIRNAME).exists()
    assert rr.is_dir()  # root preserved


# -----------------------------------------------------------------
# list_archives
# -----------------------------------------------------------------

def test_list_archives_returns_sorted_newest_last(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    names = [
        "20260301T000000Z_cc.zip",  # newest
        "20260101T000000Z_aa.zip",  # oldest
        "20260201T000000Z_bb.zip",  # middle
    ]
    _drop_archives(rr, names)
    out = list_archives(rr)
    assert [p.name for p in out] == sorted(names)
    # Newest last by lex
    assert out[-1].name == "20260301T000000Z_cc.zip"
    assert out[0].name == "20260101T000000Z_aa.zip"


def test_list_archives_missing_archive_dir_returns_empty(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    assert list_archives(rr) == []


def test_list_archives_filters_to_zip_files(tmp_path: Path) -> None:
    rr = tmp_path / "report"
    rr.mkdir()
    arch = rr / ARCHIVE_DIRNAME
    arch.mkdir()
    (arch / "valid.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    (arch / "stray.txt").write_text("not an archive", encoding="utf-8")
    (arch / "subdir").mkdir()
    out = list_archives(rr)
    assert [p.name for p in out] == ["valid.zip"]


# -----------------------------------------------------------------
# Default keep
# -----------------------------------------------------------------

def test_default_keep_constant() -> None:
    assert DEFAULT_KEEP == 10
