"""Archive rotation for the unified report.

Each render archives the previous report dir into
``<report_root>/_archive/<UTC-ts>_<spec-hash>.zip`` BEFORE overwriting,
then prunes oldest archives so only the last ``max_keep`` remain.

Default ``max_keep=10`` is chosen to satisfy the "last 10 streaks" rule
the user asked for. ``_archive/`` itself is never archived (the zipping
walk skips it explicitly).
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


ARCHIVE_DIRNAME = "_archive"
DEFAULT_KEEP = 10


def _stable_archive_name(timestamp: str, spec_hash: str) -> str:
    """Filename pattern: ``YYYY-MM-DDTHHMMSSZ_<8hex>.zip``.

    Sortable lexicographically — newest sorts last.
    """
    safe_ts = timestamp.replace(":", "").replace("-", "")
    h = (spec_hash or "00000000")[:8]
    return f"{safe_ts}_{h}.zip"


def archive_current(
    report_root: Path,
    *,
    timestamp: str,
    spec_hash: str = "",
    max_keep: int = DEFAULT_KEEP,
) -> Optional[Path]:
    """Archive the current report contents to ``_archive/`` then prune.

    Returns the new archive path, or None if the report dir was empty
    / didn't exist (nothing to archive).

    Excludes from the archive:
        - ``_archive/`` itself (would cause infinite recursion)
        - Anything under a top-level ``state/`` dir whose name starts
          with ``rebuild_`` (rebuild-only scratch — opt-in to skip)
    """
    report_root = Path(report_root)
    if not report_root.is_dir():
        return None

    # Anything to archive?
    payload_files = [
        p for p in report_root.rglob("*")
        if p.is_file() and ARCHIVE_DIRNAME not in p.relative_to(report_root).parts
    ]
    if not payload_files:
        return None

    archive_dir = report_root / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_name = _stable_archive_name(timestamp, spec_hash)
    archive_path = archive_dir / archive_name

    # If a file by this name exists (re-render in the same second), append a
    # disambiguating suffix.
    suffix = 1
    while archive_path.exists():
        archive_path = archive_dir / f"{archive_name[:-4]}_{suffix}.zip"
        suffix += 1

    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
    ) as zf:
        for p in payload_files:
            arcname = p.relative_to(report_root).as_posix()
            zf.write(p, arcname=arcname)

    logger.info(
        "report archive: wrote %s (%d files, %.1f KB)",
        archive_path, len(payload_files), archive_path.stat().st_size / 1024.0,
    )

    prune_archive(report_root, max_keep=max_keep)
    return archive_path


def prune_archive(report_root: Path, *, max_keep: int = DEFAULT_KEEP) -> list[Path]:
    """Keep only the newest ``max_keep`` archives by lexicographic name
    (timestamp-prefixed). Returns the list of removed paths.

    ``max_keep <= 0`` prunes everything.
    """
    report_root = Path(report_root)
    archive_dir = report_root / ARCHIVE_DIRNAME
    if not archive_dir.is_dir():
        return []
    archives = sorted(
        [p for p in archive_dir.iterdir() if p.is_file() and p.suffix == ".zip"],
        key=lambda p: p.name,
    )
    n_keep = max(0, int(max_keep))
    if len(archives) <= n_keep:
        return []
    to_drop = archives[: len(archives) - n_keep]
    removed: list[Path] = []
    for p in to_drop:
        try:
            p.unlink()
            removed.append(p)
            logger.info("report archive: pruned %s", p.name)
        except OSError as e:  # pragma: no cover - filesystem race
            logger.warning("report archive prune failed %s: %s", p, e)
    return removed


def list_archives(report_root: Path) -> list[Path]:
    """Newest-last list of archive zips currently on disk."""
    report_root = Path(report_root)
    archive_dir = report_root / ARCHIVE_DIRNAME
    if not archive_dir.is_dir():
        return []
    return sorted(
        [p for p in archive_dir.iterdir() if p.is_file() and p.suffix == ".zip"],
        key=lambda p: p.name,
    )


def wipe_payload(report_root: Path, *, preserve_archive: bool = True) -> None:
    """Delete every file under ``report_root`` except the ``_archive/`` tree.

    Used right after archiving to clear the slate before the new render.
    """
    report_root = Path(report_root)
    if not report_root.is_dir():
        return
    for child in list(report_root.iterdir()):
        if preserve_archive and child.name == ARCHIVE_DIRNAME:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:  # pragma: no cover
                pass


__all__ = [
    "ARCHIVE_DIRNAME", "DEFAULT_KEEP",
    "archive_current", "prune_archive", "list_archives", "wipe_payload",
]
