"""Manifest schema for the unified HTML report.

The manifest is the SINGLE source of truth for what lives under
``artifacts/report/``. Section emitters return ``SectionRecord``s; the
renderer assembles them into a ``ReportManifest`` and writes
``manifest.json`` next to ``index.html``.

Anything in ``figs/`` or ``tables/`` whose path is NOT listed in the
manifest is an orphan and gets pruned on the next render.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1


@dataclass
class SectionRecord:
    """One section's contribution to the report.

    ``html_fragment`` is the rendered HTML inserted verbatim under
    ``<section id="{name}">``. ``files`` lists every artifact the
    emitter wrote, as POSIX-style paths RELATIVE to the report root
    (e.g. ``"figs/calibration/reliability.png"``). The renderer uses
    that list to wipe orphans without disturbing live files.
    """

    name: str
    title: str
    order: int
    html_fragment: str
    files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SectionRecord":
        return cls(
            name=str(d["name"]),
            title=str(d["title"]),
            order=int(d["order"]),
            html_fragment=str(d.get("html_fragment", "")),
            files=list(d.get("files", []) or []),
            metadata=dict(d.get("metadata", {}) or {}),
        )


@dataclass
class RunMeta:
    """The header information shown at the top of every report."""

    run_id: str
    spec_name: str
    spec_hash: str
    mode: str = "backtest"  # "backtest" | "cv"
    accepted: bool = True
    blocked_reasons: list[str] = field(default_factory=list)
    state_hash: str = ""
    use_plotly: bool = True
    git_sha: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunMeta":
        return cls(
            run_id=str(d.get("run_id", "")),
            spec_name=str(d.get("spec_name", "")),
            spec_hash=str(d.get("spec_hash", "")),
            mode=str(d.get("mode", "backtest")),
            accepted=bool(d.get("accepted", True)),
            blocked_reasons=list(d.get("blocked_reasons", []) or []),
            state_hash=str(d.get("state_hash", "")),
            use_plotly=bool(d.get("use_plotly", True)),
            git_sha=str(d.get("git_sha", "")),
            note=str(d.get("note", "")),
        )


@dataclass
class ReportManifest:
    """The complete description of the current report on disk."""

    schema_version: int = SCHEMA_VERSION
    generated_at_utc: str = ""
    run_meta: RunMeta = field(default_factory=lambda: RunMeta(
        run_id="", spec_name="", spec_hash="",
    ))
    sections: dict[str, SectionRecord] = field(default_factory=dict)

    @staticmethod
    def now_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def upsert(self, record: SectionRecord) -> None:
        self.sections[record.name] = record

    def ordered(self) -> list[SectionRecord]:
        return sorted(self.sections.values(), key=lambda r: (r.order, r.name))

    def all_listed_files(self) -> set[str]:
        """Every relative path the manifest references — for orphan cleanup."""
        listed: set[str] = set()
        for rec in self.sections.values():
            for p in rec.files:
                listed.add(p.replace("\\", "/"))
        return listed

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_json(self, *, indent: int = 2) -> str:
        body = {
            "schema_version": int(self.schema_version),
            "generated_at_utc": str(self.generated_at_utc),
            "run_meta": self.run_meta.to_dict(),
            "sections": [r.to_dict() for r in self.ordered()],
        }
        return json.dumps(body, indent=indent, default=str)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, text: str) -> "ReportManifest":
        d = json.loads(text)
        secs = {}
        for raw in d.get("sections", []) or []:
            rec = SectionRecord.from_dict(raw)
            secs[rec.name] = rec
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            generated_at_utc=str(d.get("generated_at_utc", "")),
            run_meta=RunMeta.from_dict(d.get("run_meta", {}) or {}),
            sections=secs,
        )

    @classmethod
    def read(cls, path: Path) -> Optional["ReportManifest"]:
        path = Path(path)
        if not path.is_file():
            return None
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None


__all__ = ["SCHEMA_VERSION", "ReportManifest", "RunMeta", "SectionRecord"]
