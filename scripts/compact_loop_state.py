"""Compact loop state: LEDGER, BACKLOG, plot artifacts, branches, MLflow runs.

Runs at the end of every round (or via `make compact`). Idempotent.

What it compacts:
  - LEDGER.md:    if > 30 entries hot, archive oldest to LEDGER_ARCHIVE.md
  - BACKLOG.md:   strip "ACCEPTED roundNNN" annotations for items accepted >5 rounds ago
  - RESEARCH/diagrams/round_NNN/:  move dirs older than current_round - 7 to _archive/
  - Branches:     delete agent/round-NNN-* with status=kill in LEDGER and round_id < N-7
  - MLflow:       delete artifact files (keep metrics) for runs > 30 days old
  - .tmp/:        nuke anything older than 24h

What it never touches:
  - KILL_LIST.md  (append-only, kept indefinitely so we never re-litigate)
  - Accepted branches before they're merged to main
  - The most recent N rounds (always hot)

Safety: dry-run by default. Pass --apply to actually write/delete.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "RESEARCH"
LEDGER = RESEARCH / "LEDGER.md"
LEDGER_ARCHIVE = RESEARCH / "LEDGER_ARCHIVE.md"
BACKLOG = RESEARCH / "BACKLOG.md"
DIAGRAMS = RESEARCH / "diagrams"
DIAGRAMS_ARCHIVE = DIAGRAMS / "_archive"
TMP = REPO / ".tmp"
MLRUNS = REPO / "experiments" / "mlruns"


HOT_LEDGER_LINES = 30
HOT_ROUND_DIRS = 7
DEAD_BRANCH_AGE_ROUNDS = 7
ACCEPTED_ANNOTATION_AGE = 5
MLFLOW_ARTIFACT_KEEP_DAYS = 30
TMP_KEEP_HOURS = 24


@dataclass
class CompactionAction:
    kind: str
    detail: str
    applied: bool = False


def _git(cmd: list[str]) -> str:
    p = subprocess.run(["git"] + cmd, cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip()


def _ledger_lines() -> list[str]:
    if not LEDGER.exists():
        return []
    return [ln for ln in LEDGER.read_text(encoding="utf-8").splitlines() if "|" in ln and ln[:4].isdigit()]


def _current_round() -> int:
    """Highest round id seen in LEDGER."""
    rounds: list[int] = []
    for ln in _ledger_lines():
        m = re.search(r"\| (\d{3}) \|", ln)
        if m:
            rounds.append(int(m.group(1)))
    return max(rounds) if rounds else 0


def compact_ledger(actions: list[CompactionAction], apply: bool) -> None:
    """If LEDGER has > HOT_LEDGER_LINES data rows, move oldest to archive."""
    if not LEDGER.exists():
        return
    text = LEDGER.read_text(encoding="utf-8")
    lines = text.splitlines()
    data_idx = [i for i, ln in enumerate(lines) if "|" in ln and ln[:4].isdigit()]
    if len(data_idx) <= HOT_LEDGER_LINES:
        return

    overflow = data_idx[:-HOT_LEDGER_LINES]
    archived_lines = [lines[i] for i in overflow]
    keep_lines = [ln for i, ln in enumerate(lines) if i not in set(overflow)]

    actions.append(CompactionAction("ledger_archive",
                                      f"archiving {len(archived_lines)} oldest LEDGER rows "
                                      f"(keeping last {HOT_LEDGER_LINES} hot)"))
    if apply:
        prior = LEDGER_ARCHIVE.read_text(encoding="utf-8") if LEDGER_ARCHIVE.exists() else (
            "# Archived LEDGER entries (compacted from LEDGER.md)\n\n")
        LEDGER_ARCHIVE.write_text(prior + "\n".join(archived_lines) + "\n", encoding="utf-8")
        LEDGER.write_text("\n".join(keep_lines) + "\n", encoding="utf-8")
        actions[-1].applied = True


def compact_backlog(actions: list[CompactionAction], apply: bool) -> None:
    """Strip 'ACCEPTED roundNNN' annotations for items accepted > N rounds ago."""
    if not BACKLOG.exists():
        return
    text = BACKLOG.read_text(encoding="utf-8")
    cur = _current_round()
    pat = re.compile(r"(### H-\d+.*?— ACCEPTED round (\d{3})$)", re.MULTILINE)
    matches = list(pat.finditer(text))
    if not matches:
        return
    stale = [m for m in matches if cur - int(m.group(2)) > ACCEPTED_ANNOTATION_AGE]
    if not stale:
        return
    actions.append(CompactionAction("backlog_strip_annotation",
                                      f"stripping ACCEPTED annotations from {len(stale)} items "
                                      f"older than current_round - {ACCEPTED_ANNOTATION_AGE}"))
    if apply:
        new = text
        for m in stale:
            stripped = m.group(0).replace(f"— ACCEPTED round {m.group(2)}", "")
            new = new.replace(m.group(0), stripped)
        BACKLOG.write_text(new, encoding="utf-8")
        actions[-1].applied = True


def compact_diagram_dirs(actions: list[CompactionAction], apply: bool) -> None:
    """Move RESEARCH/diagrams/round_NNN dirs older than current_round - HOT_ROUND_DIRS."""
    if not DIAGRAMS.exists():
        return
    cur = _current_round()
    cold_threshold = cur - HOT_ROUND_DIRS
    cold = []
    for d in DIAGRAMS.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"round_(\d+)$", d.name)
        if not m:
            continue
        rid = int(m.group(1))
        if rid <= cold_threshold:
            cold.append(d)
    if not cold:
        return
    actions.append(CompactionAction("diagrams_archive",
                                      f"archiving {len(cold)} round_NNN dirs older than "
                                      f"round {cold_threshold}"))
    if apply:
        DIAGRAMS_ARCHIVE.mkdir(exist_ok=True)
        for d in cold:
            target = DIAGRAMS_ARCHIVE / d.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(d), str(target))
        actions[-1].applied = True


def compact_dead_branches(actions: list[CompactionAction], apply: bool) -> None:
    """Delete branches whose round was killed (per LEDGER) and is > N rounds old."""
    cur = _current_round()
    branches = _git(["branch", "--list", "agent/round-*"]).splitlines()
    branches = [b.lstrip("* ").strip() for b in branches if b.strip()]
    killed_rounds = set()
    for ln in _ledger_lines():
        if "kill" in ln.split("|")[3].strip().lower():
            m = re.search(r"\| (\d{3}) \|", ln)
            if m:
                killed_rounds.add(int(m.group(1)))
    to_delete = []
    for b in branches:
        m = re.match(r"agent/round-(\d{3})-", b)
        if not m:
            continue
        rid = int(m.group(1))
        if rid in killed_rounds and (cur - rid) >= DEAD_BRANCH_AGE_ROUNDS:
            to_delete.append(b)
    if not to_delete:
        return
    actions.append(CompactionAction("branches_delete",
                                      f"deleting {len(to_delete)} dead branches older than "
                                      f"{DEAD_BRANCH_AGE_ROUNDS} rounds"))
    if apply:
        for b in to_delete:
            _git(["branch", "-D", b])
        actions[-1].applied = True


def compact_mlflow_artifacts(actions: list[CompactionAction], apply: bool) -> None:
    """Delete artifact files (keep metrics + tags) for runs > N days old."""
    if not MLRUNS.exists():
        return
    now = dt.datetime.now()
    cutoff = now - dt.timedelta(days=MLFLOW_ARTIFACT_KEEP_DAYS)
    pruned_bytes = 0
    pruned_files = 0
    for art_dir in MLRUNS.glob("**/artifacts"):
        if not art_dir.is_dir():
            continue
        # mtime check on parent run dir
        run_dir = art_dir.parent
        try:
            mtime = dt.datetime.fromtimestamp(run_dir.stat().st_mtime)
        except FileNotFoundError:
            continue
        if mtime > cutoff:
            continue
        for f in art_dir.rglob("*"):
            if f.is_file():
                pruned_bytes += f.stat().st_size
                pruned_files += 1
                if apply:
                    f.unlink()
    if pruned_files == 0:
        return
    actions.append(CompactionAction("mlflow_prune_artifacts",
                                      f"pruning {pruned_files} artifact files "
                                      f"({pruned_bytes/1e6:.1f} MB) > {MLFLOW_ARTIFACT_KEEP_DAYS}d old"))
    if apply:
        actions[-1].applied = True


def compact_tmp(actions: list[CompactionAction], apply: bool) -> None:
    if not TMP.exists():
        return
    cutoff = dt.datetime.now() - dt.timedelta(hours=TMP_KEEP_HOURS)
    pruned = 0
    for f in TMP.rglob("*"):
        if not f.is_file():
            continue
        try:
            mtime = dt.datetime.fromtimestamp(f.stat().st_mtime)
        except FileNotFoundError:
            continue
        if mtime < cutoff:
            pruned += 1
            if apply:
                f.unlink()
    if pruned == 0:
        return
    actions.append(CompactionAction("tmp_prune",
                                      f"pruning {pruned} files in .tmp/ older than {TMP_KEEP_HOURS}h"))
    if apply:
        actions[-1].applied = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="actually apply compaction (default: dry-run)")
    args = parser.parse_args()

    actions: list[CompactionAction] = []
    compact_ledger(actions, args.apply)
    compact_backlog(actions, args.apply)
    compact_diagram_dirs(actions, args.apply)
    compact_dead_branches(actions, args.apply)
    compact_mlflow_artifacts(actions, args.apply)
    compact_tmp(actions, args.apply)

    mode = "APPLIED" if args.apply else "DRY RUN - pass --apply to commit"
    print(f"\nCompaction {mode}\n")
    if not actions:
        print("  (nothing to compact)")
    for a in actions:
        marker = "OK" if a.applied else "--"
        print(f"  {marker} {a.kind:30} {a.detail}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
