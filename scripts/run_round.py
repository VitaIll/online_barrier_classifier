#!/usr/bin/env python3
"""Entry point for one round of the autonomous research-engineering loop.

The loop runs in a single Claude Code session per RESEARCH/LOOP_DISCIPLINE.md.
This script provides:

1. Pre-flight: clean working tree, required files present, baseline pytest
   green. The round agent calls this at Phase 0.
2. Next round id: max(`round-NNN-accepted` tag) + 1, zero-padded.

The actual research work — picking from BACKLOG, spawning sub-agents,
writing code, running experiments, deciding accept/iterate/kill — is done by
Claude Code in the session, not by this script.

Usage:
    python scripts/run_round.py --preflight
    python scripts/run_round.py --next-round-id   # e.g. "003"
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def is_clean_tree() -> bool:
    rc, out, _ = _run(["git", "status", "--porcelain"])
    return rc == 0 and out == ""


def current_branch() -> str:
    _, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out


def next_round_id() -> str:
    """Highest existing `round-NNN-accepted` tag + 1, zero-padded to 3 digits.

    The in-session loop tags every accepted round (no per-round branches) —
    so tags are the source of truth for round numbering.
    """
    rc, out, _ = _run(["git", "tag", "--list", "round-*-accepted"])
    if rc != 0:
        return "001"
    max_n = -1
    for t in out.splitlines():
        m = re.match(r"round-(\d{3})-accepted", t)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{max_n + 1:03d}"


def preflight() -> int:
    print("[run_round] preflight starting", flush=True)

    if not is_clean_tree():
        print("[run_round] FAIL: working tree is not clean; commit or stash first", file=sys.stderr)
        return 2

    branch = current_branch()
    if branch != "master":
        print(f"[run_round] FAIL: not on master (currently {branch}); the in-session loop runs only on master", file=sys.stderr)
        return 5

    # Verify the RESEARCH directory and its mandatory files exist.
    required = [
        "RESEARCH/CONSTITUTION.md",
        "RESEARCH/ROUND_TEMPLATE.md",
        "RESEARCH/LOOP_DISCIPLINE.md",
        "RESEARCH/AGENTS.md",
        "RESEARCH/BACKLOG.md",
        "RESEARCH/LEDGER.md",
    ]
    missing = [p for p in required if not (REPO_ROOT / p).exists()]
    if missing:
        print(f"[run_round] FAIL: missing files: {missing}", file=sys.stderr)
        return 3

    # Sanity: BACKLOG is non-empty (has at least one queued item).
    backlog = (REPO_ROOT / "RESEARCH" / "BACKLOG.md").read_text(encoding="utf-8")
    if "Status: queued" not in backlog and "**Status**: queued" not in backlog:
        print("[run_round] WARN: no `queued` items in BACKLOG; round will run housekeeper")

    # Run the test suite once. A red baseline halts the round before anything else.
    rc, out, err = _run(["python", "-m", "pytest", "-q", "tests/"])
    print(out)
    if err:
        print(err, file=sys.stderr)
    if rc != 0:
        print(f"[run_round] FAIL: pytest red on baseline (rc={rc})", file=sys.stderr)
        return 4

    print(f"[run_round] preflight OK; next round id = {next_round_id()}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Round runner")
    parser.add_argument("--preflight", action="store_true", help="run preflight checks")
    parser.add_argument("--next-round-id", action="store_true", help="print the next round id")
    args = parser.parse_args()

    if args.next_round_id:
        print(next_round_id())
        return 0

    if args.preflight:
        return preflight()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
