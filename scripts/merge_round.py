"""Round merge/kill mechanizer — implements RESEARCH/GIT_DISCIPLINE.md.

Two subcommands:
  merge  — fast-forward the current `agent/round-NNN-*` branch onto master,
           run a post-merge smoke pytest, tag `round-NNN-accepted`, delete
           the branch, push to origin if it exists.
  kill   — append KILL_LIST.md, delete the branch, leave master untouched.

The round agent is responsible for writing the LEDGER row BEFORE invoking
this script (the CRITIC checklist already verifies one is present in diff).

Usage:
    python scripts/merge_round.py merge --hypothesis H-005
    python scripts/merge_round.py kill --hypothesis H-005 --reason "BSS 0.04 < 0.10 threshold"

Exit codes:
    0 — success
    1 — operation failed (smoke test red, etc.); state recovered
    2 — refused (wrong branch, FF impossible, etc.); state untouched
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KILL_LIST = REPO / "RESEARCH" / "KILL_LIST.md"


def _git(args: list[str], check: bool = True) -> str:
    p = subprocess.run(["git"] + args, cwd=REPO, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(p.stderr or p.stdout).strip()}")
    return p.stdout.strip()


def _current_branch() -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"])


def _round_id_from_branch(branch: str) -> str | None:
    m = re.match(r"^agent/round-(\d{3})-[a-z0-9-]+$", branch)
    return m.group(1) if m else None


def _has_origin() -> bool:
    p = subprocess.run(["git", "remote"], cwd=REPO, capture_output=True, text=True)
    return "origin" in (p.stdout or "").splitlines()


def _is_ff_possible(branch: str) -> bool:
    """master must be ancestor of branch (i.e., branch is master + N commits)."""
    p = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "master", branch],
        cwd=REPO, capture_output=True, text=True,
    )
    return p.returncode == 0


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + line + "\n", encoding="utf-8")


def cmd_merge(args: argparse.Namespace) -> int:
    branch = _current_branch()
    rid = _round_id_from_branch(branch)
    if not rid:
        print(f"refusing: branch {branch!r} is not agent/round-NNN-<slug>", file=sys.stderr)
        return 2

    # Make sure working tree is clean — uncommitted state is a sign the round
    # didn't finish properly.
    dirty = _git(["status", "--porcelain"])
    if dirty:
        print("refusing: working tree is dirty (uncommitted changes):", file=sys.stderr)
        print(dirty, file=sys.stderr)
        print("commit or discard, then re-run.", file=sys.stderr)
        return 2

    if not _is_ff_possible(branch):
        print(f"refusing: master is not ancestor of {branch}; FF impossible (race)",
              file=sys.stderr)
        print("  resolution: kill this round, re-open from current master.")
        return 2

    head_sha = _git(["rev-parse", "--short", "HEAD"])
    head_msg = _git(["log", "-1", "--pretty=%s"])
    print(f"merging {branch} ({head_sha}) -> master")

    # Move to master and fast-forward.
    _git(["checkout", "master"])
    try:
        _git(["merge", "--ff-only", branch])
    except RuntimeError as e:
        print(f"FF merge failed: {e}", file=sys.stderr)
        return 2

    new_sha = _git(["rev-parse", "--short", "HEAD"])

    # Post-merge smoke. Aborts via ORIG_HEAD on failure so master stays sane.
    if not args.skip_smoke:
        print("post-merge smoke: pytest -q tests/")
        smoke = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/"],
            cwd=REPO, capture_output=False,
        )
        if smoke.returncode != 0:
            print("smoke FAILED; rolling master back via ORIG_HEAD", file=sys.stderr)
            _git(["reset", "--hard", "ORIG_HEAD"])
            print("master rolled back; round must be killed and re-opened from new master.",
                  file=sys.stderr)
            # Re-attach to the failing branch so the agent can act on it.
            _git(["checkout", branch])
            return 1

    # Tag the accepted round.
    tag = f"round-{rid}-accepted"
    _git(["tag", "-f", tag, new_sha])
    print(f"tagged {tag} -> {new_sha}")

    # Delete the now-merged branch.
    _git(["branch", "-d", branch])
    print(f"deleted branch {branch}")

    # Push to origin if it exists.
    if not args.no_push and _has_origin():
        try:
            _git(["push", "origin", "master"])
            _git(["push", "origin", tag])
            print("pushed master + tag to origin")
        except RuntimeError as e:
            print(f"push failed (non-fatal, master is correct locally): {e}",
                  file=sys.stderr)

    print(f"\nround-{rid} ACCEPTED. master={new_sha}. {head_msg[:60]}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    branch = _current_branch()
    rid = _round_id_from_branch(branch)
    if not rid:
        print(f"refusing: branch {branch!r} is not agent/round-NNN-<slug>", file=sys.stderr)
        return 2

    head_sha = _git(["rev-parse", "--short", "HEAD"], check=False) or "?"
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    reason = args.reason or "no reason given"

    line = f"[round-{rid}] {now} sha={head_sha} hypothesis={args.hypothesis or '?'}: {reason}"
    _append(KILL_LIST, line)
    print(f"appended KILL_LIST: {line}")

    # Move off the branch before deleting it.
    _git(["checkout", "master"])
    _git(["branch", "-D", branch])
    print(f"deleted branch {branch}")

    print(f"\nround-{rid} KILLED. master untouched.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("merge", help="FF agent/round-NNN-* onto master, tag, delete branch.")
    pm.add_argument("--hypothesis", default="?", help="hypothesis ID (e.g., H-005)")
    pm.add_argument("--no-push", action="store_true", help="skip push to origin")
    pm.add_argument("--skip-smoke", action="store_true",
                    help="skip post-merge pytest (only for empty-diff rounds)")
    pm.set_defaults(func=cmd_merge)

    pk = sub.add_parser("kill", help="Delete branch, append KILL_LIST, leave master untouched.")
    pk.add_argument("--hypothesis", default="?", help="hypothesis ID (e.g., H-005)")
    pk.add_argument("--reason", required=True, help="one-line reason for the kill")
    pk.set_defaults(func=cmd_kill)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
