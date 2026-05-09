"""Single-command loop health/observability dashboard.

Usage:
    python scripts/loop_status.py
    make loop-status

Reads the heartbeat, LEDGER tail, BACKLOG head, KILL_LIST tail, and git state
to produce a one-page status digest. Run any time to see if the loop is alive
and what it's been doing.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "RESEARCH"
HEARTBEAT = RESEARCH / ".loop_heartbeat.json"
LEDGER = RESEARCH / "LEDGER.md"
BACKLOG = RESEARCH / "BACKLOG.md"
KILL_LIST = RESEARCH / "KILL_LIST.md"


def _git(cmd: list[str]) -> str:
    p = subprocess.run(["git"] + cmd, cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip()


def _color(s: str, c: str) -> str:
    codes = {"red": "31", "green": "32", "yellow": "33", "blue": "34",
             "magenta": "35", "cyan": "36", "bold": "1", "dim": "2"}
    if not sys.stdout.isatty():
        return s
    return f"\033[{codes[c]}m{s}\033[0m"


def _heartbeat_summary() -> str:
    if not HEARTBEAT.exists():
        return _color("[no heartbeat — daemon never started]", "yellow")
    hb = json.loads(HEARTBEAT.read_text())
    started = hb.get("started_at") or hb.get("finished_at") or "?"
    try:
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
        age_str = f"{int(age.total_seconds())}s ago"
        if age.total_seconds() > 3600:
            age_str = f"{age.total_seconds()/3600:.1f}h ago"
    except Exception:
        age_str = "?"
    status = hb.get("status", "?")
    color = {"ok": "green", "running": "cyan", "fail": "red", "halted": "red", "error": "red"}.get(status, "yellow")
    return (f"{_color('STATUS', 'bold')}: {_color(status.upper(), color)}  "
            f"round_index={hb.get('round_index', '?')}  "
            f"last_event={age_str}  "
            f"consecutive_failures={hb.get('consecutive_failures', 0)}")


def _ledger_tail(n: int = 8) -> list[str]:
    if not LEDGER.exists():
        return []
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    data = [ln for ln in lines if "|" in ln and "date" not in ln.lower()]
    return data[-n:]


def _backlog_top(n: int = 5) -> list[str]:
    if not BACKLOG.exists():
        return []
    text = BACKLOG.read_text(encoding="utf-8")
    items = re.findall(r"^### (H-\d+).*?\n.*?\*\*Status\*\*:\s*(\w+)", text, re.MULTILINE | re.DOTALL)
    queued = [(hid, st) for hid, st in items if st in ("queued", "in_progress")]
    return [f"{hid:8}  {st}" for hid, st in queued[:n]]


def _kill_tail(n: int = 5) -> list[str]:
    if not KILL_LIST.exists():
        return []
    return [ln for ln in KILL_LIST.read_text().splitlines() if ln.startswith("[round")][-n:]


def _branches() -> dict:
    out = _git(["branch", "--list", "agent/round-*"])
    branches = [b.lstrip("* ").strip() for b in out.splitlines()]
    return {"count": len(branches), "names": branches[-5:]}


def _accept_kill_rate() -> dict:
    """Count outcomes from the LEDGER."""
    if not LEDGER.exists():
        return {}
    text = LEDGER.read_text(encoding="utf-8")
    rows = [ln.split("|") for ln in text.splitlines() if "|" in ln and ln[:4].isdigit()]
    counter = {}
    for r in rows:
        if len(r) >= 4:
            status = r[3].strip()
            counter[status] = counter.get(status, 0) + 1
    return counter


def main() -> int:
    print()
    print(_color("=" * 78, "dim"))
    print(_color("  AUTONOMOUS LOOP STATUS", "bold"))
    print(_color("=" * 78, "dim"))
    print()
    print("  " + _heartbeat_summary())
    print()

    branches = _branches()
    rates = _accept_kill_rate()
    rate_summary = "  ".join(f"{k}={v}" for k, v in sorted(rates.items()))
    print(f"  {_color('Branches', 'bold')}: {branches['count']} agent/round-* total")
    print(f"  {_color('LEDGER counts', 'bold')}: {rate_summary or '(empty)'}")
    print()

    print(_color("  Recent rounds (LEDGER tail):", "bold"))
    tail = _ledger_tail(6)
    if not tail:
        print("    (empty)")
    for ln in tail:
        # Trim the long note column for terminal readability.
        parts = ln.split("|")
        if len(parts) >= 8:
            short = " | ".join(p.strip() for p in parts[:7])
            note = parts[7].strip()[:80]
            print(f"    {short} | {note}{'…' if len(parts[7].strip()) > 80 else ''}")
        else:
            print(f"    {ln[:140]}")
    print()

    print(_color("  Top of BACKLOG:", "bold"))
    top = _backlog_top(5)
    if not top:
        print("    (empty)")
    for item in top:
        print(f"    {item}")
    print()

    kills = _kill_tail(5)
    if kills:
        print(_color("  Recent KILL_LIST entries:", "bold"))
        for ln in kills:
            print(f"    {ln[:140]}")
        print()

    cur_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    cur_sha = _git(["rev-parse", "--short", "HEAD"])
    print(f"  {_color('Git', 'bold')}: branch={cur_branch} sha={cur_sha}")
    open_prs = _git(["log", "--oneline", "-5"]).splitlines()
    print(f"  {_color('Last 5 commits', 'bold')}:")
    for c in open_prs:
        print(f"    {c}")
    print()
    print(_color("=" * 78, "dim"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
