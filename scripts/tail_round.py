"""Live-tail the most recently active Claude Code round session.

Pretty-prints thinking summaries, assistant text, tool calls, and tool results
as the round agent works. The Claude Code app writes one JSON Lines file per
session under `~/.claude/projects/<sanitized-cwd>/`. This script auto-detects
the most-recently-modified session log (excluding the user's interactive chat)
and tails it.

Usage:
    python scripts/tail_round.py                # auto-detect active round
    python scripts/tail_round.py --session <id> # specific session
    python scripts/tail_round.py --tail 40      # initial backlog of N lines
    python scripts/tail_round.py --no-thinking  # hide thinking summaries
    python scripts/tail_round.py --tools-only   # only tool calls + results

Stop with Ctrl+C. Safe to run multiple times concurrently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make stdout resilient to non-cp1252 chars on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PROJECT_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "C--Users-vitil-OneDrive-Desktop-online-barrier-classifier"
)
# This conversation's session ID prefix — exclude from "active round" detection.
USER_CHAT_PREFIX = "1b8a55a1"


def _color(s: str, c: str) -> str:
    if not sys.stdout.isatty():
        return s
    codes = {
        "red": "31", "green": "32", "yellow": "33", "blue": "34",
        "magenta": "35", "cyan": "36", "gray": "90",
        "bold": "1", "dim": "2",
    }
    return f"\033[{codes[c]}m{s}\033[0m"


def find_active_session(exclude_prefix: str = USER_CHAT_PREFIX) -> Optional[Path]:
    if not PROJECT_DIR.exists():
        return None
    candidates = [
        p for p in PROJECT_DIR.glob("*.jsonl")
        if not p.name.startswith(exclude_prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_subagents(session_path: Path) -> list[Path]:
    """Subagent JSONL files for this session."""
    sub_dir = session_path.parent / session_path.stem / "subagents"
    if not sub_dir.exists():
        return []
    return sorted(sub_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)


def truncate(text: str, n: int = 280) -> str:
    text = text.replace("\n", " | ").strip()
    if len(text) <= n:
        return text
    return text[:n] + _color(f"...+{len(text)-n}", "gray")


def format_event(event: dict, *, show_thinking: bool, tools_only: bool) -> Optional[str]:
    typ = event.get("type")

    if typ == "user":
        if tools_only:
            return None
        msg = event.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "tool_result":
                        # Show tool result tail
                        tc = c.get("content")
                        if isinstance(tc, list):
                            for t in tc:
                                if isinstance(t, dict) and t.get("type") == "text":
                                    parts.append(f"[result] {t.get('text', '')}")
                        elif isinstance(tc, str):
                            parts.append(f"[result] {tc}")
            text = " | ".join(parts)
        else:
            text = str(content) if content else ""
        if not text:
            return None
        return _color("<- user ", "blue") + " " + truncate(text)

    if typ == "assistant":
        msg = event.get("message") or {}
        content = msg.get("content", [])
        if not isinstance(content, list):
            return None
        out_lines: list[str] = []
        for c in content:
            if not isinstance(c, dict):
                continue
            ct = c.get("type")
            if ct == "thinking" and show_thinking:
                think = c.get("thinking", "")
                if think:
                    out_lines.append(_color("~ think ", "magenta") + " " + truncate(think, 300))
            elif ct == "text":
                t = c.get("text", "")
                if t:
                    out_lines.append(_color("-> text ", "green") + " " + truncate(t, 350))
            elif ct == "tool_use":
                name = c.get("name", "?")
                inp = c.get("input", {})
                inp_str = json.dumps(inp, ensure_ascii=False) if inp else ""
                out_lines.append(_color(f"$ tool  ", "yellow") + _color(name, "bold") + " " + truncate(inp_str, 200))
        if not out_lines:
            return None
        if tools_only:
            out_lines = [l for l in out_lines if "$ tool" in l]
            if not out_lines:
                return None
        return "\n".join(out_lines)

    return None


def follow(path: Path, *, initial_tail: int, show_thinking: bool, tools_only: bool) -> int:
    print(_color(f"=== Tailing {path.name} ({path.stat().st_size // 1024} KB)", "cyan"))
    subagents = find_subagents(path)
    if subagents:
        print(_color(f"   {len(subagents)} sub-agent(s): {', '.join(p.stem[-8:] for p in subagents)}", "gray"))
    print()

    # Print last N events as backlog
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 1
    for line in lines[-initial_tail:]:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        out = format_event(ev, show_thinking=show_thinking, tools_only=tools_only)
        if out:
            print(out)

    # Tail-follow
    last_size = path.stat().st_size
    print(_color("... following — Ctrl+C to stop", "dim"))
    while True:
        try:
            try:
                current_size = path.stat().st_size
            except FileNotFoundError:
                time.sleep(2)
                continue
            if current_size > last_size:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_size)
                    new_data = f.read()
                last_size = current_size
                for line in new_data.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out = format_event(ev, show_thinking=show_thinking, tools_only=tools_only)
                    if out:
                        print(out, flush=True)
            time.sleep(2)
        except KeyboardInterrupt:
            print()
            print(_color("... stopped", "dim"))
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", help="explicit session ID (without .jsonl)")
    parser.add_argument("--tail", type=int, default=20, help="initial backlog lines")
    parser.add_argument("--no-thinking", action="store_true", help="hide thinking summaries")
    parser.add_argument("--tools-only", action="store_true", help="only tool calls")
    args = parser.parse_args()

    if args.session:
        path = PROJECT_DIR / f"{args.session}.jsonl"
    else:
        path = find_active_session()
    if not path or not path.exists():
        print(f"No session log found under {PROJECT_DIR}", file=sys.stderr)
        return 1

    return follow(
        path,
        initial_tail=args.tail,
        show_thinking=not args.no_thinking,
        tools_only=args.tools_only,
    )


if __name__ == "__main__":
    sys.exit(main())
