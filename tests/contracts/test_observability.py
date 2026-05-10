"""Contract tests for wagie.observability — structured JSON logging.

Asserts:
  - WAGIE_LOG_FORMAT=json switches the root handler to the JSON formatter
  - Each log line is valid JSON with the documented schema
  - StageError context is unpacked into the JSON record
  - The default text formatter still works when env var is absent
  - `wagie info` under WAGIE_LOG_FORMAT=json emits valid JSON to stderr
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wagie.core.errors import StageError
from wagie.observability import JsonFormatter, configure_logging


# -----------------------------------------------------------------
# JsonFormatter — direct unit tests
# -----------------------------------------------------------------


def _emit_one(record_args: dict, *, exc=None) -> dict:
    """Emit a single record through JsonFormatter and parse the JSON line."""
    fmt = JsonFormatter()
    rec = logging.makeLogRecord({
        "name": record_args.get("name", "wagie.test"),
        "levelno": record_args.get("levelno", logging.INFO),
        "levelname": record_args.get("levelname", "INFO"),
        "msg": record_args.get("msg", "hello"),
        "args": record_args.get("args", ()),
        **{k: v for k, v in record_args.items()
           if k not in {"name", "levelno", "levelname", "msg", "args"}},
    })
    if exc is not None:
        try:
            raise exc
        except BaseException:
            rec.exc_info = sys.exc_info()
    return json.loads(fmt.format(rec))


def test_json_formatter_emits_required_fields():
    out = _emit_one({"msg": "test message", "name": "wagie.engine",
                     "levelname": "WARNING", "levelno": logging.WARNING})
    assert isinstance(out, dict)
    assert set(out) >= {"ts", "level", "logger", "msg"}
    assert out["msg"] == "test message"
    assert out["logger"] == "wagie.engine"
    assert out["level"] == "WARNING"
    # ts is ISO-8601-ish (contains "T" and timezone marker).
    assert "T" in out["ts"]


def test_json_formatter_surfaces_extra_fields():
    out = _emit_one({"msg": "x", "stage_name": "online_arf",
                     "n_seen": 42, "phase": "predict"})
    assert "extra" in out
    assert out["extra"]["stage_name"] == "online_arf"
    assert out["extra"]["n_seen"] == 42
    assert out["extra"]["phase"] == "predict"


def test_json_formatter_unpacks_stage_error_context():
    err = StageError(
        stage_name="boom",
        op="update",
        original=RuntimeError("inner"),
        context={"bar_ts_ns": 12345, "label": 1},
    )
    out = _emit_one({"msg": "stage failure"}, exc=err)
    assert "exc" in out
    assert out["exc"]["type"] == "StageError"
    # StageError-specific surfacing.
    assert out["exc"]["stage_name"] == "boom"
    assert out["exc"]["op"] == "update"
    assert out["exc"]["context"]["bar_ts_ns"] == 12345
    assert out["exc"]["context"]["label"] == 1
    assert isinstance(out["exc"]["trace"], list)
    assert any("RuntimeError" in line or "StageError" in line
               for line in out["exc"]["trace"])


def test_json_formatter_handles_non_serializable_extras():
    """Arbitrary objects in extra= must not crash the formatter."""
    class Weird:
        def __repr__(self):
            return "<Weird>"

    out = _emit_one({"msg": "x", "weird": Weird()})
    assert out["extra"]["weird"] == "<Weird>"


def test_json_formatter_handles_bytes_in_extras():
    out = _emit_one({"msg": "x", "blob": b"hi"})
    assert out["extra"]["blob"] == "hi"


def test_json_formatter_omits_extra_block_when_no_extras():
    """Empty extra is dropped — keep records compact."""
    out = _emit_one({"msg": "plain"})
    assert "extra" not in out


# -----------------------------------------------------------------
# configure_logging — env-var dispatch
# -----------------------------------------------------------------


def _grab_root_handler():
    root = logging.getLogger()
    return next((h for h in root.handlers
                 if getattr(h, "_wagie_handler", False)), None)


def test_configure_logging_json_via_env_var(monkeypatch):
    monkeypatch.setenv("WAGIE_LOG_FORMAT", "json")
    stream = io.StringIO()
    configure_logging(stream=stream)
    h = _grab_root_handler()
    assert h is not None
    assert isinstance(h.formatter, JsonFormatter)
    # Restore default for subsequent tests.
    configure_logging(format_="text", stream=stream)


def test_configure_logging_text_default(monkeypatch):
    monkeypatch.delenv("WAGIE_LOG_FORMAT", raising=False)
    stream = io.StringIO()
    configure_logging(stream=stream)
    h = _grab_root_handler()
    assert h is not None
    # Plain Formatter (not our JsonFormatter).
    assert not isinstance(h.formatter, JsonFormatter)


def test_configure_logging_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("WAGIE_LOG_FORMAT", "json")
    stream = io.StringIO()
    configure_logging(format_="text", stream=stream)
    h = _grab_root_handler()
    assert not isinstance(h.formatter, JsonFormatter)


def test_configure_logging_idempotent_replaces_own_handler():
    """Calling configure_logging twice shouldn't accumulate duplicate handlers."""
    stream = io.StringIO()
    configure_logging(format_="json", stream=stream)
    configure_logging(format_="json", stream=stream)
    root = logging.getLogger()
    n_ours = sum(1 for h in root.handlers
                 if getattr(h, "_wagie_handler", False))
    assert n_ours == 1


def test_configure_logging_preserves_third_party_handlers():
    """We must not nuke pytest's caplog handler."""
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    sentinel.foo_marker = True  # type: ignore[attr-defined]
    root.addHandler(sentinel)
    try:
        configure_logging(format_="json", stream=io.StringIO())
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)


def test_log_record_round_trips_through_json():
    """End-to-end: stream a record through configure_logging(json) and parse."""
    stream = io.StringIO()
    configure_logging(format_="json", stream=stream, level=logging.DEBUG)
    log = logging.getLogger("wagie.test_e2e")
    log.warning("propagation works", extra={"trace_id": "abc"})
    line = stream.getvalue().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "propagation works"
    assert payload["level"] == "WARNING"
    assert payload["extra"]["trace_id"] == "abc"


# -----------------------------------------------------------------
# CLI: `wagie info` under WAGIE_LOG_FORMAT=json
# -----------------------------------------------------------------


def _run_wagie_cli(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "wagie", *args],
        capture_output=True, text=True, encoding="utf-8",
        env=env, timeout=120,
    )


def test_cli_info_under_json_log_format_emits_valid_json():
    """`WAGIE_LOG_FORMAT=json wagie info` — any log record on stderr must parse
    as JSON. (stdout is the human-readable info dump; we check stderr only.)
    """
    cp = _run_wagie_cli(["info"], {"WAGIE_LOG_FORMAT": "json"})
    assert cp.returncode == 0, f"stdout={cp.stdout!r} stderr={cp.stderr!r}"
    # stdout still carries the human header.
    assert cp.stdout.startswith("wagie ")
    # If anything was emitted on stderr (it's optional for `info`), each
    # non-empty line MUST be parseable JSON with the documented schema.
    for line in cp.stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        assert {"ts", "level", "logger", "msg"} <= set(rec)


def test_cli_info_internal_flag_lists_internal_symbols():
    """`wagie info --internal` adds an internal symbols block to stdout."""
    cp = _run_wagie_cli(["info", "--internal"], {})
    assert cp.returncode == 0, f"stderr={cp.stderr!r}"
    assert "internal symbols:" in cp.stdout


def test_cli_info_default_does_not_list_internal_symbols():
    cp = _run_wagie_cli(["info"], {})
    assert cp.returncode == 0
    assert "internal symbols:" not in cp.stdout
