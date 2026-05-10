"""wagie.observability — structured logging.

The default ``logging.basicConfig(INFO, "%s %s %s")`` emits a plain text line
that is fine for a single-developer shell session but useless when shipping
log records into a structured store. This module adds an opt-in JSON
formatter selected by the ``WAGIE_LOG_FORMAT`` env var:

    WAGIE_LOG_FORMAT=json   → one JSON record per line on stderr
    (anything else)         → fall back to the human-readable text formatter

Records carry the standard fields (timestamp, level, logger, message) plus
any ``extra=`` payload the call site provided AND the exception info when
present. ``StageError.context`` is recursively unpacked so failures show up
with rich context.

Usage:
    from wagie.observability import configure_logging
    configure_logging()        # respects WAGIE_LOG_FORMAT

The CLI (``src/wagie/__main__.py``) calls this at startup; tests can call it
explicitly with ``format_="json"``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
import traceback
from typing import Any, Optional


# Standard LogRecord attrs we DON'T want to surface in the `extra` block —
# they have first-class fields in our JSON envelope or are noise.
_STD_LOGRECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per LogRecord on a single line.

    Schema:
        {
          "ts": ISO-8601 UTC,
          "level": "INFO",
          "logger": "wagie.engine",
          "msg": "...",
          "extra": { ...any kwargs the caller passed via logger.X(..., extra={}) },
          "exc": { "type": "...", "msg": "...", "trace": [...] }   # only if exception
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = _dt.datetime.fromtimestamp(
            record.created, tz=_dt.timezone.utc
        ).isoformat()
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Surface user-supplied `extra=` kwargs.
        extras = {
            k: _safe(v) for k, v in record.__dict__.items()
            if k not in _STD_LOGRECORD_ATTRS and not k.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        # Exception info (chained).
        if record.exc_info:
            etype, evalue, etb = record.exc_info
            payload["exc"] = {
                "type": etype.__name__ if etype else None,
                "msg": str(evalue) if evalue is not None else "",
                "trace": traceback.format_exception(etype, evalue, etb),
            }
            # If the exception is a StageError, surface its rich context.
            try:
                from wagie.core.errors import StageError  # local import: cycle-safe
                if isinstance(evalue, StageError):
                    payload["exc"]["stage_name"] = evalue.stage_name
                    payload["exc"]["op"] = evalue.op
                    payload["exc"]["context"] = {
                        k: _safe(v) for k, v in evalue.context.items()
                    }
            except Exception:
                pass
        return json.dumps(payload, default=_safe, separators=(",", ":"))


def _safe(v: Any) -> Any:
    """Best-effort JSON-safe coercion."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return repr(v)
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    try:
        return str(v)
    except Exception:
        return repr(v)


def configure_logging(
    level: int = logging.INFO,
    format_: Optional[str] = None,
    *,
    stream=None,
) -> None:
    """Install a single root handler matching the chosen format.

    Args:
        level: log level (default INFO).
        format_: "json" or anything else (text). If None, read from
                 ``WAGIE_LOG_FORMAT`` env var; default "text".
        stream: target stream (default ``sys.stderr``); useful for tests.

    Idempotent — replaces any existing root handlers we own. Anything the
    test harness installed (pytest's caplog handler, etc.) is preserved.
    """
    if format_ is None:
        format_ = os.environ.get("WAGIE_LOG_FORMAT", "text").strip().lower()
    stream = stream if stream is not None else sys.stderr

    root = logging.getLogger()
    # Drop only handlers we previously installed (tagged with our marker).
    for h in list(root.handlers):
        if getattr(h, "_wagie_handler", False):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream)
    handler._wagie_handler = True  # type: ignore[attr-defined]
    if format_ == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)


__all__ = ["JsonFormatter", "configure_logging"]
