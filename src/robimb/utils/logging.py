"""Structured logging utilities emitting JSON Lines payloads."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from uuid import uuid4

__all__ = [
    "JsonLogFormatter",
    "configure_json_logger",
    "flush_handlers",
    "generate_trace_id",
    "log_event",
]


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - exercised via tests
        message = record.getMessage()
        payload: MutableMapping[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "message": message,
        }

        trace_id = getattr(record, "trace_id", None)
        if trace_id:
            payload["trace_id"] = trace_id

        event = getattr(record, "event", None) or message
        payload["event"] = event

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, Mapping):
            payload.update(extra_fields)

        return json.dumps(payload, ensure_ascii=False)


def configure_json_logger(log_path: Path | None, level: int = logging.INFO) -> logging.Logger:
    """Configure the project logger with a JSONL handler."""

    logger = logging.getLogger("robimb")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(JsonLogFormatter())
    else:
        handler = logging.NullHandler()

    handler.setLevel(level)
    logger.addHandler(handler)
    return logger


def flush_handlers(logger: logging.Logger) -> None:
    """Ensure all handlers flush their buffers."""

    for handler in logger.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:  # pragma: no cover - defensive
                continue


def generate_trace_id() -> str:
    """Return a unique trace identifier suitable for correlating log events."""

    return uuid4().hex


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    trace_id: str | None = None,
    level: int = logging.INFO,
    message: str | None = None,
    **fields: Any,
) -> str:
    """Emit a structured event on the provided ``logger``."""

    event_trace_id = trace_id or generate_trace_id()
    extra = {
        "trace_id": event_trace_id,
        "event": event,
        "extra_fields": fields,
    }

    logger.log(level, message or event, extra=extra)
    return event_trace_id

