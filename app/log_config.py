"""Project-wide ``structlog`` configuration.

Provides a single :func:`configure_logging` call that the CLI entry points
(``minimal-agent run`` / ``minimal-agent serve``) invoke once at startup,
plus a thin :func:`get_logger` wrapper that every other module uses.

All log output goes to **stderr** so that the ``run`` CLI can still pipe
the model's text result on **stdout** without interleaving log lines.

Configuration is driven entirely by environment variables:

``LOG_LEVEL``
    One of ``DEBUG``, ``INFO`` (default), ``WARNING``, ``ERROR``,
    ``CRITICAL``. Case-insensitive.

``LOG_FORMAT``
    ``auto`` (default), ``console``, or ``json``. ``auto`` uses a
    human-friendly console renderer when stderr is a TTY and falls back
    to one-line JSON otherwise (handy for ``docker logs`` / log
    aggregators).

``FEISHU_DEBUG``
    Legacy alias. When set (any non-empty value other than ``0``/``false``)
    forces ``LOG_LEVEL=DEBUG`` regardless of the ``LOG_LEVEL`` setting,
    matching the historical behavior where ``FEISHU_DEBUG=1`` cranked the
    SDK *and* the application logs up to debug.

The configuration is **idempotent**: calling :func:`configure_logging`
multiple times re-applies the current env settings, which is convenient
for unit tests that mutate ``os.environ`` between cases.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _resolve_level() -> int:
    raw = (os.environ.get("LOG_LEVEL") or "").strip().upper()
    legacy_debug = (os.environ.get("FEISHU_DEBUG") or "").strip().lower()
    if legacy_debug and legacy_debug not in ("0", "false", "no", "off"):
        return logging.DEBUG
    if raw in _LEVELS:
        return _LEVELS[raw]
    return logging.INFO


def _resolve_format() -> str:
    """Return one of ``'console'`` or ``'json'``.

    ``auto`` resolves to ``console`` when stderr is a TTY (interactive
    developer terminal) and ``json`` otherwise (containers, CI, redirected
    output) — JSON is one-line and easy to ingest.
    """
    raw = (os.environ.get("LOG_FORMAT") or "auto").strip().lower()
    if raw in ("console", "json"):
        return raw
    try:
        is_tty = bool(sys.stderr.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - exotic streams
        is_tty = False
    return "console" if is_tty else "json"


def _build_renderer(fmt: str) -> Any:
    if fmt == "json":
        return structlog.processors.JSONRenderer()
    # ``colors=False`` keeps logs readable in non-TTY contexts where
    # ``LOG_FORMAT=console`` was forced (e.g. ``docker compose logs``).
    try:
        is_tty = bool(sys.stderr.isatty())
    except (AttributeError, ValueError):  # pragma: no cover
        is_tty = False
    return structlog.dev.ConsoleRenderer(colors=is_tty)


def configure_logging() -> None:
    """Configure ``structlog`` (and the stdlib ``logging`` root) from env.

    Safe to call multiple times — each call re-reads the env and resets
    the global config. Stdlib ``logging`` is also configured so that any
    third-party library that uses it (notably ``lark_oapi``) shares the
    same destination and level.
    """
    level = _resolve_level()
    fmt = _resolve_format()
    renderer = _build_renderer(fmt)

    # Send everything to stderr; CLI subcommands write their *result*
    # (model text / image path) to stdout, so keeping logs on stderr is
    # what lets ``run`` participate in shell pipelines cleanly.
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    # Replace any existing handlers so re-configuring (e.g. between tests
    # or after dotenv mutates the env) doesn't accumulate duplicates.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger, optionally bound to ``logger=<name>``.

    ``name`` should normally be ``__name__`` so log records carry the
    originating module (e.g. ``app.bot``). Returned loggers are cheap;
    callers can store them as module-level globals.
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


__all__ = ["configure_logging", "get_logger"]
