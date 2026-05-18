"""Test package marker + structlog test setup.

We deliberately *do not* call :func:`app.log_config.configure_logging` here
because that wires log output to real stderr. Instead, install a quiet
structlog configuration that:

- runs wrapper-class filtering at ``DEBUG`` so every ``log.debug(...)``
  call still flows into the processor chain;
- routes the final rendered output to a no-op factory so unit tests stay
  silent;
- keeps the stdlib ``logging`` root quiet too (``lark_oapi`` and any
  other library that uses ``logging`` would otherwise leak into test
  output).

This lets tests that need to assert on log records use the canonical
``structlog.testing.capture_logs()`` context manager, which replaces the
processor chain with a recorder that bypasses our no-op factory entirely.
"""
from __future__ import annotations

import logging

import structlog


class _NullPrintLogger:
    """A logger that swallows every output method structlog might call."""

    def msg(self, message):
        return None

    # structlog's PrintLogger exposes one method per level; we mirror that
    # so any future structlog version that calls .debug / .warning / ...
    # directly still gets a no-op.
    debug = info = warning = error = critical = fatal = msg
    log = msg


class _NullPrintLoggerFactory:
    def __call__(self, *args, **kwargs):
        return _NullPrintLogger()


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=False),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=_NullPrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logging.getLogger().setLevel(logging.CRITICAL)
