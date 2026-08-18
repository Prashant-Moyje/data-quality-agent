"""Structured logging.

WHY structlog: agent runs are multi-step and non-deterministic. When something
goes wrong you need to know *which step, which tool, which run_id*. Key-value
logs make that greppable; f-string logs do not.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "data_detective"):
    return structlog.get_logger(name)
