"""Lightweight logging helpers.

Human-readable logs go to stderr; structured machine-readable logs go to stdout
as one JSON object per line when ``--json`` is requested by the CLI. We avoid a
heavy logging framework on purpose.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False


def get_logger(name: str = "spinescoutx", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger writing human messages to stderr."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("spinescoutx")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logger


def emit_json(record: dict[str, Any]) -> None:
    """Print one JSON object to stdout (machine-readable CLI log line)."""
    sys.stdout.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    sys.stdout.flush()
