"""Filesystem path helpers and run-directory conventions.

No absolute paths are hardcoded anywhere in SpineScoutX; everything is resolved
relative to user-provided roots or the current working directory.
"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def repo_root() -> Path:
    """Best-effort repository root (the directory containing pyproject.toml).

    Falls back to the current working directory if no marker is found.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def run_dir(runs_root: str | Path, run_id: str) -> Path:
    """Return the directory for a given run id under ``runs_root``."""
    return Path(runs_root) / run_id


def new_run_id(prefix: str, timestamp: str) -> str:
    """Compose a stable run id from a prefix and a caller-supplied timestamp.

    The timestamp is passed in (never read from the clock here) so that callers
    control reproducibility and so this function stays pure/testable.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in prefix)
    return f"{safe}-{timestamp}"
