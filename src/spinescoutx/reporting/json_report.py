"""JSON serialisation for SpineScoutX finding graphs and metric summaries.

All writers produce pretty-printed, key-sorted JSON for reproducible diffs.
Research-only: no clinical or diagnostic content is added here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.paths import ensure_dir
from .finding_graph import FindingGraph, finding_graph_to_dict


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` as pretty, sorted JSON and return the path."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path


def write_json_report(graph: FindingGraph, path: str | Path) -> Path:
    """Serialise a :class:`FindingGraph` to a JSON file and return the path."""
    return _write_json(finding_graph_to_dict(graph), path)


def read_json_report(path: str | Path) -> dict[str, Any]:
    """Read a JSON report file into a dict."""
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON report {in_path} must be an object, got {type(data).__name__}.")
    return data


def write_metrics_summary(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write a metrics dict to a JSON file (pretty, sorted) and return the path."""
    return _write_json(dict(metrics), path)


__all__ = [
    "read_json_report",
    "write_json_report",
    "write_metrics_summary",
]
