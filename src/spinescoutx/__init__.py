"""SpineScoutX: anatomy-grounded lumbar MRI finding-graph system (research-only).

SpineScoutX is a research and educational prototype. It is NOT diagnostic, NOT
clinically validated, and NOT for medical decision-making.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public, dependency-light vocabulary re-exports. Heavy modules (models, training)
# are imported lazily by the CLI / callers to keep ``import spinescoutx`` cheap.
from .constants import (
    BASE_CONDITIONS,
    CONDITIONS,
    LEVELS,
    SEVERITIES,
    evidence_region_for,
    split_condition,
)

__all__ = [
    "__version__",
    "CONDITIONS",
    "BASE_CONDITIONS",
    "LEVELS",
    "SEVERITIES",
    "split_condition",
    "evidence_region_for",
]
