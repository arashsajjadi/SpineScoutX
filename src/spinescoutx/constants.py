"""Shared label vocabulary and anatomy mappings for SpineScoutX.

This module is the single source of truth for condition / level / severity
strings and the anatomy-region mapping used by the evidence-consistency metric.
Every other module imports names from here so the vocabulary stays consistent.

Medical note: the "conditions" below are the RSNA 2024 Lumbar Spine
Degenerative Classification finding types. SpineScoutX grades *findings*; it does
not diagnose disease. The anatomy regions used for evidence scoring are derived
from SPIDER anatomical masks (vertebra / disc / spinal canal), which are NOT
pathology masks and do NOT include foraminal or lateral-recess labels. Where a
finding's true region is unavailable, we use a clearly-flagged approximation.
"""

from __future__ import annotations

from typing import Final

# --- Disc / motion-segment levels (RSNA ordering, superior -> inferior) ---------
LEVELS: Final[tuple[str, ...]] = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")
LEVEL_TO_INDEX: Final[dict[str, int]] = {lv: i for i, lv in enumerate(LEVELS)}

# --- Severity grades (ordinal) --------------------------------------------------
SEVERITIES: Final[tuple[str, ...]] = ("normal_mild", "moderate", "severe")
SEVERITY_TO_INDEX: Final[dict[str, int]] = {s: i for i, s in enumerate(SEVERITIES)}
SEVERE_INDEX: Final[int] = SEVERITY_TO_INDEX["severe"]
NUM_SEVERITY_CLASSES: Final[int] = len(SEVERITIES)

# RSNA competition sample weights per severity (normal/mild=1, moderate=2, severe=4).
# Used by the weighted log-loss metric and may seed class-weighted training loss.
SEVERITY_SAMPLE_WEIGHTS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0)

# --- Sides ----------------------------------------------------------------------
SIDES: Final[tuple[str, ...]] = ("left", "right")

# --- Conditions (RSNA finding types) --------------------------------------------
# Full per-side condition keys exactly as they appear in RSNA labels.
CONDITIONS: Final[tuple[str, ...]] = (
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
)
CONDITION_TO_INDEX: Final[dict[str, int]] = {c: i for i, c in enumerate(CONDITIONS)}

# Base (side-stripped) condition families.
BASE_CONDITIONS: Final[tuple[str, ...]] = (
    "spinal_canal_stenosis",
    "neural_foraminal_narrowing",
    "subarticular_stenosis",
)


def split_condition(condition: str) -> tuple[str, str | None]:
    """Return (base_condition, side) for a full condition key.

    >>> split_condition("left_neural_foraminal_narrowing")
    ('neural_foraminal_narrowing', 'left')
    >>> split_condition("spinal_canal_stenosis")
    ('spinal_canal_stenosis', None)
    """
    for side in SIDES:
        prefix = f"{side}_"
        if condition.startswith(prefix):
            return condition[len(prefix) :], side
    return condition, None


# --- Anatomy classes (SPIDER-derived, simplified) -------------------------------
# Our segmenter collapses SPIDER's per-vertebra/per-disc labels into 4 semantic
# classes. Index 0 is background. Foreground class order is fixed and shared with
# the anatomy-prior channel layout below.
ANATOMY_CLASSES: Final[tuple[str, ...]] = ("background", "vertebra", "disc", "spinal_canal")
ANATOMY_CLASS_TO_INDEX: Final[dict[str, int]] = {c: i for i, c in enumerate(ANATOMY_CLASSES)}
NUM_ANATOMY_CLASSES: Final[int] = len(ANATOMY_CLASSES)
FOREGROUND_ANATOMY_CLASSES: Final[tuple[str, ...]] = ("vertebra", "disc", "spinal_canal")

# Anatomy-prior input channels for the anatomy-guided classifier, in fixed order.
# (disc, canal, vertebra) — disc first because most RSNA findings are disc-level.
ANATOMY_PRIOR_CHANNELS: Final[tuple[str, ...]] = ("disc", "spinal_canal", "vertebra")
NUM_ANATOMY_PRIOR_CHANNELS: Final[int] = len(ANATOMY_PRIOR_CHANNELS)

# --- Evidence target-region mapping (for Anatomical Evidence Consistency) -------
# Maps a base condition to the anatomy region whose mask defines "on-target"
# heatmap mass, plus a source tag. "anatomy" => region is a real SPIDER class.
# "approximate" => SPIDER has no label for this region; we approximate it and the
# AEC value MUST be reported with target_region_source="approximate".
REGION_SOURCE_ANATOMY: Final[str] = "anatomy"
REGION_SOURCE_APPROXIMATE: Final[str] = "approximate"


def evidence_region_for(condition: str) -> tuple[str, str | None, str]:
    """Map a full condition key to (region_name, side, region_source).

    - spinal_canal_stenosis -> the real "spinal_canal" anatomy mask.
    - *_neural_foraminal_narrowing -> side-aware foraminal approximation.
    - *_subarticular_stenosis -> side-aware lateral-recess approximation.

    The foraminal / lateral-recess regions are NOT present in SPIDER, so they are
    flagged "approximate". Callers must surface that flag in any report.
    """
    base, side = split_condition(condition)
    if base == "spinal_canal_stenosis":
        return "spinal_canal", None, REGION_SOURCE_ANATOMY
    if base == "neural_foraminal_narrowing":
        return f"foraminal_{side}", side, REGION_SOURCE_APPROXIMATE
    if base == "subarticular_stenosis":
        return f"lateral_recess_{side}", side, REGION_SOURCE_APPROXIMATE
    raise ValueError(f"Unknown condition: {condition!r}")


# --- Uncertainty flags ----------------------------------------------------------
UNCERTAINTY_FLAGS: Final[tuple[str, ...]] = (
    "high_confidence",
    "moderate_confidence",
    "review_required",
)

# Standard non-diagnostic limitations attached to every finding graph / report.
RESEARCH_LIMITATIONS: Final[tuple[str, ...]] = (
    "Uses non-commercial public research data (RSNA non-commercial; SPIDER CC BY 4.0).",
    "Not diagnostic and not for medical decision-making.",
    "Not clinically validated; no prospective or external clinical evaluation.",
    "Segmentation priors are ANATOMY masks (vertebra/disc/canal), not pathology masks.",
    "Foraminal and lateral-recess evidence regions are approximations, not labeled anatomy.",
)
