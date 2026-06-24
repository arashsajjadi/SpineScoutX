"""Evidence Stability — how stable a grader's prediction is under plausible
localizer / slice / crop perturbation, computed from the AUTO route only.

The oracle->auto gap exists because the localizer is imperfect. A *single* auto
crop hides this: two crops a few pixels apart can grade differently. Evidence
stability re-runs the **same** grader on ``K`` plausible perturbations of the auto
localization (in-plane crop-centre jitter + slice shift, drawn from the localizer's
*own* measured error scale — never from ground-truth coordinates) and summarises how
much the prediction moves. A finding whose ``P(severe)`` swings across plausible
crops is, by construction, less trustworthy than one that is invariant.

This module holds the **pure** scoring/grading logic (no I/O, no torch) so it is
unit-testable without data; the re-cropping + grader forward live in
``scripts/run_evidence_stability.py`` and ``data.perturb_crops``.

Research-only. Not diagnostic. Stability is a research reliability signal, not
triage advice. No ground-truth coordinates are used to generate perturbations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# perturbation configuration (per route; scales set from the localizer's own
# measured in-plane error, NOT from any per-case ground truth)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PerturbConfig:
    """Localizer-uncertainty perturbation knobs for one route.

    ``xy_sigma`` / ``tail_sigma`` are in pixels and chosen to match the route's
    measured auto-vs-localizer in-plane error scale (canal median ~2.5 px, heavy
    tail; axial leveling is noisier -> larger slice jitter). ``slice_jitter`` is the
    max |instance offset| sampled uniformly. These describe the *localizer*, not the
    test case, so using them introduces no ground-truth leakage.
    """

    k: int = 8
    xy_sigma: float = 4.0
    tail_prob: float = 0.15
    tail_sigma: float = 14.0
    slice_jitter: int = 1
    max_offset: float = 48.0
    # grade thresholds
    stable_pstd: float = 0.05
    unstable_flip: float = 1.0 / 3.0
    unstable_range: float = 0.40

    def to_json(self) -> dict:
        return asdict(self)


# Per-route defaults. Subarticular gets a larger slice jitter because axial
# leveling is the known bottleneck (~0.43 ±1-slice hit), so plausible level
# uncertainty spans more slices.
# Scales are matched to each route's *measured* localizer error so instability reflects
# plausible error for THAT route (not a common scale): canal in-plane median ~2.5 px with a
# heavy tail; the foraminal localizer is clean (~2.2 px median, crop-hit 0.999) -> gentler
# in-plane jitter; subarticular's dominant uncertainty is the axial level/slice (±1-slice hit
# ~0.43) -> larger slice jitter, moderate in-plane (the paramedian offset is approximate).
DEFAULT_CONFIGS: dict[str, PerturbConfig] = {
    "spinal_canal_stenosis": PerturbConfig(k=8, xy_sigma=4.0, tail_sigma=14.0, slice_jitter=1),
    "left_neural_foraminal_narrowing": PerturbConfig(
        k=8, xy_sigma=3.0, tail_sigma=8.0, tail_prob=0.12, slice_jitter=1
    ),
    "right_neural_foraminal_narrowing": PerturbConfig(
        k=8, xy_sigma=3.0, tail_sigma=8.0, tail_prob=0.12, slice_jitter=1
    ),
    "left_subarticular_stenosis": PerturbConfig(k=8, xy_sigma=5.0, tail_sigma=14.0, slice_jitter=2),
    "right_subarticular_stenosis": PerturbConfig(
        k=8, xy_sigma=5.0, tail_sigma=14.0, slice_jitter=2
    ),
}


def config_for(condition: str) -> PerturbConfig:
    return DEFAULT_CONFIGS.get(condition, PerturbConfig())


def sample_offsets(cfg: PerturbConfig, rng: np.random.Generator) -> list[tuple[float, float, int]]:
    """Draw ``cfg.k`` ``(dx, dy, ds)`` perturbations of the auto localization.

    In-plane: isotropic ``N(0, xy_sigma)`` with a heavy-tail mixture (prob
    ``tail_prob`` -> ``N(0, tail_sigma)``), clamped to ``+/- max_offset``. Slice:
    uniform integer in ``[-slice_jitter, slice_jitter]``. The list never includes
    the zero offset (the baseline is graded separately from the original cache).
    """
    out: list[tuple[float, float, int]] = []
    for _ in range(int(cfg.k)):
        sig = cfg.tail_sigma if rng.random() < cfg.tail_prob else cfg.xy_sigma
        dx = float(np.clip(rng.normal(0.0, sig), -cfg.max_offset, cfg.max_offset))
        dy = float(np.clip(rng.normal(0.0, sig), -cfg.max_offset, cfg.max_offset))
        ds = (
            int(rng.integers(-cfg.slice_jitter, cfg.slice_jitter + 1))
            if cfg.slice_jitter > 0
            else 0
        )
        out.append((dx, dy, ds))
    return out


# --------------------------------------------------------------------------- #
# stability statistics + grade (pure)
# --------------------------------------------------------------------------- #


def normalized_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a 3-class prob vector, normalised to [0, 1] (log base 3)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def stability_stats(probs: np.ndarray) -> dict:
    """Summarise prediction movement across perturbations.

    ``probs`` is ``(1 + K, 3)``: **row 0 is the baseline** (original auto crop),
    rows ``1..K`` are the perturbed crops. Returns P(severe) dispersion, mean
    entropy, severity-flip rate and agreement rate *relative to the baseline
    prediction* (computed over the K perturbations only).
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3 or probs.shape[0] < 2:
        raise ValueError("probs must be (1+K, 3) with K>=1 perturbations")
    base, pert = probs[0], probs[1:]
    base_pred = int(np.argmax(base))
    pert_pred = np.argmax(pert, axis=1)
    p_sev = probs[:, 2]
    ent = np.array([normalized_entropy(r) for r in probs])
    return {
        "baseline_pred": base_pred,
        "baseline_p_severe": float(base[2]),
        "p_severe_mean": float(p_sev.mean()),
        "p_severe_std": float(p_sev.std()),
        "p_severe_min": float(p_sev.min()),
        "p_severe_max": float(p_sev.max()),
        "p_severe_range": float(p_sev.max() - p_sev.min()),
        "entropy_mean": float(ent.mean()),
        "severity_flip_rate": float(np.mean(pert_pred != base_pred)),
        "agreement_rate": float(np.mean(pert_pred == base_pred)),
        "k_perturb": int(pert.shape[0]),
    }


def stability_grade(stats: dict, cfg: PerturbConfig) -> str:
    """Map stats to ``stable`` / ``mildly_unstable`` / ``unstable``."""
    flip = stats["severity_flip_rate"]
    rng_ = stats["p_severe_range"]
    std = stats["p_severe_std"]
    if flip >= cfg.unstable_flip or rng_ >= cfg.unstable_range:
        return "unstable"
    if flip == 0.0 and std <= cfg.stable_pstd:
        return "stable"
    return "mildly_unstable"


def instability_score(stats: dict) -> float:
    """Single continuous instability score in [0, 1] for triage/ROC analysis.

    Blends the severity-flip rate (does the decision change?) with the P(severe)
    range (how far does the severe probability swing?). Higher = less stable.
    """
    return float(0.5 * stats["severity_flip_rate"] + 0.5 * min(stats["p_severe_range"], 1.0))


# review reasons (must be a subset of ALLOWED_REVIEW_REASONS in the schema once
# registered there)
def stability_review_reasons(grade: str, condition: str) -> tuple[str, ...]:
    """Route-aware review reasons triggered by instability."""
    if grade == "stable":
        return ()
    reasons = ["evidence_unstable"]
    if grade == "unstable":
        if "subarticular" in condition:
            reasons.append("axial_candidate_disagreement")
        elif "foraminal" in condition:
            reasons.append("foraminal_slice_disagreement")
        else:
            reasons.append("route_unstable")
    return tuple(reasons)


def route_quality(grade: str, localizer_confidence: float | None) -> str:
    """Aggregate route-quality signal from stability + localizer/scorer confidence.

    ``good`` = stable and (no conf info or conf>=0.6); ``weak`` = unstable or very
    low localizer confidence; else ``fair``. Purely a reliability descriptor.
    """
    lc = localizer_confidence
    if grade == "unstable" or (lc is not None and lc < 0.35):
        return "weak"
    if grade == "stable" and (lc is None or lc >= 0.60):
        return "good"
    return "fair"
