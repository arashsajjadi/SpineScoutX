# v1.2 — Real Case Viewer + Evidence Intelligence + Capability: plan

> Research-only · not diagnostic. Phase-0 plan for v1.2. No code/merge in this phase.

## Why the current README/cards are insufficient
- **The key question is unanswered.** A visitor still cannot see, for one study:
  *input/evidence route → model prediction → held-out reference label → correct/wrong →
  failure mode*. The v1.1 cards show the model's **output** but never place it next to the
  **held-out reference** with an explicit **correctness** verdict.
- **Cards are too compressed.** Measured: the v1.1 case cards are **1143×559 px** with up to
  10 findings crammed into one tiny-font table — unreadable on GitHub. The mission target is
  **wide (≥1600×900), large-font, one-concept-per-card**, sectioned (summary / evidence route
  / prediction-vs-reference / safety / correctness).
- **Charts ≠ a case viewer.** v1.1 added charts and structured cards but no per-case
  "viewer" that a non-expert can read standalone.

## What "real case viewer" means here
A per-study, anonymized object + rendered card(s) that show, side by side:
1. **Case summary** (category, max P(severe), review count, worst failure mode, route quality).
2. **Evidence route** (which view/localizer placed each crop; provenance = auto, no GT).
3. **Prediction vs held-out reference** — the model's severity estimate + P(severe) +
   confidence next to the **held-out reference label**, with a **correctness** verdict
   (exact_correct / severe_correct / severe_false_negative / severe_false_positive /
   uncertain_review).
4. **Safety/review** (evidence stability, route quality, review_required + reasons).
5. **Correctness/failure note** (one-line plain-language summary).

## How held-out reference labels are shown SAFELY
- The reference label is the RSNA grading target for that (study, level, side). It is
  **never an input** to the model (auto inference uses no GT coordinates and no GT severity);
  it is shown **only** for transparency, clearly tagged `held_out_reference_label` and
  visually separated from the prediction column. Correctness is **derived** (prediction vs
  reference) by code, never hardcoded. No raw DICOM pixels, no identifiers (hashed `case_*`).

## Model-improvement attempts worth trying (evidence-backed priority)
- **Evidence intelligence v2 (do):** add an **instability *type*** per finding
  (crop_sensitive / slice_sensitive / axial_candidate_sensitive / route_sensitive) measured
  by isolating slice-only vs in-plane-only perturbation (reuses the `sample_offsets(mode=...)`
  added in v1.1), plus condition-specific thresholds and a severe-FN-targeted review eval.
  This is a real triage-intelligence upgrade and feeds the viewer.
- **Axial scorer v2 (evidence-backed decision, not blind retrain):** v1.1 already showed (real
  locked-test data) the robust grader tolerates leveling noise → bounded grading payoff. v1.2
  will **complete the slice-vs-in-plane attribution** to quantify the grading-relevant slice
  share, and decide: escalate to a trained stack-sequence v2 only if slice dominates;
  otherwise record the honest negative + keep v1. (No multi-hour blind retrain that risks the
  mandatory merge for a payoff the evidence bounds.)
- **Right-foraminal (do, bounded):** v1.1 diagnosed it as sample-limited (56% of misses are
  confidently-normal; specialist non-decisive). Re-confirm + lean on the evidence-stability
  review mitigation rather than re-run the same non-decisive specialist.
- **Similar-case retrieval (do, bounded):** explanation-only kNN on deployed-grader penultimate
  features; metadata only; never changes predictions.

## What will be skipped and why
- **A from-scratch multi-hour axial v2 training** — bounded payoff (v1.1 evidence) + would
  risk the mandatory merge; gated behind the attribution result instead.
- **External labeled validation** — not feasible (RSNA strips scanner metadata; SPIDER lacks
  the 5 graded labels), documented in v1.1; v1.2 keeps the internal domain-shift stress test.

## Gates / safety (unchanged)
pytest + ruff + format + build + doctor + forbidden-file/large-file/claim scans + link checks;
research-only, no clinical claims; no DICOMs/weights/runs/outputs/caches/identifiers committed;
no GT in auto inference; no locked-test tuning; merge to `main` via PR with a merge commit
(tag-preserving) only after all gates pass.
