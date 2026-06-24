# Axial-first v1 + showcase — plan + audit

> Research-only. Not diagnostic. Priority: **core capability first, presentation second**
> (~70–80% effort on accuracy/auto coverage, ~20–30% on README/showcase). The mission
> fails if it only makes the README prettier without advancing real model capability.

## Current state
- **Auto coverage 3/5:** spinal canal stenosis (sagittal-T2, robust auto-trained), left &
  right neural foraminal narrowing (sagittal-T1, oracle-trained on auto). Subarticular L/R
  remain oracle-only.
- **Locked-test protocol** (`splits_v1`) in place; all auto claims retrain on `train`,
  evaluate once on `test`, with cluster-bootstrap CIs.

## Why subarticular is the real scientific blocker (Priority 1)
Subarticular stenosis is graded on **axial-T2**. The gating sub-problem is assigning each
lumbar level to the correct axial slice with no GT. Pure DICOM-geometry matching
(sagittal-disc-z → axial-slice-z) was measured insufficient: **27.5% within ±1 slice**
(median 2 slices / 12.8 mm). A new diagnostic (this milestone, dev split) shows *why* and
*how to fix it*: the geometry error is a large systematic **per-level bias** (−2.1 slices
at L1/L2 → +4.0 at L5/S1) plus noise — but each level sits at a **predictable normalized
z-position** in the stack (l1/l2 0.82 … l5/s1 0.14). So the fix is a **coordinate-supervised
axial level scorer** (appearance ⊕ normalized-z, monotonic decoding), not more geometry.
The in-plane subarticular centre is also highly consistent per side (left x/cols 0.549,
right 0.456, y 0.52; std ~0.02–0.04) → a fixed supervision-derived offset replaces an
in-plane localizer.

## Why right-foraminal needs refinement (Priority 2)
Right-foraminal auto severe recall (0.660) trails left (0.788) on the locked test despite a
clean localizer. Audit candidates: laterality mapping, parasagittal slice choice, side
asymmetry, shared-vs-specialist grader, model selection.

## Documentation / presentation problems (Priority 2, after core)
README is dry, text-heavy, and does not convey value in 30 s: no visual pipeline, no
coverage badge, no results cards, no example finding graph, no gallery. Technical depth is
fine in `docs/`, but the top README needs a visual-first overhaul (honest about blocked
findings).

## Execution order (accuracy first, showcase second)
1. **Axial subarticular** coordinate-supervised route → unlock 4/5 or 5/5, or a stronger
   *supervised* blocker (gated on the level scorer's dev slice-hit vs the 27.5% baseline).
2. **Right-foraminal** refinement (specialist / slice-selection / threshold).
3. **Router** = best grader per condition; **Safety Mode v4** across available auto findings.
4. **Visual README + gallery + assets + report v4** (the 20–30%), honest about coverage.
5. Quality gate, push, honest tags. **v1.0 only if 5/5 real auto locked-test results exist.**

Provenance on every metric: oracle (upper bound) vs auto (real inference); split (dev/test);
n / n_severe / 95% CI. No GT coordinates at auto inference.
