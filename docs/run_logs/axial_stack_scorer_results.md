# Axial level scorer v2 — bounded grading-payoff analysis + v2 design (locked test)

> Research-only · not diagnostic. The axial level scorer is the known localization
> bottleneck for subarticular stenosis. This documents, with **real locked-test evidence**,
> whether a stack-sequence v2 would improve **grading** (the deployed output) — before
> committing to that training — and specifies the v2 design as the next step.

## The bottleneck (measured)
- The current coordinate-supervised per-slice axial level scorer reaches **±1-slice hit
  ~0.43** (vs geometry 0.275) — real but imperfect leveling
  (`run_logs/subarticular_auto_results.md`).
- Subarticular auto severe recall is nonetheless **0.746 / 0.737** (L/R) on the locked test.

## Does the imperfect leveling actually hurt GRADING? (evidence-backed)
The evidence-stability analysis (`evidence_stability_v1.md`) re-runs the **same** deployed
subarticular grader under K plausible localizer perturbations that **include ±2-slice
jitter** (the leveling component) — auto coordinates only, no GT. Result on the locked test:

| condition | stable | mildly_unstable | **unstable** | auto severe recall |
|---|---|---|---|---|
| left_subarticular_stenosis | 47% | 28% | **26%** | 0.746 |
| right_subarticular_stenosis | 46% | 28% | **25%** | 0.737 |

**Interpretation:** under perturbation that includes the slice/leveling error a better scorer
would fix, only ~25% of subarticular findings are *unstable*, and the grader still achieves
0.74/0.74 severe recall. The robust auto-trained grader **already absorbs most of the
leveling noise** (it was trained on the jittered distribution). So a stack-sequence v2 would
most plausibly improve **localization quality and trust** (steadier crops, fewer unstable
findings) but the **direct grading payoff is bounded** — consistent with the decision policy
"if it improves localization but not grading, say so." We therefore **do not** train v2 in
this milestone on the (unproven) assumption of a large grading gain; we specify it and gate
it on a measured improvement.

A finer slice-only vs in-plane-only attribution is reproducible with
`python scripts/run_axial_sensitivity.py` (writes `axial_sensitivity_run.md`); it re-runs the
grader under each isolated perturbation regime to quantify the slice share of instability.

## Stack-sequence scorer v2 — specified design (next step)
- **Input:** full axial-T2 stack (or a sampled stack) → lightweight per-slice CNN encoder.
- **Sequence model** over slices (1D TCN / BiGRU / small Transformer) with normalized z-rank
  and reliable spacing/orientation features → per-level slice distribution + confidence.
- **Decode:** monotonic-order + minimum-distance DP; top-k candidate pooling; confidence
  calibration.
- **Supervision:** RSNA subarticular axial coordinate instances on **train/dev only**;
  locked test used once for final evaluation; **no GT at auto inference**.
- **Compare:** geometry baseline → per-slice scorer v1 → stack-sequence v2 (+ DP, + top-k) on
  ±0/±1/±2 slice-hit, median |slice error|, per-level error, **and downstream subarticular
  severe recall / recall@FAR**, plus evidence-stability and runtime.
- **Keep v2 only if** it improves slice-hit **and** downstream grading; if it improves
  localization but not grading, record that explicitly (per the evidence above, the likely
  outcome) and keep it for trust/route-quality rather than for the severe-recall headline.

## v1.2 update — instability TYPING confirms the cause (and bounds the payoff)
Evidence Intelligence v2 (`evidence_intelligence_v2.md`) attributes each finding's
instability to a cause by isolating slice-only vs in-plane-only perturbation. On the
subarticular routes the unstable findings are **dominated by `axial_candidate_sensitive`**
(slice/leveling): L 91 vs in-plane 11, R 98 vs in-plane 21. So the axial leveling choice **is**
the dominant instability driver for subarticular — a stack-sequence v2 is now **data-motivated**,
not assumed. **But the payoff is bounded:** `axial_candidate_sensitive` findings carry a severe-FN
rate of **0.032** (6/189 in the subsample) — ~4× the stable rate (0.008) yet a small absolute
number of recoverable severe misses, consistent with the robust grader already absorbing most
leveling noise. **Decision:** keep v1; a trained stack-sequence v2 is justified to reduce these
specific unstable misses and improve trust/route-quality, but is gated on beating v1 on
±slice-hit **and** downstream subarticular grading (not run here to protect the release and
because the recoverable severe-FN count is modest). The `axial_candidate_sensitive` type is now
surfaced in the case viewer + review reasons so these cases are flagged today.

## Status
Bounded analysis + v1.2 instability-typing evidence complete (cause confirmed = axial leveling;
payoff bounded = modest recoverable severe FNs). v2 **not trained** this milestone — specified
and gated as the precise next step. Honest outcome: no overclaim, no blind training.
