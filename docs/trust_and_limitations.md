# Trust & limitations — how much can you rely on SpineScoutX?

> Research-only · not diagnostic · not clinically validated · not for medical
> decision-making. **Category: academic research prototype — NOT an FDA/CE-cleared medical
> product.** This page states plainly what is reliable, what is not, and why.

## What is reasonably reliable (within the stated scope)
- **Locked-test discipline.** All headline numbers are on a **patient-level locked test**
  (`splits_v1`: train 1382 / dev 296 / test 296 studies) used once for final eval; models
  are selected on `dev`, never on `test`. Historical seed-1337 val results are kept separate.
- **Auto = real inference.** Reported severe recall is on **auto-localized** crops with **no
  ground-truth coordinates at inference**; every metric is tagged oracle (upper bound) vs
  auto, with cluster-bootstrap 95% CIs and paired tests.
- **Severe-first behaviour.** Canal auto severe recall 0.830 [0.725, 0.929]; subarticular
  0.746 / 0.737; a Safety Mode frontier + `review_required` layer make the severe-vs-review
  trade-off explicit.
- **Auditable outputs.** Finding graphs are deterministic, schema-validated, and proven
  model-derived (not templated) by CI tests; no diagnosis/treatment wording; anonymized.

## Real case viewer (v1.2) — see correctness for yourself
- Every showcased study now shows the **model prediction next to the held-out reference label**
  with a **code-derived** correctness verdict. The reference is shown for transparency only and
  is **never** a model input (auto inference uses no GT). This makes the system's mistakes
  (severe false negatives / positives) visible, not hidden behind aggregate metrics.
- **Similar research-case retrieval is explanation-only** — it never changes a prediction. It
  retrieves severity-relevant neighbours (severity agreement ≈0.73–0.89) but is **side-agnostic**
  (same-side rate ≈chance): the image embedding captures morphology, not laterality.
- **Instability typing (v2)** names the cause of an unstable finding but does not make it correct;
  it is a triage/explanation aid. **Right-foraminal remains the weakest route** — reconfirmed in
  v1.2 as signal/sample-limited (its instability is `slice_sensitive`, i.e. best-slice choice).

## Evidence-aware reliability (v1.1) — honest summary
- **Evidence stability** (re-grading under plausible localizer perturbation) is a *real*
  signal: it predicts errors (AUROC 0.80) and severe FNs (0.71) above chance — **but it is
  largely redundant with confidence**, so we do **not** claim it beats confidence in general.
  It **does** add severe-FN triage value on the two weakest **right-side** routes, and it is
  surfaced as `evidence_stability` / `route_quality` / the `evidence_unstable` review reason.
- **Calibration is a documented negative.** The graders are already well-calibrated (test ECE
  0.03–0.08); a dev-fit temperature does not transfer, so **no temperature is applied** in the
  deployed path. We report this rather than ship a calibration that doesn't help.
- **Internal domain-shift only.** Severe recall is robust to in-plane resolution and matrix
  size, mildly sensitive to slice thickness, and **weaker at L5-S1 (0.579) than L4-L5
  (0.868)**. This is *internal* robustness, not external generalization.

## What is NOT reliable / NOT done
- **No external validation.** Single dataset (RSNA LumbarDISC); **no** other-institution,
  prospective, scanner-diversity, or reader-study validation. External labeled validation is
  **not feasible** with available legal data (RSNA strips scanner/vendor/field-strength;
  SPIDER lacks the five graded labels) — so it was **not performed**.
- **No clinical validation, no regulatory clearance.** Not a medical device.
- **Right-foraminal is weaker** (auto severe recall 0.660 [0.524, 0.788]) and trails left
  (0.788); the gap is within CI overlap (n_severe ≈ 53) and a right-specialist did not
  decisively help — the limit is sample size.
- **Axial level scorer is imperfect** (±1 axial-slice hit 0.43). Subarticular auto works
  because the grader **tolerates** the leveling noise, not because leveling is solved; a
  better scorer is the main pending improvement.
- **Modest severe counts** (≈52–138 per condition) → wide CIs; small differences are not
  decisive (we say so).
- **No PACS/RIS workflow validation, no latency/robustness testing in a clinical setting.**
- **Anatomy ≠ pathology.** SPIDER masks are anatomy; foraminal/subarticular evidence regions
  are approximate.

## Why locked-test helps (but is not enough)
A locked test prevents tuning-on-the-test-set optimism and gives honest out-of-sample
estimates. It does **not** establish generalization to new institutions, scanners,
populations, or prospective use — that requires external and prospective studies which have
**not** been done here.

## How to read a finding graph
`severity_estimate` is the argmax of a calibrated 3-class probability; `P(severe)` and
`calibrated_confidence` quantify uncertainty; `review_required` (deliberately conservative)
flags low-confidence / borderline / model-disagreement / axial-level-uncertain findings for
**human research review**, not triage. `auto` provenance = real inference; `reference_label`
is a held-out research target shown for transparency only.

## One-line summary
Strong, honestly-measured **research** results on one locked test; **not** a validated or
clinical system. Use it to study anatomy-grounded auto-inference and severe-safety, not to
make any medical decision.
