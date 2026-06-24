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

## What is NOT reliable / NOT done
- **No external validation.** Single dataset (RSNA LumbarDISC); **no** other-institution,
  prospective, scanner-diversity, or reader-study validation.
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
