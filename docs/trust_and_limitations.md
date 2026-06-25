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

## v1.4 — raw-accuracy audit (honest negative)
- A ruthless accuracy audit found **no bug depressing severe recall**; the deployed metrics are
  correct and reproduce exactly. So there is **no "free" accuracy** to recover from the pipeline.
- A direct test showed the v1.3 localization gain does **not** raise subarticular grading; raw
  severe recall is **grader-capacity / training-data limited**, not bug/decode limited.
- **Raw severe recall is unchanged in v1.4** (macro 0.752). We did not tag an accuracy upgrade.
  The next real gain needs more severe-class data or a higher-capacity (e.g. MIL) grader —
  honestly out of reach for a bounded sprint. This bounds how much to trust further "tuning".

## v1.3 — real evidence viewer + honest capability wins
- **Real evidence viewer:** committed cards are **pixel-free** (derived signals: auto crop
  centre, slice, mean intensity) to respect the RSNA data licence; the full real-pixel viewer
  runs locally only. Each card shows prediction vs **held-out reference** (never an input) with
  code-derived correctness. See [`real_evidence_asset_policy.md`](run_logs/real_evidence_asset_policy.md).
- **Two real, locked-test gains:** evidence-v3 severe-FN detection AUROC 0.833→0.863; axial
  decode ±1 slice-hit 0.432→0.487 (no retrain, β dev-selected). **Baseline severe recall is
  unchanged** — these improve *triage* and *localization*, not the headline recall.
- **Honest negatives kept:** evidence stability alone is redundant with confidence (only the v3
  *combination* helps); right-foraminal accuracy remains sample-limited (no specialist retrain);
  retrieval is side-aware only via metadata (the embedding does not encode laterality); the axial
  grading payoff is bounded (the robust grader already tolerates leveling noise).

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

---

## v1.9 update — convergent findings (v1.4–v1.8c)

The following limitations were confirmed across nine strategies and five release versions:

- **Label quality is the binding ceiling.** v1.4 proved no pipeline bug. v1.5 proved MIL
  and localization don't transfer to grading. v1.6 proved external data, SSL, anatomy prior,
  and larger backbone don't help (decisive losses). v1.7 proved label cleaning is rejected
  by dev (original labels win). v1.8b/v1.8c proved SAM2.1 and real MedSAM2 morphometry are
  redundant with the image grader. The raw accuracy ceiling (macro 0.752) requires
  **expert re-annotation**, not further model engineering.
- **BiGRU localization win does not transfer to grading.** v1.5 BiGRU improved axial ±1-
  slice-hit from 0.487 → 0.616 (decisive localization win), but re-cropping with improved
  crops did not improve grading (subarticular test 0.742→0.702). Grading is grader/data-
  limited, not localization-limited.
- **Triage improves effective safety but has a ceiling.** v1.7 triage reaches effective
  foraminal severe recall 0.933 at 15% budget, but captures only 76% of severe FN (22/29);
  the remaining 24% (7/29) are not surfaced by the deployed uncertainty signals. The deployed
  grader argmax is unchanged.
- **Expert re-annotation has not happened yet.** The v1.7 704-case review pack was created
  (338 right-foraminal + 366 left-foraminal; 87 R-for severe FN) and is awaiting expert
  radiologist labels. Raw accuracy will not improve until those labels (and ideally a clean
  test re-read) are available.
- **Real MedSAM2 is not better than SAM2.1 for this task.** v1.8c proved real MedSAM2
  morphometry (AUROC 0.551) is weaker than SAM2.1 (AUROC 0.687) and redundant with the
  image grader. Medical foundation model segmentation is not the missing signal.
- **v1.9 is a packaging sprint only.** No new accuracy claims. No new locked-test reads.
  Best raw model remains v1.0 deployed reference (macro 0.752).
