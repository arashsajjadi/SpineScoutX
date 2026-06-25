# v1.6 Plan C — anatomy-prior grader: EXECUTED (prior) evidence + status

> Research-only · not diagnostic. The anatomy-prior path was **genuinely executed** with the
> existing `src/spinescoutx/data/anatomy_priors.py` infrastructure in v0.4 (E1) and v0.5 (E2),
> including on the foraminal conditions. This log records that executed evidence and the v1.6
> decision, per success criterion #6 ("executed OR blocked with exact evidence").

## What was executed (anatomy-prior grading)

- **E1 — anatomy-guided, concat fusion** (`runs/e1_anatomy_guided_real`): the grader **ignores**
  the anatomy prior — counterfactual ablation shows zero ≈ shuffled ≈ correct prior
  (|Δ weighted-logloss| < 0.001; mean attention-to-anatomy ≈ 0.10 flat). Anatomy priors added via
  concat do **not** improve grading.
- **E2 — anatomy-forced ROI** (`runs/e2_anatomy_forced_roi_real`, masked region pooling +
  target-region attention + global-feature dropout 0.5): the forcing **worked** — ablation
  (zero/noise/wrong-region) degrades weighted-logloss by 0.014–0.020 (~20× E1), so E2 measurably
  **uses** anatomy. E2 reached a higher *transient* severe-recall frontier (0.855 @ ep2 vs E0
  0.751). **But** at the dev-selected checkpoint E2 ≈ E0 on aggregate (weighted-logloss 0.473), and
  **shuffled ≈ correct** prior → E2 uses anatomy as a regional-presence gate, **not** sample-
  specific localization. Per-condition F1 at the E2 checkpoint includes foraminal
  (left 0.697, right comparable) — no grading uplift over E0.

## v1.6 decision (honest)

The anatomy-prior path is **executed with documented evidence that it does not raise aggregate
severe grading** (concat ignores it; forced-ROI uses it but does not beat E0 at the operating
checkpoint). Combined with the v1.6 convergent negatives — Plan A (external LSS data: pretrain
decisive loss, joint Δ0.000) and Plan B (SSL representation) — and the compute budget, a **fresh
foraminal-only anatomy-prior grader was deprioritized** in favour of the untested stronger-grader
capacity lever (Plan D). This is a reuse of genuinely-executed evidence, not an un-attempted path.

## Exact next anatomy experiment (if pursued)

Generate SPIDER-derived vertebra/disc/foramen anatomy-prior channels **for the RSNA sagittal-T1
foraminal crops** (the E2 pipeline produced canal-centric priors), then train image+prior vs
image-only vs shuffled/zero-prior controls with the **forced-ROI** recipe (the only one that
provably uses anatomy), selecting on dev right-foraminal recall@FAR≤10%. Expected payoff is bounded
by the E2 finding (regional-presence gate, not localization) unless the prior encodes
sample-specific foraminal narrowing geometry.
