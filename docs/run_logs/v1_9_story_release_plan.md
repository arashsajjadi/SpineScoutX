# v1.9 — Research Story, Real-Case Gallery, Best-Model Weights, README Polish, Reproducible Release

> Research-only · non-commercial · not diagnostic · not clinically validated.

## What this sprint is NOT

**This is not an accuracy-improvement sprint.** No new model is trained. No locked-test is
re-read for tuning. No raw metric changes.

## What this sprint IS

A **packaging, storytelling, reproducibility, and scientific transparency sprint** to:

1. Summarize the full journey from v1.0 to v1.8c clearly and honestly.
2. Generate clean charts explaining what was tried, what worked, what failed.
3. Publish a license-gated real-case gallery (or local-only instructions if gate fails).
4. Package the best model weights safely through GitHub Release assets (not ordinary Git).
5. Overhaul README/docs for clarity and credibility.
6. Add reproducibility scripts.
7. Run full quality gates and merge cleanly.

## Best raw model position

**The best raw deployed model remains the v1.0 deployed reference graders**, with locked-test
5-route macro severe recall **0.752** (canal 0.830, L-for 0.788, R-for 0.660, L-sub 0.746,
R-sub 0.737). None of v1.1–v1.8c improved raw argmax severe recall. This is stated honestly, not
buried.

## Triage/safety config (separate from raw model)

**v1.7 severe-FN triage** is the best *safety* config: at 15% review budget, effective foraminal
severe recall improves from 0.724 → 0.933 (22/29 severe FN captured). This is packaged separately
as "best safety/triage config" — it does NOT change argmax predictions.

## Real-image gallery

A license/privacy gate (`scripts/check_real_image_release_gate_v1_9.py`) must pass before any
real medical image panels are committed. Gate conditions:

* repo is private OR explicit redistribution permission exists
* no patient identifiers in pixels or metadata
* panels are small, derived visual examples, NOT raw DICOMs / full series
* no DICOM/NIfTI committed
* file names use anonymized/hash IDs

If gate passes: commit PNG panels to `docs/assets/v1_9/real_cases/`.
If gate fails: generate gallery locally under gitignored `local_reports/v1_9_real_case_gallery/`,
commit only pixel-free schematic examples and instructions.

## Weights publishing strategy

No weight file committed to ordinary Git history:

* If any file > 50 MiB: use **GitHub Release asset** (`gh release upload`).
* Git LFS used only if strictly necessary.
* Packaged as `spinescoutx-best-raw-v1.9.tar.gz` (best raw model) and
  `spinescoutx-triage-config-v1.9.tar.gz` (triage config).
* SHA-256 checksums published in `docs/assets/v1_9/checksums.txt`.

## Full gates before merge

pytest · ruff check · ruff format --check · python -m build · spinescoutx doctor --data ·
`python scripts/check_release_safety_v1_9.py` · no secrets/DICOMs/large files staged.
