# v1.5 — Full Raw Accuracy Offensive: plan

> Research-only · not diagnostic. v1.4 proved no bug + no decode→grading transfer but **did not
> execute new training**. v1.5 **executes real retraining**: candidate-bag MIL graders for the
> weak routes + a full axial stack scorer. An *executed* negative is acceptable; an
> analysis-only negative is not.

## Baseline (locked-test auto severe recall, main `1f7fd91`)
canal 0.830 · L-for 0.788 · **R-for 0.660** · L-sub 0.746 · R-sub 0.737 · **macro 0.752**.

## Mandatory experiments (hard-stops if skipped)
1. **Candidate bags** (foraminal L/R sagittal-T1 top-k; subarticular L/R axial top-k), gitignored.
2. **Right-foraminal MIL** grader — real training, dev-selected, locked-test once.
3. **Subarticular axial MIL** grader — real training, dev-selected, locked-test once.
4. **Axial stack-sequence scorer (BiGRU)** — real training attempt (slice-hit + downstream).

## Secondary (scout / bounded / documented)
External labeled data (scout; RSNA terms restrict redistribution — likely RSNA-only); SSL
pretraining (document cost/feasibility); exhaustive loss/aug sweeps (bounded to best configs);
ensemble/router (only if a specialist improves a route).

## Compute / disk / GPU plan
RTX 5080 16 GB (2.5 GB used), 304 GB disk free. Bags cached as 224² 2.5D crops (gitignored,
`data/cache/v1_5_candidate_bags/`). MIL = ConvNeXt-Tiny encoder (warm-start from the deployed
grader) over K=5 crops → attention pool → 3-class head; bounded epochs + small config sweep.

## Rollback / checkpoints
`git tag v1.5-precache-start` (set). Commit after every safe milestone (builder, each model,
each eval) so any step is revertible. Tags pushed only after a stable, gated result.

## Selection rules (strict)
**dev** selects every model/config/threshold; **locked-test once** per final selected family.
No test tuning; no hidden GT in auto inference; reference labels never an input. Cluster-bootstrap
CIs; report FAR with any recall change; reproduce baseline first (done in v1.4, exact).

## What counts as success (≥1)
R-for severe recall +≥0.05 · subarticular macro +≥0.03 · overall macro +≥0.03 · high-confidence
severe-FN −≥20% (without FAR blow-up) · recall@FAR≤10% materially up on ≥2 weak routes.

## What counts as a valid negative
All mandatory trainings **executed** (bags built, MIL trained for R-for + subarticular, axial
stack attempted) with logged metrics, and **no** target met → `v1.5.0-full-retraining-negative-
result`. Not acceptable: skipping training, or stopping at "sample-limited" without executing.
