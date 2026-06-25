# v1.7 teacher-distillation fallback (Phase 7) — EXECUTED NEGATIVE

> Research-only · not diagnostic. License-clean lightweight teacher = **ensemble of SpineScoutX's
> own foraminal models** (deployed grader + v1.6 baseline + v1.6 LSS), applying public RSNA-2024
> *strategies* (per-finding models, ensembling, soft probabilities) — **no external code/weights
> copied**. Student = fresh convnext_tiny trained CE(original label) + KL(student‖teacher) + severe
> upweight; dev-selected on right-foraminal recall@FAR≤10; locked-test once. Reproduce:
> `build_kaggle_teacher_distillation_v1_7.py`.

## Result (locked-test)

| | dev R-for recall@FAR≤10 | TEST L-for severe | TEST R-for severe | TEST foraminal macro |
|---|---|---|---|---|
| deployed / v1.6 baseline | ~0.79 | 0.788 / 0.769 | 0.660 / 0.679 | 0.724 |
| **teacher-distilled student** | 0.854 | **0.538** | **0.453** | **0.496** |

## Verdict — NEGATIVE

The student's high **dev** recall@FAR≤10 (0.854) **did not generalise**: locked-test foraminal
severe recall **collapsed to 0.496** (vs baseline 0.724). The distilled student converged to a
conservative operating point (low FAR), so its argmax severe recall is far below baseline — the same
dev→test overfit / conservative-collapse seen for the v1.6 LSS and convnext-small arms. An ensemble
teacher built from models that are all ~0.66–0.72 on right-foraminal cannot exceed that ceiling;
distillation reproduces, and here worsens, it. No accuracy gain.

## Kaggle-strategy review (writeups only; no code/weights reused)

RSNA-2024 top solutions use: sagittal/axial separation, disc-level + series-level heads, large
ensembles + TTA, and soft probabilities. We reuse the *ideas* (per-finding model, ensembling, soft
targets) via our own models — they do not move the in-domain severe ceiling, consistent with v1.4–
v1.6. The binding constraint remains label quality, not ensembling/distillation.
