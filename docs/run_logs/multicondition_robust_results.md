# Multi-condition locked-test baselines + view-routing taxonomy

> Research-only. Not diagnostic. All-condition E0 retrained on splits_v1 `train`,
> selected on `dev`, evaluated ONCE on the locked `test`. Cluster-bootstrap 95% CIs.
> **Non-canal numbers are oracle (GT-coordinate) UPPER BOUNDS** — the auto-localized
> distribution cannot yet be generated for foraminal/subarticular (see taxonomy).

## Locked-test per-condition (oracle crops)

| condition | n / severe | severe recall [95% CI] | wll | severe AUROC | GT view |
|---|---|---|---|---|---|
| spinal_canal_stenosis | 1463 / 53 | 0.925 [0.849, 0.985] | 0.405 | 0.983 | sagittal_t2 |
| left_neural_foraminal_narrowing | 1480 / 52 | 0.769 [0.638, 0.891] | 0.551 | 0.973 | sagittal_t1 |
| right_neural_foraminal_narrowing | 1470 / 53 | 0.811 [0.712, 0.906] | 0.509 | 0.975 | sagittal_t1 |
| left_subarticular_stenosis | 1434 / 138 | 0.790 [0.718, 0.857] | 0.525 | 0.967 | axial_t2 |
| right_subarticular_stenosis | 1434 / 137 | 0.854 [0.778, 0.920] | 0.599 | 0.958 | axial_t2 |

**Canal AUTO (real inference) locked-test severe recall: 0.830 [0.725, 0.929]** (the only condition with a working auto-localizer; see `canal_locked_test.md`).

## View-routing feasibility taxonomy (the answer to 'does v0.9 generalize to all 5?')

| condition | GT view | auto-localization status |
|---|---|---|
| spinal_canal_stenosis | sagittal_t2 | available (sagittal-T2 disc localizer; v0.9 auto recipe) |
| left_neural_foraminal_narrowing | sagittal_t1 | BLOCKER: needs parasagittal-T1 side-aware localizer |
| right_neural_foraminal_narrowing | sagittal_t1 | BLOCKER: needs parasagittal-T1 side-aware localizer |
| left_subarticular_stenosis | axial_t2 | BLOCKER: needs axial-T2 localizer + level matching |
| right_subarticular_stenosis | axial_t2 | BLOCKER: needs axial-T2 localizer + level matching |

## Honest conclusion
- **1/5 conditions (canal)** has a working auto-localizer; the v0.9 robust recipe applies
  and is confirmed on the locked test (`canal_locked_test.md`).
- **2/5 (foraminal L/R)** are graded on **sagittal-T1** parasagittal side-specific
  slices. v0.9's 'slice doesn't matter' finding is canal-specific; foraminal needs
  correct parasagittal-T1 slice selection — a side-aware T1 localizer is the next step.
- **2/5 (subarticular L/R)** are graded on **axial-T2**. SPIDER has no axial anatomy and
  no axial localizer exists; an axial localizer + level matching is needed.
- So 'generalize v0.9 to all five' is **gated by view-specific localization**, not by the
  grading recipe. This is a routing/localization frontier, documented (not faked).
  The oracle baselines above bound what each condition could reach once auto-
  localization for its view exists.

Artifacts: `outputs/real/multicondition_robust_results.json`. Reproduce:
`python scripts/run_multicondition_v1.py`.
