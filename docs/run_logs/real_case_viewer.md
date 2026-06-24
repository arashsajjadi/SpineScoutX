# Real case viewer pack

> Research-only · not diagnostic. Each card shows, for one anonymized locked-test study:
> input/evidence route → model prediction → **held-out reference label** → derived
> correctness → safety/review. No DICOM pixels, no identifiers (`case_*`). The held-out
> reference is shown for transparency only and is NEVER a model input.

| card | case_id | category | highest P(severe) | correct/ref | severe errors |
|---|---|---|---|---|---|
| `case_canal_correct_severe` | case_2ab10d949b | correct_severe | 0.73 | 18/25 | 0 |
| `case_left_foraminal_correct_severe` | case_1d9f0d7393 | correct_severe | 0.84 | 17/25 | 0 |
| `case_right_foraminal_hard` | case_66e55e9f11 | hard_right_foraminal | 0.88 | 7/25 | 11 |
| `case_subarticular_correct` | case_7723c07b18 | correct_severe | 0.65 | 21/25 | 0 |
| `case_axial_unstable` | case_0cec5cde30 | false_negative | 0.90 | 16/25 | 2 |
| `case_model_disagreement` | case_a574c8d563 | model_disagreement | 0.37 | 22/25 | 0 |
| `case_review_required` | case_f8490523c2 | false_negative | 0.89 | 14/25 | 10 |
| `case_mostly_normal` | case_edb5f4253e | mostly_normal | 0.05 | 25/25 | 0 |

Cards: `docs/assets/cases/*.png` (1800×1000, large-font, sectioned). Legend:
`prediction_vs_reference_legend.png`. Full JSON/MD pack: `outputs/real/case_viewer_pack/`
(gitignored). Schema: `case_viewer_v1` (`reporting/case_viewer.py`). Reproduce:
`python scripts/make_real_case_viewer_pack.py`.
