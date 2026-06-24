# Real-data evidence case viewer (v1.3)

> Research-only · not diagnostic. Each card shows, for a real locked-test case (hashed
> `case_*`): **A** real derived evidence signals (auto crop centre, slice, mean intensity
> — pixel-free), **B** prediction vs **held-out reference** + code-derived correctness,
> **C** evidence-v3 safety/review + side-aware (v2) similar cases. No DICOMs, no
> identifiers, no GT at inference (`real_evidence_asset_policy.md`). The full real-pixel
> viewer is generated locally under `outputs/real/evidence_case_viewer/` (gitignored).

| card | case_id | category |
|---|---|---|
| `case_canal_correct_severe` | case_2ab10d949b | correct_severe |
| `case_left_foraminal_correct_severe` | case_1d9f0d7393 | correct_severe |
| `case_right_foraminal_hard` | case_66e55e9f11 | hard_right_foraminal |
| `case_subarticular_correct` | case_7723c07b18 | correct_severe |
| `case_axial_unstable` | case_0cec5cde30 | false_negative |
| `case_model_disagreement` | case_a574c8d563 | model_disagreement |
| `case_review_required` | case_f8490523c2 | false_negative |
| `case_mostly_normal` | case_edb5f4253e | mostly_normal |

Reproduce: `python scripts/make_real_evidence_case_viewer.py`.
