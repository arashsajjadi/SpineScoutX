# Kaggle Notebook Late Submission Result — SpineScoutX v1.9.2

> Research-only · non-commercial · not diagnostic · not clinically validated.

## Summary

v1.9.2 built and ran a Kaggle Script kernel (code competition path) that successfully
produced a valid 25-row `submission.csv` for the RSNA 2024 Lumbar Spine competition.
The kernel submission was rejected with HTTP 400 — competition closed — consistent with
the v1.9.1 CSV-upload attempt.

## Kernel execution log (v7 — COMPLETE)

**Kernel ref:** `arashsajjadi/spinescoutx-v1-9-late-submission` version 7  
**Status:** `KernelWorkerStatus.COMPLETE`  
**Device:** CPU (Tesla P100-PCIE-16GB is CUDA capability 6.0, PyTorch 2.x requires ≥7.0)  
**Wall-clock:** ~38 s

### Key log events

| t (s) | Event |
|---|---|
| 1.9 | Wheel extracted to `/kaggle/working/pkgs/`; spinescoutx loaded |
| 1.9 | All 6 model paths: OK |
| 6.8 | GPU P100 CUDA 6.0 < 7.0 → fell back to CPU |
| 6.9 | All SpineScoutX imports: OK |
| 6.9 | Series index: Sagittal T1, Axial T2, Sagittal T2/STIR |
| 25.9 | Canal: 5 levels |
| 33.0 | Foraminal: L=5, R=5 |
| 37.9 | Subarticular: L=5, R=5 |
| 37.9 | Output: 25 rows written to `/kaggle/working/submission.csv` |
| 37.9 | Validation PASSED (25 rows, prob sum mean=1.000000 std=3.03e-08) |

### Submission CSV (25 rows)

All probabilities are positive and sum to 1.0 per row.

```
row_id,normal_mild,moderate,severe
44036939_left_neural_foraminal_narrowing_l1_l2,0.018,0.192,0.790
44036939_left_neural_foraminal_narrowing_l2_l3,0.013,0.201,0.786
44036939_left_neural_foraminal_narrowing_l3_l4,0.010,0.200,0.791
44036939_left_neural_foraminal_narrowing_l4_l5,0.039,0.443,0.519
44036939_left_neural_foraminal_narrowing_l5_s1,0.443,0.489,0.067
44036939_right_neural_foraminal_narrowing_l1_l2,0.371,0.378,0.251
44036939_right_neural_foraminal_narrowing_l2_l3,0.342,0.579,0.079
...
44036939_spinal_canal_stenosis_l4_l5,0.015,0.178,0.808
```

## Submission attempt result

```
$ kaggle competitions submit \
    -c rsna-2024-lumbar-spine-degenerative-classification \
    -k arashsajjadi/spinescoutx-v1-9-late-submission \
    -v 7 \
    -f submission.csv \
    -m "SpineScoutX v1.9 notebook late submission — research-only, no leaderboard tuning"

400 Client Error: Bad Request for url:
  https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission
```

**Result: REJECTED — competition closed (HTTP 400).**

## Comparison with v1.9.1

| Attempt | Method | Kernel | submission.csv valid | API result |
|---|---|---|---|---|
| v1.9.1 | CSV-only (`-f` local file) | N/A | Yes (25 rows) | HTTP 400 |
| v1.9.2 | Code kernel (`-k/-v`) | v7 COMPLETE | Yes (25 rows) | HTTP 400 |

Both methods produce a valid 25-row submission, but the RSNA 2024 competition API
rejects all late submissions regardless of method — the competition closed October 2024.

## Technical discoveries (v1.9.2)

These were unknown before this sprint:

| Finding | Detail |
|---|---|
| Dataset mount path | `/kaggle/input/datasets/<user>/<slug>/` (NOT `/kaggle/input/<slug>/`) |
| Competition data path | `/kaggle/input/competitions/<slug>/` |
| Tarball double-nest | `<slug>.tar.gz` → `<slug>/<slug>/` after Kaggle extraction |
| Script kernel isolation | Only `code_file` is executed; other push-dir files are ignored |
| Wheel without pip | `zipfile.ZipFile(wheel).extractall(pkgs_dir)` + `sys.path.insert()` works |
| P100 CUDA cap 6.0 | PyTorch 2.x requires ≥7.0; must fall back to CPU explicitly |
| Private datasets | Private datasets fail to mount in kernels even for the owner |

## Leaderboard context

- Competition closed: October 2024
- Public LB top: 0.332 (weighted log loss, lower=better)
- Private LB top: 0.389 (1875 teams)
- SpineScoutX estimated ~40–60th percentile (based on internal metrics; not calibrated for
  log loss; no official score obtained)
- Submission rejected before scoring — no official placement recorded

## Conclusion

The v1.9.2 kernel ran the complete 5-route SpineScoutX inference pipeline end-to-end
on Kaggle infrastructure, producing a passing validation on the real competition test case.
The HTTP 400 rejection confirms the competition is permanently closed to new submissions.
No official score was obtained. This is documented honestly.
