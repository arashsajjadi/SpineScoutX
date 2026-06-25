# v1.9 Model Package Verification — Kaggle Submission

> Research-only · non-commercial · not diagnostic.

## Package location

Local: `outputs/real/v1_9_packages/spinescoutx-best-raw-v1.9.tar.gz` (495.4 MiB)
GitHub Release: `v1.9.0-research-story-best-model`

## SHA-256 verification

```
452154208c346ea05529cd47f8913c54d22b150c87bd1b1ed1d612bffcedc9ba
```

Status: ✅ MATCHES `docs/assets/v1_9/checksums.txt`

## Model run directories used (local)

| Key | Path | Present |
|---|---|---|
| canal_grader | `runs/v1_canal_auto_robust` | ✅ |
| foraminal_grader | `runs/v1_foraminal_oracle_ctrl` | ✅ |
| subarticular_grader | `runs/v1_subarticular_auto_robust` | ✅ |
| canal_localizer | `runs/l0_disc_localizer_real` | ✅ |
| foraminal_localizer | `runs/lf_foraminal_localizer` | ✅ |
| subarticular_scorer | `runs/axial_level_scorer` | ✅ |

## Notes

- Local `runs/` weights used directly (same as tarball content).
- SHA-256 verified against published checksums.
- No weights committed to git history.
