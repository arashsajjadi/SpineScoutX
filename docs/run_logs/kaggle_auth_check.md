# Kaggle Auth Check — SpineScoutX v1.9 Late Submission

> Research-only · non-commercial · not diagnostic.

## Auth status

- `~/.kaggle/kaggle.json` present; permissions 600
- username: `arash`
- key: present (37 chars, not printed)
- Kaggle CLI version: 2.2.2

## Verification

```
kaggle competitions files -c rsna-2024-lumbar-spine-degenerative-classification
```

Result: authenticated; competition files listed (sample_submission.csv + test DICOMs visible).

## Sample submission file

`sample_submission.csv` (2545 bytes) — available via Kaggle API.

## Notes

- Token source: `~/.kaggle/kaggle.json` (pre-existing from v1.8b work).
- No token printed, committed, or exposed.
- Token file in `/home/arash/Documents/api_kaggle.txt` is local-only; not in repo.
