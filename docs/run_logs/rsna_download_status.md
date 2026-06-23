# RSNA / LumbarDISC download + manifest status

> Research-only, non-commercial. RSNA data is **not** committed or redistributed.

## Acquisition (Phase 2)
- Source: Kaggle competition `rsna-2024-lumbar-spine-degenerative-classification`
  (official; rules accepted under the user's account).
- Auth: Kaggle `KGAT_` token → `~/.kaggle/access_token`. The plaintext token file
  was deleted after a confirmed download.
- Downloaded ~29 GB zip → extracted to `data/raw/rsna` (gitignored). Total 34 GB,
  **147,218** DICOMs across **1,975** train studies (+1 sample test study).
- `spinescoutx doctor --data` → RSNA **READY**, SPIDER **READY**.

## Manifest / crop cache (Phase 3)
Command: `spinescoutx prepare-rsna --rsna-root data/raw/rsna --out data/cache/rsna`
(2.5D crops, 224², decode-once-per-series, cache-first/resume-safe).

| Quantity | Value |
|---|---|
| Studies with localizers | 1,974 |
| Series | 6,291 |
| Findings / crops | 48,692 |
| Patient/study split | 1,579 train / 395 val |
| Severity | normal_mild 37,626 · moderate 7,950 · **severe 3,081 (6.3%)** · unlabeled 35 |
| Localizer sequence | sagittal_t1 19,724 · axial_t2 19,220 · sagittal_t2 9,748 |

## Integrity checks (validated on a 300-study sanity pass + full run)
- **No study-level leakage** between train/val (verified: 0 studies in >1 split).
- Crops are `(3, 224, 224)` float32 ∈ [0,1]; 2.5D pad notes recorded
  (`dup_nearest_slice` only at slice edges).
- All conditions/levels/sides canonicalize; sequence types classified.
- The 35 unlabeled crops (`severity_index = -1`) are dropped before the loss.

Manifest report (machine-readable) at `outputs/real/rsna_manifest_report.json`
(gitignored). No raw data, DICOMs, crops, or caches are committed.
