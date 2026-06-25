# LSS-MRI AISSLab — manual download (fallback for no-network environments)

> Research-only · non-commercial (CC BY 4.0). Do **not** commit the data; `data/external/` is
> gitignored. In this session the automated download succeeded, so these steps are the fallback.

1. Open the Mendeley record: https://data.mendeley.com/datasets/rgb77xm3jf/4
2. Download **`LSS MRI AISSLab Dataset_V0.2.zip`** (≈2.04 GB).
3. Place it at: `data/external/lss_mri_aisslab/lss_mri_aisslab_v0.2.zip`
4. Verify integrity (must match):
   `sha256sum data/external/lss_mri_aisslab/lss_mri_aisslab_v0.2.zip`
   → `592a294f93d575a16bccc2681c793eb1cfc6679fa2746ac50cbc8970f806b4b1`
5. Run `python scripts/prepare_lss_mri_aisslab.py` (verifies + extracts `Foramina_Detection/`).
6. Run `python scripts/prepare_lss_foraminal_v1_6.py` (builds the gitignored crop cache + manifest).

Direct (no-login) URL used by the script, from the Mendeley public-files API
(`/public-api/datasets/rgb77xm3jf/files?folder_id=root&version=4`):
`https://data.mendeley.com/public-files/datasets/rgb77xm3jf/files/6d9a0116-925d-4111-acb0-1e679f7dfd71/file_downloaded`

If the host is unreachable (firewalled CI/cron), Plan A is **blocked**; the offensive falls back to
Plan B (self-supervised pretraining on RSNA+SPIDER) per the v1.6 plan.
