# v1.8b — foundation model download audit (Phase 2)

> Research-only · not diagnostic. Downloaded with the user's local HF token (never printed) into
> **gitignored** `data/models/` (7.2 GB total; weights never committed). Reproduce:
> `download_foundation_seg_models_v1_8b.py`.

| model | HF id | gating | local size | status | notes |
|---|---|---|---|---|---|
| **MedSAM2** (Plan A) | `wanglab/MedSAM2` | ungated | 0.16 GB | downloaded ✓ | ships original-SAM2 `.pt` (MedSAM2_latest.pt); state-dict loads; **inference needs the `sam2` package** (not yet installed) or a transformers Sam2 state-dict graft |
| **SAM 3** (Plan B) | `facebook/sam3` | manual (already accepted) | 6.9 GB | **downloaded ✓ (accessible)** | transformers `Sam3Model` + `model.safetensors`; the v1.8b smoke used a wrong processor kwarg (`text=`) — an **API mismatch, not a license block**; the open-vocab text-prompt API is wired up in Phase 5 |
| **SAM 2.1** (fallback) | `facebook/sam2.1-hiera-base-plus` | ungated | 0.65 GB | **downloaded ✓ + smoke OK** | clean transformers `Sam2Model` box-prompt inference; the practical foundation-seg workhorse |

## Outcome

All three foundation models are **locally available**. **Neither MedSAM2 nor SAM3 is license-blocked**
(MedSAM2 ungated; SAM3 manual-gate already accepted via the user's HF approval). SAM 2.1 runs
cleanly out of the box via `transformers` and is the immediate workhorse; MedSAM2 (Plan A) and SAM 3
(Plan B, open-vocab) are wired in Phases 4–5 and QC'd against the reusable SPIDER U-Net (Plan C).
Weights/configs all stay gitignored.
