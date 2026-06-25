# v1.8c — real MedSAM2 smoke test (Phase 3)

> Research-only · not diagnostic. **Proves real MedSAM2 was used, not `facebook/sam2.1-*`.**
> Reproduce: `smoke_real_medsam2_v1_8c.py`.

## Proof of real MedSAM2

| evidence | value |
|---|---|
| model class module | **`sam2.modeling.sam2_base`** (official sam2, NOT transformers) |
| predictor class module | **`sam2.sam2_image_predictor`** |
| runtime type | `in_process_sam2` (VisionServeX `MedSAM2Runtime`) |
| checkpoint | **`MedSAM2_latest.pt`** (`wanglab/MedSAM2`) |
| config | `configs/sam2.1/sam2.1_hiera_t.yaml` |

## Smoke result

A real RSNA foraminal crop (3×224×224 → HxWx3) segmented with a center box prompt produced a
**(224,224)** mask, area frac **0.075** (plausible foraminal-sized region). No images committed.
This is genuine MedSAM2 inference via the official `sam2` predictor + the MedSAM2 checkpoint — the
v1.8b SAM2.1-fallback flaw is corrected.
