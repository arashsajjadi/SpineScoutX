# v1.8c — VisionServeX + MedSAM2 capability audit (Phase 1)

> Research-only · not diagnostic. Reproduce: `audit_visionservex_medsam2_v1_8c.py`.

## VisionServeX (local)

- Located at `/home/arash/PycharmProjects/VisionServeX`; the `visionservex` package (**v3.23.0**) is
  importable in this environment.
- **It already provides a real MedSAM2 backend:** `visionservex/medical/medsam2_runtime.py`
  (`load_medsam2_runtime` → `from sam2.build_sam import build_sam2` + `SAM2ImagePredictor`;
  `segment_2d(...)`), plus `medical/medsam2_batch.py` (order-preserving batch runner) and a
  separate `sam2_runtime.py` (the **transformers `facebook/sam2.1-*` fallback** — explicitly *not*
  MedSAM2).
- License/disclaimer surfaced by the runtime: MedSAM2 is **non-commercial**, **not for diagnosis**.

## Integration decision

**Use VisionServeX's `medsam2_runtime` directly** (it cleanly wraps the official `sam2` stack +
the MedSAM2 checkpoint) — no duplicate serving code. SpineScoutX adds only a thin morphometry
adapter on top. SAM2.1 (`sam2_runtime` / the v1.8b transformers path) and SAM3 remain **comparators
only**.
