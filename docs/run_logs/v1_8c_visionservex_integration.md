# v1.8c — VisionServeX integration (Phase 11)

> Research-only · not diagnostic. Real MedSAM2 is served through **VisionServeX** — no duplicate
> model-serving code in SpineScoutX.

- **Adapter:** `src/spinescoutx/segmentation/medsam2_runner.py` wraps
  `visionservex.medical.medsam2_runtime` (`load_medsam2_runtime` → `sam2.build_sam.build_sam2` +
  `SAM2ImagePredictor`; `segment_2d`). SpineScoutX calls VisionServeX **in-process** (no blocking
  web service), so CI/tests stay offline.
- **Mockable / CI-safe:** `MedSAM2.available()` gates on `visionservex` + `sam2` + the local
  checkpoint; the module imports without any of them, so the test suite never needs weights.
- **Start command (local):** weights at `data/models/medsam2/MedSAM2_latest.pt` (gitignored);
  `python scripts/smoke_real_medsam2_v1_8c.py` validates the runtime.
- **License/disclaimer** surfaced by the runtime: MedSAM2 is **non-commercial**, **not for
  diagnosis**.

**Operational value:** the integration is real and reusable (SpineScoutX can now run any
VisionServeX MedSAM2/SAM2 backend). It is, however, **not an accuracy upgrade** (see fusion results).
