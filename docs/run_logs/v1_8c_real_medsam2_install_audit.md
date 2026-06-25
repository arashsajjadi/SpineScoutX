# v1.8c — real MedSAM2 install audit (Phase 2)

> Research-only · not diagnostic. The v1.8b blocker ("sam2 package missing") is **fixed**.

| item | value |
|---|---|
| sam2 package | **installed** `sam2==1.1.0` (+ hydra-core 1.3.3, iopath 0.1.10, portalocker 3.2.0) |
| MedSAM2 checkpoint | `data/models/medsam2/MedSAM2_latest.pt` (≈156 MB, gitignored, from v1.8b HF download `wanglab/MedSAM2`) |
| config resolved | `configs/sam2.1/sam2.1_hiera_t.yaml` (MedSAM2 = SAM2.1 hiera-tiny medical finetune) |
| loader | VisionServeX `load_medsam2_runtime` → `sam2.build_sam.build_sam2` + `sam2.sam2_image_predictor.SAM2ImagePredictor` |
| env impact | additive pip install into the existing env; no SpineScoutX deps broken (gates pass) |

Nothing from `sam2` or the checkpoint is committed (gitignored). Real MedSAM2 imports and builds.
