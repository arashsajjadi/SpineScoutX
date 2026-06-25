# v1.8c — real MedSAM2 segmentation QC (Phase 5)

> Research-only · not diagnostic. **Real MedSAM2** (`sam2.modeling.sam2_base` + `MedSAM2_latest.pt`,
> config `sam2.1_hiera_t.yaml`) via VisionServeX runtime, center-box prompt, all 19,700 RSNA
> foraminal crops. Masks/features gitignored. Reproduce: `run_real_medsam2_segmentation_v1_8c.py`.

| metric | real MedSAM2 | v1.8b SAM2.1 (ref) |
|---|---|---|
| findings | 19,700 | 19,700 |
| seg-failure rate | **0.0%** | 0.0% |
| mean prompt/IoU score | 0.422 | 0.575 |
| throughput | ~32 crops/s (623 s total) | ~19 crops/s |
| area frac — severe vs non-severe | 0.0700 vs 0.0721 (flat) | 0.0814 vs 0.0823 (flat) |
| mask contrast — severe vs non-severe | −0.038 vs −0.014 | (contrast carried SAM2.1 signal) |

Real MedSAM2 reliably segments every crop, but — like SAM2.1 — the foraminal opening **area is flat**
across severities; MedSAM2's masks are *different* (lower mean score, different contrast sign) but
not more severity-informative.
