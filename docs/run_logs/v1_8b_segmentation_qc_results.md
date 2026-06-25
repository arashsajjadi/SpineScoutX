# v1.8b — segmentation QC results (Phase 7)

> Research-only · not diagnostic. SAM2.1 (`facebook/sam2.1-hiera-base-plus`, transformers) box-prompt
> segmentation of the central foraminal-opening proxy on all 19,700 RSNA foraminal findings. RSNA
> has **no ground-truth masks**, so QC is shape/area/confidence plausibility (the SPIDER U-Net is the
> GT-mask reference for canal/disc; foramen has no GT). Reproduce:
> `run_foundation_segmentation_v1_8b.py`.

## SAM2.1 foraminal segmentation QC

| metric | value |
|---|---|
| findings segmented | 19,700 (train+dev+test) |
| segmentation-failure rate | **0.0%** (every crop produced a mask) |
| mean SAM2 IoU-confidence | 0.575 |
| throughput | ~19 crops/s (batch 8, RTX 5080, shared GPU) |
| mask area frac — severe vs non-severe | 0.0814 vs 0.0823 (**flat**) |
| min-opening proxy — severe vs non-severe | 0.3705 vs 0.3688 (**flat**) |

**Key QC finding:** masks are produced reliably (0% failure) but the **opening AREA does not separate
severe from non-severe** — a center-box SAM2.1 prompt segments the central object, not a calibrated
foraminal aperture. The discriminative morphometry is **intensity contrast** within vs around the
mask (see Phase 9), not geometry. MedSAM2 (medical SAM2) and SAM3 (open-vocab) were downloaded
(Phase 2) as comparators; given SAM2.1 already segments every crop at 0% failure and the limiting
factor is **calibration/redundancy** (not raw mask availability), the morphometry hypothesis is
testable directly on the SAM2.1 masks (Phases 9–11) — and the answer below does not depend on the
specific SAM variant.

## Source selection

For foraminal morphometry, **SAM2.1** is the QC-selected source (0% failure, reliable masks). The
SPIDER U-Net remains the validated canal/disc/vertebra source (Dice ≈0.884) for any future
canal/subarticular morphometry. No source produced a calibrated *aperture* mask on RSNA without GT.
