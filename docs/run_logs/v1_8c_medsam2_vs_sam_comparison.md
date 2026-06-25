# v1.8c — real MedSAM2 vs SAM2.1 vs SAM3 comparison (Phase 6)

> Research-only · not diagnostic. Reproduce: `compare_medsam2_sam21_sam3_v1_8c.py` (and the
> morphometry-only signal scripts).

| property | real MedSAM2 | SAM2.1 (v1.8b) | SAM3 |
|---|---|---|---|
| executed on RSNA foraminal | **yes (19,700, 0% fail)** | yes (19,700, 0% fail) | downloaded; not needed |
| mean segmentation score | 0.422 | 0.575 | — |
| foraminal opening area severe-vs-non-severe | flat | flat | — |
| **right-foraminal morphometry-only dev AUROC** | **0.551 (GBM)** | **0.687 (GBM)** | — |
| redundant with image grader | yes (fusion no raw gain) | yes (fusion α=0) | — |

**Decision: real MedSAM2 is NOT better than SAM2.1** for spine morphometry — it is *worse* on the
target right-foraminal route (AUROC 0.551 vs 0.687) and equally redundant. So the v1.8b SAM2.1
fallback did **not** bias the conclusion; running real MedSAM2 strengthens the negative.
