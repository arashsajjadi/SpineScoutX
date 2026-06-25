# v1.8c — final real MedSAM2 results

> Research-only · not diagnostic · not clinically validated. `splits_v1`; dev selects; **locked-test
> read once** (the fusion eval). **No model deployed** (no raw gain); deployed 5/5 unchanged (macro
> 0.752).

## Locked-test foraminal severe recall — NO improvement

| arm | L-for | R-for | macro |
|---|---|---|---|
| deployed | 0.788 | 0.660 | 0.724 |
| v1.8b SAM2.1 fusion | 0.788 | 0.660 | 0.724 |
| **real MedSAM2 fusion** | **0.788** | **0.660** | **0.724** |

## Why (corrected vs v1.8b)

v1.8b ran SAM2.1 fallback; **v1.8c ran REAL MedSAM2** (proven: `sam2.modeling.sam2_base` +
`MedSAM2_latest.pt`). Real MedSAM2 segments every crop (0% fail) but its foraminal morphometry is
**weaker** than SAM2.1's (right-foraminal dev AUROC 0.551 vs 0.687) and equally **redundant** with
the image grader → fusion adds nothing (Δ+0.000), triage adds nothing. So the v1.8b SAM2.1 fallback
did **not** flatter or bias the conclusion: **segmentation-morphometry — even from a real medical
foundation model — is not the missing signal.**

## Verdict / tag

No raw severe-recall improvement, no triage improvement → **`v1.8c.0-real-medsam2-negative-result`**.
The one durable positive is **operational**: a real, reusable VisionServeX MedSAM2 integration.
