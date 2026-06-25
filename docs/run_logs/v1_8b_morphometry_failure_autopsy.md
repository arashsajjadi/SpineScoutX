# v1.8b — segmentation-morphometry failure autopsy

> Research-only · not diagnostic. The MedSAM2/SAM3.1/SAM2.1 → morphometry → fusion path executed
> end-to-end with no raw severe-recall gain. This autopsy answers the exact questions.

- **Did MedSAM2 produce usable masks?** MedSAM2 (Plan A) was **downloaded** (ungated, `.pt`); its
  inference needs the `sam2` package (not installed). **SAM2.1** (same SAM2 family) was used instead
  and produced masks on **100% of crops (0% failure)** — mask *availability* was never the blocker.
- **Did SAM3.1 produce usable masks?** SAM3 (Plan B) was **downloaded and is accessible** (manual
  gate already accepted; 6.9 GB, transformers); it was not required because the limiting factor is
  **redundancy**, not mask quality (below).
- **Did the spine-specific fallback win QC?** Not needed — SAM2.1 segmented every crop; the SPIDER
  U-Net (Dice ≈0.884) remains the validated canal/disc reference for future work.
- **Did morphometry-only contain signal?** **Yes** — right-foraminal dev AUROC 0.687, but from
  **intensity contrast**, not opening geometry (mask **area is flat** severe-vs-non-severe).
- **Did fusion improve dev?** **No** — the dev α-sweep chose α=0 (pure deployed grader).
- **Did fusion generalize to locked-test?** N/A — α=0 means fusion ≡ baseline (Δ+0.000).
- **Which anatomical measurements failed?** The **foraminal aperture geometry** (area / min-opening):
  a center-box SAM2.1 prompt segments the central object, **not a calibrated foraminal aperture**, so
  geometry is flat across severities. Only the (redundant) intensity contrast carried signal.
- **What remains as the next non-clinic path?** A **calibrated aperture segmenter** — a foramen-
  specific model trained on the LSS foraminal masks (or MedSAM2 with proper medical prompts) — could
  give true geometry. **But the redundancy result bounds this:** the deployed image grader already
  extracts the foraminal signal from the same pixels, so even a perfect aperture mask is unlikely to
  add much. Converging with v1.4–v1.7, the binding constraint is **in-domain severe-label quantity/
  quality**, not derived features. The honest next lever is expert re-annotation (v1.7 review pack) +
  a clean-labelled test set — not more segmentation-derived evidence.
