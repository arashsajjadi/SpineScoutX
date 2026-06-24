# External-validation literature scan

> Research-only. This document positions SpineScoutX's design choices against the
> RSNA 2024 competition, the peer-reviewed/arXiv literature, and screening-metric
> conventions. Citations were verified by fetching the source page where marked
> *(verified)*; everything else is explicitly labelled as inferred or unconfirmed.
> Nothing here is a clinical claim.

## How our choices line up with prior work

**1. Localize-then-classify is the standard, and we follow it.** The two-stage
"predict disc-level keypoints, then grade per level" pattern was near-universal among
RSNA 2024 winners. The 2nd-place solution's stage-1 head reportedly hit ~99% of disc
coordinates within 5 px; the 3rd-place used a CenterNet keypoint detector. Our
disc-level heatmap localizer (median 2.5 px, PCK@32 0.94) is the same design family
and the same role: it is what lets us measure the honest **oracle→auto** gap instead
of reporting GT-cropped numbers as if they were inference.

**2. Sagittal-only / per-level was the winning norm — and axial gave little.** Most
top solutions modelled sagittal only and per-crop/per-level. Notably the 2nd-place
team **built an axial pipeline and dropped it** because the cross-validation gain was
too small. This is direct external corroboration of our own honest finding: on the
**canal** condition, study-level multi-view graph reasoning (E3) did **not** beat the
per-region anatomy-forced model (E2). We did not over-fit a narrative onto a null
result; the field saw the same thing.

**3. Our E3 canal numbers are credible and competitive.** **M-SCAN** (Batra et al.,
arXiv:2503.01634, 2025) is a multi-view sagittal+axial cross-attention grader
reporting **canal-stenosis AUROC 0.971**. Our E3 severe-canal AUROC is **0.972** on
the RSNA val split — essentially the same ballpark as published multi-view
cross-attention work. (M-SCAN's abstract says "1,975 unique studies" but does *not*
name RSNA, so we treat the dataset linkage as inferred, not asserted.)

**4. Graph-over-levels has precedent; anatomy-mask graph attention appears novel.**
Baur et al. (J. Imaging Inform. Med., 2024) build a CNN→point-cloud→**GNN over disc
levels** for Pfirrmann grading, so cross-level graph reasoning is a recognised idea.
But across the verified RSNA 2024 solutions we found **no** top entry combining
explicit anatomy-segmentation masks with graph/cross-level attention — so E3's
anatomy-graph design is novel even though, on this single condition, it was not
superior.

**5. SPIDER-as-anatomy-prior is published; SPIDER→RSNA-2024 transfer appears novel.**
van der Graaf et al. (European Radiology, 2024, DOI 10.1007/s00330-024-11080-0)
explicitly reuse a **SPIDER-trained U-Net as an anatomy prior** for downstream canal
stenosis grading (Dice 0.92 canal). This validates the SPIDER→grading transfer
concept. A *published* SPIDER→RSNA-2024 transfer specifically could not be verified,
so our cross-dataset use of SPIDER for RSNA appears to be a novel combination.

## Severe-first operating frontier is mainstream — and we should add AUPRC/MCC
"Sensitivity at a fixed false-alarm budget" is exactly the standard CADe framing:
**FROC / CPM = average sensitivity at fixed false-positives per scan** (LUNA16; Park,
*Bioengineering* 2024). Related recognised framings: **partial AUC** in the
high-specificity region and **decision-curve analysis / net benefit** (Vickers et
al.). Reporting guidelines (CLAIM, Mongan et al. 2020; Hicks et al., *Sci Rep* 2022)
recommend, for imbalanced tasks: AUROC, **AUPRC** (emphasised under imbalance),
sensitivity/specificity at operating points, PPV/NPV, **MCC**, and **calibration** —
and warn against bare accuracy. Our severe frontier already reports severe AUROC,
**severe AP (= AUPRC)**, recall at fixed false-alarm budgets, and **ECE** — i.e. the
recommended set. A natural follow-up is to add sensitivity@fixed-specificity and MCC.

## Verified citations
- **The RSNA LumbarDISC Dataset** — Richards et al. — Radiology: AI / arXiv:2506.09162 — 2025 — https://arxiv.org/abs/2506.09162 *(verified; train 1,981 / public 272 / private 444, matching our ~1,975)*
- **M-SCAN: multi-view cross-attention canal-stenosis grading** — Batra, Gumber, Kumar — arXiv:2503.01634 — 2025 — https://arxiv.org/abs/2503.01634 *(verified; RSNA linkage inferred from "1,975 studies")*
- **3D imaging + GNN for Pfirrmann grading** — Baur et al. — J. Imaging Inform. Med. — 2024 — DOI 10.1007/s10278-024-01251-2 *(verified via PMC)*
- **Multi-attention CNN for lumbar spinal stenosis** — Lin et al. — Bioengineering 11(10):1021 — 2024 *(verified via PMC)*
- **DeepSPINE: segmentation-prior + multi-task grading** — Lu et al. — arXiv:1807.10215 — 2018 — https://arxiv.org/abs/1807.10215 *(verified)*
- **SPIDER dataset & benchmark** — van der Graaf et al. — Scientific Data 11:264 — 2024 — https://www.nature.com/articles/s41597-024-03090-w *(verified; Zenodo 8009680)*
- **Canal-stenosis grading using SPIDER U-Net prior** — van der Graaf et al. — European Radiology — 2024 — DOI 10.1007/s00330-024-11080-0 *(verified via PMC)*
- **CADe metrics review (FROC/CPM/pAUC)** — Park — Bioengineering 11(11):1165 — 2024 *(verified via PMC)*
- **LUNA16 evaluation (CPM = sensitivity at fixed FP/scan)** — https://luna16.grand-challenge.org/Evaluation/ *(verified)*
- **CLAIM checklist for medical-imaging AI** — Mongan, Moy, Kahn — Radiology: AI — 2020 *(verified via PMC)*
- **Evaluation metrics for medical AI** — Hicks et al. — Sci Rep 12:5979 — 2022 *(verified via PMC)*
- **Liu et al.**, lumbar degenerative classification on RSNA 2024 — Int. J. Comput. Intell. Syst. — 2025 — DOI 10.1007/s44196-025-01098-7 *(DOI/authors corroborated; paywalled abstract only)*

## Could not verify (stated honestly)
- Top RSNA 2024 private-leaderboard scores (Kaggle pages JS/reCAPTCHA-walled).
- The exact `any_severe_spinal` extra weight in the competition metric.
- Any *published* SPIDER→RSNA-2024 transfer (our combination appears novel).
- M-SCAN's dataset identity (RSNA inferred, not stated).

## Takeaways for SpineScoutX
- The **localizer-first, honest-inference** stance and the **severe-first frontier**
  are both squarely in line with field best practice.
- E3's **anatomy-mask graph attention** and the **SPIDER→RSNA prior** appear to be
  genuinely novel combinations — independent of whether E3 wins on canal.
- The honest **E2 > E3 on canal** result is consistent with the competition's own
  experience that extra views/heads added little on the dominant sagittal signal.
- Recommended metric additions: sensitivity@fixed-specificity and MCC alongside the
  severe AP / recall@FAR / ECE we already report.
