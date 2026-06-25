# v1.8c — real MedSAM2 failure autopsy

> Research-only · not diagnostic.

- **Was real MedSAM2 installed and used?** **Yes** — `sam2==1.1.0` installed; ran via VisionServeX
  `medsam2_runtime` (`sam2.modeling.sam2_base` + `SAM2ImagePredictor` + `MedSAM2_latest.pt`), proven
  in the smoke report. The v1.8b "SAM2.1 fallback" flaw is corrected.
- **Did real MedSAM2 improve mask QC over SAM2.1?** No — both 0% failure; MedSAM2's mean score is
  lower (0.422 vs 0.575) and the foraminal opening area is equally flat.
- **Did it improve dev morphometry signal?** No — *worse* on the target (right-foraminal GBM dev
  AUROC 0.551 vs SAM2.1 0.687).
- **Did it add complementary signal to the image grader?** No — fusion Δ+0.000; the morphometry is a
  coarse, redundant view of the same pixels.
- **Did fusion improve dev/test?** No (locked-test severe recall unchanged).
- **Did triage improve?** No (≈ deployed-only severe-FN capture; ≤ v1.7's disagreement triage).
- **Was VisionServeX useful?** Yes operationally — it provided the real MedSAM2 runtime so no
  serving code was duplicated; but it did not change the accuracy conclusion.
- **What remains as the next non-clinic path?** None on the *derived-feature* axis — MIL,
  localization, external data, SSL, anatomy, bigger backbones, SAM2.1 morphometry, and now **real
  MedSAM2** morphometry are all redundant/bounded. The binding constraint (v1.4–v1.8c, convergent)
  is **in-domain severe-label quality**: expert re-annotation (the v1.7 review pack) + a
  clean-labelled held-out **test** set is the only lever that can move the measured ceiling.
