# v1.8c — real MedSAM2 morphometry-only signal (Phase 8)

> Research-only · not diagnostic. Reproduce: `train_real_medsam2_morphometry_only_v1_8c.py`.

| route | logistic dev AUROC | GBM dev AUROC | severe recall@FAR≤10 |
|---|---|---|---|
| right-foraminal | 0.487 | **0.551** | 0.146 |
| (v1.8b SAM2.1 right-foraminal) | 0.636 | **0.687** | 0.208 |

**Real MedSAM2 morphometry is weaker than SAM2.1 on the target route** and far below the deployed
image grader. **Hard decision (per plan): no complementary signal → fusion confirmed but no
locked-test gain expected.** Top features were geometric (centroid-y, area, compactness) — the
discriminative contrast signal that SAM2.1 had is weaker on MedSAM2's masks.
