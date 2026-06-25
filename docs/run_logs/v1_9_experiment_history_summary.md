# SpineScoutX experiment history summary (v1.0 → v1.8c)

> Research-only · non-commercial · not diagnostic · not clinically validated.
> Protocol: `splits_v1`; **dev/train select**; **locked-test read once** per final candidate.

## Current best metrics

| Route | Locked-test severe recall |
|---|---|
| Spinal canal | 0.830 |
| Left foraminal | 0.788 |
| Right foraminal | 0.660 |
| Left subarticular | 0.746 |
| Right subarticular | 0.737 |
| **5-route macro** | **0.752** |

**Best safety/triage config (v1.7):** effective foraminal severe recall 0.724 → **0.933** at 15% review budget.

## Experiment history

### v1.0 — Auto-robust five-finding system ✅

**Hypothesis:** Coordinate-supervised localizer + robust auto-training unlocks all 5 routes.

**Method:** Coordinate-supervised axial level scorer, robust auto-training (oracle vs auto gap), Safety Mode v4 router; 5 graders: canal, L/R foraminal, L/R subarticular.

**Headline:** All 5 routes graded automatically. Canal 0.830 / L-for 0.788 / R-for 0.660 / L-sub 0.746 / R-sub 0.737; 5-route macro 0.752. KEY: robust-auto training benefit ∝ oracle→auto gap — recovers canal & subarticular.

**Raw metric delta:** baseline (0.752 macro set here)

**Why it matters:** Established the system; set the raw accuracy ceiling.

### v1.1 — Intelligent evidence + evidence stability 🛡

**Hypothesis:** Evidence-stability scoring (re-grading under localizer perturbation) adds triage signal.

**Method:** Evidence-stability scoring (no GT), finding_graph_v5, Safety v5 calibration trial, domain-shift audit, right-foraminal diagnosis.

**Headline:** Evidence stability AUROC 0.80 on dev but largely redundant with confidence. Adds triage only on 2 weakest right-side routes. Calibration NEGATIVE → raw probs kept. Right-foraminal diagnosed: 56% misses are confidently-normal, sample-limited (n≈53 severe).

**Raw metric delta:** +0.000 (no raw change)

**Why it matters:** Evidence intelligence framework; diagnosed right-foraminal weak point.

### v1.2 — Real case viewer + evidence intelligence v2 ❌

**Hypothesis:** Making intelligence visible improves trust and reveals failure modes.

**Method:** Real case viewer (1800×1000 cards, prediction→held-out reference→code-derived correctness), instability typing (foraminal=best-slice, subarticular=leveling), similar-case retrieval, README redesign.

**Headline:** Similar-case retrieval: severity-agree 0.73–0.89 but same-side ~chance (embedding = morphology not laterality). Viewer shows full intelligence chain. Axial v2 NOT trained (bounded payoff).

**Raw metric delta:** +0.000 (no raw change)

**Why it matters:** Visualization and explainability infrastructure.

### v1.3 — Axial localization upgrade + evidence intelligence v3 🛡

**Hypothesis:** Train-derived positional prior + BiGRU improves localization; new evidence signals add triage.

**Method:** Axial decode v2: monotonic positional-prior decoder (β dev-selected, NO CNN retrain). Evidence v3: conf+stability+retrieval_conflict+near_severe. Side-aware retrieval v2, real evidence pixel-free viewer.

**Headline:** REAL LOCALIZATION WIN: ±1-slice-hit 0.432→0.487, medAE 2→1. Severe-FN AUROC 0.833→0.863 (new signals add independently). HEADLINE: 5/5 severe recall UNCHANGED (gains are localization+triage, not argmax grading).

**Raw metric delta:** +0.000 argmax grading; ±1-slice-hit +0.055

**Why it matters:** Best localization (BiGRU); real evidence viewer; evidence AUROC improved.

### v1.4 — Raw accuracy war room — rigorous audit ❌

**Hypothesis:** There may be a bug or pipeline flaw suppressing severe recall.

**Method:** Ruthless accuracy audit (audit_accuracy_pipeline.py, 13/13 checks), collect_probs alignment check (0/4384 mismatches), latent bug B1 fix (silent key-collision → now raises), subarticular recrop test.

**Headline:** RIGOROUS NEGATIVE: no severe-recall-corrupting bug found. Bug B1 fixed (ZERO metric delta). v1.3 localization does NOT transfer to subarticular grading (paired: dev +0.004/test −0.040). 5/5 macro severe recall UNCHANGED at 0.752.

**Raw metric delta:** +0.000 (no raw change; bug B1 had zero metric delta)

**Why it matters:** Proved no pipeline bug; confirmed ceiling is data/model-limited.

### v1.5 — Full raw accuracy offensive — MIL + BiGRU axial stack ❌

**Hypothesis:** Candidate-bag MIL and BiGRU axial-sequence model will improve grading.

**Method:** Candidate-bag MIL (K=5 auto crops + attention pooling), baseline+MIL ensemble, BiGRU axial level-sequence refiner, recrop→regrade pipeline.

**Headline:** EXECUTED NEGATIVE on grading: MIL dev +0.125 recall@FAR10 R-for but test 0.660→0.453 (overfit); subarticular MIL collapsed to 0.000. DECISIVE LOCALIZATION WIN: BiGRU ±1-slice-hit 0.487→0.616, medAE 2→1 — NOT deployed. BiGRU recrop does NOT propagate to grading (confirms grader/data-limited, not loc-limited).

**Raw metric delta:** +0.000 grading (MIL and ensemble both negative)

**Why it matters:** BiGRU is best-ever localization; proved MIL/localization not the bottleneck.

### v1.6 — Safe adaptive accuracy offensive — external data + SSL + anatomy + stronger grader ❌

**Hypothesis:** External LSS data, SimCLR SSL, anatomy prior, or bigger backbone (ConvNeXt) will improve severe grading.

**Method:** Plan A: LSS-MRI AISSLab external data (CC BY 4.0, 208 real severe foramina) — pretrain + joint. Plan B: SimCLR SSL. Plan C: anatomy prior. Plan D: convnext_small backbone.

**Headline:** ALL FOUR EXECUTED NEGATIVE. LSS pretrain: DECISIVE LOSS (L −0.192/R −0.151); joint (+179 severe): Δ0.000. SimCLR: non-convergent. Anatomy: executed prior evidence (no gain). ConvNeXt: DECISIVE LOSS (L −0.173/R −0.208). ROOT CAUSE convergent: severe-label QUANTITY + QUALITY, not capacity or representation.

**Raw metric delta:** +0.000 (all four approaches negative or decisively worse)

**Why it matters:** Exhausted external data, SSL, anatomy, capacity axes; confirmed label ceiling.

### v1.7 — Hard-case label repair + noise-aware + triage safety upgrade 🛡

**Hypothesis:** Cleaning mislabeled hard cases and noise-aware training will improve raw recall.

**Method:** Hard-case mining (21 confident-normal severe misses + 1550 borderlines), 704-case local review pack, provisional soft-label cleaning, noise-aware retraining, teacher distillation, severe-FN triage router.

**Headline:** Raw accuracy NEGATIVE: label cleaning dev < original labels; noise-aware no gain; teacher distillation collapsed test macro to 0.496. TRIAGE WIN: at 15% review budget effective foraminal severe recall 0.724 → 0.933 (22/29 severe FN captured). 704-case review pack created for expert re-annotation.

**Raw metric delta:** +0.000 argmax; effective recall +0.209 at 15% review budget (triage only)

**Why it matters:** Best safety/triage config; review pack for expert re-annotation; confirmed label quality is the binding ceiling.

### v1.8b — SAM2.1 segmentation-morphometry (note: incorrectly called MedSAM2; corrected in v1.8c) ❌

**Hypothesis:** Foundation model segmentation morphometry adds complementary foraminal signal.

**Method:** Planned MedSAM2 but sam2 package missing → ran Transformers SAM2.1 fallback. Box-prompt segmentation of all 19,700 foraminal crops; 13 morphometric features; GBM morphometry-only model; late fusion; triage router.

**Headline:** Morphometry HAS standalone signal (R-for dev AUROC 0.687, from contrast not area). But REDUNDANT with image grader → fusion dev α=0 (no gain), locked-test foraminal severe recall UNCHANGED (0.788/0.660/macro 0.724). NOTE: this ran SAM2.1, not real MedSAM2 — corrected in v1.8c.

**Raw metric delta:** +0.000 (morphometry redundant with grader)

**Why it matters:** Established morphometry pipeline; proved SAM morphometry not the missing signal.

### v1.8c — Real MedSAM2 + VisionServeX morphometry (correction sprint) ❌

**Hypothesis:** Real MedSAM2 medical model (not SAM2.1 fallback) may give better masks and morphometry signal.

**Method:** Installed sam2==1.1.0; ran real MedSAM2 via VisionServeX medsam2_runtime (sam2.modeling.sam2_base + MedSAM2_latest.pt). Proved real: module path + checkpoint hash + smoke report. Segmented all foraminal crops (0% fail, ~32/s); GBM morphometry; late fusion; triage.

**Headline:** REAL MedSAM2 EXECUTED (v1.8b flaw corrected). Real MedSAM2 morphometry WEAKER than SAM2.1 (R-for dev AUROC 0.551 vs 0.687). Fusion locked-test UNCHANGED (L 0.788/R 0.660/macro 0.724; paired Δ +0.000). Triage no gain. Durable positive: reusable VisionServeX MedSAM2 integration.

**Raw metric delta:** +0.000 (even real MedSAM2 morphometry is redundant/bounded)

**Why it matters:** Proved real MedSAM2 not better than SAM2.1 morphometry; confirmed SAM2.1 fallback did not bias v1.8b conclusion; reusable VisionServeX integration.

## Convergent conclusion

v1.4–v1.8c convergent: the weak-route severe ceiling is bound by in-domain severe-label quality and label ambiguity, NOT by bugs, localizer, MIL, external data, SSL, anatomy prior, larger backbone, SAM2.1 morphometry, or real MedSAM2 morphometry. Only remaining lever: expert re-annotation of the v1.7 704-case review pack + a clean-labelled held-out test set re-read.

Legend: ✅ improved raw accuracy · 🛡 improved safety/triage · ❌ executed negative
