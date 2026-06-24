# v1.3 — Accuracy Jump + Real Evidence Viewer + Generalization: plan

> Research-only · not diagnostic. Phase-0 plan. No code/merge in this phase.

## 1. Current baseline (locked-test auto severe recall, frozen at v1.2 / main `8091e6e`)
| finding | route | severe recall [95% CI] |
|---|---|---|
| spinal canal stenosis | sagittal-T2 | 0.830 [0.725, 0.929] |
| left neural foraminal narrowing | sagittal-T1 | 0.788 [0.673, 0.892] |
| right neural foraminal narrowing | sagittal-T1 | **0.660** [0.524, 0.788] |
| left subarticular stenosis | axial-T2 | 0.746 [0.674, 0.815] |
| right subarticular stenosis | axial-T2 | 0.737 [0.667, 0.807] |

## 2. Current weak routes
- **Right neural foraminal** — weakest; characterized (v1.0–v1.2) as **signal/sample-limited**:
  specialist non-decisive; 56% of severe misses are *confidently normal*; per-level
  thresholding does not help; n_severe≈53; instability is `slice_sensitive` (best-slice).
- **Axial level scorer** — ±1-slice hit **0.43**; subarticular grading works because the robust
  grader *tolerates* leveling noise. Instability typing (v1.2) confirms subarticular instability
  is dominated by `axial_candidate_sensitive` (leveling), but with a small severe-FN payoff.

## 3. Why right-foraminal + axial are the two highest-value model targets
Right-foraminal is the only route materially below the pack; axial leveling is the only
*localization* component still far from solved (0.43). Both are the documented bottlenecks.
**Honest expectation (evidence-based):** right-foraminal accuracy is sample-limited (low odds of
a decisive train win → the real win is better severe-FN *triage*); axial **localization** has a
real shot at improvement via a better **decoder** (equal-spacing/gap-regularized monotonic DP +
top-k) without retraining the CNN — a bounded, evaluable change. A full stack-sequence model
needs new full-stack pixel caching (large) and the v1.1/v1.2 evidence bounds its *grading*
payoff, so it is gated behind the decode result.

## 4. What "real-data evidence viewer" means here
For real locked-test cases (hashed `case_*`): the **input/evidence route**, the **model
prediction** (severity estimate + P(severe) + confidence + stability + route quality), the
**held-out reference label** (transparency only — never an input), the **code-derived
correctness**, the **review_required reason**, and **why** (instability type / route quality /
retrieval). The full version includes real derived evidence crops; the committed version is
pixel-free (see §5–6).

## 5. Raw-data / license / privacy constraints
RSNA LumbarDISC is public **non-commercial research** competition data; redistribution of the
imagery (even derived) is restricted. SPIDER is CC BY 4.0 (anatomy masks, different task). The
repo is private. To avoid **any** redistribution question we treat committed assets as
**pixel-free**.

## 6. May safe derived evidence crops be committed?
**Decision: NO (conservative).** We generate the full real-pixel evidence viewer **locally**
under gitignored `outputs/real/evidence_case_viewer/`, and commit only **pixel-free** assets:
the real structured prediction-vs-reference card + real **derived scalar signals** (crop-centre
offset, slice index, mean intensity — metadata-free, non-reconstructive) + a **schematic**
evidence-route diagram. This addresses "looks synthetic" by making cards unmistakably
real-case-derived, without committing medical pixels. Full policy: `real_evidence_asset_policy.md`.

## 7. What will be trained / retrained / changed
- **Axial decode v2** (no CNN retrain): gap-regularized monotonic DP + top-k pooling; eval
  ±slice-hit on locked test. Optional small BiGRU logit-refiner if the decode result is promising.
- **Evidence intelligence v3**: a combined **severe-FN risk score** (P(severe), confidence,
  entropy, stability, instability type, route quality, retrieval conflict); dev-tuned thresholds.
- **Retrieval v2**: side/level metadata filtering + back-off + `retrieval_conflict`.
- **Right-foraminal**: hard-case audit + apply the v3 risk score to its severe-FN *review*
  capture (real triage win); reconfirm the accuracy limit (no new specialist train — it has been
  non-decisive four times; re-running is low-value and risks the release).

## 8. What will be skipped if low-value
- A full from-scratch axial **stack-sequence model with new pixel caching** (gated behind the
  decode result; v1.1/v1.2 bound its grading payoff).
- Another right-foraminal **specialist grader** retrain (consistently non-decisive).
- Committing any **medical pixels** (policy §6).

## Gates / safety (unchanged)
pytest + ruff + format + build + doctor + forbidden-file/large-file/claim scans + link checks;
research-only; no GT in auto inference; reference never an input; no locked-test tuning; no
DICOMs/weights/runs/outputs/caches/identifiers committed; merge to `main` via PR (merge commit,
tag-preserving) only after all gates pass.
