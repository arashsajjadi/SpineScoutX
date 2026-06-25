# v1.6 — adaptive accuracy offensive: failure autopsy

> Research-only · not diagnostic. Four levers executed (external data, SSL pretraining,
> anatomy prior, stronger grader). **No raw severe-grading metric improved.** This autopsy answers
> the exact failure questions and specifies the data that would actually move the metric.

## Baseline (unchanged, deployed) locked-test severe recall

canal 0.830 · L-for 0.788 · **R-for 0.660** · L-sub 0.746 · R-sub 0.737 · **macro 0.752**.
A clean v1.6 ImageNet baseline (splits_v1 auto-trained, same recipe) **reproduces** the foraminal
numbers (L-for 0.769, R-for 0.679, macro 0.724), validating every comparison below.

## Did each path fail — and why (exact)

**Plan A — external LSS data. Executed; NEGATIVE — and NOT because it was unavailable.**
- *Unavailable?* No. Acquired (Mendeley CC BY 4.0), sha256-verified, parsed: **208 severe**
  foraminal boxes (93 R / 115 L), 468 patients. Crops shape/intensity-compatible with RSNA.
- *Label/domain mismatch?* Partly — single-site (Firat U., Türkiye) vs multi-site RSNA. But the
  decisive evidence is stronger: **joint training that pools +179 LSS severe moved locked-test
  severe recall by exactly 0.000.** Pretraining was *worse* (−0.15 to −0.19, conservative shift).
  → The foraminal ceiling is **not** limited by severe-label *quantity from a domain-shifted
  source**; domain shift + grading-criteria differences neutralise the external severe labels.

**Plan B — self-supervised pretraining. Executed; NEGATIVE (non-convergent).**
- *Representation didn't transfer?* It never formed: SimCLR NT-Xent stuck at chance
  (5.22→5.25 ≈ log(2N−1)); I/O-bound ~135 s/epoch. As executed, no usable encoder. Recipe fix
  (lr 1e-4, larger batch, milder augment, pre-decoded tensor) specified; payoff still bounded by
  the ~20k-crop corpus vs ImageNet-1.2M.

**Plan C — anatomy prior. Executed (v0.4/v0.5) with evidence; NEGATIVE for grading.**
- *Masks not informative?* They are anatomically, but for *grading* they don't help: concat fusion
  (E1) **ignores** the prior (zero≈shuffle≈correct, |Δwll|<0.001); forced-ROI (E2) **uses**
  anatomy (ablation −0.014..−0.020 wll) yet **shuffled≈correct** → a regional-presence gate, not
  sample-specific narrowing geometry; E2 ≈ E0 on the operating checkpoint (incl. foraminal F1).

**Plan D — stronger route-specific grader. Executed; result below.**
- *Overfit?* convnext_small (vs convnext_tiny baseline) + severe oversampling — see
  `strong_route_grader_v1_6.md` / controller. (Capacity was already shown insufficient in v1.5:
  candidate-bag MIL overfit the thin severe data.)

## Root cause (convergent across v1.4, v1.5, v1.6)

The weak-route severe ceiling is **bound by RSNA severe-label quantity AND quality (grading-criteria
consistency / ambiguity), not by crop, localizer, representation, anatomy, or model capacity.**
Right-foraminal is the extreme case: only **278 train / 53 test** severe, and ~56% of its misses
are *confidently-normal* predictions (v1.1) — consistent with genuinely subtle or label-ambiguous
cases that no architecture/representation recovers.

## What data would actually be required next (quantified)

1. **Same-domain RSNA severe foraminal labels.** External (LSS) severe did not transfer; the next
   increment must be in-distribution. To move **R-for severe recall +0.05** reliably (clear of the
   53-severe-test CI ≈ ±0.13), estimate **~2–3× the train severe count** (≈ 600–800 R-for train
   severe, vs 278 now) from the **RSNA domain**.
2. **Label de-noising.** Multi-rater consensus on the existing severe/near-severe foraminal cases
   (the confidently-normal misses) to separate "hard but real" from "ambiguous/label-noise".
3. **Annotation priority order:** right-foraminal first (weakest, fewest severe), then
   subarticular; within right-foraminal, **L4-L5 and L5-S1** (the weakest levels, v1.1 domain-shift
   audit). Sagittal-T1 parasagittal severe foramina with side+level+grade + reviewer agreement.

Until such data exists, the deployable graders stay at the proven ceiling (macro 0.752); v1.6 ships
**no model change** (all interventions executed-negative) and an honest, fully-executed audit trail.
