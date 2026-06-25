# v1.6 Plan A — LSS→RSNA foraminal transfer: EXECUTED NEGATIVE

> Research-only · not diagnostic. Protocol: `splits_v1`; **dev** selects (foraminal-macro
> recall@FAR≤10%); **RSNA locked-test read once** per model. Cluster bootstrap CIs, n_boot=2000.
> RSNA auto-foraminal crops; identical convnext_tiny recipe (weighted-CE, 18ep, freeze-2). Only the
> encoder init / data pool changes between arms.

## Arms

1. **Baseline** — ImageNet-init, RSNA-only (`rsna_baseline`).
2. **Transfer** — LSS supervised foraminal pretraining (2978 crops incl. 208 severe; lss_dev
   recall@FAR10 0.966) → RSNA fine-tune (`rsna_lss`, mode A).
3. **Joint** — ImageNet-init, RSNA train **+ LSS lss_train pooled** (16332 crops, **743 severe** =
   564 RSNA + 179 LSS, +32%), severe oversampling (`rsna_joint`, mode B).

## Locked-test (severe recall, argmax)

| arm | L-for | R-for | foraminal macro |
|---|---|---|---|
| deployed reference | 0.788 | 0.660 | 0.724 |
| **baseline (ImageNet)** | **0.769** [0.654,0.873] | **0.679** [0.559,0.800] | **0.724** |
| transfer (LSS pretrain) | 0.577 [0.439,0.711] | 0.528 [0.400,0.667] | 0.553 |
| joint (LSS+RSNA) | 0.769 [0.646,0.881] | 0.679 [0.545,0.818] | 0.724 |

Paired deltas vs baseline (locked-test):
- **transfer**: L-for severe recall **−0.192 [−0.346,−0.041] DECISIVE**, R-for **−0.151
  [−0.275,−0.036] DECISIVE**. recall@FAR10: R-for +0.019 (n.s.), L-for −0.077 (n.s.).
- **joint**: severe recall **±0.000** on both sides (identical); R-for recall@FAR10 0.774→0.679,
  L-for 0.846→0.808 (both slightly down).

## Verdict — NEGATIVE

External LSS foraminal data improves neither foraminal route:
- **Pretraining hurts** — the LSS-initialised grader fine-tunes to a *more conservative* operating
  point (R-for argmax FAR 0.080→0.026), collapsing argmax severe recall (−0.15 to −0.19) while its
  severe *ranking* is unchanged (recall@FAR10 flat) → LSS features add no discriminative severe
  signal that survives fine-tuning; domain shift (single-site Türkiye scanner vs multi-site RSNA).
- **Joint adds nothing** — pooling 179 external severe crops moves locked-test severe recall by
  **exactly 0.000**. The RSNA foraminal ceiling (R-for ≈ 0.68) is **not** limited by simple severe
  label *quantity* from a domain-shifted source.

The clean baseline (ImageNet, splits_v1 auto-trained) **reproduces the deployed foraminal
performance** (macro 0.724 = deployed; R-for 0.679 ≈ deployed 0.660), validating the recipe.

**→ Plan A executed-negative. Adaptive switch to Plan B (self-supervised representation).**
Reproduce: `train_foraminal_grader_v1_6.py` (lss_pretrain / rsna_baseline / rsna_lss / rsna_joint),
`compare_foraminal_transfer_v1_6.py`.
