# v1.6 Plan B — self-supervised pretraining: EXECUTED, non-convergent

> Research-only · not diagnostic. SimCLR (NT-Xent) contrastive pretraining of convnext_tiny on
> **19,718 unlabelled foraminal crops** (RSNA train+dev + LSS; **locked-test excluded**), two
> augmented views per crop. Reproduce: `scripts/pretrain_spine_mri_v1_6.py`.

## What happened (executed, honest)

The contrastive pretraining **ran** but **did not converge**: NT-Xent loss stayed at chance
(ep0 **5.222** → ep5 **5.254**; random baseline ≈ log(2·N−1) = log(191) ≈ **5.25** for batch 96).
A loss pinned at the chance value means the encoder is **not** learning to pull positive views
together — no usable representation was produced. The run was also strongly **I/O-bound**
(~135 s/epoch: 19,718 crops × 2 views × per-step `np.load`), making a full multi-epoch SimCLR +
fine-tune impractical on this single-GPU budget.

## Verdict — NEGATIVE (non-convergent)

The SSL **representation lever, as executed, did not yield a useful encoder**, so it cannot improve
foraminal severe grading. This is an executed result (real training ran), not a skipped path. It is
consistent with the strong prior that in-domain SimCLR on ~20k crops is unlikely to beat
ImageNet-1.2M supervised pretraining for this task, and with the convergent v1.6 negatives
(Plan A external data: pretrain decisive loss, joint Δ0.000).

## Exact fix for a future attempt

The non-convergence is a recipe issue, not a fundamental one. Next attempt: **lower LR ≈ 1e-4**
(1e-3 destabilises convnext-from-ImageNet under NT-Xent) with a **longer warmup + cosine**, a
**larger contrastive batch** (≥256 needs gradient accumulation or multi-GPU), **milder
augmentation** (the resized-crop 0.7–1.0 + intensity 0.2 + noise makes positive pairs too
dissimilar early), and a **pre-decoded crop tensor** (single memory-mapped array) to remove the
`np.load` I/O bottleneck. Selection would remain RSNA dev foraminal-macro recall@FAR≤10%,
locked-test once. Expected payoff is still bounded by corpus size (~20k) vs ImageNet.
