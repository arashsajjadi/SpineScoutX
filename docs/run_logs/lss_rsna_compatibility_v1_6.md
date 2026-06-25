# v1.6 LSS↔RSNA compatibility audit (Plan A)

> Research-only · not diagnostic. Quantifies whether LSS foraminal crops can be mixed with / used
> to pretrain the RSNA sagittal-T1 foraminal grader. Reproducible:
> `scripts/audit_external_data_v1_6.py`.

## View / geometry

| property | LSS | RSNA foraminal |
|---|---|---|
| view | sagittal (parasagittal foraminal box) | sagittal-T1 (parasagittal best slice) |
| crop | 2.5D box+context, resized | 2.5D auto-localized, resized |
| shape | (3, 224, 224) | (3, 224, 224) ✓ |
| range | [0, 1] | [0, 1] ✓ |
| condition | RFS/LFS → right/left foraminal | right/left foraminal ✓ |
| levels | L1-L2..L5-S1 ✓ | L1-L2..L5-S1 ✓ |

## Intensity (mean over ~400 crops)

| | mean | std | p5 | p95 |
|---|---|---|---|---|
| LSS | 0.318 | 0.189 | 0.048 | 0.648 |
| RSNA | 0.293 | 0.233 | 0.006 | 0.807 |

Distributions are **close** (both normalized T1-ish sagittal foraminal crops); LSS has slightly
lower contrast / narrower range. A mild residual domain shift remains (single-site LSS vs
multi-site RSNA), handled by per-crop [0,1] normalization + light intensity augmentation at
fine-tune time.

## Transfer mode decision

Modes considered: (A) supervised LSS pretraining → RSNA fine-tune; (B) joint LSS+RSNA with a
dataset embedding; (C) LSS encoder-only pretraining; (D) LSS for hard/severe representation;
(E) reject.

**Chosen: A (supervised LSS foraminal pretraining → RSNA fine-tuning).** Justification:
crops are shape/intensity-compatible; LSS carries 208 graded severe foramina (severe-aware
pretraining signal); identical architecture (convnext_tiny + level/condition embeddings) makes the
LSS-trained encoder a drop-in warm start. The experiment trains the RSNA foraminal grader twice —
**ImageNet-init (baseline) vs LSS-init (transfer)** — same recipe, splits_v1 train, **dev-selected**
on foraminal-macro recall@FAR≤10%, **locked-test read once**. Mode B (joint) is held as a fallback
variant if A is marginal. Mode E (reject) is not warranted — compatibility is sufficient.
