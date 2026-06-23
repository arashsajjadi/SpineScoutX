# E0 — image-only baseline (REAL RSNA val) results

> Research-only, not diagnostic. Real metrics on the held-out **study-level** val
> split (9,745 crops / 395 studies). Run artifacts are gitignored; regenerate via
> `spinescoutx train-classifier --config configs/real_e0_baseline_rsna.yaml`.

Model: ConvNeXt-Tiny (timm, pretrained), 2.5D 224² crops + level/condition
embeddings; frozen-backbone warmup then gentle fine-tune (backbone lr ×0.2);
class-weighted CE; AMP on RTX 5080. Best by val weighted log loss @ epoch 3
(early-stopped; later epochs overfit — honest).

## Headline metrics (best checkpoint)
| metric | value |
|---|---|
| weighted log loss (RSNA-style) | **0.4621** |
| macro F1 | 0.706 |
| balanced accuracy | 0.763 |
| **severe recall** | **0.751** |
| severe FNR | 0.249 |
| severe one-vs-rest AUROC | **0.971** |
| ECE (top-label) | **0.027** |

Confusion (rows=true normal_mild/moderate/severe, cols=pred):
`[[6506, 1087, 43], [226, 1052, 257], [9, 134, 431]]` — severe: 431/574 caught,
only 9 severe→normal_mild.

## Per-condition F1
| condition | F1 |
|---|---|
| left_subarticular_stenosis | 0.723 |
| right_neural_foraminal_narrowing | 0.704 |
| right_subarticular_stenosis | 0.701 |
| left_neural_foraminal_narrowing | 0.690 |
| spinal_canal_stenosis | 0.625 |

## Per-level F1
l1_l2 0.660 · l2_l3 0.689 · **l3_l4 0.716** · l4_l5 0.680 · l5_s1 0.666

## Failure analysis
310 flagged failures (severe false negatives + high-confidence wrong) in
`outputs/real/e0_failure_cases.csv`; full breakdown in
`outputs/real/e0_error_analysis.md`; confusion + Grad-CAM examples in
`outputs/real/e0_confusion_matrix.png` / `e0_examples_grid_real.png` (gitignored).
Spinal-canal-stenosis is the hardest condition (F1 0.625). This is the **baseline**
against which the anatomy-guided model (E1) is measured.
