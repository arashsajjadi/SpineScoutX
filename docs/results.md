# Results — SpineScoutX (real data)

> Research-only, not diagnostic, not clinically validated, not for medical
> decision-making. All numbers below are **real**, measured on held-out data; run
> artifacts are gitignored and regenerable. Nothing is fabricated, nothing
> cherry-picked.

## Research question
Can explicit anatomical priors from a real SPIDER-trained segmenter improve
disc-level lumbar degenerative finding grading on RSNA/LumbarDISC?

**Short answer (this implementation): no meaningful improvement, and the
anatomy-guided model largely ignores the priors — shown by the ablation.** Details
below; we report this honestly rather than overclaim the tiny aggregate edge.

## Data
RSNA: 1,974 studies → 48,692 localizer crops (2.5D, 224²), patient/study-level
split (38,947 train / 9,745 val), **0 leakage**, severe ≈ 6.3%. SPIDER: 218
patients / 10,338 slices (official split).

## E4 — SPIDER anatomy segmentation (real)
mean Dice **0.884** · canal **0.902** · vertebra **0.903** · disc 0.846 (official
SPIDER val). Anatomy masks only (not pathology). Used to generate RSNA anatomy
priors (48,692; vertebra present 99.8%, disc 85%, canal 47%).

## E0 vs E1 — disc-level grading (real RSNA val, 9,745 crops)
| metric | E0 image-only | E1 anatomy-guided | better |
|---|---|---|---|
| weighted log loss ↓ | 0.4621 | **0.4579** | E1 (−0.0042) |
| macro F1 ↑ | 0.706 | **0.717** | E1 |
| balanced accuracy ↑ | **0.763** | 0.752 | E0 |
| **severe recall** ↑ | **0.751** | 0.711 | E0 |
| severe FNR ↓ | **0.249** | 0.289 | E0 |
| severe AUROC ↑ | 0.971 | 0.972 | ~tie |
| ECE ↓ | **0.027** | 0.034 | E0 |

**Mixed and marginal.** E1 edges E0 on aggregate log loss / macro-F1; E0 is better
on severe recall and calibration. No clear win for anatomy priors.

## Counterfactual anatomy ablation (E1 model, real val) — the decisive test
| anatomy mode | weighted log loss | severe recall | mean AEC |
|---|---|---|---|
| correct | 0.4579 | 0.711 | 0.099 |
| shuffled (other sample) | 0.4576 | 0.711 | 0.100 |
| zero | 0.4584 | 0.723 | 0.101 |
| noise | 0.4580 | 0.711 | 0.100 |

Δ vs correct: |Δ weighted log loss| < 0.001 for every perturbation; severe recall
unchanged (zero even +0.012). **Zeroing, shuffling, or noising the anatomy prior
barely changes anything ⇒ E1 does not meaningfully use the anatomy branch.** By the
project's own interpretation rules (`zero ≈ correct ⇒ branch ignored`,
`correct ≈ shuffled ⇒ anatomy not semantically used`), the small E1>E0 edge is
**not attributable to anatomy** — more plausibly extra head capacity / training
variance.

## Evidence consistency (AEC)
Mean AEC ≈ **0.10** and **flat across all anatomy perturbations** — consistent with
the model ignoring anatomy. AEC is also region-size sensitive (the canal/foraminal
target regions are a small fraction of the crop), so a low absolute AEC is partly
expected; the *flatness across ablation modes* is the informative signal. Foraminal
/ subarticular AEC regions are **approximate** (SPIDER lacks those labels).

## Calibration
Both models are well-calibrated out of the box (E0 ECE 0.027, E1 ECE 0.034);
temperature scaling (fit on val — a research-demo simplification) leaves ECE
essentially unchanged (E1 0.034→0.031, T≈1.13; E0 T≈0.98). Uncertainty flags
(`high`/`moderate`/`review_required`) are attached to every finding.

## Why v0.5 — "anatomy-forced" (motivation)

The v0.4 ablation is the crux: concat-fusion (`[image crop ⊕ anatomy channels] → classifier`)
let the model **ignore** anatomy (zero ≈ shuffle ≈ correct). Concatenated mask channels are an
*optional* input the network can down-weight to zero. v0.5 changes the **structure** so anatomy
is not bypassable: the anatomy masks define **regions**, and features are **region-pooled** from
those masks (canal/disc/vertebra), with global-feature dropout forcing the head to rely on the
region features. If the mask is zeroed or shuffled, the region features change by construction, so
the prediction must change too — making the ablation a genuine test of anatomy usage. See
`docs/technical_report.md` and the E2 results/ablation below once available.

## Honest interpretation
- **Did anatomy priors help classification?** Not meaningfully — a <1% log-loss /
  macro-F1 edge that the ablation shows is **not** due to anatomy; severe recall is
  worse for E1.
- **Did they improve evidence consistency (AEC)?** No — AEC is low and flat.
- **Did calibration help?** Both are already well-calibrated; temperature scaling
  adds little.
- **What worked:** a strong real E0 baseline (severe AUROC 0.971), real E4
  segmentation (Dice 0.884), and a **rigorous ablation that prevents a false
  "anatomy helps" claim**.
- **Why the negative result is useful:** it demonstrates that naive
  channel-concatenation of anatomy priors is insufficient; future work should force
  reliance on anatomy (e.g., anatomy-conditioned attention, region-pooled features,
  auxiliary AEC loss) rather than concluding anatomy is useless.

## Failure cases (shown, not hidden)
E0: 310 flagged (severe false negatives + high-confidence wrong); spinal-canal
stenosis is the hardest condition (F1 0.625). E1: 474 flagged. E4: disc Dice (0.846)
lags canal/vertebra. See `outputs/real/*_failure_cases.csv`, `*_error_analysis.md`,
and the figures under `outputs/real/figures/` (all gitignored; regenerable).

## Runtime (RTX 5080)
RSNA prep 48,692 crops ≈ 5 min; anatomy priors ≈ 3 min; E0/E1 train ≈ 30–45 min
each (AMP, early-stopped); E4 inference 0.85 ms/slice. Crops/priors/runs are
cached, gitignored, and resumable.
