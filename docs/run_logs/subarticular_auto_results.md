# Axial subarticular auto route — UNLOCKED (locked test, splits_v1)

> Research-only. Not diagnostic. Auto inference reads NO GT coordinates: the axial-T2
> series is chosen from the series index, each level's axial slice from the
> **coordinate-supervised axial level scorer** (appearance ⊕ normalized-z, monotonic
> decoding), and the left/right lateral-recess crop centre from a fixed
> supervision-derived offset. Graders retrained on splits_v1 `train`, selected on `dev`
> auto, evaluated ONCE on locked `test`. Cluster-bootstrap 95% CIs.

## From geometry blocker → supervised scorer → robust grader
The prior pure-geometry matcher hit only **27.5% within ±1** axial slice. The
coordinate-supervised scorer improves this to **43% within ±1 / 65% within ±2** (locked
test) — better but still imperfect, because adjacent axial slices are genuinely ambiguous.
**The decisive step is the grader, not perfect leveling:** training the grader on the same
(imperfectly leveled) auto crops makes it *robust to the leveling error* — exactly the
canal lesson (train == inference).

## Severe recall [95% CI] on locked test (auto), per side
| grader | left subarticular (n=1434, sev=138) | right subarticular (n=1431, sev=137) |
|---|---|---|
| oracle-trained control (auto) | 0.246 [0.176, 0.326] | 0.365 [0.286, 0.444] |
| **auto-trained robust (auto)** | **0.746 [0.674, 0.815]** | **0.737 [0.667, 0.807]** |
| oracle ceiling (auto-robust on oracle crops) | 0.768 | 0.766 |

Paired robust − control (auto, same nodes): **left +0.500 [0.403, 0.591]** (McNemar 71/2,
p=6e-19); **right +0.372 [0.286, 0.462]** (McNemar 55/4, p=2e-12) — both decisive.
recall@FAR≤10%: left 0.732, right 0.708. The robust grader recovers **~96–97% of the
oracle ceiling** despite the imperfect axial level scorer.

## Honest verdict — subarticular auto-inference is REAL (coverage → 5/5)
- Deployable axial subarticular **auto** severe recall: **left 0.746, right 0.737** on the
  locked test, no GT at inference, with the highest severe counts of any condition (138/137).
- The oracle-trained grader **collapses** on auto subarticular (0.25/0.37) — the axial
  route is *only* viable with robust auto-training, which is the headline.
- **Transparency:** the level scorer is imperfect (±1 slice-hit 0.43); the strong end-to-end
  result comes from the grader tolerating that noise, not from perfect leveling. A better
  axial level scorer (e.g. a stack-sequence model) is a future lever to push higher.

Artifacts: `outputs/real/subarticular_auto_results.json`,
`outputs/real/axial_level_scorer_results.json`. Reproduce:
`python scripts/run_axial_level_scorer.py && python scripts/run_subarticular_locked_test.py`.
