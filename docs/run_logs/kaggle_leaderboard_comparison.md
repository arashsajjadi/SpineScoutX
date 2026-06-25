# Kaggle Leaderboard Comparison — SpineScoutX v1.9

> Research-only · non-commercial · not diagnostic.
> SpineScoutX is NOT an official competition entry.
> No submission was accepted (competition closed Oct 2024).

## Competition

RSNA 2024 Lumbar Spine Degenerative Classification
`rsna-2024-lumbar-spine-degenerative-classification`
Deadline: 2024-10-08 · Teams: 1874 · Prize: $50,000

## Metric

**Weighted log loss** (lower = better):
- `normal_mild` weight: 1× (approximately)
- `moderate` weight: 2× (approximately)
- `severe` weight: 4× (approximately)

This is **NOT** the same as our internal severe recall metric.
A model with good recall can have poor log loss if poorly calibrated.

## Leaderboard data

### Private LB (final rankings, from `kaggle competitions leaderboard --show`)

| Rank | Team | Private score |
|---|---|---|
| 1 | Avengers | 0.388936 |
| 2 | IanPan-Kevin-Yuji-Bartley | 0.389591 |
| 3 | SonySpine s & tkmn & Moyashii | 0.391374 |
| 4 | SPINE CHART | 0.398424 |
| 5 | Two people | 0.399791 |
| 6 | NVSpine | 0.401065 |
| 7 | HLIP | 0.401322 |
| 8 | K_mataro | 0.403199 |
| 9 | Adam Narai | 0.403633 |
| 10 | Preferred Spine | 0.406792 |

### Public LB distribution (from downloaded CSV, 1875 teams)

| Percentile | Score |
|---|---|
| Top-1 (best) | 0.3319 |
| Top-5 | 0.3392 |
| Top-10 | 0.3480 |
| Top-25 | 0.3971 |
| Top-50 (median) | 0.5752 |
| Bottom-25 (75th %ile) | 0.8990 |
| Bottom-10 (90th %ile) | 1.0433 |

## SpineScoutX estimate

**SpineScoutX did not produce a scored submission** (competition closed).
The following is an honest qualitative estimate only.

### What we know

- SpineScoutX v1.9 generates real model probabilities for all 5 conditions.
- Internal metric: 5-route macro severe recall = **0.752** (argmax).
- The model was NOT optimized for log loss or calibration.
- Calibration trial in v1.1 was NEGATIVE → raw softmax probabilities kept.
- The model tends to predict high probability for `severe` class (designed for recall).

### Estimated log-loss range

| Baseline | Expected score | Approximate rank |
|---|---|---|
| Uniform (1/3, 1/3, 1/3) | ≈1.099 | ~1725/1875 (bottom 8%) |
| Label-frequency prior | ≈0.85–0.95 | ~1500–1600/1875 |
| SpineScoutX v1.9 (estimated) | ≈0.55–0.85 | ~800–1500/1875 |
| Median competitor | 0.575 | 937/1875 |
| Top-10 competitors | ≤0.407 | ≤10/1875 |

### Why SpineScoutX is likely below median

1. **Not log-loss optimized**: graders trained with cross-entropy for class balance,
   but inference threshold was calibrated for severe recall (argmax).
2. **No temperature scaling**: calibration sprint (v1.1) showed no improvement.
3. **Right-foraminal weakness**: our R-for severe recall is 0.660, weakest route.
4. **Single test study**: the visible test has only 25 rows; score variance is high.
5. **Pipeline not end-to-end**: separate localizer → grader chain accumulates errors.

### What top competitors likely did

- End-to-end trained 3D/2D models (e.g., SegFormer, MaxViT, 3D CNN).
- Explicit log-loss calibration (temperature scaling, label smoothing).
- Large ensemble of diverse models.
- Competition-specific augmentation + test-time augmentation.
- Many submission iterations with public LB feedback.

## Honest assessment

SpineScoutX is a **research pipeline** built for clinical decision-support
concepts, not for Kaggle leaderboard optimization. Its internal metric
(severe recall + triage) is more clinically relevant than log loss.

We would likely score in the **40th–60th percentile range** on this
competition — competitive with below-median teams, but far behind
top-5 performers who used large ensembles and competition-tuned systems.

The more important research insight is the convergent finding (v1.4–v1.8c):
the binding ceiling is **label quality**, not model architecture, and no
amount of competition tuning could fix that without expert re-annotation.

## Raw summary

| Item | Value |
|---|---|
| Submission accepted? | No (competition closed, 400 error) |
| Kaggle score | N/A |
| Private LB best score | 0.388936 (Avengers) |
| Public LB best score | 0.331905 (IanPan-Kevin-Yuji-Bartley) |
| Public LB median | 0.575187 |
| Total teams | 1875 |
| SpineScoutX estimated rank | ~900–1400/1875 |
| SpineScoutX internal macro severe recall | 0.752 |
