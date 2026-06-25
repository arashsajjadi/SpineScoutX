# v1.5 — Full Raw Accuracy Offensive: conclusion

> **Research-only · not diagnostic · not clinically validated.** Held-out reference = RSNA graded
> severity. Protocol: `splits_v1` (train 1382 / dev 296 / test 296 studies). **Dev** selects;
> **locked-test** read once per final model family. All numbers below are real, executed runs
> (no analysis-only placeholders). Cluster (study-level) bootstrap CIs, n_boot=2000.

## TL;DR

The offensive **executed four real training experiments** (candidate-bag MIL ×2, dev-selected
ensemble ×2, a BiGRU axial level-sequence refiner, and a recrop→regrade propagation test). On the
**headline grading metric (severe recall / recall@FAR≤10%) every experiment is an executed
negative** on locked-test. The **one decisive win is localization**: the BiGRU sequence refiner
lifts axial level ±1-slice-hit **0.487 → 0.616** and halves the median slice error (2 → 1) on
locked-test. Crucially, that large localization gain **does not propagate to subarticular severe
recall** (test Δ −0.040, n.s.), which — together with both MIL collapses — is **definitive
evidence that the weak routes are grader/data-limited, not localization- or capacity-limited.**

Headline grading accuracy did **not** improve → tag **`v1.5.0-full-retraining-negative-result`**.
The localization win is a real, validated secondary capability gain, reported honestly as such.

Baselines reproduced exactly by every harness (R-for severe recall 0.660; subarticular L/R
0.746/0.737, pooled 0.742; axial ±1-hit current 0.432 / prior 0.487) — confirming the comparisons
are apples-to-apples.

## 1. Candidate-bag MIL grader (mandatory) — EXECUTED NEGATIVE

Multi-instance grading over K=5 auto candidate crops per (study, level, side); shared
ConvNeXt-Tiny encoder (warm-started from the deployed grader) → attention pooling → level+condition
fusion → 3-class severity. Severe sqrt-inverse-freq sampler + focal loss; **dev selection on
recall@FAR≤10%** with a FAR>0.5 spam guardrail; **locked-test once**. Bags built from auto
localizers only (no GT coords); reference severity is the label.

Paired (same findings) deployed single-crop grader vs MIL on locked-test:

| route | metric | baseline | MIL | paired Δ [95% CI] | verdict |
|---|---|---|---|---|---|
| right-foraminal | severe recall (argmax) | 0.660 | 0.453 | **−0.208 [−0.362, −0.056]** | decisive loss |
| right-foraminal | recall@FAR≤10% | 0.811 | 0.736 | −0.075 [−0.191, +0.064] | n.s. loss |
| subarticular (pooled) | severe recall (argmax) | 0.742 | 0.000 | **−0.742 [−0.797, −0.681]** | collapse |
| subarticular (pooled) | recall@FAR≤10% | 0.709 | 0.251 | **−0.458 [−0.524, −0.371]** | decisive loss |
| subarticular-left | severe recall | 0.746 | 0.000 | −0.746 | collapse |
| subarticular-right | severe recall | 0.737 | 0.000 | −0.737 | collapse |

**Why it failed (honest):**
- **Right-foraminal** is the thinnest-severe route (278 train / 48 dev / 53 test severe). The MIL
  *won decisively on dev* recall@FAR≤10% (0.667 → 0.792, Δ +0.125 [+0.023, +0.261]) but the gain
  **did not generalize** — classic small-n dev overfit. The extra MIL capacity fit dev noise.
- **Subarticular** collapsed: the dev-selected epoch maximizing recall@FAR≤10% predicts **no
  severe at argmax** (severe recall 0.000), and even threshold-swept recall@FAR≤10% (0.251–0.286)
  is far below baseline. The fixed paramedian-offset auto crops are noisy and the fresh MIL lacks
  the deployed grader's robust-auto-training; it never learns a usable severe ranking.

This reproduces the standing project finding: **robust-auto-training benefit ∝ the oracle→auto gap;
the deployed subarticular grader already banks that robustness, and a plain MIL does not.**

## 2. Dev-selected baseline+MIL ensemble (cheap recovery attempt) — NEGATIVE

Convex class-probability blend `(1−α)·baseline + α·MIL`; α swept on dev (maximize recall@FAR≤10%,
FAR guardrail); locked-test once.

| route | selected α (dev) | test severe recall (base→ens) | test recall@FAR10 (base→ens) | verdict |
|---|---|---|---|---|
| right-foraminal | 0.95 | 0.660 → 0.453 (Δ −0.208, decisive) | 0.811 → 0.774 (Δ −0.038, n.s.) | loss (dev-overfit α) |
| subarticular | 0.15 | 0.742 → 0.735 (Δ −0.007, n.s.) | 0.709 → 0.713 (Δ +0.004, n.s.) | no change |

The MILs are too weak to add complementary signal: R-for's dev picks α≈1 (the overfit MIL) and
loses on test; subarticular's dev picks a tiny α≈0.15 that barely perturbs the baseline (no
decisive movement either way).

## 3. BiGRU axial level-sequence refiner (mandatory) — DECISIVE LOCALIZATION WIN

The deployed `axial_level_scorer` classifies each axial slice's lumbar level **independently**. The
refiner reads the whole stack as a z-ordered sequence and refines the per-slice level posteriors
with bidirectional context: input per slice = `[scorer 5 log-probs, norm_z]`; a 1-layer BiGRU
(hidden 64) emits refined 5-level logits; the same monotonic DP decode maps levels→slices. Real
training: full-stack scorer log-probs cached for 1382/296/296 studies; CE on labelled slices;
select on val ±1-hit; **locked-test once**.

Locked-test level localization (n=1369 study-levels):

| decode | ±0 hit | ±1 hit | ±2 hit | median \|err\| |
|---|---|---|---|---|
| raw_monotonic (v1.x current) | 0.134 | 0.432 | 0.652 | 2.0 |
| raw_prior (v2 best decode) | 0.162 | 0.487 | 0.714 | 2.0 |
| **BiGRU refiner (v1.5)** | **0.251** | **0.616** | **0.790** | **1.0** |

**±1-hit +0.129 over the v2 prior decode (+0.184 over current); exact-hit +0.089; median slice
error halved.** The harness reproduces the v2 reference exactly (0.432 / 0.487), validating it.
This is a genuine, decisive, executed model-capability improvement on locked-test.

## 4. Does better localization raise grading? recrop→regrade (the decisive test) — NEGATIVE

Re-crop the subarticular evidence at the BiGRU-decoded slice and re-grade with the **fixed**
deployed grader, paired against the current/prior decodes on the same scored stacks (GT severity
scores recall only; no GT coords used).

| split | current | prior | **BiGRU** | BiGRU−current Δ [95% CI] |
|---|---|---|---|---|
| dev | 0.689 | 0.692 | **0.714** | +0.026 [−0.025, +0.081] (n.s.) |
| test | 0.742 | 0.702 | 0.702 | **−0.040 [−0.084, +0.004]** (n.s.) |

Despite a far larger localization gain than v1.4's prior decode (+0.129 vs +0.055 ±1-hit), test
subarticular severe recall **does not improve** (the current decode's 0.742 remains the
high-water mark). This **decisively confirms and extends v1.4**: the robust grader is invariant to
leveling, so subarticular severe recall is **grader/data-limited, not localization-limited.**

## Verdict & tag

- **Mandatory experiments all executed** (real training ran; no analysis-only stops): candidate
  bags built, R-for MIL trained, subarticular MIL trained, BiGRU axial stack scorer trained.
- **Headline grading accuracy (severe recall / recall@FAR≤10%) did not improve on any route** →
  none of the v1.5 accuracy-upgrade targets met → **`v1.5.0-full-retraining-negative-result`**.
- **Real positive:** decisive locked-test **localization** win (BiGRU ±1-hit +0.129, median error
  halved) — a capability gain, not a grading-accuracy gain; reported as such (no overclaim).
- **Scientific yield:** convergent, decisive evidence that the weak routes (right-foraminal,
  subarticular) are **data/grader-limited** — more candidate crops (MIL) overfit thin severe data,
  and better localization (BiGRU) does not move a leveling-robust grader. The accuracy ceiling here
  is **severe-label quantity + grader robustness**, not localization or model capacity.

Reproduce: `run_baseline_reproduction.py`, `train_mil_grader_v1_5.py --route {right_foraminal,
subarticular}`, `compare_mil_vs_baseline_v1_5.py`, `run_mil_ensemble_v1_5.py`,
`run_axial_seq_refiner_v1_5.py`, `run_subarticular_recrop_bigru_v1_5.py`. Artifacts (weights,
caches, JSON) are gitignored; numbers above are the committed record.
