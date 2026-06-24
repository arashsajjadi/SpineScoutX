# v1.4 Raw Accuracy War Room — conclusion (rigorous negative)

> Research-only · not diagnostic. The honest bottom line: **no raw severe-recall improvement
> was achievable in this bounded sprint**, and we prove *why*. No accuracy-upgrade tag.

## What was attempted (and the result)
| effort | result |
|---|---|
| **Ruthless accuracy audit** (code + on-real-data + logic) | **No severe-recall-corrupting bug.** 13/13 on-data invariants PASS (incl. collect_probs alignment 0/4384 mismatches), 8/8 logic tests PASS. One real *latent* bug (B1) found + fixed with **zero metric delta** (no duplicate keys in any deployed manifest). |
| **Baseline reproduction** | All 5 routes reproduce the v1.3 numbers **exactly** (deterministic inference) → the reported metrics are real, not artifacts. |
| **v1.3-localization → subarticular grading** (paired re-crop) | **dev Δ +0.004, test Δ −0.040.** Better localization does **not** transfer to better grading — the robust grader tolerates leveling noise. Decode is **not** the bottleneck. |
| **Right-foraminal retrain** | **Not repeated** (non-decisive 5× across v1.0–v1.3; n_severe≈53; 56% of misses confidently-normal). The v1.4 audit confirms no bug depresses it, so a recipe change cannot manufacture absent signal. |

## Raw severe recall — UNCHANGED (locked-test auto)
| finding | v1.3 | v1.4 | Δ |
|---|---|---|---|
| spinal canal | 0.830 | 0.830 | +0.000 |
| left foraminal | 0.788 | 0.788 | +0.000 |
| right foraminal | 0.660 | 0.660 | +0.000 |
| left subarticular | 0.746 | 0.746 | +0.000 |
| right subarticular | 0.737 | 0.737 | +0.000 |
| **macro** | **0.752** | **0.752** | **+0.000** |

(No grader was retrained; deployed graders are unchanged, so deployed recall is unchanged. The
re-crop experiment was an *inference-time* test that did not improve recall and is not deployed.)

## Why the ceiling exists (the proven bottleneck)
1. **Not a bug** — the metric/label/side/crop/train-eval/split logic is correct (audit) and the
   baselines reproduce exactly. There is no "free" accuracy hiding in the pipeline.
2. **Not the localizer decode** — a real localization gain (v1.3 ±1-hit 0.43→0.49) does **not**
   raise subarticular grading (dev +0.004 / test −0.040). The grader already absorbs leveling noise.
3. **Sample/signal limit** — right-foraminal severe recall is bounded by **n_severe≈53** and the
   fact that **56% of its severe misses are confidently-normal** (no visual signal the grader can
   exploit). This is a *data* limit, not a tuning/architecture limit.

## Next bottleneck (what a real raw-accuracy jump now requires)
Raw severe recall is **grader-capacity / training-data limited**, so the next gains need real
investment beyond a bounded sprint:
- **More severe-class training data** (especially right-foraminal and subarticular severe cases)
  or external data — the single highest-leverage lever.
- **A retrained grader on the v1.3-improved crops** (train AND eval on the prior-decoder
  distribution, not just eval) — the re-crop test only changed eval; a matched retrain might
  recover the small loss, but the payoff is bounded by (1)–(3).
- **Multi-candidate / MIL grading** (top-k crops per finding) trained end-to-end — a genuine
  capacity increase, not tested here (full retrain, out of sprint scope).

## Tagging decision (honest)
Per the hard principle "no accuracy improvement, no accuracy release tag", v1.4 ships as
**`v1.4.0-accuracy-audit-negative-result`** — a rigorous accuracy audit + proof of the ceiling +
a latent-bug fix + integrity tests, with **no raw-metric change**. No `*-accuracy-upgrade` tag.

Reproduce: `python scripts/audit_accuracy_pipeline.py` ·
`python scripts/run_baseline_reproduction.py` · `python scripts/run_subarticular_recrop_v1_4.py`.
