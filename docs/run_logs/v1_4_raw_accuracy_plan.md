# v1.4 — Raw Accuracy War Room: plan

> Research-only · not diagnostic. This sprint targets **raw model performance**, not docs.
> No accuracy improvement ⇒ no accuracy tag.

## 1. Current baseline (locked-test auto severe recall, frozen at v1.3 / main `61ad496`)
| finding | route | severe recall [95% CI] |
|---|---|---|
| spinal canal stenosis | sagittal-T2 | 0.830 [0.725, 0.929] |
| left neural foraminal narrowing | sagittal-T1 | 0.788 [0.673, 0.892] |
| right neural foraminal narrowing | sagittal-T1 | **0.660** [0.524, 0.788] |
| left subarticular stenosis | axial-T2 | 0.746 [0.674, 0.815] |
| right subarticular stenosis | axial-T2 | 0.737 [0.667, 0.807] |

## 2. Weakest conditions
Right neural foraminal (0.660) ≫ then subarticular (~0.74). Macro severe recall ≈ 0.752.

## 3. Likely bottlenecks (from v1.0–v1.3 evidence)
- **Right-foraminal**: signal/sample-limited — specialist non-decisive **5×**; 56% of severe
  misses are confidently-normal; n_severe≈53; per-level thresholding doesn't help.
- **Subarticular**: axial leveling imperfect (±1-hit improved 0.43→0.49 in v1.3 via a decode
  prior, with **no retrain**); the robust grader *tolerates* leveling noise (bounded payoff —
  but the actual v1.3 localization gain has **not** been pushed through to a re-grade yet).
- Possible **pipeline bug** anywhere in label/side/crop/eval — not yet audited for accuracy.

## 4. Model / pipeline hypotheses
- **H1 (bug):** a real accuracy-relevant bug exists (class mapping / severe-recall calc / side
  handling / auto-vs-oracle provenance / train-eval preprocessing mismatch / split leakage). If
  so, fixing it is the highest-value win.
- **H2 (localization→grading):** feeding the **v1.3 better-localized** subarticular crops (v2
  positional-prior decoder) to the deployed subarticular grader raises severe recall.
- **H3 (right-foraminal training):** a class-balanced/focal loss + severe oversampling moves
  right-foraminal severe recall (low prior odds — sample-limited).

## 5. Experiments (bounded, high-value first)
1. **Accuracy-integrity audit** + on-real-data invariant tests (H1). Fix + measure any bug.
2. **Baseline reproduction + variance** (so noise ≠ improvement).
3. **v2-decoder subarticular re-crop → re-grade** (H2): regenerate auto subarticular crops with
   the v1.3 prior decoder, re-run the deployed grader, compare severe recall.
4. **Right-foraminal bounded retrain** (H3): one dev-selected loss/sampling run, locked-test once
   — only if budget remains; else rigorous negative.
5. Generalization stress on any change (no single-number wins that break strata).

## 6. Strict selection rules
- **dev** selects every method/threshold/hyperparameter; **locked-test evaluated once**.
- No test-derived thresholds/augmentation/ensembling. No hidden GT in auto inference.
- Cluster-bootstrap CIs by study; paired deltas; report FAR alongside any recall change.
- Reproduce the baseline first; a change inside run-to-run noise is **not** an improvement.

## 7. What counts as a real improvement
- right-foraminal severe recall **+≥0.05** abs (dev-selected, honest CI); or
- subarticular severe recall **+≥0.03** abs; or
- macro severe recall **+≥0.03** abs; or
- material drop in high-confidence severe FNs **without** unacceptable FAR; or
- a real bug fixed with a measurable metric delta.

## 8. What does NOT count
- README/gallery/explanation changes; severe-FN *detection* AUROC (already done in v1.3);
- recall gains purchased by exploding FAR (must report FAR); changes within reproduction noise;
- anything selected on the locked test.

## Gates / safety (unchanged)
pytest + ruff + format + build + doctor + forbidden-file/large-file/claim scans + link checks;
research-only; no GT in auto inference; reference never an input; no locked-test tuning; no
DICOMs/weights/runs/outputs/caches/identifiers committed; merge to `main` via PR (merge commit)
after gates. **Tag `*-accuracy-upgrade` only if a raw/headline metric improves; else
`v1.4.0-accuracy-audit-negative-result`.**
