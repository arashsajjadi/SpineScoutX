# v1.7 noise-aware foraminal retraining (Phase 5) — EXECUTED NEGATIVE on raw grading

> Research-only · not diagnostic. Trained the deployed convnext_tiny foraminal architecture on
> RSNA `splits_v1` with label-quality modes (raw RSNA labels never modified on disk; soft labels /
> weights from the gitignored provisional parquet, train+dev only). **Dev** selects on
> right-foraminal recall@FAR≤10 (anti-spam guardrails: reject FAR>0.5 / all-severe / all-normal).
> Reproduce: `train_noise_aware_foraminal_v1_7.py`.

## Dev result (mode selection)

| mode | what it changes | dev R-for recall@FAR≤10 |
|---|---|---|
| **A** original labels (baseline) | one-hot, class-weighted CE | **0.792** (dev-best) |
| C provisional soft labels | soft-CE (moderate→soft-severe etc.) | 0.750 |
| F severe-FN upweight | ×2 weight on severe | 0.750 |
| G hybrid soft+ordinal+weights | (budget-deferred this run) | — |

**Dev-selection chose mode A (original labels).** The provisional soft-label and severe-upweight
modes did **not** beat the original-label baseline on dev — the soft modes are severe-spammy (dev
FAR 0.48–0.83, caught by the guardrail), i.e. moving moderate→severe mass raises false alarms
without a net recall@FAR gain. G (the hybrid) was deferred for compute budget; it inherits the same
spammy soft-label behaviour as C and is not expected to beat A (documented, not hidden).

## Verdict — NEGATIVE

Because the dev-selected model is the **original-label** mode A — whose recipe reproduces the v1.6
ImageNet baseline (locked-test R-for **0.679** / foraminal macro **0.724**) — provisional label
cleaning **does not improve foraminal severe grading**, and no fresh locked-test peek is spent on
the equivalent model. This is expected: the locked-test severe labels themselves are unchanged, so
cleaning only the *train* labels cannot raise recall measured against the (possibly noisy) test
labels. The honest lever remains **human re-annotation** (the review pack) + a *test-label* review,
which this run cannot perform. Triage (Phase 8) is the deployable safety win in the meantime.
