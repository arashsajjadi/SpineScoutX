# v1.4 accuracy pipeline audit — code + on-real-data

> Research-only · not diagnostic. A ruthless audit run **before** more training, to find any
> bug depressing or inflating locked-test severe recall. Verdict: **no severe-recall-corrupting
> bug**; one real latent bug fixed (zero metric delta today); baselines reproduce exactly.

## Method
- **Code review** of all 8 risk areas (metric calc, class mapping, collect_probs alignment,
  side/laterality, crop logic, train/eval preprocessing parity, model selection + loss, split
  leakage) — `src/spinescoutx/{evaluation,data,training}/...` + the locked-test scripts.
- **On-real-data invariants** (`scripts/audit_accuracy_pipeline.py`, **13/13 PASS**): split
  disjointness, auto-provenance (no hidden GT), **collect_probs (y_true,probs) alignment to
  manifest labels** (0/1480 mismatches on canal; 0/1470 R-foraminal; 0/1434 L-subarticular),
  severe recall recomputed two independent ways (exact match), crop-side matches condition,
  severe == class index 2.
- **Pure-logic tests** (`tests/test_accuracy_pipeline_integrity.py`, **8/8 PASS**): severe=index
  2, severe-recall definition, recall@FAR monotone/bounded, crop-bounds no off-by-one + needs_pad,
  x/y (col/row) convention, 2.5D channel order, foraminal side convention (+x = patient-left).

## Findings
### BUG B1 (real, latent, low-severity) — FIXED, zero metric delta
`collect_probs` keyed its result dict by `"study|level"`; a manifest with >1 row per
(study, level) — e.g. a combined sided manifest — would **silently overwrite** rows and bias the
metric. **Harmless today:** every deployed call pre-filters to one condition+side (verified
**0 duplicate keys** in all four deployed caches × splits), so no headline number is affected.
**Fix:** `gap_decomposition.collect_probs` now **raises** on duplicate keys (loud, not silent).
Metric delta: **none** (no duplicates exist in the deployed path; re-audit still 13/13 PASS).

### Caveats (not bugs, documented)
- **C1 — `recall_at_far*` is an in-sample oracle-threshold metric** (threshold swept on the eval
  set): an *upper bound* on a fixed deployed threshold. The bootstrap re-sweeps per resample
  (internally honest) and it is applied identically to all models (comparisons fair). The
  **headline `severe_recall` is argmax-based** and has no such issue. Reported, not changed.
- **C2 — robust consistency-loss weights moderate+severe** (`targets>=1`): matches the docstring
  and is **not used in the locked-test path** (those train on auto-distribution crops via
  `RsnaCropDataset`, consistency_weight 0). No impact.
- **C3 — `severe_fnr` returns 0.0 (not NaN) on a severe-empty split**: a latent foot-gun for the
  `severe_aware` selection metric on very-low-prevalence conditions; dev sets here are
  severe-rich, so no practical impact. Low priority.

## What this means for accuracy
The reported v1.3 locked-test severe recall is **correct, not a bug artifact** — metric math,
label mapping, eval alignment, laterality, crop geometry, train/eval parity, model selection, and
split discipline are all sound, and the baselines reproduce to ≤0.005 (deterministic inference,
see `v1_4_baseline_reproduction.md`). **There is no "free" accuracy hiding in a bug.** Raw
improvement must therefore come from data/model capacity, not a fix — which directs the rest of
v1.4 to targeted experiments (and bounds expectations honestly).

Reproduce: `python scripts/audit_accuracy_pipeline.py` ·
`python -m pytest tests/test_accuracy_pipeline_integrity.py`.
