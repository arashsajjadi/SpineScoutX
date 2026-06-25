# v1.7 provisional label cleaning (PROVISIONAL — not ground truth)

> Research-only · not diagnostic. **Provisional soft labels + ambiguity flags + sample weights** for train+dev only; raw RSNA labels are never overwritten; locked-test is never touched. This is a fallback for human review, NOT a substitute. Reproduce: `scripts/provisional_label_cleaning_v1_7.py`.

- train+dev findings: **16740**
- ambiguity-flagged: **263** (severe: 16 of 671; right-foraminal severe: 10)
- severe upweighted: **655**
- rules applied: {'keep_original': 15822, 'severe_confirmed_upweight': 633, 'moderate_with_severe_evidence': 247, 'severe_poor_evidence': 11, 'severe_baseline_upweight': 22, 'severe_ambiguous_models_disagree': 5}
- mean sample weight by class (0/1/2): {0: 1.0, 1: 1.0, 2: 1.47}

Soft-label rules never flip a label; they soften ambiguous severe cases, sharpen model-confirmed severe cases, and move probability mass for moderate-with-severe-evidence cases. Used by `train_noise_aware_foraminal_v1_7.py --mode provisional` (Phase 5).
