# v1.7 — failure autopsy (raw accuracy not improved; triage safety upgrade achieved)

> Research-only · not diagnostic. The label-repair offensive executed mining + review pack +
> provisional cleaning + noise-aware retraining + teacher distillation + triage. Raw severe recall
> did not improve; this autopsy answers the exact questions and quantifies the human-review need.

## Did the hard-case pack identify label ambiguity? — YES

Train+dev mining surfaced **21 confidently-normal severe misses** (true severe predicted normal_mild
with p_nm≥0.5, p_sev≤0.2; 12 right-foraminal), **177 severe FN** (87 right), and **1550 moderate/
severe borderline** findings. The moderate/severe boundary holds most of the ambiguity — consistent
with the v1.6 conclusion that the ceiling is label-quality bound.

## How many severe FN are confidently normal / likely mislabelled or ambiguous?

On **locked-test** (n=2950 foraminal): **29 severe FN**, of which **6 are high-confidence** (true
severe predicted *confidently* normal_mild). These 6 are the prime mislabel/ambiguity suspects:
every model (deployed + 4 v1.6 variants) agrees they look normal, so triage cannot surface them
(0/6 captured at 5–10% budget, 3/6 at 15–20%). Train+dev: ~21 such confidently-normal severe.

## Did provisional label cleaning help dev? Did it generalise?

**No.** Dev-selection across modes chose mode A (original labels, R-for recall@FAR≤10 0.792); the
provisional soft-label (0.750) and severe-upweight (0.750) modes lost on dev (soft labels became
severe-spammy, dev FAR 0.48–0.83). So there was nothing to generalise — cleaning the **train**
labels cannot raise recall measured against unchanged (possibly noisy) **test** labels.

## Did reviewed labels exist? What exact human step is needed?

**No reviewed labels in this run.** A complete **704-case review pack** was generated
(`review_packs/v1_7_hard_cases/`, local-only). Exact step: a radiologist sets the true severity for
each case (priority: right-foraminal severe FN → L4-L5/L5-S1 → left-foraminal), saves
`review_sheet_reviewed.csv`; `ingest_review_labels_v1_7.py` then versions corrected labels for
Phase-6 retraining. **A test-set re-read is also required** to lift the measured ceiling, since the
locked-test labels are themselves part of the noise and may not be modified under this protocol.

## Annotation budget + expected gain (honest estimate)

- **~250 right-foraminal hard cases** (the pack already contains 338) covers essentially all
  right-foraminal severe-FN + the moderate/severe borderline around them.
- Expected gain is **bounded by where the corrected labels live**: correcting **train** labels for
  the ~12–21 confidently-normal severe + the ~745 right-foraminal borderline may sharpen the
  decision boundary, but the locked-test number only moves if the **test** severe labels are also
  re-read. Rough scaling: **100 reviewed** (train) → likely <+0.02 R-for (boundary sharpening only);
  **250** → +0.02–0.05 *if* a meaningful fraction of confidently-normal severe are genuine and
  consistently re-cropped; **500 + a test re-read** → the first credible path to a +0.05 R-for gain
  by removing label noise on both train and the evaluation set.
- **Annotate first:** right-foraminal severe FN at **L4-L5 / L5-S1**, then the moderate/severe
  borderline, then left-foraminal.

## Net

Raw accuracy is label-noise-bound and cannot be moved by algorithmic cleaning alone (dev rejects it)
or by distillation (test collapse). The **deployable outcome is the triage safety upgrade**
(effective severe recall 0.724 → 0.933 at 15% review) plus the **review pack** that makes the real
fix — expert re-annotation of train **and** test — executable.
