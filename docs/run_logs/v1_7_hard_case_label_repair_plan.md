# v1.7 — Hard-Case Label Repair + Active Re-annotation + Noise-Aware Offensive: plan

> **Research-only · non-commercial · not diagnostic · not clinically validated · not for medical
> decision-making.** Held-out reference = RSNA graded severity. Protocol: `splits_v1`
> (train 1382 / dev 296 / test 296 studies). **Dev/train select; RSNA locked-test read once** per
> final selected system. **Never** use locked-test labels for training/selection and **never**
> modify locked-test labels.

## Why v1.7 is a DATA/LABEL-QUALITY sprint (not a model sprint)

v1.4 (audit), v1.5 (MIL + localization), and v1.6 (external LSS data + SSL + anatomy + bigger
backbone) all proved the weak-route severe ceiling is **not** moved by model/representation/
external-data/capacity changes. v1.6's autopsy localised the binding constraint to **in-domain
severe-label quantity AND quality** — right-foraminal has only 278 train / 53 test severe, and
~56% of its misses are *confidently-normal* predictions (label ambiguity / genuinely subtle cases).
**The next scientifically valid lever is in-domain label quality + severe-class signal.**

## Plan (label-quality first, model second)

1. **Hard-case mining** — find the cases most likely to unlock severe recall (right-foraminal
   severe FN first; confidently-normal misses; moderate/severe borderline; model-disagreement;
   retrieval-conflict; low route-quality; + controls). Reuse the v1.6 multi-model prediction
   artifacts (deployed grader, v1.6 baseline/LSS/joint/small, v1.5 MIL) for the disagreement signal.
2. **Local-only radiology review pack** — a real second-read product (HTML + CSV + JSONL + image
   panels) for expert re-annotation. **The entire pack is gitignored / local-only.**
3. **Human-review ingestion** — validate + version corrected labels separately; never overwrite
   raw labels; never use test labels. If no reviewed file exists, write the review-needed handoff.
4. **Provisional algorithmic label-cleaning (fallback)** — if no human labels: soft labels +
   ambiguity flags + sample weights from ensemble agreement / disagreement / retrieval / confidence
   / route-quality. **Marked provisional; not ground truth; never flips labels blindly; train/dev
   only.**
5. **Noise-aware foraminal retraining** — train *after* label repair (soft-label CE, ordinal,
   severe-FN upweight, ambiguity downweight) from the deployed architecture first (not a bigger
   model). Dev selects (right-foraminal recall@FAR≤10 + severe recall + high-conf severe-FN);
   locked-test once.
6. **Teacher-distillation fallback** + **severe-FN triage fallback** if raw accuracy doesn't move.
7. Final eval, autopsy, docs, gates, merge.

## Non-negotiables (no repeats of failed paths)

No internal-MIL-only retry; no localizer-only retry; no LSS-only retry; no "bigger backbone" as the
main plan; no README/gallery sprint; no analysis-only negative; **"we need more labels" is not a
stopping point — build the re-labeling system + review pack.**

## Safety / hygiene

- **Local review panels may contain imaging pixels but MUST stay gitignored / local-only**
  (`review_packs/`, `data/labels/`, `outputs/`, `runs/` all gitignored). Committed artifacts =
  code, docs, configs, summaries, and **pixel-free** synthetic/schematic examples only.
- No DICOM/NIfTI/PNG-JPG panels/HTML pack/raw crops/masks/caches/checkpoints/weights/parquet
  prediction dumps committed. No patient identifiers. Repo private.
- **Final deployment only if raw locked-test severe recall improves.** A review pack is a valid
  milestone; a severe-FN triage improvement is a *safety* upgrade, not an accuracy upgrade. No
  overclaiming; no hidden-GT use; no locked-test tuning.

## Success / tagging

`v1.7.0-reviewed-label-accuracy-upgrade` (reviewed labels improve locked-test raw severe recall) ·
`v1.7.0-noise-aware-accuracy-upgrade` (provisional/noise-aware training improves it) ·
`v1.7.0-triage-safety-upgrade` (raw flat but severe-FN triage metrics improve) ·
`v1.7.0-hardcase-review-pack` (pack created, no human labels yet) ·
`v1.7.0-label-repair-negative-result` (all of pack + provisional cleaning + retraining + teacher +
triage executed/blocked with evidence, no raw gain). **No accuracy-upgrade tag without a real
locked-test raw metric improvement.**
