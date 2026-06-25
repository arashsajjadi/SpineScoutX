# v1.6 LSS-MRI AISSLab audit (Plan A)

> Research-only · not diagnostic. Numbers from `data/cache/lss_foraminal/audit.json` (crop cache),
> reproducible via `scripts/audit_external_data_v1_6.py`.

## Foraminal label distribution (RSNA-mapped)

- **2978 foraminal crops** from **468 patients** (1 box skipped: degenerate bbox).
- By side: **right 1396**, left 1582.
- By RSNA severity index: **0 (normal/mild) 2516 · 1 (moderate) 254 · 2 (severe) 208**.
- **Severe (208)** by side: **right 93**, left 115.
- By level: l1_l2 571 · l2_l3 686 · l3_l4 794 · l4_l5 649 · l5_s1 278.
- Patient-level split: lss_train 2527 (179 severe) · lss_dev 451 (29 severe).

## Key correction (severe count)

A pre-acquisition web summary claimed "47 Severe" total. **That is wrong for the released V0.2
archive.** The dataset's own GitHub README states the distribution (Normal 67.45% ≈ 2009/2979) and
**our parse of the actual XML matches it exactly** — including **208 Severe (grade-3) foraminal
boxes**. This is independently verified by the per-box `<name>` decoding (`RFS3`/`LFS3` = severe).

**Implication for strategy:** LSS is **not** merely a representation resource — it is a genuine
**foraminal severe-label augmentation** source. RSNA has ~278 right-foraminal train-severe; LSS
adds 93 right + 115 left external severe foramina with precise boxes. This makes the LSS→RSNA
transfer (Plan A) a real shot at improving right-foraminal severe recall, not just morphology.

## Caveat

LSS severe is still scarce in absolute terms (208) and from a single hospital (Firat University,
Türkiye) → domain shift vs RSNA (US/multi-site). Compatibility quantified in
`lss_rsna_compatibility_v1_6.md`; transfer measured on **RSNA dev**, confirmed on **locked-test
once**.
