# v1.8b — final segmentation-morphometry results

> Research-only · not diagnostic · not clinically validated. Protocol: `splits_v1`; **dev** selects;
> **RSNA locked-test read once** for the dev-selected fusion. Paired cluster bootstrap, n_boot=2000.
> **No model is deployed from v1.8b** (no raw gain); deployed 5/5 graders unchanged (macro 0.752).

## Pipeline executed

MedSAM2 + SAM3 + SAM2.1 downloaded (gitignored); **SAM2.1** segmented all 19,700 RSNA foraminal
findings at **0% failure** → 13 morphometric features → morphometry-only signal check → late fusion
→ morphometry triage.

## Raw accuracy (locked-test foraminal severe recall) — NO improvement

| arm | L-for | R-for | foraminal macro |
|---|---|---|---|
| deployed reference | 0.788 | 0.660 | 0.724 |
| **morphometry fusion** (dev-selected α = **0.0**) | 0.788 | 0.660 | 0.724 |

The dev α-sweep selected **α = 0.0** (pure deployed grader): adding any morphometry weight does **not**
improve dev right-foraminal recall@FAR≤10, so the fusion equals the baseline (paired Δ **+0.000**).

## Morphometry-only signal vs redundancy

Morphometry **does** carry severity signal (right-foraminal dev AUROC **0.687**, from intensity
**contrast**, not area) — but it is **weaker than and redundant with** the image grader (which sees
the same pixels). Hence fusion gains nothing.

## Triage (Phase 11) — morphometry adds nothing

Severe-FN capture at review budgets, deployed-only vs +morphometry (locked-test, 29 foraminal
severe-FN): 10% 0.38→0.48, **15% 0.55→0.55, 20% 0.66→0.59** — morphometry does **not** reliably add
to severe-FN capture beyond the grader's own uncertainty.

## Verdict / tag

**Segmentation-morphometry is NOT the missing signal** — it is real but redundant with the image
grader (criterion #7 answered). No raw severe-recall improvement, no fusion gain, no triage gain →
**`v1.8b.0-morphometry-negative-result`**. All Plans A–D executed; nothing deployed. Autopsy:
`v1_8b_morphometry_failure_autopsy.md`.
