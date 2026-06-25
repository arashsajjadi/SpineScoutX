# v1.5 external-data acquisition scout

> Research-only · not diagnostic. Searched for legal/public lumbar-MRI datasets to add training
> signal. Conclusion: **no new legal-to-redistribute dataset with the matching 5-finding severity
> labels is quickly integrable**; v1.5 proceeds **RSNA-only** with candidate-bag MIL (a real
> capacity increase needing no new labels).

## Candidates evaluated
| dataset | license / access | views / labels | compatible? | usable here? |
|---|---|---|---|---|
| **RSNA LumbarDISC** | non-commercial; Kaggle + RSNA MIRA; redistribution restricted | sag-T1/T2 + axial-T2; canal/subarticular/foraminal severity per level | **yes (already the training set)** | already used; cannot redistribute |
| **SPIDER** | CC BY 4.0; Zenodo 10159290 / HF | sag-T1/T2 segmentations + some IVD gradings | partial (anatomy, not the 5 graded findings; no axial subarticular) | already used for anatomy; **possible SSL pretraining** (CC BY) |
| **VerSe** | CC; vertebra segmentation (CT-centric) | spine CT/MR vertebrae | no (no degenerative severity; CT) | not usable for grading |
| **MRI-CORE foundation model** (arXiv 2506.12186) | model weights, not data | MRI foundation features | maybe (pretrained backbone) | out of scope (download + adapt; not a quick integration) |

## Decision
- **No new labeled dataset** with the RSNA five findings is both legally redistributable and
  label-compatible for a quick supervised add. RSNA terms forbid committing/redistributing its
  imagery (consistent with the pixel-free policy); SPIDER lacks the matching graded labels.
- **v1.5 trains RSNA-only.** The capacity increase comes from **candidate-bag MIL** (aggregating
  K localized crops per finding) — a real model-capacity lever that needs **no new labels**.
- **SSL pretraining** on SPIDER + RSNA train/dev slices (CC BY for SPIDER) is feasible in
  principle but expensive; documented as a future lever, not run in this bounded sprint (it would
  not change the severe-class label ceiling that limits the weak routes).

## Compliance
No external data downloaded or committed. RSNA/SPIDER imagery remains gitignored; no
redistribution. See `configs/data/external_candidates_v1_5.yaml`.

Sources: [RSNA LumbarDISC (Radiology:AI)](https://pubs.rsna.org/doi/10.1148/ryai.250480) ·
[LumbarDISC arXiv 2506.09162](https://arxiv.org/abs/2506.09162) ·
[SPIDER (Zenodo 10159290)](https://zenodo.org/records/10159290) ·
[MRI-CORE foundation model (arXiv 2506.12186)](https://arxiv.org/pdf/2506.12186).
