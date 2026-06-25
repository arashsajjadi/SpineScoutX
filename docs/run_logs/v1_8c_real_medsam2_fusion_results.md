# v1.8c — real MedSAM2 fusion + triage results (Phases 9-10)

> Research-only · not diagnostic. Late fusion (deployed probs + real-MedSAM2 morphometry GBM), dev
> α-sweep, **locked-test once**. Reproduce: `train_real_medsam2_fusion_grader_v1_8c.py`,
> `fit_real_medsam2_triage_router_v1_8c.py`.

## Fusion (locked-test, paired) — NO raw improvement

| arm | L-for | R-for | foraminal macro |
|---|---|---|---|
| deployed | 0.788 | 0.660 | 0.724 |
| MedSAM2 fusion (dev α = 0.3) | 0.788 | 0.660 | 0.724 |

Paired Δ +0.000 on both sides (L CI [−0.053,+0.059] n.s.; R CI [+0.000,+0.000]). Even with a
non-zero dev-selected blend, the fusion flips **no** severe argmax decision on locked-test.

## Triage (locked-test, 29 foraminal severe-FN) — NO gain

Severe-FN capture, deployed-only vs +real-MedSAM2 morphometry: 5% 0.24→0.21, 10% 0.38→0.48,
**15% 0.55→0.55, 20% 0.66→0.66** — real MedSAM2 morphometry does **not** improve severe-FN capture
over the grader's own uncertainty (matches v1.8b SAM2.1).
