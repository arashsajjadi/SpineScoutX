# Similar research-case retrieval v2 — side/level-aware (explanation-only)

> Research-only · not diagnostic. v2 filters the research-case bank by same
> (condition, level, side) first (back-off to level, then condition) before the cosine
> kNN, so neighbours are anatomically matched. **Never changes a prediction.**

Overall (k=5): same-side **1.000** (v1 ≈ 0.52 = chance), same-level 1.000, severity agreement 0.794.

| grader | n | same-side | same-level | severity agreement |
|---|---|---|---|---|
| canal | 1480 | 1.000 | 1.000 | 0.899 |
| foraminal | 2950 | 1.000 | 1.000 | 0.807 |
| subarticular | 2868 | 1.000 | 1.000 | 0.726 |

## Interpretation (honest)
- v2 makes retrieval **side/level-aware** via metadata filtering (v1's embedding was
  side-agnostic at ~chance). Neighbours are now anatomically matched, so the retrieved
  severity distribution and the **retrieval_conflict** signal are meaningful.
- Still explanation-only: retrieval never votes on or changes the prediction.

Reproduce: `python scripts/run_similar_case_retrieval_v2.py`.
