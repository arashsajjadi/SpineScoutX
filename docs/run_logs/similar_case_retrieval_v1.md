# Similar research-case retrieval (explanation-only) — v1

> Research-only · not diagnostic. Top-k nearest DEV neighbours per TEST finding in the
> grader's penultimate-feature space (cosine; cached crops, no GT). **Explanation only —
> retrieval NEVER changes a prediction.** Retrieved cases are *similar research cases* —
> not a clinical reference.

Bank = dev, query = test, k = 5. Overall severity agreement (majority-retrieved ==
held-out reference) **0.792** over 7298 findings; mean
same-condition rate 0.618.

| grader | n | severity agreement | same-side rate | same-condition rate |
|---|---|---|---|---|
| canal | 1480 | 0.891 | n/a | 1.000 |
| foraminal | 2950 | 0.805 | 0.522 | 0.522 |
| subarticular | 2868 | 0.728 | 0.520 | 0.520 |

## Interpretation (honest)
- High same-condition / same-side rates mean the grader embedding groups anatomically
  similar findings — retrieval returns relevant *similar research cases*.
- Severity agreement indicates retrieved neighbours tend to share the held-out severity;
  it is a sanity check on the embedding, **not** a second predictor (we never vote).
- Surfaced in the case viewer as `similar_research_cases` (severity distribution).

Reproduce: `python scripts/run_similar_case_retrieval.py`.
