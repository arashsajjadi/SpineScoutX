# Evidence Intelligence v3 — combined severe-FN risk score (locked-test auto)

> Research-only · not diagnostic. Built from existing locked-test records (no new
> inference). v3 risk = 0.4·(1−conf) + 0.3·instability + 0.15·retrieval_conflict + 0.15·near_severe (fixed weights — NOT fitted on test). Reference labels score severe FNs only.

## Severe-FN detection AUROC (pooled, n=7298, n_severe_FN=109)
| signal | AUROC [95% CI] |
|---|---|
| confidence only | 0.833 [0.807, 0.858] |
| stability only | 0.752 [0.715, 0.787] |
| confidence + stability (v1.1) | 0.818 [0.790, 0.847] |
| **v3 combined** | **0.863 [0.838, 0.888]** |

## Severe-FN capture at matched review budget (pooled)
| budget | confidence only | stability only | v3 combined |
|---|---|---|---|
| 10% | 0.349 | 0.248 | 0.505 |
| 20% | 0.661 | 0.431 | 0.752 |
| 30% | 0.853 | 0.651 | 0.881 |

## Interpretation (honest, no overclaim)
- Pooled, **v3 **improves** severe-FN detection over confidence alone** (0.863 vs 0.833).
- Per condition, v3's value concentrates where the route is weak/uncertain; on strong
  routes confidence already saturates severe-FN detection.
- v3 adds **retrieval_conflict** and **near_severe** to confidence+stability, and feeds
  the case viewer's review reasons + the severe-FN risk surfaced per finding. It is a
  triage/explanation signal; it never changes a prediction.

Reproduce: `python scripts/run_evidence_intel_v3.py`.
