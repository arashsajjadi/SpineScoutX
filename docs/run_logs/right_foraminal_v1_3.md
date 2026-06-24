# Right neural foraminal narrowing — v1.3 audit + evidence-v3 review repair

> Research-only · not diagnostic. The weakest route. v1.3 does **not** retrain a
> specialist (non-decisive across v1.0–v1.2); it improves severe-FN **triage** via the
> evidence-intelligence v3 risk score. Reference labels score FNs only.

## Hard-case audit (locked test)
- n=1470, n_severe=53, severe FNs=18.
- **56%** of severe misses are confidently normal
  (P(severe)<0.20; mean miss P(severe) 0.204) — a signal/sample
  limit, not a thresholding knob (consistent with v1.0–v1.2).

## Severe-FN triage: evidence v3 vs confidence
- severe-FN detection AUROC: confidence **0.806** [0.720,0.881] → v3 **0.823** [0.726,0.901].

| review budget | confidence-only capture | v3 capture |
|---|---|---|
| 10% | 0.389 | 0.444 |
| 20% | 0.611 | 0.556 |
| 30% | 0.722 | 0.889 |

## Verdict (honest)
- **Accuracy unchanged** — right-foraminal severe recall remains sample/signal-limited;
  retraining a specialist has been non-decisive four times, so it was not repeated.
- **Real v1.3 gain = triage:** the v3 risk score catches right-foraminal severe FNs
  better than confidence alone for human research review. Next step for accuracy: more
  right-side severe data or a dedicated right localizer.

Reproduce: `python scripts/run_right_foraminal_v1_3.py`.
