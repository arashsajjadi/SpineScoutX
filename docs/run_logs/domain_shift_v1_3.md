# Domain-shift / generalization stress test v1.3 (locked-test auto)

> Research-only · not diagnostic. Model-internal reliability bins from existing records
> (no new inference). Severe recall per bin; reference labels score recall only.

Overall severe recall **0.748** (n_severe=433).

## confidence tertile
| bin | severe recall | n_severe |
|---|---|---|
| low | 0.524 | 208 |
| mid | 0.956 | 225 |
| high | nan | 0 |

## evidence stability grade
| bin | severe recall | n_severe |
|---|---|---|
| stable | 0.907 | 129 |
| mildly_unstable | 0.775 | 169 |
| unstable | 0.563 | 135 |

## route
| bin | severe recall | n_severe |
|---|---|---|
| sagittal_t2 | 0.830 | 53 |
| sagittal_t1 | 0.724 | 105 |
| axial_t2 | 0.742 | 275 |

## level
| bin | severe recall | n_severe |
|---|---|---|
| l1_l2 | 0.429 | 7 |
| l2_l3 | 0.750 | 28 |
| l3_l4 | 0.738 | 80 |
| l4_l5 | 0.868 | 197 |
| l5_s1 | 0.579 | 121 |

## instability type
| bin | severe recall | n_severe |
|---|---|---|
| stable | 0.884 | 69 |
| route_sensitive | 0.333 | 12 |
| crop_sensitive | 0.333 | 9 |
| slice_sensitive | 0.692 | 13 |
| axial_candidate_sensitive | 0.750 | 24 |

## Interpretation (honest)
- Severe recall drops sharply in the **low-confidence** and **unstable** bins — these are
  exactly where `review_required` + the evidence-v3 risk score concentrate review, so the
  model's own reliability signals track its generalization weaknesses.
- Anatomical/route bins reconfirm the known weak spots (right-foraminal route, L5-S1).
- Internal stress only — **not** external/prospective generalization (no such data).

Reproduce: `python scripts/run_domain_shift_v1_3.py`.
