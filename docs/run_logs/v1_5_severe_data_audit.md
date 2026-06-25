# v1.5 severe-class data audit (labels only)

> Research-only · not diagnostic. The honest training ceiling: a specialist learns
> from severe examples that exist. RSNA labels + splits_v1; no imaging decoded.

| condition | train n / severe | dev severe | test severe |
|---|---|---|---|
| left_neural_foraminal_narrowing | 6910 / **286** | 59 | 52 |
| left_subarticular_stenosis | 6747 / **643** | 131 | 138 |
| right_neural_foraminal_narrowing | 6895 / **278** | 48 | 53 |
| right_subarticular_stenosis | 6751 / **646** | 142 | 137 |
| spinal_canal_stenosis | 6910 / **349** | 68 | 53 |

## Train severe count by level (where the signal is)
| condition | l1_l2 | l2_l3 | l3_l4 | l4_l5 | l5_s1 |
|---|---|---|---|---|---|
| left_neural_foraminal_narrowing | 2 | 10 | 33 | 99 | 142 |
| left_subarticular_stenosis | 18 | 63 | 140 | 325 | 97 |
| right_neural_foraminal_narrowing | 11 | 5 | 31 | 98 | 133 |
| right_subarticular_stenosis | 20 | 50 | 142 | 313 | 121 |
| spinal_canal_stenosis | 18 | 42 | 90 | 184 | 15 |

## Interpretation
- The **train severe count** per route bounds how much a specialist/MIL can learn. Routes
  with few train-severe examples (esp. right-foraminal) are data-limited; MIL/aug
  can help robustness but cannot create absent severe signal.
- Severe examples concentrate at L4/L5 and L5/S1 (lower levels), so level-aware
  sampling targets those.

Reproduce: `python scripts/run_severe_data_audit_v1_5.py`.
