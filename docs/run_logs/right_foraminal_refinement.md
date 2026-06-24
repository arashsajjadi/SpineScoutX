# Right-foraminal refinement — specialist vs combined (locked test)

> Research-only. Not diagnostic. Right-only oracle-trained specialist vs the combined
> side-aware grader, paired on the same locked-test right-foraminal nodes (n=1470, severe=53); cluster-bootstrap CIs.

| grader | right-foraminal auto severe recall [95% CI] |
|---|---|
| combined side-aware | 0.660 [0.524, 0.788] |
| right specialist | 0.698 [0.571, 0.820] |

Paired specialist − combined: **+0.038** [-0.091, +0.149] (decisive=False); McNemar 5/3 p=0.727.

## Verdict
No decisive improvement: the L/R asymmetry is within sampling noise at this severe count; the limit is sample size, not the grader.

Reproduce: `python scripts/run_right_foraminal_refine.py`.
