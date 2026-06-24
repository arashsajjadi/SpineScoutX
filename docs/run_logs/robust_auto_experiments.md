# Robust auto-inference training (Phase 3/4) — closing the oracle→auto gap

> Research-only. Not diagnostic. Not clinically validated. Not for medical
> decision-making. Outputs are non-diagnostic severity estimates. All numbers are
> measured on the held-out canal val set; run artifacts are gitignored / regenerable.

## Question

Phase 1 proved the oracle→auto severe-recall collapse is **in-plane (crop-centre)
driven**. Can making training match inference recover it?

## Setup

E0 architecture (ConvNeXt-Tiny, 2.5D), canal-stenosis only. Every variant is **selected
and reported on the AUTO distribution** (the real-inference C4 val set: localizer xy +
geometric-mid slice; 1955 nodes, 87 severe), with cluster-bootstrap (by study) 95% CIs.
Model selection monitor = `wll + 0.20·severe_fnr` on auto val.

| variant | training data |
|---|---|
| `r_oracle_ctrl` | GT-centred canal crops, **no jitter** (control) |
| `r_jitter_level` | GT crops + level-aware Gaussian crop-centre jitter (+heavy-tail) |
| `r_jitter_empirical` | GT crops + jitter sampled from measured localizer residuals |
| `r_auto_train` | **auto-localized crops** (localizer xy + mid slice; no GT coords) |
| `r_mixed` | oracle + auto crops |
| `r_consistency` | level-aware jitter + two-view symmetric-KL consistency loss |

## Results (auto C4 val, n=1955, severe=87)

| variant | auto severe recall [95% CI] | auto wll | oracle severe recall |
|---|---|---|---|
| `r_oracle_ctrl` (control) | 0.667 [0.570, 0.754] | 0.521 | 0.736 |
| `r_jitter_level` | 0.736 [0.634, 0.833] | 0.486 | 0.793 |
| `r_jitter_empirical` | 0.678 [0.578, 0.775] | 0.463 | 0.805 |
| **`r_auto_train`** | **0.793 [0.696, 0.881]** | **0.454** | **0.874** |
| `r_mixed` | 0.632 [0.524, 0.736] | 0.396 | 0.690 |
| `r_consistency` | 0.655 [0.548, 0.755] | 0.444 | 0.747 |

Reference anchors (established all-conditions E0 on canal): auto 0.644 → oracle 0.828.

### Paired comparison (same C4 nodes — the rigorous test)

vs the **deployed all-conditions E0**:

| variant | Δ severe recall [95% CI] | decisive? | McNemar (catch→miss / miss→catch, p) | Δ wll [95% CI] |
|---|---|---|---|---|
| **`r_auto_train`** | **+0.149 [+0.050, +0.243]** | **YES** | 17 / 4, **p=0.007** | **−0.100 [−0.166, −0.040]** |
| `r_jitter_level` | +0.092 [−0.010, +0.193] | no | 13 / 5, p=0.10 | −0.068 [−0.135, −0.003] (decisive) |
| `r_jitter_empirical` | +0.034 [−0.051, +0.121] | no | 9 / 6, p=0.61 | — |
| `r_consistency` | +0.011 [−0.082, +0.106] | no | 8 / 7, p=1.0 | — |
| `r_mixed` | −0.011 [−0.103, +0.084] | no | 9 / 10, p=1.0 | −0.158 [−0.218, −0.103] (decisive) |

vs the **canal-only control** (`r_oracle_ctrl`): `r_auto_train` Δ severe recall
**+0.126 [+0.048, +0.211] (decisive)**; `r_jitter_level` +0.069 [−0.025, +0.163] (not decisive).

## Conclusion

**Yes — localizer-aware robust training recovers the severe-recall loss, and the
decisive lever is training on the real auto-localized distribution.**

- **`r_auto_train` is the decisive winner.** Auto severe recall **0.793**, a paired
  **+0.149 [+0.050, +0.243]** over the deployed E0 (McNemar p=0.007: it catches 17
  severe cases E0 missed and loses only 4) and **+0.126 [+0.048, +0.211]** over the
  canal control. It **recovers ~81%** of the established 0.644→0.828 gap, **also**
  decisively improves auto weighted log loss (−0.100), and is even the best on the
  oracle distribution (0.874) — robustness training doubles as a regularizer. This
  clears the project's success bar (≥+0.06 auto severe recall, CI excluding 0).
- **Honest nuance — not every robust recipe works.** Level-aware crop jitter
  decisively improves calibration/log-loss and *trends* toward better severe recall
  (+0.092, p=0.10) but is **not decisive on severe recall alone**. Empirical jitter,
  oracle/auto mixing, and the consistency loss did **not** decisively raise severe
  recall under severe-aware selection (mixing gives the best log-loss but trades away
  severe recall). Exposing the grader to the localizer's *actual* error structure
  (auto-train) beats approximating it with synthetic jitter.
- **Why.** The localizer error is dominated by the superior-inferior axis (σ_y up to
  ~34 px at L1/L2 vs σ_x ~8 px) — occasional vertical level-confusion, heavy-tailed
  (p99 ~103 px) — see `localizer_error_profile.md`. Auto-train matches that exact
  distribution; parametric/empirical jitter only approximates it.
- **Overfitting.** The ConvNeXt grader overfits canal in ~6–8 epochs; the oracle
  control's auto severe recall peaks (~0.71) then collapses (~0.41) as it memorizes
  perfectly-centred crops. Training on auto crops regularizes against this.

## Caveats

- Canal-only, single sagittal view; severe n=87 (CIs are wide — hence paired tests).
- The canal-only control (auto-selected) has a smaller raw gap (0.736→0.667) than the
  all-conditions E0 (0.828→0.644); the headline claim uses the **paired** comparison on
  identical nodes against both baselines, which is robust to that.
- Not yet extended to foraminal/subarticular (needs an axial localizer) or to a fresh
  held-out test split; reported as canal val.

Artifacts: `outputs/real/robust_auto_experiments.json` (incl. `paired_analysis`),
`outputs/real/figures/robust_auto_frontier.png`; runs `runs/r_*` (gitignored).
Reproduce: `python scripts/run_robust_experiments.py --epochs 18 --n-boot 2000`.
