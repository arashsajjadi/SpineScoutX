# v1.0 baseline snapshot — frozen before the v1.1 intelligence upgrade

> Research-only · not diagnostic · not clinically validated. This freezes the
> v1.0 locked-test state with provenance so every v1.1 change is measured against a
> fixed reference. **No new metric is introduced here** — every number is copied
> from an existing committed run log (cited).

## Provenance
| | |
|---|---|
| Baseline tag | `v1.0.0-auto-robust-five-finding-research` = commit `13c346d` |
| Authoritative branch at audit | `feature/spinescoutx-model-output-showcase` (`08b7d99`) |
| v1.1 work branch | `feature/spinescoutx-v1.1-intelligence` (off `08b7d99`) |
| `origin/main` at audit | `150b9d1` (v0.3-era; 51 commits behind; merge pending) |
| Test split | `splits_v1` locked **test** (296 studies; seed 20260623); dev=selection only |
| Device | RTX 5080 16GB, torch 2.11.0+cu130 |

## Per-condition locked-test AUTO metrics (frozen v1.0)
Source: `docs/run_logs/safety_mode_v4.md` (table), `subarticular_auto_results.md`,
`foraminal_auto_results.md`, `canal_locked_test.md`, model card §Performance.

| condition | deployable grader | n / n_severe | severe recall (argmax) [95% CI] | recall@FAR≤10% [CI] | FAR@90% sevR | review→severe-FN capture |
|---|---|---|---|---|---|---|
| spinal_canal_stenosis | auto-trained robust | 1480 / 53 | **0.830** [0.725, 0.929] | 0.943 [0.833, 1.000] | 0.085 | 13%→22% |
| left_neural_foraminal_narrowing | oracle-trained | 1480 / 52 | **0.788** [0.673, 0.892] | 0.885 [0.795, 0.962] | 0.123 | 16%→45% |
| right_neural_foraminal_narrowing | oracle-trained | 1470 / 53 | **0.660** [0.524, 0.788] | 0.811 [0.681, 0.898] | 0.255 | 15%→28% |
| left_subarticular_stenosis | auto-trained robust | 1434 / 138 | **0.746** [0.674, 0.815] | 0.732 [0.601, 0.808] | 0.252 | 30%→46% |
| right_subarticular_stenosis | auto-trained robust | 1434 / 137 | **0.737** [0.667, 0.807] | 0.708 [0.602, 0.779] | 0.258 | 29%→36% |

**Coverage: 5/5 findings with a real auto route.** CIs are cluster-bootstrap by study.

## Weakest condition / top bottleneck (frozen)
- **Weakest:** right neural foraminal narrowing — argmax severe recall 0.660, FAR@90% 0.255
  (worst), trails left foraminal (0.788). The L/R CIs overlap (n_severe ≈ 52–53);
  a prior right-specialist attempt was non-decisive → the limit is **sample size**.
- **Top technical bottleneck:** axial level localization. The coordinate-supervised
  axial level scorer reaches ±1 slice-hit **0.43** (vs geometry 0.275). Subarticular
  auto works because the **robust grader tolerates** level noise, not because leveling
  is solved. This is the headline target for a future scorer (v2).
- **Trust ceiling:** no external / prospective / scanner-diversity / reader-study
  validation; single dataset (RSNA LumbarDISC).

## Current output artifacts (frozen)
- Schema `finding_graph_v4` (`src/spinescoutx/reporting/finding_graph_schema.py`).
- Showcase cards in `docs/assets/showcase/` (8 cards + schema visual; structured
  finding graphs, **no DICOM pixels**).
- Safety Mode v4 dashboard + router (`scripts/run_safety_mode_v4.py`).
- Output-intelligence CI audit (`docs/run_logs/output_intelligence_audit.md`).

## v1.1 targets (what must beat / extend this baseline)
1. **New intelligence:** evidence-stability score, evaluated *against errors* (not just added).
2. **Safety Mode v5:** evidence-/route-aware review, condition-specific calibration.
3. **Weakest route:** materially improve right-foraminal **or** rigorously re-diagnose.
4. **Localization:** axial stack-sequence scorer v2 — improve ±1 slice-hit **or** document a precise negative.
5. **Generalization:** internal domain-shift stress test (external validation only if legally feasible).
6. **Release:** merge all correct work into `main`; tags reachable from `main`.
