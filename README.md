# SpineScoutX

**A research-only lumbar-MRI pipeline that reads a study with no ground-truth coordinates and
outputs a non-diagnostic, severe-safety-aware *research finding graph* — then shows that output
next to the held-out reference so you can see where it is right, uncertain, or wrong.**

> ⚠️ **Research-only · not diagnostic · not clinically validated · not for medical
> decision-making · no treatment recommendations.** Severity values are *research findings /
> severity estimates*, never a diagnosis. Public non-commercial data (RSNA LumbarDISC; SPIDER
> CC BY 4.0); no data redistributed. **5/5 findings have a real auto-inference route.**

## See what it actually does — one real locked-test case
![real case viewer](docs/assets/readme/hero_case_viewer.png)

Left = the **model prediction** (severity estimate + P(severe)); right = the **held-out
reference label** (shown for transparency only — *never* a model input); far right = the
**code-derived correctness** (✓/✗). Provenance is `auto` — **no ground-truth coordinates at
inference**. Full sectioned cards (evidence route, safety, failure note) are in the
**[case gallery](docs/gallery.md)**; here is how to read one:

![prediction vs reference](docs/assets/readme/prediction_vs_reference_card.png)

## Real evidence viewer (v1.3) — what the model *saw* → predicted → reference
![real evidence case](docs/assets/real_cases/case_right_foraminal_hard.png)

**Panel A** shows the *real derived evidence signals* for this locked-test case (auto crop
centre, slice index, mean intensity — pixel-free, no DICOMs committed); **Panel B** the model
prediction next to the **held-out reference** with code-derived correctness; **Panel C** the
evidence-v3 safety/review + side-aware similar cases. To respect the data licence, committed
cards are **pixel-free**; the full real-pixel viewer runs locally
([asset policy](docs/run_logs/real_evidence_asset_policy.md)). More cases (correct, uncertain,
wrong) in the **[gallery](docs/gallery.md)**.

## Coverage & performance (locked-test **auto** = real inference)
| finding | view route | severe recall [95% CI] |
|---|---|---|
| spinal canal stenosis | sagittal-T2 | **0.830** [0.725, 0.929] |
| left neural foraminal narrowing | sagittal-T1 | **0.788** [0.673, 0.892] |
| right neural foraminal narrowing | sagittal-T1 | **0.660** [0.524, 0.788] |
| left subarticular stenosis | axial-T2 | **0.746** [0.674, 0.815] |
| right subarticular stenosis | axial-T2 | **0.737** [0.667, 0.807] |

Every number is on a **never-tuned patient-level locked test**, auto = real inference, with
cluster-bootstrap CIs. Full results: [`docs/results.md`](docs/results.md).

> **v1.4 (raw-accuracy audit — honest negative):** a ruthless accuracy audit found **no bug
> depressing severe recall** (baselines reproduce exactly; one latent bug fixed with zero metric
> delta), and a direct test showed v1.3's localization gain does **not** transfer to grading.
> **Raw severe recall is unchanged** — the ceiling is grader-capacity/training-data limited, not
> a bug. No accuracy-upgrade tag. See
> [`v1_4_raw_accuracy_conclusion.md`](docs/run_logs/v1_4_raw_accuracy_conclusion.md).

## How it works
![input to output](docs/assets/readme/pipeline_input_to_output.png)

*Core finding:* robust auto-training helps in proportion to the oracle→auto localization gap
(it recovers canal & subarticular; foraminal's clean localizer needs none — grader chosen per
condition).

## Evidence intelligence v3 + capability wins (v1.3)
![evidence intelligence v3](docs/assets/readme/evidence_intelligence_v3_card.png)

Two real, locked-test capability wins this release:
- **Evidence intelligence v3** — a combined severe-FN risk score (confidence + stability +
  retrieval-conflict + near-severe) **improves severe-FN detection AUROC 0.833 → 0.863**
  (most on the weak subarticular routes; conf+stability alone was *below* confidence, so the
  *new* signals do the work). [`evidence_intelligence_v3.md`](docs/run_logs/evidence_intelligence_v3.md).
- **Axial decode v2** — a train-derived positional-prior monotonic decoder (no CNN retrain,
  β dev-selected) **improves axial ±1 slice-hit 0.432 → 0.487** on the locked test (geometry
  baseline 0.275). [`axial_stack_scorer_v2_results.md`](docs/run_logs/axial_stack_scorer_v2_results.md).

Each finding also carries evidence-stability + an instability *type* (crop/slice/axial-candidate/
route) and **side-aware (v2) similar cases** (same-side rate 1.00 vs v1's ~chance). Honest
context: stability alone is largely redundant with confidence — it is the *combination* (v3)
that helps. See also [`evidence_stability_v1.md`](docs/run_logs/evidence_stability_v1.md).

## Safety mode (severe-first, selective review)
![safety mode](docs/assets/readme/safety_mode_explainer.png)

Reaching 90% severe recall has an explicit false-alarm / review cost — reported, never hidden.
Calibration is a documented negative (graders already well-calibrated → raw probabilities kept).
See [`safety_mode_v5.md`](docs/run_logs/safety_mode_v5.md).

## Where it fails (shown, not hidden)
![failures](docs/assets/readme/failure_gallery_preview.png)

Right-foraminal is the weakest route (precisely characterized: 56% of its severe misses are
*confidently normal* — a signal/sample limit). Severe recall is weaker at L5-S1 (0.579) than
L4-L5 (0.868) ([domain-shift audit](docs/run_logs/external_validation_audit.md)); the axial
level scorer is imperfect (±1 slice-hit 0.43); **no external/prospective validation.** Details:
**[trust & limitations](docs/trust_and_limitations.md)**.

## Quickstart
```bash
pip install -e .
spinescoutx doctor --data                          # checks RSNA/SPIDER + deps
# data required (docs/data_setup.md). Regenerate the visible outputs:
python scripts/make_real_case_viewer_pack.py       # wide prediction-vs-reference case cards
python scripts/make_readme_assets_v12.py           # README explainer assets
python scripts/run_evidence_intel_v2.py            # instability typing
python scripts/run_safety_mode_v5.py               # evidence-aware severe-first dashboard
```

## Trust & limitations (read this)
**An academic research prototype, not an FDA/CE-cleared medical product.** No external,
prospective, or reader-study validation. Not diagnostic, not clinical. Details:
[`docs/trust_and_limitations.md`](docs/trust_and_limitations.md), [`docs/safety.md`](docs/safety.md).

## Full docs
- 🖼 [Gallery](docs/gallery.md) · 📊 [Results + CIs](docs/results.md) ·
  🧪 [Technical report](docs/technical_report.md)
- 🪪 [Model card](docs/model_card.md) · [Dataset card](docs/dataset_card.md) ·
  🔒 [Safety policy](docs/safety.md) · 🔁 [Reproducibility](docs/reproducibility.md)
- 🧩 [Finding-graph schema v5](docs/run_logs/report_schema_v5.md) ·
  [Case viewer](docs/run_logs/real_case_viewer.md) ·
  [Output audit](docs/run_logs/output_intelligence_audit.md)

## License
Code: MIT ([`LICENSE`](LICENSE)). **Datasets are NOT covered by MIT** and are not included
(RSNA non-commercial; SPIDER CC BY 4.0). No DICOMs, masks, checkpoints, caches, or runs are
committed.
