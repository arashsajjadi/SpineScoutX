# SpineScoutX

**An anatomy-grounded lumbar MRI finding-graph system for research-only degenerative spine analysis.**

> ⚠️ **Research-only.** SpineScoutX is a research and educational prototype. It is
> **not diagnostic, not clinically validated, and not for medical decision-making.**
> It uses public **non-commercial** research datasets and does not redistribute them.

---

## 1. Project summary

SpineScoutX studies whether **explicit anatomical priors** (vertebra / disc /
spinal-canal masks) improve disc-level lumbar degeneration *finding grading*. It
connects:

- disc-level degenerative **finding grading** (RSNA),
- anatomical **segmentation priors** (SPIDER),
- **anatomical evidence-consistency** scoring (AEC),
- **calibration-aware**, uncertainty-flagged output,
- deterministic **structured finding graphs** and non-diagnostic reports.

It is **not** a demo classifier and **not** a clinical product.

## 2. Research question

> Can explicit anatomical priors improve the reliability, severe-case recall,
> calibration, and visual evidence localization of disc-level lumbar degeneration
> grading?

**Hypothesis.** An anatomy-guided model using vertebra/disc/canal priors should
improve severe-case recall, calibration, and anatomical evidence consistency
versus an image-only crop classifier — *or at minimum* produce more anatomically
meaningful visual evidence even if raw classification metrics do not improve.

## 3. Novelty statement (precise)

Prior work exists. **SpineScoutX does not claim to be the first AI lumbar spine
system** — DeepSPINE and other lumbar stenosis/grading systems predate it. The
contribution is scoped to:

- anatomy-grounded **finding-graph** generation;
- **cross-dataset anatomy transfer** (SPIDER segmentation → RSNA grading);
- **anatomical evidence-consistency** scoring (AEC);
- **counterfactual anatomy ablation** (correct / shuffled / zero / noise);
- **calibration-aware, non-diagnostic structured reporting**;
- minimal, license-aware, reproducible research engineering.

## 4. Datasets

| Dataset | Role | Levels / classes | License |
|---|---|---|---|
| RSNA 2024 Lumbar Spine Degenerative Classification | disc-level grading (primary) | L1/L2…L5/S1; `normal_mild`/`moderate`/`severe`; 5 conditions | **non-commercial research only** |
| SPIDER Lumbar Spine Segmentation | anatomy priors | vertebra / disc / spinal canal | **CC BY 4.0** (attribution) |

Conditions: spinal canal stenosis; left/right neural foraminal narrowing;
left/right subarticular stenosis. See [`docs/data_setup.md`](docs/data_setup.md).

## 5. License / data-use warnings

- Source **code**: MIT (see [`LICENSE`](LICENSE)).
- **Datasets are not covered by MIT** and are **not** included. RSNA = non-commercial
  research; SPIDER = CC BY 4.0. You must obtain them yourself under their terms.
- No DICOMs, NIfTI volumes, masks, checkpoints, or caches are committed.

## 6. Quickstart

```bash
pip install -e .            # core (CPU/GPU). Add extras as needed:
pip install -e ".[dicom,parquet]"   # decode real RSNA DICOMs + parquet manifests

spinescoutx doctor                  # environment check (no data needed)
pytest -q                           # full suite on synthetic fixtures (no data needed)

# With data (see docs/data_setup.md):
spinescoutx prepare-rsna  --rsna-root data/raw/rsna   --out data/cache/rsna
spinescoutx prepare-spider --spider-root data/raw/spider --out data/cache/spider
spinescoutx train-classifier      --config configs/baseline_image_only.yaml
spinescoutx train-segmenter       --config configs/segmentation_spider.yaml
spinescoutx train-anatomy-guided  --config configs/anatomy_guided.yaml
spinescoutx ablate                --config configs/ablation.yaml
spinescoutx evaluate --run runs/e0_baseline_image_only
spinescoutx report   --study-id <ID> --run runs/e1_anatomy_guided
spinescoutx figure   --report outputs/reports/<ID>.json
```

You can run a no-data synthetic smoke of the whole pipeline by setting
`data.synthetic: true` in a config (used by the tests).

## 7. CLI commands

| Command | Purpose |
|---|---|
| `doctor` | environment & optional-dependency check |
| `prepare-rsna` / `prepare-spider` | index + cache a dataset (fails clearly if missing) |
| `train-classifier` (E0) | image-only baseline |
| `train-segmenter` (E4) | SPIDER anatomy segmenter |
| `train-anatomy-guided` (E1) | anatomy-guided classifier |
| `evaluate` | metrics for a finished run |
| `ablate` (E2/E3) | counterfactual anatomy ablations |
| `report` | finding-graph JSON + Markdown for a study |
| `figure` | visual panels from a report |
| `benchmark` | inference latency / memory for a run |

Every command supports `--help` and `--json` (machine-readable log line).

## 8. Pipeline (text diagram)

```
DICOM ingestion
  -> metadata index (series classification / selection)
  -> disc-level localizer matching
  -> disc-level 2.5D crop extraction (prev/center/next slice)
  -> SPIDER segmentation model (anatomy priors: vertebra/disc/canal)
  -> [E0] image-only classifier        \
  -> [E1] anatomy-guided classifier      >  Grad-CAM evidence -> AEC scoring
  -> calibration (ECE, temperature, uncertainty flags)
  -> deterministic finding graph (JSON)
  -> Markdown report + visual panels
```

## 9. Experiments E0–E7

| ID | Experiment |
|---|---|
| **E0** | Image-only baseline classifier |
| **E1** | Anatomy-guided classifier (image + anatomy priors + level/condition embeddings) |
| **E2** | Ablation — anatomy prior **shuffled** from another study |
| **E3** | Ablation — anatomy prior **zeroed** (also `noise`, optional) |
| **E4** | SPIDER anatomy **segmenter** |
| **E5** | **Evidence** (Grad-CAM heatmaps + AEC + leakage + peak-to-localizer) |
| **E6** | **Calibration** (ECE, reliability diagram, temperature scaling) |
| **E7** | **Runtime** (preprocessing / training / inference / memory) |

## 10. Metrics

- **Classification:** RSNA-style weighted log loss, macro-F1, per-condition &
  per-level F1, **severe recall**, severe FNR, severe one-vs-rest AUROC, confusion
  matrix, balanced accuracy, ECE.
- **Segmentation:** Dice & IoU per class, mean Dice, **canal Dice**, inference latency.
- **Evidence:** **AEC**, evidence leakage, peak-to-localizer distance, report
  completeness, uncertainty coverage, failure cases.
- **Runtime:** preprocessing / training / inference time, GPU peak memory, CPU fallback.

## 11. Reproducibility

- Deterministic seeding (`spinescoutx.utils.seed`); patient/study-level splits with
  recorded seed + timestamp; no image-level splitting; leakage check enforced.
- Config-hash + manifest-hash cache invalidation; decoded crops / masks cached.
- Typed YAML configs; best+last checkpoints only; compact metrics logs.

## 12. Limitations

- **Anatomy ≠ pathology.** SPIDER priors are anatomy masks (vertebra/disc/canal),
  **not** pathology or stenosis masks.
- **Approximate regions.** Foraminal and lateral-recess evidence regions are
  approximations (SPIDER has no such labels) and are flagged accordingly.
- **No clinical/external validation.** Public non-commercial research data only.
- Improvements are reported **only** when measured against the E0 baseline.

## 13. Forbidden claims

The following must **never** appear as claims about SpineScoutX (only inside this
section, as the list of things we do *not* claim): "diagnostic"; "clinically
validated"; "medical decision-making"; "commercial medical diagnosis";
"automatically detects all lumbar abnormalities"; "first AI system for lumbar disc
disease"; "approved"; "doctor replacement".

**Allowed public claim:** *"SpineScoutX is a research-only prototype that studies
anatomy-grounded lumbar MRI degenerative finding graphs using public
non-commercial research datasets. It is not diagnostic, not clinically validated,
and not for medical decision-making."*

## 14. Example output schema (finding graph)

```json
{
  "study_id": "12345",
  "research_only": true,
  "not_diagnostic": true,
  "dataset_source": "rsna",
  "model_version": "0.1.0",
  "run_id": "e1_anatomy_guided",
  "findings": [
    {
      "level": "l4_l5",
      "condition": "spinal_canal_stenosis",
      "side": null,
      "grade": "moderate",
      "confidence": 0.71,
      "calibrated_confidence": 0.64,
      "uncertainty_flag": "moderate_confidence",
      "evidence_consistency": 0.58,
      "evidence_region": "spinal_canal",
      "evidence_region_source": "anatomy",
      "evidence_image_path": "outputs/figures/12345_l4_l5_scs.png",
      "crop_path": "data/cache/rsna/crops/12345_l4_l5_scs.npy",
      "notes": ""
    }
  ],
  "limitations": ["...non-commercial research data...", "...not diagnostic...", "..."]
}
```

## 15. Citation / attribution

If you use SpineScoutX, please also cite the underlying datasets:

- **RSNA 2024 Lumbar Spine Degenerative Classification** (Radiological Society of
  North America) — non-commercial research use.
- **SPIDER** Lumbar Spine MRI Segmentation dataset — *van der Graaf et al.*,
  licensed CC BY 4.0 (attribution required).

```bibtex
@software{spinescoutx,
  title  = {SpineScoutX: anatomy-grounded lumbar MRI finding graphs (research-only)},
  year   = {2026},
  note   = {Research prototype. Not diagnostic; not clinically validated.}
}
```

See [`docs/technical_report.md`](docs/technical_report.md) for methods, metrics,
ablations, and **honest failure cases**.
