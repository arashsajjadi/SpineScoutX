# Model zoo — SpineScoutX weight registry

> **Research-only. Non-commercial. Not diagnostic. Not clinically validated.**
> Weights are published as GitHub Release assets — NOT committed to Git history (each
> grader is ~107 MiB; the full bundle is ~495 MiB compressed).

## Best raw model (v1.9)

**Best raw model = v1.0 deployed reference graders.** None of v1.1–v1.8c improved raw
argmax severe recall. The v1.7 triage config is a separate safety/review layer.

Locked-test 5-route macro severe recall: **0.752**

## GitHub Release — `v1.9.0-research-story-best-model`

```bash
# Download via GitHub CLI
gh release download v1.9.0-research-story-best-model \
  --repo arashsajjadi/SpineScoutX \
  --pattern "*.tar.gz" --pattern "checksums.txt"
```

Or download manually from:
`https://github.com/arashsajjadi/SpineScoutX/releases/tag/v1.9.0-research-story-best-model`

## Release assets

| Asset | Contents | Size |
|---|---|---|
| `spinescoutx-best-raw-v1.9.tar.gz` | 5 graders + localizer `best.pt` + configs + metrics | ~495 MiB |
| `spinescoutx-triage-config-v1.9.tar.gz` | v1.7 triage output JSON + docs | < 1 MiB |
| `checksums.txt` | SHA-256 for both archives | — |

SHA-256 checksums: [`docs/assets/v1_9/checksums.txt`](assets/v1_9/checksums.txt)

## Archive contents

### `spinescoutx-best-raw-v1.9.tar.gz`

```
spinescoutx-best-raw-v1.9/
  metrics.json                 # per-route locked-test severe recall
  model_card.md
  graders/
    canal/          best.pt  config.json  metrics.json
    left_foraminal/ best.pt  config.json  metrics.json
    right_foraminal/best.pt  config.json  metrics.json   # same run as left_foraminal
    left_subarticular/ best.pt  config.json  metrics.json
    right_subarticular/best.pt config.json  metrics.json  # same run as left
    localizer/      best.pt  config.json  metrics.json
```

### `spinescoutx-triage-config-v1.9.tar.gz`

```
spinescoutx-triage-config-v1.9/
  triage_summary.json    # effective recall at each review budget
  severe_fn_triage_v1_7.json
  v1_7_final_accuracy_results.md
  v1_7_failure_autopsy.md
```

## How to load

```python
import torch, json
from pathlib import Path

# Load a grader
ckpt = torch.load("graders/left_foraminal/best.pt", map_location="cpu")
cfg  = json.loads(Path("graders/left_foraminal/config.json").read_text())
# model = _build_model(cfg_obj); model.load_state_dict(ckpt["state_dict"])
# See src/spinescoutx/training/train_classifier.py:_build_model
```

## Per-route metrics

| Grader run | Finding | Locked-test severe recall |
|---|---|---|
| `v1_canal_auto_robust` | Spinal canal stenosis | 0.830 |
| `v1_foraminal_oracle_ctrl` | Left neural foraminal narrowing | 0.788 |
| `v1_foraminal_oracle_ctrl` | Right neural foraminal narrowing | 0.660 |
| `v1_subarticular_auto_robust` | Left subarticular stenosis | 0.746 |
| `v1_subarticular_auto_robust` | Right subarticular stenosis | 0.737 |

## Important notes

- Weights are **not** in Git history (> 50 MiB; GitHub Release asset only).
- Right-foraminal and left-foraminal share one trained run (`v1_foraminal_oracle_ctrl`);
  the route is distinguished at inference by the side field in the manifest.
- Same pattern for subarticular L/R (`v1_subarticular_auto_robust`).
- Backbone: ConvNeXt-Tiny (ImageNet pretrained); input: 2.5D crop 224² (prev/center/next
  sagittal or axial slice).
- These weights are non-commercial. Do not use for diagnosis or clinical decision-making.
