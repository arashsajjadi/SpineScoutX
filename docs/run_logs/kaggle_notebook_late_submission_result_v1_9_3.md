# Kaggle Offline Notebook Late Submission Result — SpineScoutX v1.9.3

> Research-only · non-commercial · not diagnostic · not clinically validated.

## Summary

v1.9.3 fixed the internet-access blocker from v1.9.2 and produced the first **accepted**
Kaggle submission for SpineScoutX. The submission is currently PENDING scoring in Kaggle's
queue.

## What v1.9.2 got wrong

v1.9.2 (kernel v7) had `enable_internet: true` in `kernel-metadata.json`. While the kernel ran
successfully and produced a valid `submission.csv`, Kaggle's competition UI rejected the submission
with:

```
Cannot submit: Your Notebook cannot use internet access in this competition.
Please disable internet in the Notebook editor and save a new version.
```

This revealed that the competition's submission requirement is not just "code kernel" but
specifically "code kernel with internet DISABLED." This rules out v1.9.1 (CSV-only, HTTP 400)
and v1.9.2 (kernel but internet-enabled, rejected by UI).

## Root cause of internet dependency (v7)

All three graders (`canal`, `left_foraminal`, `left_subarticular`) had `pretrained: true`
in their `config.json`. When `collect_probs` calls `build_backbone("convnext_tiny", ..., pretrained=True)`,
timm downloads ImageNet-pretrained `convnext_tiny` weights from HuggingFace Hub — even though
`best.pt` immediately overwrites them with fine-tuned weights. This download is unnecessary and
fails with internet disabled.

Evidence: v7 log showed:
```
Warning: You are sending unauthenticated requests to the HF Hub.
```

v8 log shows 0 HuggingFace warnings.

## Fixes in v1.9.3 (kernel v8)

### 1. Disable internet in kernel metadata

```json
"enable_internet": false
```

### 2. Set HF offline env vars (before any ML imports)

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TIMM_FUSED_ATTN"] = "0"  # avoid CUDA-only fused ops on CPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### 3. Monkey-patch `build_backbone` to force `pretrained=False`

```python
import spinescoutx.models.image_classifier as _ic
_orig_build_backbone = _ic.build_backbone
def _offline_build_backbone(name: str, in_chans: int, pretrained: bool):
    return _orig_build_backbone(name, in_chans, False)
_ic.build_backbone = _offline_build_backbone
```

`pretrained=False` prevents timm from downloading ImageNet weights. The fine-tuned weights
from `best.pt` are loaded right after, producing **identical inference results** — confirmed by
identical probabilities in v7 and v8 outputs.

## Kernel v8 execution log

**Kernel ref:** `arashsajjadi/spinescoutx-v1-9-late-submission` version 8  
**Status:** `KernelWorkerStatus.COMPLETE`  
**Internet:** disabled (`enable_internet: false`)  
**Device:** CPU (Tesla P100-PCIE-16GB CUDA cap 6.0 < PyTorch 2.x required ≥7.0)  
**Wall-clock:** ~44 s

| t (s) | Event |
|---|---|
| 2.3 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` confirmed |
| 2.3 | Wheel extracted, spinescoutx loaded |
| 7.8 | `build_backbone patched: pretrained=False forced (offline mode)` |
| 7.8 | All 6 model paths OK |
| 7.9 | GPU P100 CUDA 6.0 < 7.0 → CPU fallback |
| 7.9 | All imports OK |
| 25.1 | Canal: 5 levels |
| 31.8 | Foraminal: L=5, R=5 |
| 36.4 | Subarticular: L=5, R=5 |
| 36.4 | 25 rows output to `/kaggle/working/submission.csv` |
| 36.4 | Validation PASSED (prob sums = 1.000000, std = 3.03e-08) |

**HuggingFace warnings: 0** (v7 had 1 — the `timm` pretrained-weights download that needed internet)

## Submission attempt result

```
$ kaggle competitions submit \
    -c rsna-2024-lumbar-spine-degenerative-classification \
    -k arashsajjadi/spinescoutx-v1-9-late-submission \
    -v 8 \
    -f submission.csv \
    -m "SpineScoutX v1.9 offline notebook late submission — research-only, no leaderboard tuning"

(no error — submission accepted)

$ kaggle competitions submissions -c rsna-2024-lumbar-spine-degenerative-classification

ref: 54064897
file: submission.csv
date: 2026-06-26 03:27:13
description: SpineScoutX v1.9 offline notebook late submission — research-only, no leaderboard tuning
status: SubmissionStatus.PENDING
publicScore: —
privateScore: —
```

**Result: ACCEPTED by Kaggle API (no HTTP 400). Status: PENDING scoring.**

## Submission CSV validation (v8)

- Rows: 25 / expected 25 ✅
- Columns: `row_id, normal_mild, moderate, severe` ✅
- NaN count: 0 ✅
- Duplicate row_ids: 0 ✅
- Prob sum mean = 1.00000000, max deviation = 7.45e-08 ✅
- Identical to v7 output (offline patch does not change model outputs) ✅

## Score (post-merge update area)

_Score pending as of merge. Kaggle's scoring queue for closed competitions may take minutes
to hours. Submission ref: 54064897. Check:_

```bash
kaggle competitions submissions -c rsna-2024-lumbar-spine-degenerative-classification
```

_Public leaderboard context (for comparison when score arrives):_
- Competition top: 0.332 (weighted log loss, lower = better)
- 1875 teams competed
- SpineScoutX estimated ~40–60th percentile based on internal metrics (not calibrated for log loss)
- One submission only — no leaderboard tuning

## Version comparison

| Version | Method | Internet | Kernel COMPLETE | submission.csv valid | API result |
|---|---|---|---|---|---|
| v1.9.1 | CSV-only | N/A | N/A | Yes (25 rows) | HTTP 400 |
| v1.9.2 (v7) | Code kernel | enabled | Yes | Yes (25 rows) | Blocked by UI (internet must be disabled) |
| v1.9.3 (v8) | Code kernel | **disabled** | Yes | Yes (25 rows) | **ACCEPTED — PENDING** |

## Technical details of offline fix

The `timm` library calls HuggingFace Hub when building a model with `pretrained=True`, even
if the pretrained weights are immediately overwritten by a checkpoint load. This is because
timm's weight-download step happens before `load_state_dict`. The fix is not to disable timm
but to call `timm.create_model(pretrained=False)` which builds the architecture from locally-
available timm package code only, with no network access. We then load `best.pt` as normal.

This is semantically correct because:
1. `pretrained=True` downloads ImageNet weights → architecture + ImageNet features
2. `load_state_dict(best.pt)` overwrites ALL weights → architecture + fine-tuned features
3. `pretrained=False` skips step 1 → architecture only (random init)
4. `load_state_dict(best.pt)` overwrites ALL weights → architecture + fine-tuned features

Steps 2 and 4 produce identical models. Confirmed by bit-exact matching of v7 and v8 outputs.
