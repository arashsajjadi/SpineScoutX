"""SAM2.1 / MedSAM2 foundation-segmentation runner for RSNA crops (v1.8b).

Loads a transformers SAM2 model (``facebook/sam2.1-*`` or a MedSAM2 state-dict graft) and segments
2.5D crops with a box prompt, returning the highest-IoU candidate mask per crop. Batched for speed.
Masks are unvalidated foundation-model proxies on RSNA (no GT masks) — quality is checked downstream
in the v1.8b segmentation QC. Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_sam2(model_dir: str | Path, device: str | None = None):
    """Return (model, processor) for a transformers SAM2 checkpoint directory."""
    import transformers as tf

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = tf.Sam2Processor.from_pretrained(str(model_dir))
    model = tf.Sam2Model.from_pretrained(str(model_dir)).to(device).eval()
    return model, processor, device


def _to_rgb_uint8(chan: np.ndarray) -> np.ndarray:
    a = np.clip(chan, 0.0, 1.0)
    return np.stack([(a * 255).astype("uint8")] * 3, axis=-1)


@torch.no_grad()
def segment_center_channel(model, processor, device, crops, box=(70, 70, 154, 154)):
    """Segment a batch of 2.5D crops (each (3,H,W) in [0,1]) with one center box prompt.

    Returns a list of dicts: ``{mask (H,W) bool, iou (float)}`` — the best-IoU candidate per crop.
    """
    imgs = [_to_rgb_uint8(c[1]) for c in crops]  # center channel
    boxes = [[list(box)] for _ in imgs]
    inputs = processor(images=imgs, input_boxes=boxes, return_tensors="pt").to(device)
    out = model(**inputs)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])
    ious = out.iou_scores.cpu().numpy()  # (B, 1, 3)
    results = []
    for i, m in enumerate(masks):
        m = np.asarray(m)  # (1, 3, H, W) or (3, H, W)
        m = m[0] if m.ndim == 4 else m
        sc = ious[i].reshape(-1)
        best = int(np.argmax(sc))
        results.append({"mask": m[best].astype(bool), "iou": float(sc[best])})
    return results
