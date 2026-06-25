"""Real MedSAM2 adapter (v1.8c) — thin wrapper over VisionServeX's MedSAM2 runtime.

Uses ``visionservex.medical.medsam2_runtime`` (official ``sam2`` build + the MedSAM2 checkpoint) so
SpineScoutX runs **real MedSAM2**, not the transformers SAM2.1 fallback. The wrapper is importable
without VisionServeX/sam2 present (``available()`` reports capability) so CI stays mockable and the
test suite never needs the weights. Research-only. Not diagnostic — segmentation only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
DEFAULT_CKPT = ROOT / "data/models/medsam2/MedSAM2_latest.pt"


def available() -> bool:
    """True iff the real MedSAM2 stack (VisionServeX runtime + sam2 + checkpoint) is usable."""
    try:
        import importlib.util as u

        return (
            u.find_spec("visionservex.medical.medsam2_runtime") is not None
            and u.find_spec("sam2") is not None
            and DEFAULT_CKPT.is_file()
        )
    except Exception:  # noqa: BLE001
        return False


class MedSAM2:
    """Loaded real-MedSAM2 model; ``segment(img_chw_or_hwc, box)`` -> (mask HxW bool, score)."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CKPT, device: str | None = None):
        import torch
        from visionservex.medical.medsam2_runtime import load_medsam2_runtime

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime = load_medsam2_runtime(str(checkpoint), device=device)
        self._segment_2d = __import__(
            "visionservex.medical.medsam2_runtime", fromlist=["segment_2d"]
        ).segment_2d

    def info(self) -> dict:
        return self.runtime.info()

    def segment(self, img, box=(70, 70, 154, 154)) -> tuple[np.ndarray, float]:
        """Segment a 2.5D crop ``(3,H,W)`` (center channel used) or an HxWx3 image with a box."""
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:  # (3,H,W) crop -> HxWx3 from center channel
            chan = np.clip(arr[1], 0.0, 1.0)
            hwc = np.stack([(chan * 255).astype("uint8")] * 3, axis=-1)
        else:
            hwc = arr.astype("uint8") if arr.max() > 1.5 else (arr * 255).astype("uint8")
        res = self._segment_2d(self.runtime, hwc, boxes=[list(box)])
        if not res.segments:
            return np.zeros(hwc.shape[:2], bool), 0.0
        seg = res.segments[0]
        mask = np.asarray(seg.mask).astype(bool)
        score = float(getattr(seg, "score", getattr(seg, "confidence", 0.0)) or 0.0)
        return mask, score
