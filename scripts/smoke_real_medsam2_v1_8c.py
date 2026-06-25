#!/usr/bin/env python3
"""Real MedSAM2 smoke test (v1.8c Phase 3) — proves real MedSAM2 ran (module + checkpoint)."""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_load_tokens_v1_8b import ensure_auth  # noqa: E402

from spinescoutx.segmentation.medsam2_runner import MedSAM2, available  # noqa: E402

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
OUT = ROOT / "outputs/real/v1_8c_real_medsam2_smoke.json"


def main() -> int:
    ensure_auth()
    if not available():
        raise SystemExit("real MedSAM2 unavailable (sam2/visionservex/checkpoint missing)")
    import torch

    m = MedSAM2()
    info = m.info()
    proof = {
        "model_module": type(m.runtime._model).__module__,
        "predictor_module": type(m.runtime._predictor).__module__,
        "runtime_type": info["runtime_type"],
        "checkpoint": Path(info["checkpoint_path"]).name,
        "config": info["config_path"],
        "is_real_medsam2": type(m.runtime._model).__module__.startswith("sam2."),
        "cases": [],
    }
    # synthetic 2D + a real RSNA crop
    syn = (np.random.rand(3, 224, 224)).astype("float32")
    crops = glob.glob(str(ROOT / "data/cache/rsna_auto_foraminal/crops/*.npy"))[:1]
    for name, arr in [("synthetic", syn)] + [("rsna_crop", np.load(crops[0]).astype("float32"))]:
        t0 = time.perf_counter()
        mask, score = m.segment(arr)
        proof["cases"].append(
            {
                "case": name,
                "mask_shape": list(mask.shape),
                "area_frac": round(float(mask.mean()), 4),
                "score": round(score, 3),
                "infer_ms": round((time.perf_counter() - t0) * 1000, 1),
                "gpu_mb": round(torch.cuda.memory_allocated() / 1e6, 1)
                if torch.cuda.is_available()
                else 0,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))
    print(f"\nREAL MedSAM2: {proof['is_real_medsam2']} (module {proof['model_module']})")
    return 0 if proof["is_real_medsam2"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
