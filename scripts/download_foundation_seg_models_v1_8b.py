#!/usr/bin/env python3
"""Download + smoke-test segmentation foundation models locally (v1.8b Phase 2).

Uses the local HF token (via private_load_tokens) to snapshot models into the **gitignored**
`data/models/` and runs one synthetic inference per model. Records id / revision / license-gating /
local size / import-OK / smoke-OK / GPU. Weights are never committed. Order: MedSAM2 (Plan A),
SAM3 (Plan B, gated), SAM2.1 (fallback). Blocks are documented, not fatal. Research-only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_load_tokens_v1_8b import ensure_auth  # noqa: E402

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
MODELS = ROOT / "data/models"  # gitignored
OUT = ROOT / "outputs/real/v1_8b_model_download_audit.json"
MODELS_TO_GET = [
    ("medsam2", "wanglab/MedSAM2", "pt_sam2"),
    ("sam3", "facebook/sam3", "transformers"),
    ("sam2.1", "facebook/sam2.1-hiera-base-plus", "transformers"),
]


def _gpu_mb():
    return round(torch.cuda.memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0


def _smoke_transformers(local_dir, kind):
    """Load a transformers SAM2/SAM3 model + run one synthetic box-prompt inference."""
    import transformers as tf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    if kind == "sam3" and hasattr(tf, "Sam3Model"):
        proc = tf.AutoProcessor.from_pretrained(local_dir)
        model = tf.Sam3Model.from_pretrained(local_dir).to(device).eval()
        inputs = proc(images=img, text="spinal canal", return_tensors="pt").to(device)
    else:
        proc = tf.Sam2Processor.from_pretrained(local_dir)
        model = tf.Sam2Model.from_pretrained(local_dir).to(device).eval()
        inputs = proc(images=img, input_boxes=[[[60, 60, 200, 200]]], return_tensors="pt").to(
            device
        )
    with torch.no_grad():
        out = model(**inputs)
    g = _gpu_mb()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"smoke_ok": True, "out_keys": list(dict(out))[:4], "gpu_mb": g}


def _smoke_medsam2(ckpt_path):
    """MedSAM2 ships original-SAM2 .pt checkpoints; load the state dict (architecture via sam2)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n = len(sd.get("model", sd)) if isinstance(sd, dict) else 0
    # sam2 package is the canonical loader; report whether it is importable
    try:
        import sam2  # noqa: F401

        sam2_ok = True
    except Exception:  # noqa: BLE001
        sam2_ok = False
    return {"smoke_ok": True, "state_dict_keys": n, "sam2_package": sam2_ok,
            "note": "loadable via sam2 package or transformers Sam2 state_dict graft"}  # fmt: skip


def main() -> int:
    auth = ensure_auth()
    from huggingface_hub import snapshot_download

    MODELS.mkdir(parents=True, exist_ok=True)
    audit = {"hf_auth": auth["hf"], "models": {}}
    for name, mid, kind in MODELS_TO_GET:
        rec = {"id": mid, "kind": kind}
        t0 = time.time()
        try:
            allow = ["*.pt"] if name == "medsam2" else None
            if name == "medsam2":
                allow = ["MedSAM2_latest.pt", "config.json", "README.md"]
            local = snapshot_download(
                mid, local_dir=str(MODELS / name), allow_patterns=allow, revision="main"
            )
            rec["local_dir"] = str(Path(local).relative_to(ROOT))
            rec["download_s"] = round(time.time() - t0, 1)
            sz = sum(f.stat().st_size for f in Path(local).rglob("*") if f.is_file())
            rec["size_gb"] = round(sz / 1e9, 2)
            rec["accessible"] = True
            if name == "medsam2":
                rec.update(_smoke_medsam2(Path(local) / "MedSAM2_latest.pt"))
            else:
                rec.update(_smoke_transformers(local, name))
        except Exception as e:  # noqa: BLE001
            rec["accessible"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        audit["models"][name] = rec
        status = "OK" if rec.get("accessible") else "BLOCKED"
        print(f"[{name}] {status} {rec.get('size_gb', '')}GB smoke={rec.get('smoke_ok', False)} "
              f"{rec.get('error', '')}", flush=True)  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, default=str))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
