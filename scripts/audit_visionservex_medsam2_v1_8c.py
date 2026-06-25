#!/usr/bin/env python3
"""Audit VisionServeX + MedSAM2 capability (v1.8c Phase 1)."""

from __future__ import annotations

import importlib.util as u
import json


def main() -> int:
    caps = {
        "visionservex_importable": u.find_spec("visionservex") is not None,
        "visionservex_medsam2_runtime": u.find_spec("visionservex.medical.medsam2_runtime")
        is not None,
        "visionservex_sam2_runtime_transformers_fallback": u.find_spec("visionservex.sam2_runtime")
        is not None,
        "sam2_package": u.find_spec("sam2") is not None,
        "decision": "use VisionServeX medsam2_runtime (real sam2 build) for MedSAM2; "
        "SAM2.1/SAM3 = comparators only",
    }
    print(json.dumps(caps, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
