#!/usr/bin/env python3
"""Audit whether MedSAM2/SAM/SPINEPS/nnU-Net are used anywhere in the repo (v1.8b Phase 1)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
TERMS = ["medsam", "sam2", "sam3", "segment.?anything", "spineps", "nnunet", "nnu-net"]


def main() -> int:
    pat = re.compile("|".join(TERMS), re.IGNORECASE)
    hits = {}
    for p in (ROOT / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if pat.search(line) and "v1_8b" not in str(p):
                hits.setdefault(str(p.relative_to(ROOT)), []).append(i)
    print("foundation-seg references (excluding v1.8b):", hits or "NONE")
    rc = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "runs/e4_segmentation_spider_real"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    print("existing SPIDER segmenter tracked-files:", rc.stdout.strip() or "(weights gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
