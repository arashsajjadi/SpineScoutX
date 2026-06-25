#!/usr/bin/env python3
"""Download extra spine datasets via Kaggle (v1.8b Phase 3) into gitignored data/external/.

Currently: AxonData Foraminal Stenosis MRI Dataset. Uses the local Kaggle token (never printed).
Nothing is committed. Research-only. Not diagnostic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_load_tokens_v1_8b import ensure_auth  # noqa: E402

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
DATASETS = {"axondata": "axondata/foraminal-stenosis-mri-dataset"}


def main() -> int:
    ensure_auth()
    for name, slug in DATASETS.items():
        dest = ROOT / "data/external" / name
        dest.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
            check=False,
        ).returncode
        print(f"[{name}] {slug} -> {'OK' if rc == 0 else f'FAILED rc={rc}'} ({dest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
