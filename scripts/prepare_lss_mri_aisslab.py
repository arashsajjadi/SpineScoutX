#!/usr/bin/env python3
"""Acquire the LSS-MRI AISSLab dataset (v1.6, Plan A) — download, verify, extract.

Downloads the public Mendeley archive (DOI 10.17632/rgb77xm3jf, CC BY 4.0, non-commercial
research), verifies its SHA-256, and extracts the foraminal-detection split. If the network is
unavailable, prints exact manual-download instructions (see
``docs/run_logs/lss_manual_download_instructions.md``) and exits non-zero so Plan B can take over.
External data lives under ``data/external/`` (gitignored) and is never committed.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
DEST = ROOT / "data/external/lss_mri_aisslab"
ZIP = DEST / "lss_mri_aisslab_v0.2.zip"
# Mendeley public-files API (no login); resolved from
# https://data.mendeley.com/public-api/datasets/rgb77xm3jf/files?folder_id=root&version=4
DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/rgb77xm3jf/files/"
    "6d9a0116-925d-4111-acb0-1e679f7dfd71/file_downloaded"
)
SHA256 = "592a294f93d575a16bccc2681c793eb1cfc6679fa2746ac50cbc8970f806b4b1"
SIZE_BYTES = 2035587816
MANUAL = (
    "Manual download: visit https://data.mendeley.com/datasets/rgb77xm3jf/4 , download "
    "'LSS MRI AISSLab Dataset_V0.2.zip', place it at "
    "data/external/lss_mri_aisslab/lss_mri_aisslab_v0.2.zip, then re-run this script."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if not (ZIP.exists() and ZIP.stat().st_size == SIZE_BYTES):
        print(f"[lss] downloading {SIZE_BYTES / 1e9:.2f} GB from Mendeley ...", flush=True)
        rc = subprocess.run(
            ["curl", "-sL", "--fail", "--max-time", "1800", "-o", str(ZIP), DOWNLOAD_URL],
            check=False,
        ).returncode
        if rc != 0 or not ZIP.exists():
            print(f"[lss] DOWNLOAD FAILED (curl rc={rc}). {MANUAL}", file=sys.stderr)
            return 2
    print("[lss] verifying SHA-256 ...", flush=True)
    got = _sha256(ZIP)
    if got != SHA256:
        print(f"[lss] CHECKSUM MISMATCH got {got} expected {SHA256}. {MANUAL}", file=sys.stderr)
        return 3
    print("[lss] checksum OK; extracting Foramina_Detection/ ...", flush=True)
    rc = subprocess.run(
        ["unzip", "-o", "-q", str(ZIP), "Foramina_Detection/*", "-d", str(DEST / "extracted")],
        check=False,
    ).returncode
    if rc != 0:
        print(f"[lss] EXTRACT FAILED (unzip rc={rc}).", file=sys.stderr)
        return 4
    n_xml = len(list((DEST / "extracted/Foramina_Detection").rglob("*.xml")))
    print(f"[lss] ready: {n_xml} annotation XMLs. Next: prepare_lss_foraminal_v1_6.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
