"""Verify v1.9 release asset checksums (Phase 7).

Checks that the packaged tarballs match the committed checksums.txt. Run before
uploading to GitHub Release and after downloading to verify integrity.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
CS_FILE = ROOT / "docs/assets/v1_9/checksums.txt"
DIST = ROOT / "outputs/real/v1_9_packages"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not CS_FILE.exists():
        print(f"Checksums file not found: {CS_FILE}", file=sys.stderr)
        return 1

    expected: dict[str, str] = {}
    for line in CS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            expected[parts[1]] = parts[0]

    all_ok = True
    print("=== SpineScoutX v1.9 release asset checksum verification ===")
    for fname, exp_sha in expected.items():
        p = DIST / fname
        if not p.exists():
            print(f"  MISSING  {fname}")
            all_ok = False
            continue
        actual = _sha256(p)
        if actual == exp_sha:
            mb = p.stat().st_size / 1_048_576
            print(f"  ✅ OK     {fname}  ({mb:.1f} MiB)")
        else:
            print(f"  ❌ MISMATCH  {fname}")
            print(f"     expected: {exp_sha}")
            print(f"     actual:   {actual}")
            all_ok = False

    print(f"\nVerification {'PASSED' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
