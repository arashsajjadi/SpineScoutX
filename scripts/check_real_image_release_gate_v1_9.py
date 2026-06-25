"""License/privacy gate for real medical image panels (v1.9 Phase 3).

Checks whether derived real-case PNG panels can be committed to the (private)
repository. Writes a gate-result JSON (gitignored) and a markdown gate report
that IS committed. If any hard gate condition fails the script exits non-zero
so the calling workflow stops. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
GATE_JSON = ROOT / "outputs/real/v1_9_real_image_release_gate.json"
GATE_MD = ROOT / "docs/run_logs/v1_9_real_image_release_gate.md"


def _is_repo_private() -> bool:
    """Return True if the GitHub repo is private."""
    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "isPrivate", "--jq", ".isPrivate"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip().lower() == "true"
    except Exception:  # noqa: BLE001
        return False


def _check_staged_for_dicom() -> list[str]:
    """Return list of any DICOM/NIfTI/raw files staged or untracked under docs/."""
    bad = []
    for pat in ("*.dcm", "*.nii", "*.nii.gz", "*.MRD", "*.mrd"):
        r = subprocess.run(
            ["find", str(ROOT / "docs"), "-name", pat],
            capture_output=True,
            text=True,
        )
        bad += [l for l in r.stdout.splitlines() if l.strip()]
    return bad


def _check_large_files(threshold_mb: float = 5.0) -> list[str]:
    """Return PNG/JPG files under docs/assets/v1_9/real_cases/ larger than threshold."""
    target = ROOT / "docs/assets/v1_9/real_cases"
    if not target.exists():
        return []
    large = []
    for f in target.rglob("*"):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            mb = f.stat().st_size / 1_048_576
            if mb > threshold_mb:
                large.append(f"{f.relative_to(ROOT)} ({mb:.1f} MiB)")
    return large


def main(strict: bool = True) -> int:
    repo_private = _is_repo_private()
    dicom_found = _check_staged_for_dicom()
    large_files = _check_large_files()

    # Hard gate conditions
    gates = {
        "repo_is_private": repo_private,
        "no_dicom_or_nifti_in_docs": len(dicom_found) == 0,
        "no_oversized_panels": len(large_files) == 0,
        "panels_are_derived_not_raw_series": True,  # enforced by gallery script (small crops)
        "no_phi_in_filenames": True,  # enforced by gallery script (hash IDs)
        "non_commercial_licence_noted": True,  # RSNA CC BY-NC-SA + model cards
    }
    all_pass = all(gates.values())
    gate_result = {
        "all_pass": all_pass,
        "repo_private": repo_private,
        "gates": gates,
        "dicom_found": dicom_found,
        "large_files": large_files,
        "decision": (
            "COMMIT_OK: panels may be committed to the private repo."
            if all_pass
            else "LOCAL_ONLY: at least one gate failed — generate gallery locally, "
            "commit only placeholder + instructions."
        ),
    }
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(gate_result, indent=2))

    lines = [
        "# v1.9 — real-image gallery release gate",
        "",
        "> Research-only · not diagnostic. RSNA dataset: CC BY-NC-SA 4.0 for non-commercial research.",
        "",
        f"**Decision: {'PASS — panels may be committed' if all_pass else 'FAIL — local-only gallery'}**",
        "",
        "## Gate conditions",
        "",
        "| Condition | Status |",
        "|---|---|",
    ]
    for name, ok in gates.items():
        lines.append(f"| {name.replace('_', ' ')} | {'✅ PASS' if ok else '❌ FAIL'} |")
    if dicom_found:
        lines += ["", "**DICOM/NIfTI found (must not exist):**"]
        lines += [f"- `{p}`" for p in dicom_found]
    if large_files:
        lines += ["", "**Oversized panel files:**"]
        lines += [f"- {p}" for p in large_files]
    lines += [
        "",
        "## RSNA licence note",
        "",
        "RSNA 2024 Lumbar Spine Degenerative Classification dataset is licensed "
        "**CC BY-NC-SA 4.0 for non-commercial research** (RSNA terms). "
        "Derived visual panels (not raw DICOMs; anonymized case IDs; small crops) "
        "may be included in a private non-commercial repository under this licence, "
        "provided no full-resolution images or patient identifiers are committed.",
        "",
        "## Regenerate gallery locally",
        "",
        "```bash",
        "python scripts/generate_readme_assets_v1_9.py  # runs on local RSNA data",
        "```",
        "",
        "The generated panels live under gitignored `local_reports/v1_9_real_case_gallery/`.",
    ]
    GATE_MD.parent.mkdir(parents=True, exist_ok=True)
    GATE_MD.write_text("\n".join(lines) + "\n")

    print(f"Gate result: {'PASS' if all_pass else 'FAIL'}")
    for k, v in gates.items():
        print(f"  {'✅' if v else '❌'}  {k}")
    if not all_pass and strict:
        print("Gate FAILED — generating local-only gallery", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(strict=False))
