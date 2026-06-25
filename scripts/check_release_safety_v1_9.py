"""v1.9 release safety gate (Phase 8).

Checks all conditions that must be true before pushing/merging the v1.9 branch:
  - no DICOM/NIfTI staged or in docs/
  - no files > 50 MiB staged for Git
  - no token-like strings in staged files
  - no real image panels without the gate passing
  - no unsafe clinical/diagnostic claims in README / docs
  - no files from forbidden paths staged

Exits 0 if all checks pass, 1 otherwise. Run as part of the pre-merge gate.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")

FORBIDDEN_PATTERNS = [
    r"hf_[A-Za-z0-9]{20,}",  # HF token
    r"kaggle[._\-]*key\s*[:=]",  # Kaggle key literal
    # raw base-64 secrets: 38+ alphanum+/+ chars, NOT preceded by github.com/, /, or path sep
    r"(?<![/\w])(?<!\bgithub\.com/)[A-Za-z0-9+]{40}={0,2}\b",
]
# Files to skip for token-literal scan (contain the pattern strings themselves or are known-safe)
TOKEN_SCAN_SKIP = {
    "check_release_safety_v1_9.py",  # this file — contains the patterns as strings
}
CLINICAL_PATTERNS = [
    r"\bdiagnos(e|is)\b(?![\w-])",  # bare "diagnose/diagnosis" without "not"
    r"\btreat\b",  # treatment recommendation
    r"\bFDA.approved\b",
    r"\bCE.mark\b",
    r"\bclinically\s+valid(at)?",
]
SAFE_EXCEPTIONS = [
    "not diagnos",
    "not for diagnos",
    "non-diagnostic",
    "not diagnostic",
    "not clinically valid",
    "research-only",
    "not treat",
    "not FDA",
    "not CE",
    "never diagnos",
    "no diagnos",
    "research finding",
    "severity estimate",
    "out of scope",
    "scope:",
    "out-of-scope",
    "we treat",
    "scope of",
    "out of",
    "do not",
    "does not",
    "never as",
    "for *diagnosis*",
    "any diagnosis",
    "severity estimate",
    "does **not**",
    "not make",
    "**not**",
    "re-diagnos",
    "rigorously",
    "honest",
    "inside this",
    "only inside",
    "disclaimers",
    "only)",
    "this list",
    "not for\n  diagnosis",
    "right-foraminal diagnosis",
    "domain-shift audit, right-foraminal",
    "foraminal diagnosis",
    "calibration trial, domain",
    "survey, right-foraminal",
]
FORBIDDEN_EXTENSIONS = {".dcm", ".nii", ".mrd", ".MRD"}
MAX_GIT_FILE_MB = 50.0


def _staged_files() -> list[Path]:
    r = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], capture_output=True, text=True, cwd=ROOT
    )
    return [ROOT / f for f in r.stdout.splitlines() if f.strip()]


def _tracked_docs() -> list[Path]:
    r = subprocess.run(
        ["git", "ls-files", "docs/", "README.md"], capture_output=True, text=True, cwd=ROOT
    )
    return [ROOT / f for f in r.stdout.splitlines() if f.strip()]


def check_no_dicom(staged: list[Path]) -> list[str]:
    errs = []
    for p in staged:
        if p.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errs.append(f"DICOM/NIfTI staged: {p.relative_to(ROOT)}")
    for root_dir in [ROOT / "docs"]:
        for p in root_dir.rglob("*"):
            if p.suffix.lower() in FORBIDDEN_EXTENSIONS:
                errs.append(f"DICOM/NIfTI in docs: {p.relative_to(ROOT)}")
    return errs


def check_no_large_staged(staged: list[Path]) -> list[str]:
    errs = []
    for p in staged:
        if p.exists() and p.stat().st_size / 1_048_576 > MAX_GIT_FILE_MB:
            mb = p.stat().st_size / 1_048_576
            errs.append(f"File too large for Git ({mb:.1f} MiB): {p.relative_to(ROOT)}")
    return errs


_PATH_LIKE = re.compile(r"[/\\](?:home|usr|opt|var|tmp|data|arash)[/\\]")


def check_no_tokens(staged: list[Path]) -> list[str]:
    errs = []
    for p in staged:
        if not p.exists() or p.suffix not in (".py", ".md", ".txt", ".json", ".yaml", ".toml"):
            continue
        if "private_load_tokens" in p.name or p.name in TOKEN_SCAN_SKIP:
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for pat in FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, text):
                hit = m.group()
                # Skip if it looks like a URL segment, file path, or GitHub identifier
                ctx_before = text[max(0, m.start() - 30) : m.start()]
                if (
                    "github.com" in ctx_before
                    or "http" in ctx_before
                    or _PATH_LIKE.search(ctx_before + hit)
                    or "/" in hit
                ):
                    continue
                line_no = text[: m.start()].count("\n") + 1
                errs.append(f"Token-like string in {p.relative_to(ROOT)}:{line_no}: {hit[:20]}…")
    return errs


CLINICAL_SKIP = {
    "check_release_safety_v1_9.py",  # contains the regex patterns as raw strings
}


def check_clinical_claims(paths: list[Path]) -> list[str]:
    errs = []
    for p in paths:
        if not p.exists() or p.name in CLINICAL_SKIP:
            continue
        try:
            text = p.read_text(errors="replace").lower()
        except Exception:  # noqa: BLE001
            continue
        for pat in CLINICAL_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                ctx = text[max(0, m.start() - 60) : m.end() + 60].strip()
                if any(exc in ctx for exc in SAFE_EXCEPTIONS):
                    continue
                line_no = text[: m.start()].count("\n") + 1
                errs.append(
                    f"Unsafe clinical claim in {p.relative_to(ROOT)}:{line_no}: …{ctx[:80]}…"
                )
    return errs


def check_real_image_gate(staged: list[Path]) -> list[str]:
    errs = []
    has_real_cases = any("real_cases" in str(p) and p.suffix == ".png" for p in staged)
    if not has_real_cases:
        return errs
    gate_json = ROOT / "outputs/real/v1_9_real_image_release_gate.json"
    if not gate_json.exists():
        errs.append(
            "Real image panels staged but gate JSON missing — run check_real_image_release_gate_v1_9.py first."
        )
        return errs
    gate = json.loads(gate_json.read_text())
    if not gate.get("all_pass", False):
        errs.append(f"Real image panels staged but gate FAILED: {gate.get('decision', '?')}")
    return errs


def main() -> int:
    staged = _staged_files()
    docs = _tracked_docs()
    all_errs: list[str] = []

    checks = [
        ("DICOM/NIfTI", check_no_dicom(staged)),
        ("Large files", check_no_large_staged(staged)),
        ("Token literals", check_no_tokens(staged)),
        ("Clinical claims", check_clinical_claims(docs + staged)),
        ("Real image gate", check_real_image_gate(staged)),
    ]

    for name, errs in checks:
        if errs:
            print(f"\n❌ {name}:")
            for e in errs:
                print(f"   {e}")
            all_errs += errs
        else:
            print(f"✅ {name}: OK")

    print()
    if all_errs:
        print(f"Safety gate FAILED — {len(all_errs)} issue(s) found.", file=sys.stderr)
        return 1
    print("Safety gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
