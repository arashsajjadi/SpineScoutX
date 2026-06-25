"""Package best model weights + triage config for v1.9 release (Phase 4).

Creates two tarballs:
  - spinescoutx-best-raw-v1.9.tar.gz   (5 graders + localizer + metrics + model card)
  - spinescoutx-triage-config-v1.9.tar.gz  (triage output + config + docs)

Files > 50 MiB are NEVER committed to Git history — both tarballs go to the
GitHub Release asset upload (`gh release upload`). This script builds them and
writes SHA-256 checksums; Phase 10 uploads them.

Research-only. Not diagnostic. Best raw model = deployed reference (v1.0);
triage model = v1.7 severe-FN router (not a raw accuracy upgrade).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
STAGE = ROOT / "outputs/real/_package_staging"
DIST = ROOT / "outputs/real/v1_9_packages"
CS_FILE = ROOT / "docs/assets/v1_9/checksums.txt"
PKG_MD = ROOT / "docs/run_logs/v1_9_best_model_packaging.md"

BEST_RAW = {
    "canal": ROOT / "runs/v1_canal_auto_robust",
    "left_foraminal": ROOT / "runs/v1_foraminal_oracle_ctrl",
    "right_foraminal": ROOT / "runs/v1_foraminal_oracle_ctrl",
    "left_subarticular": ROOT / "runs/v1_subarticular_auto_robust",
    "right_subarticular": ROOT / "runs/v1_subarticular_auto_robust",
    "localizer": ROOT / "runs/lf_foraminal_localizer",
}
TRIAGE_FILES = [
    ROOT / "outputs/real/severe_fn_triage_v1_7.json",
    ROOT / "docs/run_logs/v1_7_final_accuracy_results.md",
    ROOT / "docs/run_logs/v1_7_failure_autopsy.md",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_raw() -> Path:
    stage = STAGE / "best_raw"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    metrics = {
        "description": "SpineScoutX best raw deployed model (v1.0 reference graders)",
        "research_only": True,
        "not_diagnostic": True,
        "locked_test_severe_recall": {
            "canal": 0.830, "left_foraminal": 0.788, "right_foraminal": 0.660,
            "left_subarticular": 0.746, "right_subarticular": 0.737,
            "five_route_macro": 0.752,
        },
        "reproduction_command": "python scripts/reproduce_best_metrics_v1_9.py",
        "model_card": "docs/model_card.md",
    }
    (stage / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (stage / "model_card.md").write_text(
        "# SpineScoutX best raw model\n\n"
        "**Research-only. Non-commercial. Not diagnostic. Not clinically validated.**\n\n"
        "Best raw model = deployed reference graders from v1.0 "
        "(canal/foraminal/subarticular). "
        "Locked-test 5-route macro severe recall: **0.752**.\n\n"
        "See `metrics.json` for per-route numbers. "
        "See `docs/model_card.md` in the main repo for the full model card.\n"
    )

    for route, run_dir in BEST_RAW.items():
        dest = stage / "graders" / route
        dest.mkdir(parents=True)
        for fname in ("best.pt", "config.json", "metrics.json"):
            src = run_dir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)

    tar_path = DIST / "spinescoutx-best-raw-v1.9.tar.gz"
    DIST.mkdir(parents=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(stage, arcname="spinescoutx-best-raw-v1.9")
    return tar_path


def _package_triage() -> Path:
    stage = STAGE / "triage_config"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    summary = {
        "description": "SpineScoutX v1.7 severe-FN triage config (safety, not raw accuracy)",
        "research_only": True,
        "not_diagnostic": True,
        "deployed_grader_unchanged": True,
        "triage_metric": "effective foraminal severe recall at review budget",
        "results": {
            "raw_argmax_severe_recall": 0.724,
            "effective_at_5pct_budget": 0.819,
            "effective_at_10pct_budget": 0.867,
            "effective_at_15pct_budget": 0.933,
            "effective_at_20pct_budget": 0.952,
            "severe_fn_captured_at_15pct": "22/29",
        },
        "note": (
            "Triage does NOT change argmax predictions — it ranks findings by severe-FN risk "
            "so a human reviewer can prioritise cases. Raw severe recall stays 0.724; "
            "effective recall reaches 0.933 when the flagged 15% are reviewed."
        ),
        "reproduction_command": "python scripts/train_severe_fn_triage_v1_7.py",
    }
    (stage / "triage_summary.json").write_text(json.dumps(summary, indent=2))
    for src in TRIAGE_FILES:
        if src.exists():
            shutil.copy2(src, stage / src.name)

    tar_path = DIST / "spinescoutx-triage-config-v1.9.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(stage, arcname="spinescoutx-triage-config-v1.9")
    return tar_path


def _write_checksums(paths: list[Path]) -> None:
    lines = ["# SpineScoutX v1.9 release checksums (SHA-256)\n"]
    for p in paths:
        cs = _sha256(p)
        mb = p.stat().st_size / 1_048_576
        lines.append(f"{cs}  {p.name}  ({mb:.1f} MiB)")
    CS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CS_FILE.write_text("\n".join(lines) + "\n")
    print(f"checksums → {CS_FILE}")
    for line in lines[1:]:
        print(f"  {line}")


def _write_packaging_doc(raw_path: Path, triage_path: Path) -> None:
    raw_mb = raw_path.stat().st_size / 1_048_576
    triage_mb = triage_path.stat().st_size / 1_048_576
    raw_sha = _sha256(raw_path)
    triage_sha = _sha256(triage_path)
    text = (
        "# v1.9 — best model packaging\n\n"
        "> Research-only · not diagnostic. "
        "All weights published as **GitHub Release assets** (not ordinary Git).\n\n"
        "## Best raw model — `spinescoutx-best-raw-v1.9.tar.gz`\n\n"
        f"Size: {raw_mb:.1f} MiB  |  SHA-256: `{raw_sha}`\n\n"
        "Contains: 5 grader `best.pt` + `config.json` + `metrics.json` for canal, "
        "foraminal (L+R share one run), subarticular (L+R share one run), and the "
        "foraminal localizer.\n\n"
        "**Best raw model = v1.0 deployed reference**. Locked-test 5-route macro 0.752. "
        "None of v1.1–v1.8c improved raw argmax severe recall.\n\n"
        "## Triage config — `spinescoutx-triage-config-v1.9.tar.gz`\n\n"
        f"Size: {triage_mb:.1f} MiB  |  SHA-256: `{triage_sha}`\n\n"
        "Contains: v1.7 severe-FN triage summary + triage output JSON + docs. "
        "**Does NOT change argmax predictions.** At 15% review budget, effective foraminal "
        "severe recall improves 0.724 → 0.933 (22/29 FN captured).\n\n"
        "## Upload command (after v1.9.0 tag exists)\n\n"
        "```bash\n"
        "python scripts/verify_release_assets_v1_9.py\n"
        "gh release upload v1.9.0-research-story-best-model \\\n"
        f"  outputs/real/v1_9_packages/spinescoutx-best-raw-v1.9.tar.gz \\\n"
        f"  outputs/real/v1_9_packages/spinescoutx-triage-config-v1.9.tar.gz \\\n"
        "  docs/assets/v1_9/checksums.txt\n"
        "```\n"
    )
    PKG_MD.parent.mkdir(parents=True, exist_ok=True)
    PKG_MD.write_text(text)
    print(f"wrote {PKG_MD}")


def main() -> int:
    print("packaging best raw model …")
    raw_path = _package_raw()
    print(f"  → {raw_path.name}  ({raw_path.stat().st_size / 1_048_576:.1f} MiB)")
    print("packaging triage config …")
    triage_path = _package_triage()
    print(f"  → {triage_path.name}  ({triage_path.stat().st_size / 1_048_576:.1f} MiB)")
    _write_checksums([raw_path, triage_path])
    _write_packaging_doc(raw_path, triage_path)
    print("all packages built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
