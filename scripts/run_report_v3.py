#!/usr/bin/env python3
"""Multi-condition study-level NON-DIAGNOSTIC finding-graph reports (v3).

For a handful of locked-test studies, assembles a research finding graph across all
**auto-supported** conditions (canal + L/R foraminal), with per-level severity estimates,
calibrated confidence, uncertainty flag, review reason, and provenance. Subarticular L/R
are labelled **oracle-only / blocked** (axial route not built). Findings come from the
deployable grader per condition (canal=auto-robust, foraminal=oracle-trained) on
auto-localized crops. JSON + Markdown. Research-only; not diagnostic; no treatment advice.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from spinescoutx.constants import LEVELS, RESEARCH_LIMITATIONS, SEVERITIES
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation.calibration import confidence_to_uncertainty_flag
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUTDIR = ROOT / "outputs/real/reports_v3"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_rep.parquet"
)
DISCLAIMER = (
    "research-only · not diagnostic · not clinically validated · no medical "
    "decision-making · no treatment recommendation"
)
AUTO_ROUTES = {
    "spinal_canal_stenosis": ("runs/v1_canal_auto_robust", "data/cache/rsna_auto_canal_all"),
    "left_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "right_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "left_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
    "right_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
}
BLOCKED = ()  # 5/5 auto: no blocked findings


def _study_preds(run, cond, cache, device, study):
    man = read_manifest(ROOT / cache / "manifest.parquet")
    man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
    man["study_id"] = man.study_id.astype(str)
    man = man[man.study_id == study]
    if man.empty:
        return {}
    man.to_parquet(TMP)
    preds = collect_probs(ROOT / run, TMP, ROOT / cache, device)
    return preds  # {study|level -> (y, probs3)}


def build_report(study, device) -> dict:
    findings = []
    review_needed = False
    for cond, (run, cache) in AUTO_ROUTES.items():
        preds = _study_preds(run, cond, cache, device, study)
        side = (
            "left" if cond.startswith("left") else ("right" if cond.startswith("right") else None)
        )
        for lv in LEVELS:
            key = f"{study}|{lv}"
            if key not in preds:
                continue
            y, p = preds[key]
            pred = int(np.argmax(p))
            conf = float(np.max(p))
            flag = confidence_to_uncertainty_flag(conf)
            reasons = []
            if flag == "review_required":
                reasons.append("low_confidence")
            if pred != 2 and float(p[2]) >= 0.20:
                reasons.append("near_severe_threshold")
            if reasons:
                review_needed = True
            findings.append(
                {
                    "condition": cond,
                    "side": side,
                    "level": lv,
                    "severity_estimate": SEVERITIES[pred],
                    "p_severe": round(float(p[2]), 3),
                    "confidence": round(conf, 3),
                    "uncertainty_flag": flag,
                    "review_reasons": reasons,
                    "provenance": "auto_crop (sagittal localizer; no GT at inference)",
                    "reference_label": SEVERITIES[int(y)],  # held-out research target
                }
            )
    blocked = [
        {
            "condition": c,
            "status": "oracle_only_blocked",
            "note": "axial-T2 auto route not built (level-matching blocker)",
        }
        for c in BLOCKED
    ]
    return {
        "study_id": str(study),
        "disclaimer": DISCLAIMER,
        "auto_supported_conditions": list(AUTO_ROUTES),
        "blocked_conditions": list(BLOCKED),
        "findings_auto": findings,
        "findings_blocked": blocked,
        "study_review_required": review_needed,
        "limitations": list(RESEARCH_LIMITATIONS),
    }


def _md(rep: dict) -> str:
    lines = [
        f"# Research finding graph — study {rep['study_id']} (NON-DIAGNOSTIC)",
        "",
        f"> {rep['disclaimer']}",
        "",
        f"Study-level review_required: **{rep['study_review_required']}**.",
        "Auto-supported findings (canal + L/R foraminal) graded from auto-localized crops;",
        "subarticular L/R are **oracle-only / blocked** (no axial auto route).",
        "",
        "## Auto findings (severity estimate · P(severe) · confidence · flag)",
        "| condition | side | level | severity estimate | P(severe) | conf | flag | review |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in rep["findings_auto"]:
        lines.append(
            f"| {f['condition']} | {f['side'] or '-'} | {f['level']} | {f['severity_estimate']} | "
            f"{f['p_severe']} | {f['confidence']} | {f['uncertainty_flag']} | "
            f"{','.join(f['review_reasons']) or '-'} |"
        )
    lines += ["", "## Blocked / oracle-only findings"]
    for b in rep["findings_blocked"]:
        lines.append(f"- **{b['condition']}**: {b['status']} — {b['note']}")
    lines += ["", "## Limitations"]
    lines += [f"- {x}" for x in rep["limitations"]]
    lines += ["", "_Severity estimates are research findings, not a diagnosis._"]
    return "\n".join(lines) + "\n"


def main() -> int:
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # pick locked-test studies that have at least one severe auto finding (rich reports)
    canal = read_manifest(ROOT / "data/cache/rsna_auto_canal_all" / "manifest.parquet")
    canal["study_id"] = canal.study_id.astype(str)
    test_sev = (
        canal[(canal.study_id.map(split_map) == "test") & (canal.severity == "severe")]
        .study_id.drop_duplicates()
        .tolist()[:12]
    )

    n = 0
    for study in test_sev:
        rep = build_report(study, device)
        if not rep["findings_auto"]:
            continue
        (OUTDIR / f"{study}.json").write_text(json.dumps(rep, indent=2))
        (OUTDIR / f"{study}.md").write_text(_md(rep))
        n += 1
    print(f"[report-v3] wrote {n} study reports to {OUTDIR}")
    print(f"[report-v3] example study: {test_sev[0] if test_sev else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
