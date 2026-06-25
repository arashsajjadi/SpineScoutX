"""Generate v1.9 README assets + real-case example gallery (Phase 3 / Phase 7).

Produces anonymized, pixel-safe derived example panels from RSNA crops + deployed
model predictions. Each panel: anonymized case ID, finding/side/level, GT label,
prediction, probability bar chart, correct/FN/review status, one central crop.
No patient metadata. No raw DICOMs. No full-resolution images.
Research-only. Not diagnostic. RSNA CC BY-NC-SA 4.0 non-commercial.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
DEPLOYED = ROOT / "runs/v1_foraminal_oracle_ctrl"
GATE_JSON = ROOT / "outputs/real/v1_9_real_image_release_gate.json"

OUT_COMMIT = ROOT / "docs/assets/v1_9/real_cases"
OUT_LOCAL = ROOT / "local_reports/v1_9_real_case_gallery"

SEVERITY_LABEL = {0: "Normal/Mild", 1: "Moderate", 2: "Severe"}
SEVERITY_COLOR = {0: "#2a9d5c", 1: "#e07b39", 2: "#d64545"}
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]


def _gate_ok() -> bool:
    if GATE_JSON.exists():
        return bool(json.loads(GATE_JSON.read_text()).get("all_pass", False))
    return False


def _case_id(study_id: str, level: str, condition: str) -> str:
    raw = f"{study_id}|{level}|{condition}"
    return "case_" + hashlib.sha1(raw.encode()).hexdigest()[:8]  # noqa: S324


def _load_crop(row: dict) -> np.ndarray | None:
    cp = RSNA_CACHE / row["crop_path"]
    if not cp.exists():
        return None
    arr = np.load(cp).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return arr


def _make_panel(
    row: dict,
    crop: np.ndarray,
    prob: np.ndarray,
    pred_label: int,
    status: str,
    outdir: Path,
    panel_name: str,
) -> Path:
    gt = int(row["severity_index"])
    cid = _case_id(str(row["study_id"]), str(row["level"]), str(row["condition"]))
    level_str = str(row["level"]).replace("_", " ").upper()
    cond_short = (
        "L-foraminal"
        if "left_neural" in row["condition"]
        else "R-foraminal"
        if "right_neural" in row["condition"]
        else row["condition"].replace("_", " ")
    )
    is_correct = gt == pred_label
    badge = "✅ Correct" if is_correct else ("⚠ Review" if status == "review" else "❌ FN")
    badge_color = "#2a9d5c" if is_correct else ("#e07b39" if status == "review" else "#d64545")

    fig = plt.figure(figsize=(5.2, 3.2), dpi=130)
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[1.3, 1], wspace=0.3, left=0.04, right=0.97, top=0.80, bottom=0.10
    )
    ax_img = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    img = crop[1] if crop.ndim == 3 else crop
    ax_img.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    ax_img.axis("off")

    classes = ["Normal/\nMild", "Moderate", "Severe"]
    bar_colors = [SEVERITY_COLOR[i] for i in range(3)]
    bars = ax_bar.barh(classes, prob, color=bar_colors, height=0.55, edgecolor="white")
    for bar, p in zip(bars, prob, strict=False):
        ax_bar.text(
            p + 0.02, bar.get_y() + bar.get_height() / 2, f"{p:.2f}", va="center", fontsize=9
        )
    ax_bar.set_xlim(0, 1.25)
    ax_bar.set_xlabel("Probability", fontsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(labelsize=8)

    title = (
        f"{cid}  |  {cond_short}  {level_str}\n"
        f"GT: {SEVERITY_LABEL[gt]}  →  Pred: {SEVERITY_LABEL[pred_label]}  |  {badge}"
    )
    fig.suptitle(title, fontsize=9.5, ha="center", va="top", y=0.98, color=badge_color)
    fig.text(
        0.5,
        0.01,
        "Research-only · not diagnostic · RSNA CC BY-NC-SA",
        ha="center",
        fontsize=7,
        color="#aaaaaa",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / panel_name
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _build_probs_map(device: object, sm: dict) -> dict[str, tuple[np.ndarray, int]]:
    """Return ``study|level|condition → (prob3, pred_class)`` over test split."""
    from spinescoutx.data.crops import read_manifest
    from spinescoutx.evaluation.gap_decomposition import collect_probs

    man = read_manifest(RSNA_CACHE / "manifest.parquet")
    man["study_id"] = man.study_id.astype(str)
    result: dict[str, tuple[np.ndarray, int]] = {}
    tmp = ROOT / "outputs/real/_gallery_tmp.parquet"
    for cond in FORAMINAL:
        sub = man[(man.condition == cond) & man.severity_index.isin([0, 1, 2])].copy()
        sub = sub[sub.study_id.map(sm) == "test"]
        if sub.empty:
            continue
        sub.to_parquet(tmp)
        for k, (_, p) in collect_probs(DEPLOYED, tmp, RSNA_CACHE, device).items():
            st, lv = k.split("|")
            pa = np.asarray(p, float)
            result[f"{st}|{lv}|{cond}"] = (pa, int(np.argmax(pa)))
    if tmp.exists():
        tmp.unlink()
    return result


def main() -> int:
    gate_ok = _gate_ok()
    outdir = OUT_COMMIT if gate_ok else OUT_LOCAL
    print(f"Gate: {'PASS' if gate_ok else 'FAIL'} → writing to {outdir}")

    if not RSNA_CACHE.exists() or not (RSNA_CACHE / "manifest.parquet").exists():
        print("RSNA foraminal cache not found — cannot generate gallery.", file=sys.stderr)
        _write_placeholder(gate_ok)
        return 0

    from spinescoutx.data.crops import read_manifest
    from spinescoutx.data.locked_test import load_splits_v1
    from spinescoutx.training.optim import select_device

    device = select_device("auto")
    sm = load_splits_v1(SPLITS)
    probs_map = _build_probs_map(device, sm)

    man = read_manifest(RSNA_CACHE / "manifest.parquet")
    man["study_id"] = man.study_id.astype(str)
    man["split"] = man.study_id.map(sm)
    test_man = man[man.split == "test"].reset_index(drop=True)

    panels_spec = [
        ("correct_l_foraminal_severe", "left_neural_foraminal_narrowing", 2, False, "correct"),
        ("correct_r_foraminal_severe", "right_neural_foraminal_narrowing", 2, False, "correct"),
        ("r_foraminal_severe_fn", "right_neural_foraminal_narrowing", 2, True, "fn"),
        ("l_foraminal_severe_fn", "left_neural_foraminal_narrowing", 2, True, "fn"),
        ("r_foraminal_borderline_moderate", "right_neural_foraminal_narrowing", 1, False, "review"),
        ("l_foraminal_correct_moderate", "left_neural_foraminal_narrowing", 1, False, "correct"),
        ("r_foraminal_normal_correct", "right_neural_foraminal_narrowing", 0, False, "correct"),
        ("l_foraminal_normal_correct", "left_neural_foraminal_narrowing", 0, False, "correct"),
        ("r_foraminal_fn2", "right_neural_foraminal_narrowing", 2, True, "fn"),
        ("l_foraminal_review_moderate", "left_neural_foraminal_narrowing", 1, False, "review"),
        ("r_foraminal_moderate_correct", "right_neural_foraminal_narrowing", 1, False, "correct"),
        ("l_foraminal_severe_correct2", "left_neural_foraminal_narrowing", 2, False, "correct"),
    ]

    written = []
    for idx, (name, cond, gt_sev, want_fn, status) in enumerate(panels_spec, 1):
        subset = test_man[(test_man.condition == cond) & (test_man.severity_index == gt_sev)]
        if subset.empty:
            print(f"  skip {name}: no rows")
            continue

        found = None
        for _, row in subset.sample(frac=1, random_state=42 + idx).iterrows():
            r = row.to_dict()
            key = f"{r['study_id']}|{r['level']}|{cond}"
            if key not in probs_map:
                continue
            p, pred = probs_map[key]
            is_fn = r["severity_index"] == 2 and pred != 2
            if want_fn and not is_fn:
                continue
            if not want_fn and is_fn and status == "correct":
                continue
            crop = _load_crop(r)
            if crop is None:
                continue
            found = (r, crop, p, pred)
            break

        if found is None:
            print(f"  skip {name}: no matching example")
            continue

        r, crop, p, pred = found
        panel_name = f"case_{idx:03d}_{name}.png"
        path = _make_panel(r, crop, p, pred, status, outdir, panel_name)
        written.append((panel_name, name, cond, r["level"], r["severity_index"], pred))
        print(f"  wrote {path.name}")

    _write_index(outdir, written, gate_ok)
    _write_placeholder(gate_ok)
    print(f"\n{len(written)} panels written → {outdir}")
    return 0


def _write_index(outdir: Path, panels: list, gate_ok: bool) -> None:
    lines = [
        "# v1.9 real-case gallery",
        "",
        "> Research-only · non-commercial · not diagnostic.",
        "> RSNA 2024 dataset: CC BY-NC-SA 4.0. Case IDs = anonymized SHA-1 hashes.",
        "> Panels: central slice crop + probability bar + GT/prediction. "
        "No patient metadata. No raw DICOMs.",
        "",
        "| Panel | Finding | Level | GT | Prediction |",
        "|---|---|---|---|---|",
    ]
    for fname, _, cond, level, gt, pred in panels:
        cond_s = cond.replace("_", " ")
        level_s = str(level).replace("_", " ").upper()
        gt_s = SEVERITY_LABEL.get(int(gt), str(gt))
        pred_s = SEVERITY_LABEL.get(int(pred), str(pred))
        if gate_ok:
            lines.append(f"| ![]({fname}) | {cond_s} | {level_s} | {gt_s} | {pred_s} |")
        else:
            lines.append(f"| `{fname}` | {cond_s} | {level_s} | {gt_s} | {pred_s} |")
    if not gate_ok:
        lines += [
            "",
            "> Local-only gallery — not committed. Regenerate with "
            "`python scripts/generate_readme_assets_v1_9.py`.",
        ]
    (outdir / "index.md").write_text("\n".join(lines) + "\n")


def _write_placeholder(gate_ok: bool) -> None:
    ph = ROOT / "docs/assets/v1_9/real_cases_safe_placeholder.md"
    if gate_ok:
        ph.write_text(
            "# Real-case gallery\n\n"
            "Derived anonymized example panels in `docs/assets/v1_9/real_cases/`. "
            "See `index.md` there.\n"
            "RSNA CC BY-NC-SA 4.0 non-commercial research.\n"
            "Regenerate: `python scripts/generate_readme_assets_v1_9.py`\n"
        )
    else:
        ph.write_text(
            "# Real-case gallery — local-only\n\n"
            "License/privacy gate did not permit committing panels to this repository.\n\n"
            "To generate locally:\n"
            "```bash\npython scripts/generate_readme_assets_v1_9.py\n```\n"
            "Panels are written to `local_reports/v1_9_real_case_gallery/` (gitignored).\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
