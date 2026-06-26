"""SpineScoutX v1.9 Late Submission — Kaggle Notebook Script.

Research-only. Not diagnostic. Not for medical decision-making.

Kaggle environment paths:
  Competition data:  /kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/
  Dataset assets:    /kaggle/input/spinescoutx-v1-9-best-raw-model/
  Script working dir: /kaggle/src/   (spinescoutx/ package is co-located here)
  Output dir:        /kaggle/working/

Dataset file layout (Kaggle extracts tarballs, double-nesting the root dir):
  spinescoutx-best-raw-v1.9/spinescoutx-best-raw-v1.9/graders/{canal,left_foraminal,...}
  spinescoutx-extra-models-v1.9/spinescoutx-extra-models-v1.9/{canal_localizer,axial_scorer}
  spinescoutx-0.1.0-py3-none-any.whl  (at root — not used; package is bundled in kernel)
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# spinescoutx/ is bundled alongside this script in /kaggle/src/
# /kaggle/src/ is already on sys.path in the Kaggle environment.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
RSNA_ROOT = KAGGLE_INPUT / "rsna-2024-lumbar-spine-degenerative-classification"
MODEL_ASSET = KAGGLE_INPUT / "spinescoutx-v1-9-best-raw-model"

# Kaggle extracts tarballs and wraps contents under a directory named after the archive.
# Archive spinescoutx-best-raw-v1.9.tar.gz already contains spinescoutx-best-raw-v1.9/ inside,
# so Kaggle mounts it at: <dataset>/spinescoutx-best-raw-v1.9/spinescoutx-best-raw-v1.9/
MAIN_MODELS = MODEL_ASSET / "spinescoutx-best-raw-v1.9" / "spinescoutx-best-raw-v1.9"
EXTRA_MODELS = MODEL_ASSET / "spinescoutx-extra-models-v1.9" / "spinescoutx-extra-models-v1.9"

RUN_MAP = {
    "v1_canal_auto_robust":        MAIN_MODELS / "graders" / "canal",
    "v1_foraminal_oracle_ctrl":    MAIN_MODELS / "graders" / "left_foraminal",
    "v1_subarticular_auto_robust": MAIN_MODELS / "graders" / "left_subarticular",
    "lf_foraminal_localizer":      MAIN_MODELS / "graders" / "localizer",
    "l0_disc_localizer_real":      EXTRA_MODELS / "canal_localizer",
    "axial_level_scorer":          EXTRA_MODELS / "axial_scorer",
}

OUTPUT_CSV = KAGGLE_WORKING / "submission.csv"

CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]
LEVELS = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")
UNIFORM = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
FORAMINAL_COND = {
    "left": "left_neural_foraminal_narrowing",
    "right": "right_neural_foraminal_narrowing",
}
SUBARTICULAR_COND = {
    "left": "left_subarticular_stenosis",
    "right": "right_subarticular_stenosis",
}

# ---------------------------------------------------------------------------
# Step 1 — Verify environment
# ---------------------------------------------------------------------------

print("=== Step 1: Verify environment ===")
print(f"Python: {sys.version}")
print(f"sys.path[0]: {sys.path[0]}")

import spinescoutx  # noqa: E402 (spinescoutx/ is bundled in /kaggle/src/)

print(f"spinescoutx loaded from: {spinescoutx.__file__}")

# Verify model dirs exist
for run_name, path in RUN_MAP.items():
    status = "OK" if path.exists() else "MISSING"
    print(f"  {status}: {run_name} → {path}")

# Verify competition data
test_images = RSNA_ROOT / "test_images"
sample_sub_path = RSNA_ROOT / "sample_submission.csv"
series_csv_path = RSNA_ROOT / "test_series_descriptions.csv"
for p in [test_images, sample_sub_path, series_csv_path]:
    status = "OK" if p.exists() else "MISSING"
    print(f"  {status}: {p.name}")

# ---------------------------------------------------------------------------
# Step 2 — Imports
# ---------------------------------------------------------------------------

print("\n=== Step 2: Import SpineScoutX modules ===")

from spinescoutx.data.auto_localize import load_localizer, localize_study  # noqa: E402
from spinescoutx.data.axial_level import (  # noqa: E402
    SUBARTICULAR_OFFSETS as _SUB_OFF,
    load_axial_level_scorer,
    score_and_assign_stack,
)
from spinescoutx.data.axial_match import pick_axial_t2  # noqa: E402
from spinescoutx.data.crops import CropRecord, extract_25d, write_manifest  # noqa: E402
from spinescoutx.data.dicom_io import normalize_intensity, read_dicom  # noqa: E402
from spinescoutx.data.foraminal_localize import (  # noqa: E402
    _load_foraminal_localizer,
    _localize_slice,
    pick_sagittal_t1,
    side_candidate_instances,
    slices_by_lr,
)
from spinescoutx.data.rsna_index import classify_sequence  # noqa: E402
from spinescoutx.evaluation.gap_decomposition import collect_probs  # noqa: E402
from spinescoutx.training.optim import select_device  # noqa: E402

device = select_device("auto")
print(f"Device: {device}")
print("All imports OK.")

# ---------------------------------------------------------------------------
# Step 3 — Inference helpers
# ---------------------------------------------------------------------------


def _make_crop(study, series_id, inst, condition, level, side, x, y, rel, seq):
    return CropRecord(
        study_id=study, series_id=series_id, instance_number=inst,
        condition=condition, level=level, side=side,
        severity="Normal/Mild", severity_index=0,
        x=x, y=y, crop_path=rel,
        dicom_path=str(test_images / study / series_id / f"{inst}.dcm"),
        split="test", sequence=seq, patient_id=study,
        pad_note="", coordinate_source="auto",
    )


def _run_grader(run_dir, manifest_path, cache_root):
    man = pd.read_parquet(manifest_path)
    if man.empty:
        return {}
    result = collect_probs(run_dir, manifest_path, cache_root, device)
    out = {}
    for key, (_y, prob3) in result.items():
        level = key.split("|")[1]
        out[level] = np.asarray(prob3, dtype=np.float64)
    return out


def _build_series_index():
    df = pd.read_csv(series_csv_path)
    df["study_id"] = df["study_id"].astype(str)
    df["series_id"] = df["series_id"].astype(str)
    df["sequence_type"] = df["series_description"].map(classify_sequence)
    return df


def infer_canal(study, series_index, tmp_cache):
    out_dir = tmp_cache / "canal"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)

    model, slice_size = load_localizer(RUN_MAP["l0_disc_localizer_real"], device)
    loc = localize_study(study, test_images, series_index, model, slice_size, device)
    if loc is None:
        print("[canal] localization failed — uniform prior")
        return {}

    series_id = loc["series_id"]
    inst = loc["instance_number"]
    slices = {}
    for i in (inst - 1, inst, inst + 1):
        p = test_images / study / series_id / f"{i}.dcm"
        if p.exists():
            with contextlib.suppress(Exception):
                slices[i] = normalize_intensity(read_dicom(p))
    if inst not in slices:
        print("[canal] DICOM load failed — uniform prior")
        return {}

    records = []
    for li, level in enumerate(LEVELS):
        x, y = float(loc["points"][li, 0]), float(loc["points"][li, 1])
        rel = f"crops/{study}_{series_id}_{inst}_{level}_spinal_canal_stenosis.npy"
        arr, _ = extract_25d(slices, inst, x, y, 224)
        np.save(out_dir / rel, arr.astype(np.float32))
        records.append(_make_crop(study, series_id, inst, "spinal_canal_stenosis",
                                  level, None, x, y, rel, "sagittal_t2"))

    mpath = write_manifest(records, out_dir / "manifest.parquet")
    return _run_grader(RUN_MAP["v1_canal_auto_robust"], mpath, out_dir)


def infer_foraminal(study, series_index, tmp_cache):
    out = {"left": {}, "right": {}}
    model, slice_size = _load_foraminal_localizer(RUN_MAP["lf_foraminal_localizer"], device)
    series_id = pick_sagittal_t1(series_index, study, test_images)
    if series_id is None:
        print("[foraminal] no sagittal_t1 — uniform prior")
        return out

    lr = slices_by_lr(test_images, study, series_id)
    if len(lr) < 3:
        print("[foraminal] too few slices — uniform prior")
        return out

    for side, cond in FORAMINAL_COND.items():
        out_dir = tmp_cache / f"foraminal_{side}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "crops").mkdir(exist_ok=True)

        cands = side_candidate_instances(lr, side)
        if not cands:
            print(f"[foraminal-{side}] no candidates — uniform prior")
            continue

        best_inst, best_pts, best_mc = None, None, -1.0
        for inst in cands:
            pts, conf = _localize_slice(
                test_images, study, series_id, inst, model, slice_size, device
            )
            if pts is None:
                continue
            mc = float(np.mean(conf))
            if mc > best_mc:
                best_inst, best_pts, best_mc = inst, pts, mc

        if best_inst is None:
            print(f"[foraminal-{side}] localization failed — uniform prior")
            continue

        slices = {}
        for i in (best_inst - 1, best_inst, best_inst + 1):
            p = test_images / study / series_id / f"{i}.dcm"
            if p.exists():
                with contextlib.suppress(Exception):
                    slices[i] = normalize_intensity(read_dicom(p))
        if best_inst not in slices:
            continue

        records = []
        for li, level in enumerate(LEVELS):
            x, y = float(best_pts[li, 0]), float(best_pts[li, 1])
            rel = f"crops/{study}_{series_id}_{best_inst}_{level}_{cond}.npy"
            arr, _ = extract_25d(slices, best_inst, x, y, 224)
            np.save(out_dir / rel, arr.astype(np.float32))
            records.append(_make_crop(study, series_id, best_inst, cond,
                                      level, side, x, y, rel, "sagittal_t1"))

        mpath = write_manifest(records, out_dir / "manifest.parquet")
        out[side] = _run_grader(RUN_MAP["v1_foraminal_oracle_ctrl"], mpath, out_dir)

    return out


def infer_subarticular(study, series_index, tmp_cache):
    out = {"left": {}, "right": {}}
    model, slice_size = load_axial_level_scorer(RUN_MAP["axial_level_scorer"], device)
    ax_series = pick_axial_t2(series_index, study, test_images)
    if ax_series is None:
        print("[subarticular] no axial_t2 series — uniform prior")
        return out

    res = score_and_assign_stack(model, test_images, study, ax_series, slice_size, device)
    if res is None:
        print("[subarticular] level scoring failed — uniform prior")
        return out

    for side, cond in SUBARTICULAR_COND.items():
        out_dir = tmp_cache / f"subarticular_{side}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "crops").mkdir(exist_ok=True)

        ox, oy = _SUB_OFF[side]
        records = []
        for level in LEVELS:
            if level not in res:
                continue
            inst = res[level]["instance"]
            slices = {}
            for i in (inst - 1, inst, inst + 1):
                p = test_images / study / ax_series / f"{i}.dcm"
                if p.exists():
                    with contextlib.suppress(Exception):
                        slices[i] = normalize_intensity(read_dicom(p))
            if inst not in slices:
                continue
            h, w = slices[inst].shape
            x, y = ox * w, oy * h
            rel = f"crops/{study}_{ax_series}_{inst}_{level}_{cond}.npy"
            arr, _ = extract_25d(slices, inst, x, y, 224)
            np.save(out_dir / rel, arr.astype(np.float32))
            records.append(_make_crop(study, ax_series, inst, cond,
                                      level, side, x, y, rel, "axial_t2"))

        if not records:
            print(f"[subarticular-{side}] no crops — uniform prior")
            continue

        mpath = write_manifest(records, out_dir / "manifest.parquet")
        out[side] = _run_grader(RUN_MAP["v1_subarticular_auto_robust"], mpath, out_dir)

    return out


def assemble_submission(study, canal_probs, foraminal_probs, sub_probs, sample_df):
    rows = []
    for _, srow in sample_df.iterrows():
        row_id = str(srow["row_id"])
        rest = row_id[len(study) + 1:]
        parts = rest.rsplit("_", 2)
        if len(parts) < 3:
            prob = UNIFORM.copy()
        else:
            level = f"{parts[-2]}_{parts[-1]}"
            cond = rest[:-(len(level) + 1)]
            if cond == "spinal_canal_stenosis":
                prob = canal_probs.get(level, UNIFORM)
            elif cond == "left_neural_foraminal_narrowing":
                prob = foraminal_probs["left"].get(level, UNIFORM)
            elif cond == "right_neural_foraminal_narrowing":
                prob = foraminal_probs["right"].get(level, UNIFORM)
            elif cond == "left_subarticular_stenosis":
                prob = sub_probs["left"].get(level, UNIFORM)
            elif cond == "right_subarticular_stenosis":
                prob = sub_probs["right"].get(level, UNIFORM)
            else:
                prob = UNIFORM.copy()
        rows.append({
            "row_id": row_id,
            "normal_mild": float(prob[0]),
            "moderate": float(prob[1]),
            "severe": float(prob[2]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 — Run inference
# ---------------------------------------------------------------------------

print("\n=== Step 4: Run inference ===")

series_index = _build_series_index()
print(f"Series index:\n{series_index.to_string(index=False)}")

sample_df = pd.read_csv(sample_sub_path)
studies = sorted(sample_df["row_id"].str.split("_").str[0].unique())
print(f"Test studies: {studies}")

all_rows = []
with tempfile.TemporaryDirectory(prefix="ssx_cache_") as tmpdir:
    cache = Path(tmpdir)
    for study in studies:
        print(f"\n=== Study {study} ===")

        canal_probs = infer_canal(study, series_index, cache / study / "canal")
        print(f"[canal] {len(canal_probs)} levels")

        foraminal_probs = infer_foraminal(study, series_index, cache / study / "for")
        print(f"[foraminal] L={len(foraminal_probs['left'])} R={len(foraminal_probs['right'])}")

        sub_probs = infer_subarticular(study, series_index, cache / study / "sub")
        print(f"[subarticular] L={len(sub_probs['left'])} R={len(sub_probs['right'])}")

        study_sample = sample_df[sample_df["row_id"].str.startswith(study + "_")]
        study_df = assemble_submission(study, canal_probs, foraminal_probs, sub_probs, study_sample)
        all_rows.append(study_df)

submission = pd.concat(all_rows, ignore_index=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(OUTPUT_CSV, index=False)
print(f"\n[output] {OUTPUT_CSV} ({len(submission)} rows)")

# ---------------------------------------------------------------------------
# Step 5 — Validate
# ---------------------------------------------------------------------------

print("\n=== Step 5: Validate submission ===")
sums = submission[["normal_mild", "moderate", "severe"]].sum(axis=1)
nan_count = submission[["normal_mild", "moderate", "severe"]].isna().sum().sum()
dup_count = submission["row_id"].duplicated().sum()
bad = (np.abs(sums - 1.0) > 1e-6).sum()
missing = set(sample_df["row_id"].astype(str)) - set(submission["row_id"].astype(str))
extra = set(submission["row_id"].astype(str)) - set(sample_df["row_id"].astype(str))

issues = []
if bad:
    issues.append(f"prob_sum ≠ 1 in {bad} rows")
if nan_count:
    issues.append(f"NaN in {nan_count} cells")
if dup_count:
    issues.append(f"{dup_count} duplicate row_ids")
if missing:
    issues.append(f"{len(missing)} missing row_ids")
if extra:
    issues.append(f"{len(extra)} extra row_ids")

passed = len(issues) == 0
print(f"Passed: {passed}")
print(f"Rows: {len(submission)} / expected {len(sample_df)}")
print(f"Prob sum mean={sums.mean():.6f} std={sums.std():.2e}")
if issues:
    for iss in issues:
        print(f"  ISSUE: {iss}")
    raise SystemExit("Validation FAILED")

print("Validation PASSED")
print(f"\nDone — {OUTPUT_CSV} is ready.")
