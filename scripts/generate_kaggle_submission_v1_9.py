"""Generate Kaggle submission CSV for RSNA 2024 Lumbar Spine competition.

Runs SpineScoutX v1.9 best raw graders on the Kaggle competition test data
and produces a valid submission.csv.

Routes:
  canal        : l0_disc_localizer_real  +  v1_canal_auto_robust
  foraminal L/R: lf_foraminal_localizer  +  v1_foraminal_oracle_ctrl
  subarticular : axial_level_scorer      +  v1_subarticular_auto_robust

Kaggle metric: weighted log loss (NOT severe recall — see plan doc).
Research-only. Not diagnostic. Not for medical decision-making.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# --- default paths (local runs/ weights) ---------------------------------
_DEFAULT_RUNS: dict[str, Path] = {
    "canal_grader": ROOT / "runs/v1_canal_auto_robust",
    "foraminal_grader": ROOT / "runs/v1_foraminal_oracle_ctrl",
    "subarticular_grader": ROOT / "runs/v1_subarticular_auto_robust",
    "canal_localizer": ROOT / "runs/l0_disc_localizer_real",
    "foraminal_localizer": ROOT / "runs/lf_foraminal_localizer",
    "subarticular_scorer": ROOT / "runs/axial_level_scorer",
}
_DEFAULT_RSNA = ROOT / "data/raw/rsna"
_DEFAULT_OUT = ROOT / "submissions/spinescoutx_v1_9_late_submission.csv"
_DEFAULT_VAL_OUT = ROOT / "submissions/spinescoutx_v1_9_late_submission_validation.json"

CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]
LEVELS = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")
SUBARTICULAR_OFFSETS = {"left": (0.549, 0.524), "right": (0.456, 0.524)}
SUBARTICULAR_COND = {
    "left": "left_subarticular_stenosis",
    "right": "right_subarticular_stenosis",
}
FORAMINAL_COND = {
    "left": "left_neural_foraminal_narrowing",
    "right": "right_neural_foraminal_narrowing",
}
UNIFORM = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------


def _unpack_tarball(tarball: Path, dest: Path) -> Path:
    """Unpack tarball into dest; return root content directory."""
    with tarfile.open(tarball) as tf:
        tf.extractall(dest)
    tops = [p for p in dest.iterdir() if p.is_dir()]
    return tops[0] if len(tops) == 1 else dest


def _resolve_runs(model_package: Path | None) -> dict[str, Path]:
    """Return run dirs from tarball or from local runs/."""
    if model_package is None or not model_package.exists():
        return dict(_DEFAULT_RUNS)
    tmpdir = Path(tempfile.mkdtemp(prefix="ssx_kaggle_"))
    print(f"[unpack] extracting {model_package.name} → {tmpdir}")
    root = _unpack_tarball(model_package, tmpdir)
    runs: dict[str, Path] = {}
    for key, default in _DEFAULT_RUNS.items():
        candidate = root / default.name
        if candidate.exists():
            runs[key] = candidate
        elif default.exists():
            runs[key] = default
        else:
            raise FileNotFoundError(
                f"Cannot find model for {key}: checked {candidate} and {default}"
            )
    return runs


def _build_test_series_index(rsna_root: Path) -> pd.DataFrame:
    from spinescoutx.data.rsna_index import classify_sequence

    p = rsna_root / "test_series_descriptions.csv"
    df = pd.read_csv(p)
    df["study_id"] = df["study_id"].astype(str)
    df["series_id"] = df["series_id"].astype(str)
    df["sequence_type"] = df["series_description"].map(classify_sequence)
    return df


def _make_crop_record(
    study: str,
    series_id: str,
    inst: int,
    condition: str,
    level: str,
    side: str | None,
    x: float,
    y: float,
    crop_path: str,
    images_dir: Path,
    sequence: str,
):
    from spinescoutx.data.crops import CropRecord

    return CropRecord(
        study_id=study,
        series_id=series_id,
        instance_number=inst,
        condition=condition,
        level=level,
        side=side,
        severity="Normal/Mild",
        severity_index=0,
        x=x,
        y=y,
        crop_path=crop_path,
        dicom_path=str(images_dir / study / series_id / f"{inst}.dcm"),
        split="test",
        sequence=sequence,
        patient_id=study,
        pad_note="",
        coordinate_source="auto",
    )


def _write_manifest(records: list, out_dir: Path) -> Path:
    from spinescoutx.data.crops import write_manifest

    if not records:
        return out_dir / "manifest.parquet"
    return write_manifest(records, out_dir / "manifest.parquet")


def _run_grader(
    run_dir: Path, manifest_path: Path, cache_root: Path, device
) -> dict[str, np.ndarray]:
    """collect_probs wrapper → {level: prob3}. Returns {} on empty manifest."""
    from spinescoutx.evaluation.gap_decomposition import collect_probs

    man = pd.read_parquet(manifest_path)
    if man.empty:
        return {}
    result = collect_probs(run_dir, manifest_path, cache_root, device)
    out = {}
    for key, (_y, prob3) in result.items():
        level = key.split("|")[1]
        out[level] = np.asarray(prob3, dtype=np.float64)
    return out


# -------------------------------------------------------------------------
# canal inference
# -------------------------------------------------------------------------


def infer_canal(
    study: str,
    images_dir: Path,
    series_index: pd.DataFrame,
    canal_localizer_run: Path,
    canal_grader_run: Path,
    tmp_cache: Path,
    device,
) -> dict[str, np.ndarray]:
    """Returns {level: prob3} for spinal_canal_stenosis."""
    import contextlib

    from spinescoutx.data.auto_localize import load_localizer, localize_study
    from spinescoutx.data.crops import extract_25d
    from spinescoutx.data.dicom_io import normalize_intensity, read_dicom

    out_dir = tmp_cache / "canal"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)

    model, slice_size = load_localizer(canal_localizer_run, device)
    loc = localize_study(study, images_dir, series_index, model, slice_size, device)
    if loc is None:
        print("[canal] localization failed — using uniform prior")
        return {}

    series_id = loc["series_id"]
    inst = loc["instance_number"]
    slices: dict[int, np.ndarray] = {}
    for i in (inst - 1, inst, inst + 1):
        p = images_dir / study / series_id / f"{i}.dcm"
        if p.exists():
            with contextlib.suppress(Exception):
                slices[i] = normalize_intensity(read_dicom(p))
    if inst not in slices:
        print("[canal] DICOM load failed — using uniform prior")
        return {}

    records = []
    for li, level in enumerate(LEVELS):
        x, y = float(loc["points"][li, 0]), float(loc["points"][li, 1])
        rel = f"crops/{study}_{series_id}_{inst}_{level}_spinal_canal_stenosis.npy"
        arr, _pad = extract_25d(slices, inst, x, y, 224)
        np.save(out_dir / rel, arr.astype(np.float32))
        records.append(
            _make_crop_record(
                study,
                series_id,
                inst,
                "spinal_canal_stenosis",
                level,
                None,
                x,
                y,
                rel,
                images_dir,
                "sagittal_t2",
            )
        )

    mpath = _write_manifest(records, out_dir)
    return _run_grader(canal_grader_run, mpath, out_dir, device)


# -------------------------------------------------------------------------
# foraminal inference
# -------------------------------------------------------------------------


def infer_foraminal(
    study: str,
    images_dir: Path,
    series_index: pd.DataFrame,
    foraminal_localizer_run: Path,
    foraminal_grader_run: Path,
    tmp_cache: Path,
    device,
) -> dict[str, dict[str, np.ndarray]]:
    """Returns {'left': {level: prob3}, 'right': {level: prob3}}."""
    import contextlib

    from spinescoutx.data.crops import extract_25d
    from spinescoutx.data.dicom_io import normalize_intensity, read_dicom
    from spinescoutx.data.foraminal_localize import (
        _load_foraminal_localizer,
        _localize_slice,
        pick_sagittal_t1,
        side_candidate_instances,
        slices_by_lr,
    )

    out: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
    model, slice_size = _load_foraminal_localizer(foraminal_localizer_run, device)

    series_id = pick_sagittal_t1(series_index, study, images_dir)
    if series_id is None:
        print("[foraminal] no sagittal_t1 series found — using uniform prior")
        return out

    lr = slices_by_lr(images_dir, study, series_id)
    if len(lr) < 3:
        print("[foraminal] too few slices — using uniform prior")
        return out

    for side, cond in FORAMINAL_COND.items():
        out_dir = tmp_cache / f"foraminal_{side}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "crops").mkdir(exist_ok=True)

        cands = side_candidate_instances(lr, side)
        if not cands:
            print(f"[foraminal-{side}] no candidates — using uniform prior")
            continue

        best_inst, best_pts, best_mc = None, None, -1.0
        for inst in cands:
            pts, conf = _localize_slice(
                images_dir, study, series_id, inst, model, slice_size, device
            )
            if pts is None:
                continue
            mc = float(np.mean(conf))
            if mc > best_mc:
                best_inst, best_pts, best_mc = inst, pts, mc
        if best_inst is None:
            print(f"[foraminal-{side}] localization failed — using uniform prior")
            continue

        slices: dict[int, np.ndarray] = {}
        for i in (best_inst - 1, best_inst, best_inst + 1):
            p = images_dir / study / series_id / f"{i}.dcm"
            if p.exists():
                with contextlib.suppress(Exception):
                    slices[i] = normalize_intensity(read_dicom(p))
        if best_inst not in slices:
            print(f"[foraminal-{side}] DICOM load failed — using uniform prior")
            continue

        records = []
        for li, level in enumerate(LEVELS):
            x, y = float(best_pts[li, 0]), float(best_pts[li, 1])
            rel = f"crops/{study}_{series_id}_{best_inst}_{level}_{cond}.npy"
            arr, _pad = extract_25d(slices, best_inst, x, y, 224)
            np.save(out_dir / rel, arr.astype(np.float32))
            records.append(
                _make_crop_record(
                    study,
                    series_id,
                    best_inst,
                    cond,
                    level,
                    side,
                    x,
                    y,
                    rel,
                    images_dir,
                    "sagittal_t1",
                )
            )

        mpath = _write_manifest(records, out_dir)
        out[side] = _run_grader(foraminal_grader_run, mpath, out_dir, device)

    return out


# -------------------------------------------------------------------------
# subarticular inference
# -------------------------------------------------------------------------


def infer_subarticular(
    study: str,
    images_dir: Path,
    series_index: pd.DataFrame,
    scorer_run: Path,
    subarticular_grader_run: Path,
    tmp_cache: Path,
    device,
) -> dict[str, dict[str, np.ndarray]]:
    """Returns {'left': {level: prob3}, 'right': {level: prob3}}."""
    import contextlib

    from spinescoutx.data.axial_level import SUBARTICULAR_OFFSETS as _OFF
    from spinescoutx.data.axial_level import (
        load_axial_level_scorer,
        score_and_assign_stack,
    )
    from spinescoutx.data.axial_match import pick_axial_t2
    from spinescoutx.data.crops import extract_25d
    from spinescoutx.data.dicom_io import normalize_intensity, read_dicom

    out: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
    model, slice_size = load_axial_level_scorer(scorer_run, device)

    ax_series = pick_axial_t2(series_index, study, images_dir)
    if ax_series is None:
        print("[subarticular] no axial_t2 series — using uniform prior")
        return out

    res = score_and_assign_stack(model, images_dir, study, ax_series, slice_size, device)
    if res is None:
        print("[subarticular] level scoring failed — using uniform prior")
        return out

    for side, cond in SUBARTICULAR_COND.items():
        out_dir = tmp_cache / f"subarticular_{side}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "crops").mkdir(exist_ok=True)

        ox, oy = _OFF[side]
        records = []
        for level in LEVELS:
            if level not in res:
                continue
            inst = res[level]["instance"]
            slices: dict[int, np.ndarray] = {}
            for i in (inst - 1, inst, inst + 1):
                p = images_dir / study / ax_series / f"{i}.dcm"
                if p.exists():
                    with contextlib.suppress(Exception):
                        slices[i] = normalize_intensity(read_dicom(p))
            if inst not in slices:
                continue
            h, w = slices[inst].shape
            x, y = ox * w, oy * h
            rel = f"crops/{study}_{ax_series}_{inst}_{level}_{cond}.npy"
            arr, _pad = extract_25d(slices, inst, x, y, 224)
            np.save(out_dir / rel, arr.astype(np.float32))
            records.append(
                _make_crop_record(
                    study,
                    ax_series,
                    inst,
                    cond,
                    level,
                    side,
                    x,
                    y,
                    rel,
                    images_dir,
                    "axial_t2",
                )
            )

        if not records:
            print(f"[subarticular-{side}] no crops — using uniform prior")
            continue

        mpath = _write_manifest(records, out_dir)
        out[side] = _run_grader(subarticular_grader_run, mpath, out_dir, device)

    return out


# -------------------------------------------------------------------------
# assembly + validation
# -------------------------------------------------------------------------


def assemble_submission(
    study: str,
    canal_probs: dict[str, np.ndarray],
    foraminal_probs: dict[str, dict[str, np.ndarray]],
    subarticular_probs: dict[str, dict[str, np.ndarray]],
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Map per-condition per-level probabilities → Kaggle submission rows."""
    rows = []
    for _, srow in sample_submission.iterrows():
        row_id = str(srow["row_id"])
        rest = row_id[len(study) + 1 :]
        parts = rest.rsplit("_", 2)
        if len(parts) < 3:
            prob = UNIFORM.copy()
        else:
            level = f"{parts[-2]}_{parts[-1]}"
            cond = rest[: -(len(level) + 1)]
            if cond == "spinal_canal_stenosis":
                prob = canal_probs.get(level, UNIFORM)
            elif cond == "left_neural_foraminal_narrowing":
                prob = foraminal_probs["left"].get(level, UNIFORM)
            elif cond == "right_neural_foraminal_narrowing":
                prob = foraminal_probs["right"].get(level, UNIFORM)
            elif cond == "left_subarticular_stenosis":
                prob = subarticular_probs["left"].get(level, UNIFORM)
            elif cond == "right_subarticular_stenosis":
                prob = subarticular_probs["right"].get(level, UNIFORM)
            else:
                prob = UNIFORM.copy()

        rows.append(
            {
                "row_id": row_id,
                "normal_mild": float(prob[0]),
                "moderate": float(prob[1]),
                "severe": float(prob[2]),
            }
        )
    return pd.DataFrame(rows)


def validate_submission(df: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Validate submission CSV against sample submission."""
    issues = []

    if list(df.columns) != ["row_id", "normal_mild", "moderate", "severe"]:
        issues.append(f"Wrong columns: {list(df.columns)}")

    sample_ids = set(sample["row_id"].astype(str))
    sub_ids = set(df["row_id"].astype(str))
    missing = sample_ids - sub_ids
    extra = sub_ids - sample_ids
    if missing:
        issues.append(f"Missing row_ids: {missing}")
    if extra:
        issues.append(f"Extra row_ids: {extra}")

    if df["row_id"].duplicated().sum():
        issues.append(f"Duplicate row_ids: {df['row_id'].duplicated().sum()}")

    nan_count = df[["normal_mild", "moderate", "severe"]].isna().sum().sum()
    if nan_count:
        issues.append(f"NaN values: {nan_count}")

    neg_count = (df[["normal_mild", "moderate", "severe"]] < 0).sum().sum()
    if neg_count:
        issues.append(f"Negative probabilities: {neg_count}")

    sums = df[["normal_mild", "moderate", "severe"]].sum(axis=1)
    bad_sums = (np.abs(sums - 1.0) > 1e-6).sum()
    if bad_sums:
        issues.append(f"Rows where prob sum ≠ 1.0: {bad_sums}")

    prob_vals = df[["normal_mild", "moderate", "severe"]].to_numpy()
    if np.isinf(prob_vals).any():
        issues.append("Infinite probability values found")

    return {
        "passed": len(issues) == 0,
        "n_rows": len(df),
        "n_expected": len(sample),
        "issues": issues,
        "prob_sum_mean": float(sums.mean()),
        "prob_sum_std": float(sums.std()),
    }


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SpineScoutX Kaggle submission v1.9")
    ap.add_argument(
        "--model-package",
        type=Path,
        default=None,
        help="spinescoutx-best-raw-v1.9.tar.gz (optional; uses runs/ if omitted)",
    )
    ap.add_argument(
        "--kaggle-data",
        type=Path,
        default=_DEFAULT_RSNA,
        help="RSNA competition data root (default: data/raw/rsna/)",
    )
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    rsna_root = args.kaggle_data
    test_images_dir = rsna_root / "test_images"
    sample_sub_path = rsna_root / "sample_submission.csv"

    if not test_images_dir.exists():
        print(f"ERROR: test_images not found at {test_images_dir}")
        return 1
    if not sample_sub_path.exists():
        print(f"ERROR: sample_submission.csv not found at {sample_sub_path}")
        return 1

    sample_sub = pd.read_csv(sample_sub_path)
    studies = sorted(sample_sub["row_id"].str.split("_").str[0].unique())
    print(f"[info] {len(studies)} test study/studies: {studies}")

    from spinescoutx.training.optim import select_device

    device = select_device("auto")
    print(f"[info] device: {device}")

    runs = _resolve_runs(args.model_package)
    for key, path in runs.items():
        if not path.exists():
            print(f"ERROR: model run directory missing: {path} ({key})")
            return 1

    series_index = _build_test_series_index(rsna_root)
    print(f"[info] test series index:\n{series_index.to_string(index=False)}")

    tmp_cache = Path(tempfile.mkdtemp(prefix="ssx_kaggle_cache_"))
    print(f"[info] tmp cache: {tmp_cache}")

    all_rows: list[pd.DataFrame] = []

    for study in studies:
        print(f"\n=== Study {study} ===")

        print("[canal] running inference...")
        canal_probs = infer_canal(
            study,
            test_images_dir,
            series_index,
            runs["canal_localizer"],
            runs["canal_grader"],
            tmp_cache,
            device,
        )
        print(f"[canal] probs for {len(canal_probs)} levels")

        print("[foraminal] running inference...")
        foraminal_probs = infer_foraminal(
            study,
            test_images_dir,
            series_index,
            runs["foraminal_localizer"],
            runs["foraminal_grader"],
            tmp_cache,
            device,
        )
        n_l = len(foraminal_probs["left"])
        n_r = len(foraminal_probs["right"])
        print(f"[foraminal] left={n_l} right={n_r} levels")

        print("[subarticular] running inference...")
        subarticular_probs = infer_subarticular(
            study,
            test_images_dir,
            series_index,
            runs["subarticular_scorer"],
            runs["subarticular_grader"],
            tmp_cache,
            device,
        )
        n_sl = len(subarticular_probs["left"])
        n_sr = len(subarticular_probs["right"])
        print(f"[subarticular] left={n_sl} right={n_sr} levels")

        study_sample = sample_sub[sample_sub["row_id"].str.startswith(study + "_")]
        study_df = assemble_submission(
            study,
            canal_probs,
            foraminal_probs,
            subarticular_probs,
            study_sample,
        )
        all_rows.append(study_df)

    submission = pd.concat(all_rows, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"\n[output] {args.output} ({len(submission)} rows)")

    val_result = validate_submission(submission, sample_sub)

    print("\n=== Validation ===")
    print(f"Passed: {val_result['passed']}")
    print(f"Rows: {val_result['n_rows']} / expected {val_result['n_expected']}")
    print(f"Prob sum mean={val_result['prob_sum_mean']:.6f} std={val_result['prob_sum_std']:.2e}")
    if val_result["issues"]:
        for iss in val_result["issues"]:
            print(f"  ISSUE: {iss}")

    val_out = _DEFAULT_VAL_OUT
    val_out.parent.mkdir(parents=True, exist_ok=True)
    val_result["submission_path"] = str(args.output)
    val_result["n_studies"] = len(studies)
    val_out.write_text(json.dumps(val_result, indent=2))
    print(f"[output] validation: {val_out}")

    if not val_result["passed"]:
        print("\nERROR: validation FAILED — do not submit")
        return 1

    print("\nDone — submission ready for upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
