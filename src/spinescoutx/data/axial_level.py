"""Coordinate-supervised axial level scorer: data prep, training, monotonic decoding.

Trains :class:`AxialLevelScorer` on GT-labelled subarticular axial slices (each slice's
lumbar level + its normalized z-rank within the stack), then assigns levels to slices at
inference by running the scorer over the whole axial stack and decoding with a **monotonic
level-ordering** constraint (l1/l2 highest z … l5/s1 lowest). GT is used for supervision and
QC only — never at auto inference. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import LEVELS
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .axial_match import SUBARTICULAR, axial_z_by_instance, pick_axial_t2

log = get_logger()
_SPLIT_MAP = {"train": "train", "dev": "val", "test": "test"}

# Supervision-derived normalized in-plane centre of the subarticular zone per side,
# measured from GT axial coordinates (left x/cols 0.549±0.023, right 0.456±0.021,
# y/rows 0.524±0.04 both). The lateral recesses sit at a very consistent paramedian
# position on axial slices, so a fixed offset replaces an in-plane localizer.
SUBARTICULAR_OFFSETS = {"left": (0.549, 0.524), "right": (0.456, 0.524)}
SUBARTICULAR_COND = {
    "left": "left_subarticular_stenosis",
    "right": "right_subarticular_stenosis",
}


def prepare_axial_level_data(
    rsna_root: str | Path,
    out_cache: str | Path,
    split_map: dict[str, str],
    *,
    slice_size: int = 128,
    limit_studies: int | None = None,
) -> dict[str, Any]:
    """Cache one labelled axial slice per (study, level): image + level + normalized z-rank."""
    import cv2

    from .dicom_io import normalize_intensity, read_dicom
    from .rsna_index import RsnaPaths, build_series_index
    from .rsna_labels import load_coordinates

    rsna_root = Path(rsna_root)
    out = Path(out_cache)
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)
    ensure_dir(out / "slices")

    coords = load_coordinates(rsna_root)
    coords["study_id"] = coords.study_id.astype(str)
    coords["series_id"] = coords.series_id.astype(str)
    series = build_series_index(rsna_root)
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    sub = coords[coords.condition.isin(SUBARTICULAR)].copy()

    studies = sorted(sub.study_id.unique())
    if limit_studies is not None:
        studies = studies[: int(limit_studies)]

    rows: list[dict[str, Any]] = []
    skipped = 0
    for study in studies:
        ax_series = pick_axial_t2(series, study, images_dir)
        if ax_series is None:
            skipped += 1
            continue
        azs = axial_z_by_instance(images_dir, study, ax_series)
        if len(azs) < 5:
            skipped += 1
            continue
        zsorted = sorted(azs, key=lambda i: azs[i])
        rank = {inst: r for r, inst in enumerate(zsorted)}
        n = len(zsorted)
        g = sub[(sub.study_id == study) & (sub.series_id.astype(str) == str(ax_series))]
        for lv in LEVELS:
            gl = g[g.level == lv]
            if gl.empty:
                continue
            inst = int(gl.instance_number.median())
            if inst not in rank:
                continue
            dpath = images_dir / study / ax_series / f"{inst}.dcm"
            try:
                img = normalize_intensity(read_dicom(dpath))
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
            rel = f"slices/{study}_{ax_series}_{inst}_{lv}.npy"
            np.save(out / rel, resized.astype(np.float32))
            rows.append(
                {
                    "study_id": study,
                    "series_id": ax_series,
                    "instance_number": inst,
                    "level": lv,
                    "level_idx": LEVELS.index(lv),
                    "norm_z": float(rank[inst] / (n - 1)),
                    "slice_path": rel,
                    "split": _SPLIT_MAP.get(split_map.get(study, "train"), "train"),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "axial_level_manifest.parquet", index=False)
    return {
        "out_cache": str(out),
        "slice_size": slice_size,
        "n_rows": len(frame),
        "skipped": skipped,
        "split": {s: int((frame.split == s).sum()) for s in ("train", "val", "test")}
        if len(frame)
        else {},
    }


class AxialLevelDataset:
    """(axial slice [1,H,W], norm_z, level_idx) from the cached level manifest."""

    def __init__(self, frame: pd.DataFrame, cache_root: str | Path, slice_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.slice_size = int(slice_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        r = self.frame.iloc[idx]
        img = np.load(self.cache_root / str(r["slice_path"])).astype(np.float32)
        return {
            "image": torch.from_numpy(img[None]),
            "norm_z": torch.tensor([float(r["norm_z"])], dtype=torch.float32),
            "level_idx": torch.tensor(int(r["level_idx"]), dtype=torch.long),
        }


def train_axial_level_scorer(
    cache: str | Path,
    run_dir: str | Path,
    *,
    slice_size: int = 128,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    """Train the scorer (CE), select on dev accuracy. Returns best metrics + checkpoint."""
    import torch
    from torch.utils.data import DataLoader

    from ..models.axial_level_scorer import build_axial_level_scorer
    from ..training.optim import EarlyStopping, build_optimizer, build_scheduler, select_device
    from ..utils.seed import seed_everything

    seed_everything(1337)
    cache = Path(cache)
    out = ensure_dir(run_dir)
    dev_t = select_device(device)
    use_amp = dev_t.type == "cuda"
    man = pd.read_parquet(cache / "axial_level_manifest.parquet")
    tr = AxialLevelDataset(man[man.split == "train"], cache, slice_size)
    va = AxialLevelDataset(man[man.split == "val"], cache, slice_size)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=4)
    model = build_axial_level_scorer().to(dev_t)
    opt = build_optimizer(model, lr, 1e-4)
    sch = build_scheduler(opt, epochs, len(tl))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    lossfn = torch.nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=6, mode="max")
    best_acc, best, hist = -1.0, {}, []
    for ep in range(epochs):
        model.train()
        for b in tl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logit = model(b["image"].to(dev_t), b["norm_z"].to(dev_t))
                loss = lossfn(logit, b["level_idx"].to(dev_t))
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sch.step()
        # dev accuracy
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for b in vl:
                pr = model(b["image"].to(dev_t), b["norm_z"].to(dev_t)).argmax(1).cpu()
                correct += int((pr == b["level_idx"]).sum())
                tot += len(pr)
        acc = correct / max(tot, 1)
        hist.append({"epoch": ep, "dev_acc": acc})
        log.info("[axial-level] epoch %d dev_level_acc=%.3f", ep, acc)
        if stopper.step(acc) or best_acc < 0:
            best_acc, best = acc, {"dev_level_acc": acc, "epoch": ep}
            torch.save(
                {"state_dict": model.state_dict(), "slice_size": slice_size}, out / "best.pt"
            )
        if stopper.should_stop:
            break
    (out / "metrics.json").write_text(json.dumps({"best": best, "history": hist}, indent=2))
    return {"best": best, "checkpoint": str(out / "best.pt")}


def load_axial_level_scorer(run_dir: str | Path, device):
    import torch

    from ..models.axial_level_scorer import build_axial_level_scorer

    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device)
    model = build_axial_level_scorer().to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, int(ckpt["slice_size"])


def assign_levels_monotonic(log_probs: np.ndarray) -> dict[int, int]:
    """Assign each level a slice index (into the z-ascending stack) maximizing total
    log P(level|slice) under the monotonic constraint l1/l2 (highest z) … l5/s1 (lowest).

    ``log_probs`` is ``[n_slices, 5]`` with slices sorted by z ascending. Returns
    ``{level_idx -> slice_index}``. DP over strictly-increasing slice positions for the
    level order l5/s1→l1/l2.
    """
    n = log_probs.shape[0]
    order = [4, 3, 2, 1, 0]  # place l5/s1 at lowest slice rank, up to l1/l2
    neg = -1e9
    dp = np.full((5, n), neg)
    back = np.full((5, n), -1, dtype=int)
    dp[0] = log_probs[:, order[0]]
    for k in range(1, 5):
        best, bi = neg, -1
        for i in range(n):
            if i > 0 and dp[k - 1, i - 1] > best:
                best, bi = dp[k - 1, i - 1], i - 1
            if best > neg:
                dp[k, i] = best + log_probs[i, order[k]]
                back[k, i] = bi
    end = int(np.argmax(dp[4]))
    idxs = [0] * 5
    idxs[4] = end
    for k in range(4, 0, -1):
        idxs[k - 1] = int(back[k, idxs[k]])
    return {order[k]: idxs[k] for k in range(5)}


def score_and_assign_stack(model, images_dir, study, ax_series, slice_size, device) -> dict | None:
    """Run the scorer over an axial stack and return {level -> (instance, norm_z, conf)}."""
    import cv2
    import torch

    from .dicom_io import normalize_intensity, read_dicom

    azs = axial_z_by_instance(images_dir, study, ax_series)
    if len(azs) < 5:
        return None
    zsorted = sorted(azs, key=lambda i: azs[i])
    n = len(zsorted)
    logps = np.full((n, 5), -20.0)
    for r, inst in enumerate(zsorted):
        try:
            img = normalize_intensity(read_dicom(images_dir / study / ax_series / f"{inst}.dcm"))
        except Exception:  # noqa: BLE001
            continue
        resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            logit = model(
                torch.from_numpy(resized[None, None]).float().to(device),
                torch.tensor([[r / (n - 1)]], dtype=torch.float32).to(device),
            )
            logps[r] = torch.log_softmax(logit, dim=1)[0].cpu().numpy()
    assign = assign_levels_monotonic(logps)
    probs = np.exp(logps)
    out = {}
    for lvl_idx, sidx in assign.items():
        inst = zsorted[sidx]
        out[LEVELS[lvl_idx]] = {
            "instance": int(inst),
            "norm_z": float(sidx / (n - 1)),
            "conf": float(probs[sidx, lvl_idx]),
            "zsorted": zsorted,
            "slice_index": int(sidx),
        }
    return out


def prepare_rsna_subarticular_auto_crops(
    rsna_root: str | Path,
    scorer_run: str | Path,
    out_cache: str | Path,
    *,
    studies: list[str] | None = None,
    crop_size: int = 224,
    device: str = "auto",
) -> dict[str, Any]:
    """Auto subarticular crops (L/R) on the scorer-assigned axial slice per level, cropped
    at the supervision-derived fixed paramedian offset. Reads NO GT coordinates."""
    import contextlib

    from ..constants import SEVERITY_TO_INDEX
    from ..training.optim import select_device
    from .crops import CropRecord, extract_25d, write_manifest
    from .dicom_io import normalize_intensity, read_dicom
    from .rsna_index import RsnaPaths, build_series_index
    from .rsna_labels import load_labels

    rsna_root = Path(rsna_root)
    out = Path(out_cache)
    ensure_dir(out / "crops")
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)
    device_t = select_device(device)
    model, slice_size = load_axial_level_scorer(scorer_run, device_t)

    labels = load_labels(rsna_root)
    labels["study_id"] = labels.study_id.astype(str)
    sl = labels[labels.condition.isin(SUBARTICULAR)].copy()
    series = build_series_index(rsna_root)
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    if studies is None:
        studies = sorted(sl.study_id.unique())
    studies = [str(s) for s in studies]

    records: list[CropRecord] = []
    skipped = 0
    for study in studies:
        ax_series = pick_axial_t2(series, study, images_dir)
        if ax_series is None:
            skipped += 1
            continue
        res = score_and_assign_stack(model, images_dir, study, ax_series, slice_size, device_t)
        if res is None:
            skipped += 1
            continue
        g = sl[sl.study_id == study]
        for side, cond in SUBARTICULAR_COND.items():
            ox, oy = SUBARTICULAR_OFFSETS[side]
            cg = g[g.condition == cond]
            for _, r in cg.iterrows():
                lv = str(r.level)
                if lv not in res:
                    continue
                inst = res[lv]["instance"]
                slices: dict[int, np.ndarray] = {}
                for i in (inst - 1, inst, inst + 1):
                    p = images_dir / study / ax_series / f"{i}.dcm"
                    if p.exists():
                        with contextlib.suppress(Exception):
                            slices[i] = normalize_intensity(read_dicom(p))
                if inst not in slices:
                    skipped += 1
                    continue
                h, w = slices[inst].shape
                x, y = ox * w, oy * h
                rel = f"crops/{study}_{ax_series}_{inst}_{lv}_{cond}.npy"
                if not (out / rel).exists():
                    arr, pad = extract_25d(slices, inst, x, y, crop_size)
                    np.save(out / rel, arr.astype(np.float32))
                else:
                    pad = ""
                sev = str(r.severity)
                records.append(
                    CropRecord(
                        study_id=study,
                        series_id=ax_series,
                        instance_number=int(inst),
                        condition=cond,
                        level=lv,
                        side=side,
                        severity=sev,
                        severity_index=SEVERITY_TO_INDEX.get(sev, -1),
                        x=float(x),
                        y=float(y),
                        crop_path=rel,
                        dicom_path=str(images_dir / study / ax_series / f"{inst}.dcm"),
                        split="auto",
                        sequence="axial_t2",
                        patient_id=study,
                        pad_note=pad,
                        coordinate_source="auto",
                    )
                )
    manifest = write_manifest(records, out / "manifest.parquet")
    return {
        "out_cache": str(out),
        "scorer_run": str(scorer_run),
        "n_studies": len(studies),
        "n_auto_crops": len(records),
        "skipped": skipped,
        "manifest": str(manifest),
    }
