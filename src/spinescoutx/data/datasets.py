"""Torch dataset adapters over cached crop / segmentation manifests.

These are the *real-data* path: they read cached numpy arrays produced by the
caching pipeline and yield the exact dict schema used by the synthetic datasets,
so training code is agnostic to the source. No DICOM / volume decoding happens
here; only cached ``.npy`` arrays are read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..constants import (
    CONDITION_TO_INDEX,
    LEVEL_TO_INDEX,
    NUM_ANATOMY_PRIOR_CHANNELS,
    NUM_SEVERITY_CLASSES,
)
from .crops import CropRecord, frame_to_records


class AnatomyCacheMissingError(FileNotFoundError):
    """Raised when guided mode is requested but the anatomy mask cache is absent."""


def _load_array(path: Path) -> np.ndarray:
    """Load a cached ``.npy`` array, raising a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Cached crop array not found: {path}")
    return np.load(path).astype(np.float32)


def _to_3chw(arr: np.ndarray, crop_size: int) -> np.ndarray:
    """Coerce a cached crop array to a (3, H, W) float32 tensor-ready array."""
    if arr.ndim == 2:
        arr = np.repeat(arr[None, :, :], 3, axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Unexpected crop channel count: {arr.shape}")
    else:
        raise ValueError(f"Unexpected crop array ndim: {arr.ndim}")
    if arr.shape[-2:] != (crop_size, crop_size):
        raise ValueError(
            f"Cached crop spatial size {arr.shape[-2:]} != expected ({crop_size}, {crop_size})"
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


class RsnaCropDataset(Dataset):
    """Adapter over a cached RSNA crop manifest yielding the severity item schema."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        cache_root: str | Path,
        crop_size: int,
        use_25d: bool,
        guided: bool,
        anatomy_cache_root: str | None = None,
    ) -> None:
        self.records: list[CropRecord] = frame_to_records(manifest_df)
        self.cache_root = Path(cache_root)
        self.crop_size = int(crop_size)
        self.use_25d = bool(use_25d)
        self.guided = bool(guided)
        self.anatomy_cache_root = Path(anatomy_cache_root) if anatomy_cache_root else None
        if self.guided and self.anatomy_cache_root is None:
            raise AnatomyCacheMissingError(
                "guided=True requires anatomy_cache_root, but none was provided. "
                "Build the anatomy mask cache first or set guided=False."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _load_anatomy(self, rec: CropRecord) -> np.ndarray:
        """Load the cached anatomy-prior array for a record (guided mode only)."""
        assert self.anatomy_cache_root is not None  # guarded in __init__
        path = self.anatomy_cache_root / rec.crop_path
        if not path.exists():
            raise AnatomyCacheMissingError(
                f"guided=True but anatomy mask cache is missing for crop "
                f"{rec.crop_path!r} (looked in {path}). Build the anatomy cache "
                "first; the dataset will not fabricate anatomy channels."
            )
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[None, :, :], NUM_ANATOMY_PRIOR_CHANNELS, axis=0)
        if arr.shape[0] != NUM_ANATOMY_PRIOR_CHANNELS:
            raise ValueError(
                f"Anatomy cache for {rec.crop_path!r} has {arr.shape[0]} channels, "
                f"expected {NUM_ANATOMY_PRIOR_CHANNELS}"
            )
        return np.ascontiguousarray(arr, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        rec = self.records[index]
        image = _to_3chw(_load_array(self.cache_root / rec.crop_path), self.crop_size)

        if self.guided:
            anatomy = self._load_anatomy(rec)
        else:
            anatomy = np.zeros(
                (NUM_ANATOMY_PRIOR_CHANNELS, self.crop_size, self.crop_size),
                dtype=np.float32,
            )

        level_idx = LEVEL_TO_INDEX[rec.level]
        condition_idx = CONDITION_TO_INDEX[rec.condition]
        target = rec.severity_index
        return {
            "image": torch.from_numpy(image),
            "anatomy": torch.from_numpy(anatomy),
            "level_idx": torch.tensor(level_idx, dtype=torch.long),
            "condition_idx": torch.tensor(condition_idx, dtype=torch.long),
            "target": torch.tensor(int(target), dtype=torch.long),
            "study_id": rec.study_id,
        }


class SpiderSegDataset(Dataset):
    """Adapter over a cached SPIDER segmentation index yielding the seg item schema."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        cache_root: str | Path,
        crop_size: int,
    ) -> None:
        self.frame = index_df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.crop_size = int(crop_size)
        for col in ("subject_id", "image_path", "mask_path"):
            if col not in self.frame.columns:
                raise ValueError(f"SPIDER index missing required column: {col!r}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image = _load_array(self.cache_root / str(row["image_path"]))
        mask_path = self.cache_root / str(row["mask_path"])
        if not mask_path.exists():
            raise FileNotFoundError(f"Cached SPIDER mask not found: {mask_path}")
        mask = np.load(mask_path).astype(np.int64)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        if image.shape[-2:] != (self.crop_size, self.crop_size):
            raise ValueError(
                f"Cached SPIDER image size {image.shape[-2:]} != "
                f"({self.crop_size}, {self.crop_size})"
            )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image[None, :, :], dtype=np.float32)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            "subject_id": str(row["subject_id"]),
        }


def build_class_weights(targets: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency class weights over the 3 severity classes (length 3).

    Absent classes get weight 0 so they neither help nor dominate. The returned
    weights are normalized to sum to ``NUM_SEVERITY_CLASSES``.
    """
    counts = np.zeros(NUM_SEVERITY_CLASSES, dtype=np.float64)
    for t in targets:
        ti = int(t)
        if 0 <= ti < NUM_SEVERITY_CLASSES:
            counts[ti] += 1.0
    total = counts.sum()
    if total == 0:
        return torch.ones(NUM_SEVERITY_CLASSES, dtype=torch.float32)
    weights = np.where(counts > 0, total / (NUM_SEVERITY_CLASSES * counts), 0.0)
    nonzero_sum = weights.sum()
    if nonzero_sum > 0:
        present = (counts > 0).sum()
        weights = weights * (present / nonzero_sum)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(targets: Sequence[int]) -> WeightedRandomSampler:
    """Build a per-sample WeightedRandomSampler from inverse class frequencies."""
    counts = Counter(int(t) for t in targets)
    sample_weights = np.array(
        [1.0 / counts[int(t)] for t in targets],
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
