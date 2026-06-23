"""Deterministic in-memory synthetic datasets for offline smoke tests and CI.

The synthetic classification data plants a *learnable* signal so a small model
trains above chance: the severity label is correlated with the mean intensity of
a bright patch placed in the image AND with the anatomy-prior channels, so both
the image branch and the anatomy branch carry information about the target. The
class distribution is intentionally imbalanced (mostly ``normal_mild``) so the
weighting / sampler code paths are exercised.

The synthetic segmentation data plants simple geometric shapes (a stack of
vertebra blocks, disc bands between them, and a thin central spinal canal) so
Dice can meaningfully improve during a smoke train.

All randomness is per-index via ``numpy.random.default_rng(seed + index)`` so an
item is fully reproducible and independent of access order.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import (
    LEVELS,
    NUM_ANATOMY_PRIOR_CHANNELS,
    NUM_SEVERITY_CLASSES,
)

# Probability of each severity class (imbalanced: mostly normal_mild).
_SEVERITY_PROBS: tuple[float, ...] = (0.7, 0.2, 0.1)
# Number of full per-side conditions used for the condition index.
_NUM_CONDITIONS: int = 5


def _sample_severity(rng: np.random.Generator) -> int:
    """Draw a severity index in 0..2 from the imbalanced class distribution."""
    return int(rng.choice(NUM_SEVERITY_CLASSES, p=np.asarray(_SEVERITY_PROBS, dtype=np.float64)))


def _make_crop_item(
    index: int,
    crop_size: int,
    seed: int,
    guided: bool,
    study_prefix: str,
) -> dict[str, object]:
    """Build one classification item with a learnable severity signal.

    The severity index controls (a) the brightness of a square patch placed in
    the image and (b) the magnitude of the anatomy-prior channels, so a small
    model has a real, learnable correlation between inputs and the target.
    """
    rng = np.random.default_rng(seed + index)
    severity = _sample_severity(rng)
    level_idx = int(rng.integers(0, len(LEVELS)))
    condition_idx = int(rng.integers(0, _NUM_CONDITIONS))

    # Base image: low-amplitude background noise so the patch stands out.
    image = rng.normal(loc=0.1, scale=0.05, size=(3, crop_size, crop_size)).astype(np.float32)

    # Planted bright patch whose mean intensity scales with severity.
    patch = max(2, crop_size // 4)
    p0 = crop_size // 2 - patch // 2
    p1 = p0 + patch
    brightness = 0.25 + 0.3 * float(severity)  # 0.25 / 0.55 / 0.85
    image[:, p0:p1, p0:p1] += brightness
    image = np.clip(image, 0.0, 1.0)

    # Anatomy-prior channels. Always present (zeros when not guided) but always
    # carry the severity signal in their magnitude when guided, so the anatomy
    # branch is informative.
    anatomy = np.zeros((NUM_ANATOMY_PRIOR_CHANNELS, crop_size, crop_size), dtype=np.float32)
    if guided:
        canal = 0.2 + 0.35 * float(severity)  # grows with severity
        band = max(1, crop_size // 8)
        c0 = crop_size // 2 - band // 2
        c1 = c0 + band
        # Channel order is (disc, spinal_canal, vertebra) per ANATOMY_PRIOR_CHANNELS.
        anatomy[0, p0:p1, :] = 0.5  # disc band
        anatomy[1, :, c0:c1] = canal  # central canal column scaled by severity
        anatomy[2, : crop_size // 3, :] = 0.4  # vertebra block

    study_id = f"{study_prefix}_{index:05d}"
    return {
        "image": torch.from_numpy(image),
        "anatomy": torch.from_numpy(anatomy),
        "level_idx": torch.tensor(level_idx, dtype=torch.long),
        "condition_idx": torch.tensor(condition_idx, dtype=torch.long),
        "target": torch.tensor(severity, dtype=torch.long),
        "study_id": study_id,
    }


class SyntheticCropDataset(Dataset):
    """In-memory severity-classification dataset with a learnable planted signal."""

    def __init__(
        self,
        n: int,
        crop_size: int,
        seed: int,
        guided: bool,
        *,
        study_prefix: str = "synthstudy",
    ) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}")
        self.n = int(n)
        self.crop_size = int(crop_size)
        self.seed = int(seed)
        self.guided = bool(guided)
        self.study_prefix = study_prefix

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.n
        if not 0 <= index < self.n:
            raise IndexError(index)
        return _make_crop_item(
            index,
            self.crop_size,
            self.seed,
            self.guided,
            self.study_prefix,
        )


def _make_seg_item(index: int, crop_size: int, seed: int, subject_prefix: str) -> dict[str, object]:
    """Build one segmentation item with geometric vertebra/disc/canal shapes."""
    rng = np.random.default_rng(seed + index)
    h = w = crop_size
    image = rng.normal(loc=0.1, scale=0.04, size=(h, w)).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.int64)

    # Central spinal canal: a thin vertical column (class 3).
    band = max(1, w // 10)
    c0 = w // 2 - band // 2
    c1 = c0 + band
    mask[:, c0:c1] = 3
    image[:, c0:c1] += 0.5

    # A small vertical jitter so shapes are not pixel-identical across items.
    offset = int(rng.integers(0, max(1, h // 8)))
    block = max(2, h // 6)
    y = offset
    toggle = 0
    while y + block <= h:
        if toggle % 2 == 0:
            # Vertebra block (class 1) on either side of the canal.
            mask[y : y + block, :c0] = 1
            mask[y : y + block, c1:] = 1
            image[y : y + block, :c0] += 0.35
            image[y : y + block, c1:] += 0.35
        else:
            # Disc band (class 2), thinner.
            disc = max(1, block // 2)
            mask[y : y + disc, :c0] = 2
            mask[y : y + disc, c1:] = 2
            image[y : y + disc, :c0] += 0.6
            image[y : y + disc, c1:] += 0.6
        toggle += 1
        y += block

    image = np.clip(image, 0.0, 1.0)
    subject_id = f"{subject_prefix}_{index:05d}"
    return {
        "image": torch.from_numpy(image[None, :, :]),
        "mask": torch.from_numpy(mask),
        "subject_id": subject_id,
    }


class SyntheticSegDataset(Dataset):
    """In-memory anatomy-segmentation dataset with learnable geometric shapes."""

    def __init__(
        self,
        n: int,
        crop_size: int,
        seed: int,
        *,
        subject_prefix: str = "synthsubj",
    ) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}")
        self.n = int(n)
        self.crop_size = int(crop_size)
        self.seed = int(seed)
        self.subject_prefix = subject_prefix

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.n
        if not 0 <= index < self.n:
            raise IndexError(index)
        return _make_seg_item(index, self.crop_size, self.seed, self.subject_prefix)


def make_synthetic_classification_data(
    n: int,
    crop_size: int,
    seed: int,
    guided: bool,
) -> tuple[SyntheticCropDataset, SyntheticCropDataset]:
    """Return (train, val) classification datasets with disjoint study ids.

    The two datasets use different study-id prefixes and different rng seeds so
    their items (and therefore their study ids) never overlap.
    """
    n_val = max(1, n // 4)
    n_train = max(1, n - n_val)
    train = SyntheticCropDataset(
        n_train,
        crop_size,
        seed,
        guided,
        study_prefix="synthstudy_train",
    )
    val = SyntheticCropDataset(
        n_val,
        crop_size,
        seed + 100_003,
        guided,
        study_prefix="synthstudy_val",
    )
    return train, val


def make_synthetic_segmentation_data(
    n: int,
    crop_size: int,
    seed: int,
) -> tuple[SyntheticSegDataset, SyntheticSegDataset]:
    """Return (train, val) segmentation datasets with disjoint subject ids."""
    n_val = max(1, n // 4)
    n_train = max(1, n - n_val)
    train = SyntheticSegDataset(
        n_train,
        crop_size,
        seed,
        subject_prefix="synthsubj_train",
    )
    val = SyntheticSegDataset(
        n_val,
        crop_size,
        seed + 100_003,
        subject_prefix="synthsubj_val",
    )
    return train, val
