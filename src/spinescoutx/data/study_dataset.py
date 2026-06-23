"""Study-grouped dataset for the E3 multi-view anatomy-graph reasoner.

Groups a canal-stenosis crop manifest by study and yields, per study, the (≤5)
disc-level nodes padded to ``NUM_LEVELS`` with a presence mask. Each node carries
the image crop, the anatomy-prior masks, the morphology feature vector, the level
index and the severity target (absent levels target = ``IGNORE_INDEX``).

Image + anatomy come from the same cached caches the crop classifiers use, keyed by
``crop_path``; morphology features are computed on the fly from the anatomy mask, so
no extra join is needed. Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..constants import LEVEL_TO_INDEX, LEVELS
from ..features.morphology import NUM_FEATURES, morphology_features
from .crops import read_manifest
from .datasets import _load_array, _to_3chw

NUM_LEVELS = len(LEVELS)
IGNORE_INDEX = -100


def _morph_vector(anatomy_mask: np.ndarray) -> np.ndarray:
    from ..features.morphology import FEATURE_NAMES

    f = morphology_features(anatomy_mask)
    return np.asarray([f[n] for n in FEATURE_NAMES], dtype=np.float32)


class StudyCanalDataset(Dataset):
    """One item per study: padded ``(NUM_LEVELS, …)`` canal-stenosis evidence nodes."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        cache_root: str | Path,
        anatomy_cache_root: str | Path,
        crop_size: int = 224,
        condition: str = "spinal_canal_stenosis",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.anatomy_cache_root = Path(anatomy_cache_root)
        self.crop_size = int(crop_size)
        df = manifest_df[manifest_df["condition"] == condition].copy()
        df = df[df["severity_index"].isin([0, 1, 2])]
        df["study_id"] = df["study_id"].astype(str)
        self.studies: list[str] = sorted(df["study_id"].unique())
        self._by_study = dict(tuple(df.groupby("study_id")))

    def __len__(self) -> int:
        return len(self.studies)

    def study_targets(self, index: int) -> list[int]:
        """Severity targets of the present levels (for class-weight estimation)."""
        g = self._by_study[self.studies[index]]
        return [int(v) for v in g["severity_index"].tolist()]

    def __getitem__(self, index: int) -> dict[str, object]:
        study = self.studies[index]
        g = self._by_study[study]

        images = np.zeros((NUM_LEVELS, 3, self.crop_size, self.crop_size), dtype=np.float32)
        anatomy = np.zeros((NUM_LEVELS, 3, self.crop_size, self.crop_size), dtype=np.float32)
        morph = np.zeros((NUM_LEVELS, NUM_FEATURES), dtype=np.float32)
        targets = np.full((NUM_LEVELS,), IGNORE_INDEX, dtype=np.int64)
        level_idx = np.arange(NUM_LEVELS, dtype=np.int64)
        mask = np.zeros((NUM_LEVELS,), dtype=bool)

        for r in g.itertuples():
            li = LEVEL_TO_INDEX[r.level]
            img = _to_3chw(_load_array(self.cache_root / r.crop_path), self.crop_size)
            anat_path = self.anatomy_cache_root / r.crop_path
            anat = (
                _to_3chw(np.load(anat_path).astype(np.float32), self.crop_size)
                if anat_path.exists()
                else np.zeros((3, self.crop_size, self.crop_size), dtype=np.float32)
            )
            images[li] = img
            anatomy[li] = anat
            morph[li] = _morph_vector(anat)
            targets[li] = int(r.severity_index)
            mask[li] = True

        return {
            "images": torch.from_numpy(images),
            "anatomy": torch.from_numpy(anatomy),
            "morph": torch.from_numpy(morph),
            "level_idx": torch.from_numpy(level_idx),
            "target": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "study_id": study,
        }


def build_study_loaders(
    manifest_path: str | Path,
    cache_root: str | Path,
    anatomy_cache_root: str | Path,
    *,
    crop_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: str = "train",
    val_split: str = "val",
):
    """Return ``(train_loader, val_loader, train_ds)`` over a canal crop manifest."""
    from torch.utils.data import DataLoader

    man = read_manifest(Path(manifest_path))
    train_df = man[man["split"] == train_split] if "split" in man else man
    val_df = man[man["split"] == val_split] if "split" in man else man
    train_ds = StudyCanalDataset(train_df, cache_root, anatomy_cache_root, crop_size)
    val_ds = StudyCanalDataset(val_df, cache_root, anatomy_cache_root, crop_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, train_ds
