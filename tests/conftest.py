"""Shared synthetic fixtures for the SpineScoutX test suite.

Every fixture here is deterministic (seeded ``numpy.random.default_rng``) and uses
tiny sizes so the whole suite stays fast on CPU with no real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spinescoutx.constants import NUM_SEVERITY_CLASSES, SEVERE_INDEX


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the suite."""
    config.addinivalue_line("markers", "slow: end-to-end synthetic training smoke tests")


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic numpy random generator."""
    return np.random.default_rng(1337)


@pytest.fixture
def repo_root_path() -> Path:
    """Absolute path to the repository root (parent of the tests directory)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def perfect_probs() -> tuple[np.ndarray, np.ndarray]:
    """(y_true, probs) where probs put ~all mass on the true class."""
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    probs = np.full((y_true.size, NUM_SEVERITY_CLASSES), 1e-9, dtype=np.float64)
    probs[np.arange(y_true.size), y_true] = 1.0 - 2e-9
    return y_true, probs


@pytest.fixture
def overconfident_logits() -> tuple[np.ndarray, np.ndarray]:
    """Over-confident logits + labels: predictions are right ~half the time.

    Logits are large in magnitude (so softmax confidence is high) but the argmax
    only matches the label for half the samples, which makes the classifier badly
    calibrated and leaves room for temperature scaling to reduce the NLL.
    """
    rng = np.random.default_rng(7)
    n = 60
    labels = rng.integers(0, NUM_SEVERITY_CLASSES, size=n).astype(np.int64)
    logits = np.zeros((n, NUM_SEVERITY_CLASSES), dtype=np.float64)
    for i in range(n):
        # Make the model confidently predict label i for the first half and a
        # deterministically wrong class for the second half.
        if i % 2 == 0:
            peak = int(labels[i])
        else:
            peak = int((labels[i] + 1) % NUM_SEVERITY_CLASSES)
        logits[i, peak] = 8.0
    return logits, labels


@pytest.fixture
def severe_index() -> int:
    """The integer index of the severe severity class."""
    return SEVERE_INDEX


# --- Shared synthetic RSNA fixtures (real DICOMs via pydicom) --------------------
RSNA_STUDIES = ["1001", "1002"]
RSNA_SERIES = "9001"
RSNA_INSTANCES = [1, 2, 3, 4, 5]


def write_synthetic_dicom(path: Path, arr: np.ndarray) -> None:
    """Write a minimal valid MR DICOM (pydicom imported lazily)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = arr.astype(np.uint16).tobytes()
    ds.save_as(str(path), little_endian=True, implicit_vr=False)


def make_synthetic_rsna_root(root: Path) -> None:
    """Create a tiny but complete RSNA tree (DICOMs + the three CSVs)."""
    import pandas as pd

    images = root / "train_images"
    g = np.random.default_rng(0)
    for study in RSNA_STUDIES:
        sdir = images / study / RSNA_SERIES
        sdir.mkdir(parents=True)
        for inst in RSNA_INSTANCES:
            write_synthetic_dicom(
                sdir / f"{inst}.dcm", (g.random((64, 64)) * 1000).astype(np.uint16)
            )

    pd.DataFrame(
        {
            "study_id": RSNA_STUDIES,
            "spinal_canal_stenosis_l1_l2": ["Normal/Mild", "Severe"],
            "left_neural_foraminal_narrowing_l1_l2": ["Moderate", "Normal/Mild"],
        }
    ).to_csv(root / "train.csv", index=False)

    coord_rows = []
    for study in RSNA_STUDIES:
        coord_rows.append([study, RSNA_SERIES, 3, "Spinal Canal Stenosis", "L1/L2", 32, 32])
        coord_rows.append(
            [study, RSNA_SERIES, 3, "Left Neural Foraminal Narrowing", "L1/L2", 20, 40]
        )
    pd.DataFrame(
        coord_rows,
        columns=["study_id", "series_id", "instance_number", "condition", "level", "x", "y"],
    ).to_csv(root / "train_label_coordinates.csv", index=False)

    pd.DataFrame(
        [[s, RSNA_SERIES, "Sagittal T2/STIR"] for s in RSNA_STUDIES],
        columns=["study_id", "series_id", "series_description"],
    ).to_csv(root / "train_series_descriptions.csv", index=False)
