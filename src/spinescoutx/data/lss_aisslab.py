"""LSS-MRI AISSLab external foraminal dataset adapter (v1.6).

Parses the LSS-MRI AISSLab sagittal lumbar-spine dataset (Mendeley rgb77xm3jf, CC BY 4.0,
non-commercial research) into RSNA-compatible foraminal grading crops for **encoder pretraining /
joint training**. Each PASCAL-VOC XML annotates one rendered sagittal PNG (512x512, grayscale)
with foraminal-stenosis boxes whose ``<name>`` encodes side+grade (``RFS0``/``LFS3`` -> Right/Left
Foraminal Stenosis, grade 0=Normal,1=Mild,2=Moderate,3=Severe) and ``<level>`` = L1-L2..L5-S1.

Grade map to the RSNA 3-class scheme: Normal/Mild -> 0 (normal_mild), Moderate -> 1, Severe -> 2.
Crops are 2.5D (the box applied to the annotated PNG + its two neighbours), matching the RSNA
sagittal-T1 foraminal grader input (3,224,224). External data is gitignored; nothing committed.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# LSS grade (0-3) -> RSNA severity index (0=normal_mild, 1=moderate, 2=severe)
LSS_TO_RSNA_SEVERITY = {0: 0, 1: 0, 2: 1, 3: 2}
LSS_GRADE_NAME = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}
LSS_SIDE = {"RFS": "right", "LFS": "left"}
LSS_CONDITION = {
    "right": "right_neural_foraminal_narrowing",
    "left": "left_neural_foraminal_narrowing",
}
RSNA_SEVERITY_NAME = {0: "normal_mild", 1: "moderate", 2: "severe"}
_NAME_RE = re.compile(r"^(RFS|LFS)\s*([0-3])$", re.IGNORECASE)
_VALID_LEVELS = {"l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"}


def normalize_level(raw: str) -> str | None:
    """``L4-L5`` / ``L5-S1`` -> ``l4_l5`` / ``l5_s1``; returns None if not a lumbar level."""
    s = raw.strip().lower().replace("-", "_").replace(" ", "")
    return s if s in _VALID_LEVELS else None


@dataclass
class LssBox:
    patient: str
    png_path: Path
    slice_name: str  # e.g. IM000003
    side: str  # left | right
    level: str  # l1_l2..l5_s1
    grade: int  # 0..3 (LSS)
    severity_index: int  # 0..2 (RSNA)
    bbox: tuple[int, int, int, int]  # xmin,ymin,xmax,ymax (in PNG pixels)


def parse_lss_xml(xml_path: Path) -> list[LssBox]:
    """Parse one PASCAL-VOC XML into LssBox records (skips non-foraminal / non-lumbar objects)."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    patient = xml_path.parent.name
    png_path = xml_path.with_suffix(".png")
    slice_name = xml_path.stem
    out: list[LssBox] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        m = _NAME_RE.match(name)
        if m is None:
            continue
        side = LSS_SIDE[m.group(1).upper()]
        grade = int(m.group(2))
        level = normalize_level(obj.findtext("level") or "")
        if level is None:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            box = (
                int(float(bb.findtext("xmin"))),
                int(float(bb.findtext("ymin"))),
                int(float(bb.findtext("xmax"))),
                int(float(bb.findtext("ymax"))),
            )
        except (TypeError, ValueError):
            continue
        out.append(
            LssBox(
                patient=patient,
                png_path=png_path,
                slice_name=slice_name,
                side=side,
                level=level,
                grade=grade,
                severity_index=LSS_TO_RSNA_SEVERITY[grade],
                bbox=box,
            )
        )
    return out


def iter_lss_boxes(detection_root: Path):
    """Yield every LssBox under ``Foramina_Detection/`` (one XML may carry several boxes)."""
    for xml_path in sorted(detection_root.rglob("*.xml")):
        yield from parse_lss_xml(xml_path)


def _slice_index(slice_name: str) -> int | None:
    m = re.search(r"(\d+)$", slice_name)
    return int(m.group(1)) if m else None


def crop_lss_25d(box: LssBox, *, crop_size: int = 224, pad_frac: float = 0.6) -> np.ndarray | None:
    """2.5D crop (3, crop_size, crop_size) in [0,1]: the box (padded) on the annotated PNG and its
    two slice neighbours (same in-plane box), mirroring the RSNA sagittal-T1 2.5D foraminal crop."""
    import cv2
    from PIL import Image

    idx = _slice_index(box.slice_name)
    if idx is None:
        return None
    folder = box.png_path.parent
    width = box.png_path.stem[: -len(str(idx))] if str(idx) in box.png_path.stem else "IM"
    xmin, ymin, xmax, ymax = box.bbox
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 1 or bh <= 1:
        return None
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    chans = []
    for di in (-1, 0, 1):
        nbr = folder / f"{width}{idx + di:06d}.png"
        path = nbr if nbr.exists() else box.png_path
        try:
            img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        except (OSError, ValueError):
            return None
        h, w = img.shape
        x0, x1 = max(0, xmin - px), min(w, xmax + px)
        y0, y1 = max(0, ymin - py), min(h, ymax + py)
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        patch = cv2.resize(patch, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        mn, mx = float(patch.min()), float(patch.max())
        chans.append((patch - mn) / (mx - mn) if mx > mn else patch * 0.0)
    return np.stack(chans).astype(np.float32)  # (3, H, W)
