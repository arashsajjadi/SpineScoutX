"""Unit tests for the LSS-MRI AISSLab foraminal adapter (v1.6, Plan A).

Synthetic PASCAL-VOC XML + PNG fixtures (no real dataset needed) so the adapter's parsing, grade/
side/level mapping, and 2.5D cropping are verified in CI. Research-only.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spinescoutx.data.lss_aisslab import (
    LSS_TO_RSNA_SEVERITY,
    crop_lss_25d,
    normalize_level,
    parse_lss_xml,
)

_XML = """<?xml version='1.0' encoding='utf-8'?>
<annotation>
  <filename>IM000003.png</filename>
  <size><width>512</width><height>512</height><depth>1</depth></size>
  <object><name>RFS3</name><bndbox><xmin>247</xmin><ymin>203</ymin>
    <xmax>279</xmax><ymax>259</ymax></bndbox><level>L4-L5</level></object>
  <object><name>LFS0</name><bndbox><xmin>120</xmin><ymin>200</ymin>
    <xmax>150</xmax><ymax>250</ymax></bndbox><level>L4-L5</level></object>
  <object><name>person</name><bndbox><xmin>1</xmin><ymin>1</ymin>
    <xmax>2</xmax><ymax>2</ymax></bndbox><level>L4-L5</level></object>
</annotation>"""


def _make_patient(tmp_path):
    pdir = tmp_path / "Foramina_Detection" / "0001"
    pdir.mkdir(parents=True)
    for i in (2, 3, 4):
        Image.fromarray((np.random.rand(512, 512) * 255).astype(np.uint8)).save(
            pdir / f"IM{i:06d}.png"
        )
    (pdir / "IM000003.xml").write_text(_XML)
    return pdir / "IM000003.xml"


def test_grade_mapping_is_clinical():
    assert LSS_TO_RSNA_SEVERITY == {0: 0, 1: 0, 2: 1, 3: 2}


def test_normalize_level():
    assert normalize_level("L4-L5") == "l4_l5"
    assert normalize_level("L5-S1") == "l5_s1"
    assert normalize_level("T12-L1") is None


def test_parse_xml_decodes_side_grade_level(tmp_path):
    boxes = parse_lss_xml(_make_patient(tmp_path))
    assert len(boxes) == 2  # 'person' object skipped
    by_side = {b.side: b for b in boxes}
    assert by_side["right"].grade == 3 and by_side["right"].severity_index == 2  # severe
    assert by_side["right"].level == "l4_l5"
    assert by_side["right"].bbox == (247, 203, 279, 259)
    assert by_side["left"].grade == 0 and by_side["left"].severity_index == 0  # normal_mild


def test_crop_is_rsna_compatible(tmp_path):
    boxes = parse_lss_xml(_make_patient(tmp_path))
    crop = crop_lss_25d(boxes[0], crop_size=224)
    assert crop is not None
    assert crop.shape == (3, 224, 224)
    assert crop.dtype == np.float32
    assert float(crop.min()) >= 0.0 and float(crop.max()) <= 1.0


def test_malformed_xml_returns_empty(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<annotation><object></annotation")  # malformed
    assert parse_lss_xml(bad) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
