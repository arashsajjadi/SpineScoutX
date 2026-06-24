"""Verify committed model-output showcase assets exist, are lightweight, and PNG."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "docs/assets/showcase"

EXPECTED = [
    "case_canal_severe_card.png",
    "case_foraminal_left_card.png",
    "case_foraminal_right_hard_card.png",
    "case_subarticular_left_card.png",
    "case_subarticular_right_card.png",
    "case_review_required_card.png",
    "case_mostly_normal_card.png",
    "finding_graph_example.png",
    "report_schema_visual.png",
]


@pytest.mark.parametrize("name", EXPECTED)
def test_showcase_card_exists_and_is_lightweight_png(name):
    p = SHOWCASE / name
    assert p.exists(), f"missing committed showcase asset: {name}"
    size = p.stat().st_size
    assert 1_000 < size < 500_000, f"{name} size {size} out of bounds (expected lightweight PNG)"
    with p.open("rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n", f"{name} is not a valid PNG"


def test_no_dicom_or_npy_committed_under_assets():
    bad = [
        p
        for p in SHOWCASE.rglob("*")
        if p.suffix.lower() in {".dcm", ".npy", ".nii", ".gz", ".pt", ".ckpt"}
    ]
    assert not bad, f"forbidden artifacts committed under showcase: {bad}"
