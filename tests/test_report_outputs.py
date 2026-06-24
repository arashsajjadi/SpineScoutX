"""Intelligence/output audit: model outputs are derived, consistent, and safe."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from spinescoutx.constants import CONDITIONS
from spinescoutx.reporting import finding_graph_schema as fg

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "outputs/real/showcase_reports"


def test_output_is_derived_from_probabilities_not_hardcoded():
    a = fg.build_finding("spinal_canal_stenosis", "l4_l5", [0.8, 0.15, 0.05])
    b = fg.build_finding("spinal_canal_stenosis", "l4_l5", [0.05, 0.15, 0.8])
    assert a["severity_estimate"] == "normal_mild" and b["severity_estimate"] == "severe"
    assert a["probabilities"]["P(severe)"] == 0.05
    assert b["probabilities"]["P(severe)"] == 0.8
    assert a["calibrated_confidence"] == 0.8 and b["calibrated_confidence"] == 0.8


def test_route_matches_condition_for_every_condition():
    expect = {
        "spinal_canal_stenosis": "sagittal_t2",
        "left_neural_foraminal_narrowing": "sagittal_t1",
        "right_neural_foraminal_narrowing": "sagittal_t1",
        "left_subarticular_stenosis": "axial_t2",
        "right_subarticular_stenosis": "axial_t2",
    }
    for cond in CONDITIONS:
        f = fg.build_finding(cond, "l4_l5", [0.6, 0.3, 0.1])
        assert f["view_route"] == expect[cond]
        assert f["crop_provenance"] in fg.ALLOWED_PROVENANCE


def test_review_required_iff_reasons_present():
    f = fg.build_finding("spinal_canal_stenosis", "l4_l5", [0.33, 0.34, 0.33])
    assert f["review_required"] == bool(f["review_reasons"])


def test_validate_catches_tampered_severity():
    f = fg.build_finding("spinal_canal_stenosis", "l4_l5", [0.8, 0.15, 0.05])
    g = fg.build_study_graph("x", split="test", findings=[f], model_version="v")
    g["findings"][0]["severity_estimate"] = "severe"  # tamper (not the argmax)
    with pytest.raises(ValueError):
        fg.validate_finding_graph(g)


def test_markdown_row_reflects_json():
    f1 = fg.build_finding("spinal_canal_stenosis", "l4_l5", [0.05, 0.15, 0.80])
    f2 = fg.build_finding("left_subarticular_stenosis", "l5_s1", [0.7, 0.2, 0.1])
    g = fg.build_study_graph("x", split="test", findings=[f1, f2], model_version="v")
    md = fg.render_markdown(g)
    for f in g["findings"]:
        assert f["severity_estimate"] in md
        assert str(f["probabilities"]["P(severe)"]) in md
        assert f["view_route"] in md


def test_generated_pack_is_valid_and_anonymized_if_present():
    files = sorted(glob.glob(str(PACK / "*.json")))
    if not files:
        pytest.skip("showcase pack not generated (gitignored); logic covered by other tests")
    for fp in files:
        g = json.loads(Path(fp).read_text())
        fg.validate_finding_graph(g)  # re-validate every committed/generated report
        assert g["case_id"].startswith("case_")
        # the filename must be the anonymized case id, not a raw study id
        assert Path(fp).stem == g["case_id"]
        # markdown must exist and reflect the JSON
        md = Path(fp).with_suffix(".md")
        assert md.exists() and g["case_id"] in md.read_text()
