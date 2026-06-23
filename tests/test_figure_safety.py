"""Regression tests: synthetic figures must be unmistakably labeled as synthetic.

These guard the Phase-2 safety requirement that a synthetic-smoke figure can never
be passed off as a real RSNA/SPIDER research result.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from spinescoutx.viz.panels import (
    SYNTHETIC_PROVENANCE,
    _stamp,
    figures_from_report,
    provenance_label,
)


def test_provenance_label_flags_synthetic_and_missing() -> None:
    assert provenance_label("synthetic") == SYNTHETIC_PROVENANCE
    assert provenance_label(None) == SYNTHETIC_PROVENANCE
    assert provenance_label("") == SYNTHETIC_PROVENANCE
    assert provenance_label("smoke") == SYNTHETIC_PROVENANCE


def test_provenance_label_clears_for_real_sources() -> None:
    assert provenance_label("rsna") is None
    assert provenance_label("spider") is None


def test_stamp_adds_research_banner_and_synthetic_watermark() -> None:
    fig = plt.figure()
    _stamp(fig, provenance_label("synthetic"))
    texts = [t.get_text() for t in fig.texts]
    assert any("Research-only" in t for t in texts)
    assert any("SYNTHETIC" in t for t in texts)
    plt.close(fig)


def test_stamp_has_no_synthetic_watermark_for_real_source() -> None:
    fig = plt.figure()
    _stamp(fig, provenance_label("rsna"))
    texts = [t.get_text() for t in fig.texts]
    assert any("Research-only" in t for t in texts)
    assert not any("SYNTHETIC" in t for t in texts)
    plt.close(fig)


def test_figures_from_report_synthetic_runs(tmp_path) -> None:
    # 1 finding (toy-card path) + low-N reliability curve (low-N warning path).
    report = {
        "dataset_source": "synthetic",
        "findings": [{"grade": "moderate"}],
        "reliability_curve": {
            "bin_confidence": [0.5, 0.9],
            "bin_accuracy": [0.4, 0.8],
            "bin_count": [3, 2],
        },
        "ece": 0.2,
    }
    paths = figures_from_report(report, tmp_path)
    assert paths and all(p.exists() for p in paths)
    assert any("findings_card" in str(p) for p in paths)
