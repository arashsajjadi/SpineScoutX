"""Tests for `spinescoutx doctor --data` dataset-readiness reporting (no data needed)."""

from __future__ import annotations

from spinescoutx.cli import _dataset_readiness, main


def test_doctor_data_runs_with_no_data(tmp_path) -> None:
    rc = main(
        [
            "doctor",
            "--data",
            "--rsna-root",
            str(tmp_path / "rsna"),
            "--spider-root",
            str(tmp_path / "spider"),
        ]
    )
    assert rc == 0


def test_dataset_readiness_reports_missing(tmp_path) -> None:
    rep = _dataset_readiness(str(tmp_path / "rsna"), str(tmp_path / "spider"))
    assert rep["rsna"]["exists"] is False
    assert rep["spider"]["exists"] is False
    # Each report lists exactly what is missing so the blocker is actionable.
    assert rep["rsna"]["missing"]
    assert rep["spider"]["missing"]
