"""Tests for foraminal sagittal-T1 side-aware localization helpers + leakage guard."""

from __future__ import annotations

import inspect

from spinescoutx.data import foraminal_localize as fl


def test_foraminal_constant():
    assert fl.FORAMINAL == (
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
    )


def test_side_candidate_instances_split_by_laterality():
    # +x = patient LEFT; lr_sorted is ascending x (right -> left)
    lr = [(i, float(i)) for i in range(10)]  # x increases with instance
    left = fl.side_candidate_instances(lr, "left")
    right = fl.side_candidate_instances(lr, "right")
    assert left and right
    # left candidates are the high-x (high instance) end; right the low-x end
    assert min(left) > max(right) or (max(left) > max(right) and min(right) < min(left))
    assert max(left) >= 5  # left draws from the upper half
    assert min(right) <= 4  # right from the lower half


def test_side_candidate_instances_empty():
    assert fl.side_candidate_instances([], "left") == []


def test_auto_crops_never_reads_gt_coordinates():
    """The auto foraminal crop path must use labels (severity) but NEVER coordinates."""
    src = inspect.getsource(fl.prepare_rsna_foraminal_auto_crops)
    assert "load_coordinates" not in src
    assert "train_label_coordinates" not in src
    assert "load_labels" in src  # severity targets are allowed


def test_localizer_data_prep_uses_coordinates_for_supervision_only():
    # Training-data prep is allowed to read GT coordinates (supervision); auto path is not.
    src = inspect.getsource(fl.prepare_foraminal_localizer_data)
    assert "load_coordinates" in src
