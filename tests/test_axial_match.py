"""Tests for axial-T2 level matching geometry (sagittal-disc-z -> axial-slice-z)."""

from __future__ import annotations

import numpy as np

from spinescoutx.data import axial_match as ax


def test_subarticular_constant():
    assert ax.SUBARTICULAR == ("left_subarticular_stenosis", "right_subarticular_stenosis")


def test_pixel_to_patient_formula():
    # IPP origin 0; column dir = +y, row dir = -z; PixelSpacing=[row=2, col=3].
    geom = {
        "ipp": np.zeros(3),
        "iop": np.array([0, 1, 0, 0, 0, -1], dtype=float),
        "ps": np.array([2.0, 3.0]),
    }
    p = ax.pixel_to_patient(geom, x_col=4, y_row=5)
    # P = IPP + Xdir*(col*colSpacing) + Ydir*(row*rowSpacing)
    #   = [0,1,0]*(4*3) + [0,0,-1]*(5*2) = [0,12,-10]
    assert np.allclose(p, [0.0, 12.0, -10.0])


def test_disc_level_z_picks_z_component():
    geom = {
        "ipp": np.array([0.0, 0.0, 100.0]),
        "iop": np.array([1, 0, 0, 0, 0, -1], dtype=float),  # row dir -> -z (S-I)
        "ps": np.array([2.0, 1.0]),
    }
    pts = np.array([[0, 0], [0, 5], [0, 10], [0, 15], [0, 20]], dtype=float)
    z = ax.disc_level_z(geom, pts)
    # z = 100 + (-1)*(row*2): rows 0,5,10,15,20 -> 100, 90, 80, 70, 60
    assert np.allclose([z["l1_l2"], z["l2_l3"], z["l5_s1"]], [100.0, 90.0, 60.0])


def test_match_levels_to_axial_nearest_z():
    level_z = {"l1_l2": 100.0, "l3_l4": 80.0, "l5_s1": 60.0}
    axial_z = {1: 102.0, 2: 90.0, 3: 79.0, 4: 61.0}  # instance -> z
    m = ax.match_levels_to_axial(level_z, axial_z, top_k=1)
    assert m["l1_l2"] == [1]  # 102 nearest 100
    assert m["l3_l4"] == [3]  # 79 nearest 80
    assert m["l5_s1"] == [4]  # 61 nearest 60


def test_match_levels_top_k():
    level_z = {"l4_l5": 70.0}
    axial_z = {1: 100.0, 2: 72.0, 3: 68.0, 4: 50.0}
    m = ax.match_levels_to_axial(level_z, axial_z, top_k=2)
    assert set(m["l4_l5"]) == {2, 3}  # two nearest to 70
