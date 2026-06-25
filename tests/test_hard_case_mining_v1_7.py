"""Unit tests for the v1.7 hard-case mining logic (pure functions, synthetic table)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spinescoutx.data.hard_case_mining import (
    build_signal_table,
    mine_groups,
    priority_score,
    select_review_set,
)

RIGHT = "right_neural_foraminal_narrowing"
LEFT = "left_neural_foraminal_narrowing"
MODEL_COLS = ["p_severe_m1", "p_severe_m2"]


def _row(key, cond, level, sev, dep, m1, m2, split="train"):
    return {
        "key": key, "study_id": key.split("|")[0], "level": level, "condition": cond,
        "side": "right" if cond == RIGHT else "left", "split": split, "severity_index": sev,
        "dep_p0": dep[0], "dep_p1": dep[1], "dep_p2": dep[2],
        "p_severe_m1": m1, "p_severe_m2": m2,
    }  # fmt: skip


def _table():
    rows = [
        # severe FN, confidently normal (deployed says normal, models too)
        _row("s1|l5_s1|" + RIGHT, RIGHT, "l5_s1", 2, [0.8, 0.15, 0.05], 0.04, 0.06),
        # severe, correctly called (control)
        _row("s2|l4_l5|" + RIGHT, RIGHT, "l4_l5", 2, [0.05, 0.1, 0.85], 0.9, 0.8),
        # moderate with severe evidence (borderline)
        _row("s3|l4_l5|" + LEFT, LEFT, "l4_l5", 1, [0.1, 0.4, 0.5], 0.7, 0.65),
        # normal, correctly called (control easy)
        _row("s4|l1_l2|" + LEFT, LEFT, "l1_l2", 0, [0.95, 0.04, 0.01], 0.02, 0.01),
        # high disagreement
        _row("s5|l3_l4|" + RIGHT, RIGHT, "l3_l4", 1, [0.4, 0.35, 0.25], 0.9, 0.05),
    ]
    return pd.DataFrame(rows)


def test_build_signal_table_columns():
    sig = build_signal_table(_table(), MODEL_COLS)
    for c in ["dep_pred", "dep_p_severe", "disagreement", "is_true_severe", "dep_correct"]:
        assert c in sig.columns
    s1 = sig[sig.key.str.startswith("s1")].iloc[0]
    assert s1.is_true_severe and s1.dep_pred == 0  # severe but predicted normal_mild


def test_severe_fn_and_borderline_groups():
    sig = build_signal_table(_table(), MODEL_COLS)
    g = mine_groups(sig)
    fn_keys = set(g["A_severe_fn"].key)
    assert any(k.startswith("s1") for k in fn_keys)  # the severe FN is found
    assert not any(k.startswith("s2") for k in fn_keys)  # correctly-called severe is NOT an FN
    border_keys = set(g["C_moderate_severe_borderline"].key)
    assert any(k.startswith("s3") for k in border_keys)  # moderate w/ severe evidence


def test_priority_prefers_right_foraminal_severe_fn():
    sig = build_signal_table(_table(), MODEL_COLS)
    s1 = sig[sig.key.str.startswith("s1")].iloc[0]  # right-for severe FN, confidently normal
    s2 = sig[sig.key.str.startswith("s2")].iloc[0]  # right-for correct severe
    assert priority_score(s1) > priority_score(s2)


def test_select_review_set_dedups_and_ranks():
    sig = build_signal_table(_table(), MODEL_COLS)
    g = mine_groups(sig)
    caps = {"A_severe_fn": 10, "C_moderate_severe_borderline": 10, "G_control_random_easy": 10}
    sel = select_review_set(sig, g, caps=caps)
    assert sel.key.is_unique
    assert "priority" in sel.columns
    assert (sel.priority.to_numpy() == np.sort(sel.priority.to_numpy())[::-1]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
