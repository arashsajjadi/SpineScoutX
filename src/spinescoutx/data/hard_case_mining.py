"""Hard-case mining for label repair (v1.7).

Given a per-finding table of the true RSNA severity + several models' class probabilities, derive
the cases most likely to unlock severe recall via re-annotation: severe false negatives,
confidently-normal severe misses, moderate/severe borderline cases, strong model-disagreement, and
high-uncertainty cases — plus controls. Pure functions on a DataFrame (no I/O, no GPU) so they are
unit-testable. The deployed grader is the *primary* model (defines the deployed prediction);
other models supply the disagreement / ensemble signal. Research-only. Not diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEV_NAMES = {0: "normal_mild", 1: "moderate", 2: "severe"}
RIGHT_FOR = "right_neural_foraminal_narrowing"
LEFT_FOR = "left_neural_foraminal_narrowing"


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    return (-(p * np.log(p)).sum(axis=1)) / np.log(p.shape[1])


def build_signal_table(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Enrich a finding table with mining signals.

    Required columns: ``key, study_id, level, condition, side, split, severity_index`` plus the
    deployed grader's ``dep_p0, dep_p1, dep_p2`` and each model's ``p_severe_<m>`` (``model_cols``).
    """
    out = df.copy()
    p = out[["dep_p0", "dep_p1", "dep_p2"]].to_numpy(dtype=float)
    out["dep_pred"] = p.argmax(1)
    out["dep_p_nm"] = p[:, 0]
    out["dep_p_mod"] = p[:, 1]
    out["dep_p_severe"] = p[:, 2]
    out["dep_entropy"] = _entropy(p)
    out["dep_margin"] = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
    sev = out[model_cols].to_numpy(dtype=float)  # each model's p_severe
    out["ens_p_severe"] = np.nanmean(sev, axis=1)
    out["disagreement"] = np.nanstd(sev, axis=1)
    out["n_models_call_severe"] = (sev >= 0.5).sum(axis=1)
    out["is_true_severe"] = out["severity_index"] == 2
    out["is_true_moderate"] = out["severity_index"] == 1
    out["dep_correct"] = out["dep_pred"] == out["severity_index"]
    return out


def mine_groups(df: pd.DataFrame, *, seed: int = 1337) -> dict[str, pd.DataFrame]:
    """Return {group -> rows}. Groups follow the v1.7 spec (A,B,C,D,F + G controls; E optional)."""
    rng = np.random.default_rng(seed)
    g: dict[str, pd.DataFrame] = {}

    # A. severe false negatives (true severe, deployed not severe) — right-foraminal first
    sev_fn = df[df.is_true_severe & (df.dep_pred != 2)]
    g["A_severe_fn"] = sev_fn.sort_values(["condition", "dep_p_nm"], ascending=[True, False])

    # B. confidently-normal severe miss (true severe, predicted normal_mild, high p_nm, low p_sev)
    g["B_confident_normal_severe_miss"] = df[
        df.is_true_severe & (df.dep_pred == 0) & (df.dep_p_nm >= 0.5) & (df.dep_p_severe <= 0.2)
    ].sort_values("dep_p_nm", ascending=False)

    # C. moderate/severe borderline (label moderate w/ high p_severe OR label severe w/ high p_mod
    #    OR strong cross-model disagreement on these two classes)
    border = df[
        (df.is_true_moderate & (df.dep_p_severe >= 0.30))
        | (df.is_true_severe & (df.dep_p_mod >= 0.30))
        | (df.severity_index.isin([1, 2]) & (df.disagreement >= 0.20))
    ]
    g["C_moderate_severe_borderline"] = border.sort_values("disagreement", ascending=False)

    # D. strong model disagreement (any class), ranked
    g["D_model_disagreement"] = df.sort_values("disagreement", ascending=False).head(400)

    # F. low route quality / high uncertainty (proxy: high entropy + small margin)
    g["F_high_uncertainty"] = df[(df.dep_entropy >= 0.7) | (df.dep_margin <= 0.15)].sort_values(
        "dep_entropy", ascending=False
    )

    # G. controls — correct severe, correct non-severe, random easy (avoid biased review packs)
    correct_sev = df[df.is_true_severe & df.dep_correct]
    correct_nonsev = df[(~df.is_true_severe) & df.dep_correct & (df.dep_margin >= 0.5)]
    easy_pool = df[df.dep_correct & (df.dep_margin >= 0.6)]
    n_rand = min(60, len(easy_pool))
    rand_easy = easy_pool.iloc[rng.permutation(len(easy_pool))[:n_rand]] if n_rand else easy_pool
    g["G_control_correct_severe"] = correct_sev
    g["G_control_correct_nonsevere"] = correct_nonsev.head(80)
    g["G_control_random_easy"] = rand_easy
    return g


def priority_score(row) -> float:
    """Re-annotation priority (higher = more likely to unlock severe recall)."""
    s = 0.0
    if row.get("is_true_severe") and row.get("dep_pred") != 2:
        s += 3.0  # severe FN
        if row.get("dep_pred") == 0:
            s += 1.5  # confidently normal/mild
        s += float(row.get("dep_p_nm", 0.0))
    s += 2.0 * float(row.get("disagreement", 0.0))
    if row.get("condition") == RIGHT_FOR:
        s += 1.0  # right-foraminal priority
    if str(row.get("level")) in ("l4_l5", "l5_s1"):
        s += 0.5  # weakest levels
    return s


def select_review_set(
    df: pd.DataFrame, groups: dict[str, pd.DataFrame], *, caps: dict
) -> pd.DataFrame:
    """Build the de-duplicated review set with per-bucket caps + priority ranking."""
    picks = []
    for name, cap in caps.items():
        if name in groups and cap > 0:
            picks.append(groups[name].head(cap).assign(_group=name))
    if not picks:
        return df.head(0)
    sel = pd.concat(picks, ignore_index=True).drop_duplicates("key")
    sel = sel.assign(priority=sel.apply(priority_score, axis=1))
    return sel.sort_values("priority", ascending=False).reset_index(drop=True)
