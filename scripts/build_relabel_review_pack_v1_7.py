#!/usr/bin/env python3
"""Build the LOCAL-ONLY hard-case radiology review pack (v1.7).

For each mined hard case, renders a second-read panel (the 2.5D foraminal crop's adjacent
parasagittal slices + the deployed severity bars + all candidate-model p_severe + the reason it was
selected) and emits a review CSV, a JSONL, and a browsable HTML viewer. **The entire output folder
`review_packs/v1_7_hard_cases/` is gitignored / local-only** — it contains imaging pixels and must
never be committed. Only a pixel-free summary (counts/IDs) is committed elsewhere.

Research-only. Not diagnostic. The review collects expert labels; it does not make any diagnosis.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
REVIEW_SET = ROOT / "outputs/real/v1_7_review_set.parquet"
PACK = ROOT / "review_packs/v1_7_hard_cases"
SEV = {0: "normal_mild", 1: "moderate", 2: "severe"}
REVIEW_LABELS = [
    "normal_mild", "moderate", "severe",
    "ambiguous_moderate_severe", "insufficient_evidence", "exclude_from_training",
]  # fmt: skip
QUESTIONS = [
    "target_truly_severe", "if_not_severe_grade", "is_ambiguous", "evidence_insufficient",
    "side_level_correct", "exclude_from_training", "reviewer_note",
]  # fmt: skip


def case_id(key: str) -> str:
    return "case_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _model_cols(df):
    return [c for c in df.columns if c.startswith("p_severe_")]


def render_panel(row, crop, out_png, model_cols):
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    titles = ["slice -1", "slice (center)", "slice +1"]
    for i in range(3):
        ax[i].imshow(crop[i], cmap="gray", vmin=0, vmax=1)
        ax[i].set_title(titles[i], fontsize=9)
        ax[i].axis("off")
    probs = [row["dep_p0"], row["dep_p1"], row["dep_p2"]]
    ax[3].bar(["nm", "mod", "sev"], probs, color=["#4c72b0", "#dd8452", "#c44e52"])
    for tag in model_cols:
        ax[3].scatter([2], [row[tag]], s=18, color="black", zorder=3)
    ax[3].set_ylim(0, 1)
    ax[3].set_title("deployed probs + model p_severe", fontsize=8)
    sub = "" if not pd.notna(row.get("priority")) else f" · prio {row['priority']:.2f}"
    fig.suptitle(
        f"{case_id(row['key'])} · {row['condition'].split('_')[0]}-foraminal {row['side']} "
        f"{row['level']} · RSNA label={SEV[int(row['severity_index'])]} · "
        f"deployed={SEV[int(np.argmax(probs))]} (p_sev {probs[2]:.2f}) · {row['_group']}{sub}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def build(limit: int = 0) -> int:
    if not REVIEW_SET.exists():
        raise SystemExit(f"missing {REVIEW_SET}; run mine_hard_cases_v1_7.py first")
    rev = pd.read_parquet(REVIEW_SET)
    man = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    man["study_id"] = man.study_id.astype(str)
    man["key"] = man.study_id + "|" + man.level.astype(str) + "|" + man.condition
    cp = man.drop_duplicates("key").set_index("key")["crop_path"].to_dict()
    if limit:
        rev = rev.head(limit)
    PACK.mkdir(parents=True, exist_ok=True)
    (PACK / "panels").mkdir(exist_ok=True)
    model_cols = _model_cols(rev)
    rows_csv, items, cards = [], [], []
    for d in rev.to_dict("records"):
        cid = case_id(d["key"])
        png = f"panels/{cid}.png"
        crop_path = cp.get(d["key"])
        ok = crop_path is not None and (RSNA_CACHE / crop_path).exists()
        if ok:
            crop = np.load(RSNA_CACHE / crop_path).astype(np.float32)
            render_panel(d, crop, PACK / png, model_cols)
        base = {
            "case_id": cid, "key": d["key"], "study_id": d["study_id"],
            "finding": d["condition"], "side": d["side"], "level": d["level"],
            "split": d["split"], "current_rsna_label": SEV[int(d["severity_index"])],
            "deployed_pred": SEV[int(np.argmax([d["dep_p0"], d["dep_p1"], d["dep_p2"]]))],
            "deployed_p_severe": round(float(d["dep_p2"]), 4),
            "deployed_p_normal_mild": round(float(d["dep_p0"]), 4),
            "deployed_entropy": round(float(d["dep_entropy"]), 4),
            "model_disagreement": round(float(d["disagreement"]), 4),
            "reason_selected": d["_group"], "priority": round(float(d["priority"]), 4),
        }  # fmt: skip
        base.update({t: round(float(d[t]), 4) for t in model_cols if pd.notna(d[t])})
        rows_csv.append({**base, **dict.fromkeys(QUESTIONS, "")})
        items.append({**base, "panel": png if ok else None, "review_labels_allowed": REVIEW_LABELS})
        cards.append((base, png if ok else None))
    pd.DataFrame(rows_csv).to_csv(PACK / "review_sheet.csv", index=False)
    with open(PACK / "review_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    _write_html(cards, model_cols)
    print(f"[review-pack] {len(cards)} cases -> {PACK} (LOCAL-ONLY, gitignored)")
    print("  review_sheet.csv · review_items.jsonl · index.html · panels/*.png")
    return 0


def _write_html(cards, model_cols):
    parts = [
        "<!doctype html><meta charset=utf-8><title>v1.7 hard-case review (LOCAL-ONLY)</title>",
        "<style>body{font-family:sans-serif;margin:18px}"
        ".card{border:1px solid #ccc;border-radius:8px;padding:10px;margin:12px 0}"
        ".meta{font-size:13px;color:#333}img{max-width:1000px}select,input{margin:2px}</style>",
        "<h2>SpineScoutX v1.7 — hard-case re-annotation (research-only, not diagnostic)</h2>",
        "<p><b>Local-only.</b> Pick the true severity for the named finding; flag ambiguity / "
        "insufficient evidence / exclude. Save answers into "
        "<code>review_sheet_reviewed.csv</code>.</p>",
    ]
    for base, png in cards:
        opts = "".join(f"<option>{lab}</option>" for lab in ["", *REVIEW_LABELS])
        ms = " · ".join(f"{t.replace('p_severe_', '')} {base.get(t, '—')}" for t in model_cols)
        parts.append(
            f"<div class=card><div class=meta><b>{html.escape(base['case_id'])}</b> · "
            f"{html.escape(base['finding'])} {base['side']} {base['level']} · "
            f"RSNA <b>{base['current_rsna_label']}</b> · deployed <b>{base['deployed_pred']}</b> "
            f"(p_sev {base['deployed_p_severe']}, p_nm {base['deployed_p_normal_mild']}) · "
            f"disagreement {base['model_disagreement']} · models[{html.escape(ms)}] · "
            f"reason {base['reason_selected']} · prio {base['priority']}</div>"
            + (f"<img src='{png}'>" if png else "<i>panel unavailable</i>")
            + f"<div>true severity: <select name='{base['case_id']}_label'>{opts}</select> "
            f"ambiguous<input type=checkbox> insufficient<input type=checkbox> "
            f"exclude<input type=checkbox> note<input type=text size=40></div></div>"
        )
    (PACK / "index.html").write_text("\n".join(parts))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all selected cases")
    return build(ap.parse_args().limit)


if __name__ == "__main__":
    raise SystemExit(main())
