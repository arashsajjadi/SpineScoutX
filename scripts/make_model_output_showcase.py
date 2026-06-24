#!/usr/bin/env python3
"""Generate the real model-output showcase pack from locked-test predictions.

Runs the deployable grader per condition (the router) on the auto (real-inference)
locked-test crops, builds validated study-level finding graphs (schema v4), and renders
JSON + Markdown + a structured **finding-graph card** (matplotlib panel of the model's
output — severity, P(severe) bars, confidence, flags, route, review reasons — NOT DICOM
pixels). Curated cases are selected by category (severe, review_required, hard, multi-
finding). Committed cards live in docs/assets/showcase/ (anonymized, lightweight); the full
pack is gitignored under outputs/real/showcase_reports/. No GT at inference; no identifiers.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spinescoutx.constants import SEVERITIES
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.reporting import finding_graph_schema as fg
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
PACK = ROOT / "outputs/real/showcase_reports"
ASSETS = ROOT / "docs/assets/showcase"
DOC = ROOT / "docs/run_logs/model_output_showcase.md"
TMP = Path(
    "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
    "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_show.parquet"
)
MODEL_VERSION = "v1.0-auto-robust-five-finding"
# condition -> (deployable run, comparison run, auto cache)
ROUTES = {
    "spinal_canal_stenosis": (
        "v1_canal_auto_robust",
        "v1_canal_oracle_ctrl",
        "rsna_auto_canal_all",
    ),
    "left_neural_foraminal_narrowing": (
        "v1_foraminal_oracle_ctrl",
        "v1_foraminal_auto_robust",
        "rsna_auto_foraminal",
    ),
    "right_neural_foraminal_narrowing": (
        "v1_foraminal_oracle_ctrl",
        "v1_foraminal_auto_robust",
        "rsna_auto_foraminal",
    ),
    "left_subarticular_stenosis": (
        "v1_subarticular_auto_robust",
        "v1_subarticular_oracle_ctrl",
        "rsna_auto_subarticular",
    ),
    "right_subarticular_stenosis": (
        "v1_subarticular_auto_robust",
        "v1_subarticular_oracle_ctrl",
        "rsna_auto_subarticular",
    ),
}
ROUTE_COLOR = {"sagittal_t2": "#2a9d8f", "sagittal_t1": "#457b9d", "axial_t2": "#9d4edd"}
SEV_COLOR = {"normal_mild": "#e9f5e9", "moderate": "#fff3cd", "severe": "#f8d7da"}


def _test_preds(cond, device):
    dep, comp, cache = ROUTES[cond]
    cache_p = ROOT / "data/cache" / cache
    man = read_manifest(cache_p / "manifest.parquet")
    man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
    man["study_id"] = man.study_id.astype(str)
    split_map = load_splits_v1(SPLITS)
    man = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
    if man.empty:
        return {}
    man.to_parquet(TMP)
    dep_p = collect_probs(ROOT / "runs" / dep, TMP, cache_p, device)
    comp_p = collect_probs(ROOT / "runs" / comp, TMP, cache_p, device)
    out = {}
    for key, (y, probs) in dep_p.items():
        study, level = key.split("|")
        comp_argmax = int(np.argmax(comp_p[key][1])) if key in comp_p else int(np.argmax(probs))
        out[(study, level)] = {
            "condition": cond,
            "level": level,
            "y": int(y),
            "probs": probs.tolist(),
            "disagree": int(np.argmax(probs)) != comp_argmax,
        }
    return out


def _axial_scores(studies, device):
    """{study -> {level -> scorer conf}} for given subarticular studies (lazy)."""
    from spinescoutx.data.axial_level import load_axial_level_scorer, score_and_assign_stack
    from spinescoutx.data.axial_match import pick_axial_t2
    from spinescoutx.data.rsna_index import RsnaPaths, build_series_index

    images = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    series = build_series_index(ROOT / "data/raw/rsna")
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    model, ss = load_axial_level_scorer(ROOT / "runs/axial_level_scorer", device)
    out = {}
    for st in studies:
        axs = pick_axial_t2(series, st, images)
        if axs is None:
            continue
        res = score_and_assign_stack(model, images, st, axs, ss, device)
        if res:
            out[st] = {lv: res[lv]["conf"] for lv in res}
    return out


def _build_graphs(device):
    by_study: dict[str, list] = {}
    for cond in ROUTES:
        for (study, _level), rec in _test_preds(cond, device).items():
            by_study.setdefault(study, []).append(rec)
    # axial scorer conf only for studies that have subarticular findings
    sub_studies = sorted(
        {s for s, recs in by_study.items() if any("subarticular" in r["condition"] for r in recs)}
    )
    axial = _axial_scores(sub_studies, device)
    graphs = {}
    for study, recs in by_study.items():
        findings = []
        for r in recs:
            als = None
            if "subarticular" in r["condition"]:
                als = axial.get(study, {}).get(r["level"])
            findings.append(
                fg.build_finding(
                    r["condition"],
                    r["level"],
                    r["probs"],
                    reference_label=SEVERITIES[r["y"]],
                    model_disagreement=r["disagree"],
                    axial_level_score=als,
                )
            )
        g = fg.build_study_graph(
            study, split="test", findings=findings, model_version=MODEL_VERSION
        )
        fg.validate_finding_graph(g)
        graphs[study] = g
    return graphs


def _select(graphs):
    """Pick curated cases by category -> {asset_name: case_id}."""
    chosen, used = {}, set()

    def pick(pred, name):
        for study, g in sorted(
            graphs.items(), key=lambda kv: -kv[1]["study_summary"]["max_p_severe"]
        ):
            if study in used:
                continue
            if pred(g):
                used.add(study)
                chosen[name] = (study, g)
                return

    def has(g, cond_sub, sev=None, review=None):
        for f in g["findings"]:
            if (
                cond_sub in f["condition"]
                and (sev is None or f["severity_estimate"] == sev)
                and (review is None or f["review_required"] == review)
            ):
                return True
        return False

    pick(lambda g: has(g, "spinal_canal", "severe"), "case_canal_severe_card")
    pick(
        lambda g: (
            has(g, "left_neural_foraminal", "severe")
            or has(g, "left_neural_foraminal", review=True)
        ),
        "case_foraminal_left_card",
    )
    pick(lambda g: has(g, "right_neural_foraminal", review=True), "case_foraminal_right_hard_card")
    pick(lambda g: has(g, "left_subarticular", "severe"), "case_subarticular_left_card")
    pick(lambda g: has(g, "right_subarticular", "severe"), "case_subarticular_right_card")
    pick(lambda g: g["study_summary"]["n_review_required"] >= 2, "case_review_required_card")
    pick(lambda g: g["study_summary"]["n_severe_estimates"] >= 2, "finding_graph_example")
    # mostly-normal / low-review case: shows the review flag is selective, not always-on
    for study, g in sorted(
        graphs.items(),
        key=lambda kv: (
            kv[1]["study_summary"]["n_review_required"],
            kv[1]["study_summary"]["max_p_severe"],
        ),
    ):
        if study in used:
            continue
        if g["study_summary"]["max_p_severe"] < 0.20 and g["study_summary"]["n_findings"] >= 5:
            used.add(study)
            chosen["case_mostly_normal_card"] = (study, g)
            break
    return chosen


def render_card(graph, path, title):
    findings = sorted(graph["findings"], key=lambda f: -f["probabilities"]["P(severe)"])[:10]
    n = len(findings)
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.46 * n))
    ax.axis("off")
    s = graph["study_summary"]
    ax.text(
        0.01,
        0.97,
        f"{title} — {graph['case_id']}",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.91,
        f"model {graph['model_version']} · split {graph['split']} · "
        f"max P(severe) {s['max_p_severe']:.2f} · {s['n_review_required']} review · "
        f"{s['n_severe_estimates']} severe est.",
        fontsize=8.5,
        transform=ax.transAxes,
        color="#333",
    )
    cols = [
        "finding",
        "level",
        "route",
        "severity est.",
        "P(severe)",
        "conf",
        "flag",
        "review reasons",
        "ref",
    ]
    xs = [0.01, 0.20, 0.27, 0.40, 0.53, 0.72, 0.78, 0.88, 0.97]
    y0 = 0.83
    for x, c in zip(xs, cols, strict=False):
        ax.text(
            x,
            y0,
            c,
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
            ha="right" if c == "ref" else "left",
        )
    row_h = (y0 - 0.04) / max(n, 1)
    for i, f in enumerate(findings):
        y = y0 - 0.03 - i * row_h
        ax.add_patch(
            plt.Rectangle(
                (0.005, y - row_h * 0.45),
                0.99,
                row_h * 0.9,
                transform=ax.transAxes,
                fc=SEV_COLOR[f["severity_estimate"]],
                ec="none",
                zorder=0,
            )
        )
        cond_short = (
            f["condition"].replace("_stenosis", "").replace("_narrowing", "").replace("_", " ")
        )
        ps = f["probabilities"]["P(severe)"]
        cells = [
            cond_short,
            f["level"],
            "",
            f["severity_estimate"],
            "",
            f"{f['calibrated_confidence']:.2f}",
            f["uncertainty_flag"][:4],
            ",".join(f["review_reasons"])[:32] or "-",
            f["reference_label"] or "-",
        ]
        for x, val, col in zip(xs, cells, cols, strict=False):
            ax.text(
                x,
                y,
                str(val),
                fontsize=7.6,
                transform=ax.transAxes,
                va="center",
                ha="right" if col == "ref" else "left",
            )
        # route badge
        ax.add_patch(
            plt.Rectangle(
                (xs[2], y - row_h * 0.28),
                0.11,
                row_h * 0.56,
                transform=ax.transAxes,
                fc=ROUTE_COLOR[f["view_route"]],
                ec="none",
            )
        )
        ax.text(
            xs[2] + 0.003,
            y,
            f["view_route"].replace("_", "-"),
            fontsize=6.5,
            color="white",
            transform=ax.transAxes,
            va="center",
        )
        # P(severe) bar
        ax.add_patch(
            plt.Rectangle(
                (xs[4], y - row_h * 0.22),
                0.16 * ps,
                row_h * 0.44,
                transform=ax.transAxes,
                fc="#e63946",
                ec="none",
            )
        )
        ax.text(xs[4] + 0.165, y, f"{ps:.2f}", fontsize=7, transform=ax.transAxes, va="center")
    ax.text(
        0.01,
        0.02,
        graph["disclaimer"]
        + "  ·  auto inference (no GT) · severity = research finding, not a diagnosis",
        fontsize=7,
        transform=ax.transAxes,
        color="#777",
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_schema_visual(path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    ax.text(
        0.5,
        0.96,
        "SpineScoutX finding-graph schema (v4) — one finding",
        ha="center",
        fontsize=13,
        fontweight="bold",
        transform=ax.transAxes,
    )
    fields = [
        ("condition / level / side", "which finding, where"),
        ("view_route", "sagittal-T2 (canal) · sagittal-T1 (foraminal) · axial-T2 (subarticular)"),
        ("crop_provenance", "auto (localizer/scorer) · oracle (GT, upper bound) · blocked"),
        ("severity_estimate", "argmax of the softmax: normal_mild · moderate · severe"),
        ("probabilities", "P(normal_mild) + P(moderate) + P(severe) ≈ 1"),
        ("calibrated_confidence + uncertainty_flag", "top-class prob → high/moderate/review"),
        (
            "review_required + review_reasons",
            "low_confidence · high_entropy · model_disagreement · axial_level_uncertainty",
        ),
        ("localizer", "route · confidence · axial_level_scorer_score"),
        (
            "reference_label",
            "held-out research target (transparency only; NOT a model input/output)",
        ),
    ]
    for i, (k, v) in enumerate(fields):
        y = 0.86 - i * 0.092
        ax.add_patch(
            plt.Rectangle(
                (0.02, y - 0.035), 0.30, 0.07, transform=ax.transAxes, fc="#e0f2f1", ec="#264653"
            )
        )
        ax.text(0.035, y, k, fontsize=9, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.35, y, v, fontsize=8.5, transform=ax.transAxes, va="center", color="#222")
    ax.text(
        0.5,
        0.02,
        "research-only · not diagnostic · not clinically validated",
        ha="center",
        fontsize=8,
        transform=ax.transAxes,
        color="#777",
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    device = select_device("auto")
    PACK.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("[showcase] building finding graphs on locked-test auto predictions...")
    graphs = _build_graphs(device)
    print(f"[showcase] {len(graphs)} study graphs built")
    chosen = _select(graphs)

    titles = {
        "case_canal_severe_card": "Canal — severe",
        "case_foraminal_left_card": "Left foraminal",
        "case_foraminal_right_hard_card": "Right foraminal (hard)",
        "case_subarticular_left_card": "Left subarticular",
        "case_subarticular_right_card": "Right subarticular",
        "case_review_required_card": "Review-required case",
        "finding_graph_example": "Multi-finding study",
        "case_mostly_normal_card": "Mostly normal/mild",
    }
    index = []
    for name, (_study, g) in chosen.items():
        (PACK / f"{g['case_id']}.json").write_text(json.dumps(g, indent=2))
        (PACK / f"{g['case_id']}.md").write_text(fg.render_markdown(g))
        render_card(g, ASSETS / f"{name}.png", titles.get(name, name))
        index.append((name, g["case_id"], g["study_summary"]))
        print(
            f"[showcase] {name}: {g['case_id']} "
            f"(maxP_sev {g['study_summary']['max_p_severe']:.2f}, "
            f"{g['study_summary']['n_review_required']} review)"
        )
    render_schema_visual(ASSETS / "report_schema_visual.png")

    _doc(index)
    print(f"[showcase] wrote {len(chosen)} cards + schema visual to {ASSETS}")
    return 0


def _doc(index):
    lines = [
        "# Model-output showcase pack",
        "",
        "> Research-only · not diagnostic. Real model outputs (finding graphs) generated by the"
        " deployable graders on locked-test auto predictions; cards render the structured output"
        " (no DICOM pixels, anonymized case ids). Full JSON/MD: `outputs/real/showcase_reports/`"
        " (gitignored). Reproduce: `python scripts/make_model_output_showcase.py`.",
        "",
        "| card | case_id | max P(severe) | severe est. | review |",
        "|---|---|---|---|---|",
    ]
    for name, cid, s in index:
        lines.append(
            f"| `{name}` | {cid} | {s['max_p_severe']:.2f} | "
            f"{s['n_severe_estimates']} | {s['n_review_required']} |"
        )
    lines += ["", "Cards: `docs/assets/showcase/*.png`. Schema: `report_schema_v4.md`."]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
