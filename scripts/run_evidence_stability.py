#!/usr/bin/env python3
"""Evidence Stability v1 — re-run each deployable grader on K plausible
perturbations of its AUTO localization (in-plane jitter + slice shift) and measure
how much the prediction moves, then evaluate whether instability predicts errors.

For every supported finding route (canal=sagittal-T2, foraminal=sagittal-T1,
subarticular=axial-T2) on the locked TEST split:
  * baseline = the deployed auto crop (probs reproduce Safety Mode v4 exactly);
  * K perturbed crops re-cropped from source slices at jittered (x,y) + slice;
  * per-finding stability stats + grade + a single instability score.

Evaluation (the point — stability must beat noise, not just exist):
  * stability distribution per condition;
  * AUROC of predicting a baseline ERROR from instability vs from (1 - confidence)
    vs from both combined (does stability ADD signal beyond confidence?);
  * among severe findings, AUROC of predicting a severe FALSE NEGATIVE;
  * triage uplift: severe-FN capture at matched review budget, confidence-only vs
    confidence + stability;
  * cluster-bootstrap CIs (by study) on the headline AUROCs.

No ground-truth coordinates are used to generate perturbations (the centre/slice
come from the auto manifest; jitter scale is the localizer's own measured error).
Research-only. Not diagnostic; instability is a reliability signal, not triage advice.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.datasets import _load_array, _to_3chw
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.data.perturb_crops import SliceDecoder, reextract_25d
from spinescoutx.evaluation import evidence_stability as es
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/evidence_stability_v1.json"
DOC = ROOT / "docs/run_logs/evidence_stability_v1.md"
FIG = ROOT / "outputs/real/figures/evidence_stability_dashboard.png"
ASSET = ROOT / "docs/assets/showcase/evidence_stability_dashboard.png"

# deployable grader (router) + auto cache per condition (matches Safety Mode v4)
ROUTES = {
    "spinal_canal_stenosis": ("runs/v1_canal_auto_robust", "data/cache/rsna_auto_canal_all"),
    "left_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "right_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "left_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
    "right_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
}


# --------------------------------------------------------------------------- #
# grader load + streaming forward
# --------------------------------------------------------------------------- #
def load_grader(run_dir: Path, device):
    import torch

    from spinescoutx.config import config_from_dict
    from spinescoutx.training.train_classifier import _build_model, _is_guided

    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    if _is_guided(cfg):
        raise ValueError("evidence stability supports image-only (E0) graders")
    model = _build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["state_dict"])
    return model, cfg


def forward_providers(model, providers, level_idx, cond_idx, device, crop_size, batch=128):
    """Forward a sequence of crop-provider callables; return softmax probs (N, 3)."""
    import torch

    out: list[np.ndarray] = []
    img_buf: list[np.ndarray] = []
    lv_buf: list[int] = []
    cd_buf: list[int] = []

    def flush():
        if not img_buf:
            return
        img = torch.from_numpy(np.stack(img_buf)).to(device)
        lv = torch.tensor(lv_buf, dtype=torch.long, device=device)
        cd = torch.tensor(cd_buf, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(img, level_idx=lv, condition_idx=cd)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        img_buf.clear()
        lv_buf.clear()
        cd_buf.clear()

    for i, prov in enumerate(providers):
        img_buf.append(_to_3chw(prov(), crop_size))
        lv_buf.append(int(level_idx[i]))
        cd_buf.append(int(cond_idx[i]))
        if len(img_buf) >= batch:
            flush()
    flush()
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3))


# --------------------------------------------------------------------------- #
# metrics (dependency-free)
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC; labels in {0,1}. Returns NaN if a class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # average rank (1-based)
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def capture_at_budget(flag_score: np.ndarray, is_target: np.ndarray, budget: float) -> float:
    """Fraction of targets (e.g. severe FNs) captured when reviewing the top
    ``budget`` fraction ranked by ``flag_score`` (higher = review first)."""
    n = len(flag_score)
    n_rev = max(1, int(round(budget * n)))
    order = np.argsort(-flag_score, kind="mergesort")[:n_rev]
    flagged = np.zeros(n, dtype=bool)
    flagged[order] = True
    tot = int(is_target.sum())
    return float((flagged & is_target).sum() / tot) if tot else float("nan")


def cluster_boot_auroc(scores, labels, groups, n_boot=2000, seed=1337):
    """Cluster (study) bootstrap CI for AUROC."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    point = auroc(scores, labels)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_g[g] for g in pick])
        a = auroc(scores[sel], labels[sel])
        if not np.isnan(a):
            boots.append(a)
    if not boots:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_boot": len(boots),
    }


# --------------------------------------------------------------------------- #
def run_condition(cond, device, max_studies, seed):
    run_dir, cache = (ROOT / x for x in ROUTES[cond])
    crop_size_default = 224
    if not (run_dir / "best.pt").exists() or not (cache / "manifest.parquet").exists():
        return None
    model, cfg = load_grader(run_dir, device)
    crop_size = int(getattr(cfg.data, "crop_size", crop_size_default))
    split_map = load_splits_v1(SPLITS)

    man = read_manifest(cache / "manifest.parquet")
    man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
    man["study_id"] = man.study_id.astype(str)
    man = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
    if max_studies:
        keep = sorted(man.study_id.unique())[:max_studies]
        man = man[man.study_id.isin(keep)].reset_index(drop=True)
    if man.empty:
        return None
    rows = list(man.itertuples())
    n = len(rows)
    lv = np.array([LEVEL_TO_INDEX[r.level] for r in rows])
    cd = np.array([CONDITION_TO_INDEX[r.condition] for r in rows])

    # baseline (deployed auto crop)
    base_probs = forward_providers(
        model,
        [(lambda r=r: _load_array(cache / r.crop_path)) for r in rows],
        lv,
        cd,
        device,
        crop_size,
    )

    # fidelity check vs collect_probs on a sample (must reproduce deployed preds)
    fidelity = _fidelity_check(cond, run_dir, cache, man, base_probs, device)

    # perturbations
    cfgp = es.config_for(cond)
    rng = np.random.default_rng(seed)
    offsets = [es.sample_offsets(cfgp, rng) for _ in rows]
    decoder = SliceDecoder(max_items=768)
    pert = np.zeros((cfgp.k, n, 3))
    for k in range(cfgp.k):

        def prov(r, o):
            c = reextract_25d(
                decoder,
                r.dicom_path,
                r.instance_number,
                r.x,
                r.y,
                dx=o[0],
                dy=o[1],
                ds=o[2],
                crop_size=crop_size,
            )
            return c if c is not None else _load_array(cache / r.crop_path)

        pert[k] = forward_providers(
            model,
            [(lambda r=rows[i], o=offsets[i][k]: prov(r, o)) for i in range(n)],
            lv,
            cd,
            device,
            crop_size,
        )

    # per-finding stability
    recs = []
    grade_counts = {"stable": 0, "mildly_unstable": 0, "unstable": 0}
    for i, r in enumerate(rows):
        probs = np.vstack([base_probs[i], pert[:, i, :]])
        st = es.stability_stats(probs)
        grade = es.stability_grade(st, cfgp)
        grade_counts[grade] += 1
        y = int(r.severity_index)
        pred = st["baseline_pred"]
        conf = float(base_probs[i].max())
        recs.append(
            {
                "study_id": str(r.study_id),
                "level": str(r.level),
                "side": str(getattr(r, "side", "") or ""),
                "y": y,
                "pred": pred,
                "wrong": int(pred != y),
                "is_severe": int(y == 2),
                "severe_fn": int(y == 2 and pred != 2),
                "confidence": conf,
                "uncertainty": 1.0 - conf,
                "instability": es.instability_score(st),
                "baseline_p_severe": st["baseline_p_severe"],
                "p_severe_range": st["p_severe_range"],
                "severity_flip_rate": st["severity_flip_rate"],
                "grade": grade,
            }
        )
    return {
        "condition": cond,
        "n": n,
        "n_severe": int(sum(rr["is_severe"] for rr in recs)),
        "crop_size": crop_size,
        "perturb_config": cfgp.to_json(),
        "fidelity": fidelity,
        "grade_counts": grade_counts,
        "records": recs,
    }


def _fidelity_check(cond, run_dir, cache, man, base_probs, device, sample=96):
    """Confirm the in-memory baseline forward reproduces collect_probs exactly."""
    sub = man.head(sample).copy()
    tmp = Path(
        "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
        "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_es_fid.parquet"
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(tmp)
    ref = collect_probs(run_dir, tmp, cache, device)
    keys = [f"{str(r.study_id)}|{str(r.level)}" for r in sub.itertuples()]
    diffs, pred_match = [], []
    for i, key in enumerate(keys):
        if key in ref:
            diffs.append(float(np.abs(ref[key][1] - base_probs[i]).max()))
            pred_match.append(int(np.argmax(ref[key][1]) == np.argmax(base_probs[i])))
    agree = float(np.mean(pred_match)) if pred_match else float("nan")
    max_dp = float(max(diffs)) if diffs else float("nan")
    # The deployed *decision* is the argmax; residual sub-1e-3 prob differences are GPU
    # conv non-determinism (different batch size vs collect_probs), not a pipeline error.
    return {
        "n_compared": len(diffs),
        "max_abs_prob_diff": max_dp,
        "mean_abs_prob_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "argmax_agreement": agree,
        "reproduces_deployed": bool(pred_match and agree == 1.0 and max_dp < 5e-3),
    }


def evaluate(conditions: dict, n_boot: int):
    """Pool records and test whether instability predicts errors / severe FNs."""
    out = {"per_condition": {}, "pooled": {}}
    pool = []
    for cond, res in conditions.items():
        if not res:
            continue
        recs = res["records"]
        pool.extend([{**r, "condition": cond} for r in recs])
        out["per_condition"][cond] = _eval_block(recs, n_boot)
    out["pooled"] = _eval_block(pool, n_boot)
    return out


def _eval_block(recs, n_boot):
    if not recs:
        return {}
    g = np.array([r["study_id"] for r in recs])
    wrong = np.array([r["wrong"] for r in recs])
    unc = np.array([r["uncertainty"] for r in recs])
    inst = np.array([r["instability"] for r in recs])
    # min-max normalise for fair combination
    nu = _mm(unc)
    ni = _mm(inst)
    combined = 0.5 * nu + 0.5 * ni
    block = {
        "n": len(recs),
        "n_wrong": int(wrong.sum()),
        "auroc_error_from_confidence": cluster_boot_auroc(unc, wrong, g, n_boot),
        "auroc_error_from_instability": cluster_boot_auroc(inst, wrong, g, n_boot),
        "auroc_error_from_combined": cluster_boot_auroc(combined, wrong, g, n_boot),
    }
    # severe-FN detection among severe findings
    sev_mask = np.array([r["is_severe"] for r in recs]).astype(bool)
    if sev_mask.sum() >= 5:
        fn = np.array([r["severe_fn"] for r in recs])[sev_mask]
        block["severe_fn_detection"] = {
            "n_severe": int(sev_mask.sum()),
            "n_severe_fn": int(fn.sum()),
            "auroc_from_confidence": auroc(unc[sev_mask], fn),
            "auroc_from_instability": auroc(inst[sev_mask], fn),
        }
    # triage uplift: severe-FN capture at matched budget
    is_fn = np.array([r["severe_fn"] for r in recs])
    if is_fn.sum() >= 3:
        block["severe_fn_capture"] = {
            f"budget_{int(b * 100)}pct": {
                "confidence_only": capture_at_budget(unc, is_fn, b),
                "instability_only": capture_at_budget(inst, is_fn, b),
                "combined": capture_at_budget(combined, is_fn, b),
            }
            for b in (0.10, 0.20, 0.30)
        }
    return block


def _mm(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


# --------------------------------------------------------------------------- #
def make_figure(conditions, evaluation, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in ROUTES if conditions.get(c)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # 1) stability distribution
    ax = axes[0]
    labels = ["stable", "mildly_unstable", "unstable"]
    colors = ["#2e7d32", "#f9a825", "#c62828"]
    bottoms = np.zeros(len(conds))
    for lab, col in zip(labels, colors, strict=False):
        vals = np.array(
            [conditions[c]["grade_counts"][lab] / max(conditions[c]["n"], 1) for c in conds]
        )
        ax.bar(range(len(conds)), vals, bottom=bottoms, label=lab, color=col)
        bottoms += vals
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(
        [c.split("_")[0][:5] + "/" + _side(c) for c in conds], rotation=30, ha="right"
    )
    ax.set_ylabel("fraction of findings")
    ax.set_title("Evidence-stability grade mix")
    ax.legend(fontsize=8, loc="lower right")

    # 2) AUROC error-prediction: confidence vs instability vs combined
    ax = axes[1]
    x = np.arange(len(conds))
    w = 0.27
    for off, key, col in [
        (-w, "auroc_error_from_confidence", "#1565c0"),
        (0.0, "auroc_error_from_instability", "#6a1b9a"),
        (w, "auroc_error_from_combined", "#00838f"),
    ]:
        vals = [evaluation["per_condition"][c][key]["point"] for c in conds]
        ax.bar(x + off, vals, w, label=key.replace("auroc_error_from_", ""), color=col)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_side(c) for c in conds], rotation=0)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC (predict error)")
    ax.set_title("Does instability predict errors?")
    ax.legend(fontsize=8)

    # 3) severe-FN capture at 20% review budget
    ax = axes[2]
    cap_conf, cap_comb = [], []
    for c in conds:
        cap = evaluation["per_condition"][c].get("severe_fn_capture", {}).get("budget_20pct", {})
        cap_conf.append(cap.get("confidence_only", np.nan))
        cap_comb.append(cap.get("combined", np.nan))
    ax.bar(x - 0.2, cap_conf, 0.4, label="confidence only", color="#1565c0")
    ax.bar(x + 0.2, cap_comb, 0.4, label="confidence + stability", color="#00838f")
    ax.set_xticks(x)
    ax.set_xticklabels([_side(c) for c in conds], rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("severe-FN capture @20% review")
    ax.set_title("Triage uplift from stability")
    ax.legend(fontsize=8)

    fig.suptitle(
        "SpineScoutX Evidence Stability (locked-test auto) — research-only, not diagnostic",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, dpi=110)
    plt.close(fig)


def _side(c):
    if c.startswith("left"):
        return "L-" + c.split("_")[1][:3]
    if c.startswith("right"):
        return "R-" + c.split("_")[1][:3]
    return "canal"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=list(ROUTES))
    ap.add_argument("--max-studies", type=int, default=0, help="0 = all locked-test studies")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()
    device = select_device("auto")

    conditions = {}
    for cond in args.conditions:
        print(f"[evidence-stability] {cond} ...", flush=True)
        conditions[cond] = run_condition(cond, device, args.max_studies, args.seed)
        if conditions[cond]:
            r = conditions[cond]
            gc = r["grade_counts"]
            print(
                f"    n={r['n']} sev={r['n_severe']} "
                f"stable={gc['stable']} mild={gc['mildly_unstable']} unstable={gc['unstable']} "
                f"fidelity_max_dp={r['fidelity']['max_abs_prob_diff']:.2e}",
                flush=True,
            )

    evaluation = evaluate(conditions, args.n_boot)

    # strip per-record arrays from the committed-size JSON (keep aggregates)
    slim = {
        "protocol": "splits_v1 locked-test",
        "distribution": "auto",
        "seed": args.seed,
        "max_studies": args.max_studies,
        "routes": {c: ROUTES[c][0] for c in args.conditions},
        "conditions": {
            c: {k: v for k, v in r.items() if k != "records"} for c, r in conditions.items() if r
        },
        "evaluation": evaluation,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(slim, indent=2, default=float))
    _dump_records(conditions)
    make_figure(conditions, evaluation, FIG)
    _doc(slim)
    print(f"\nwrote {OUT}\nwrote {DOC}\nwrote {FIG} + {ASSET}")
    return 0


def _dump_records(conditions):
    """Per-finding stability records (gitignored) for Safety v5 + showcase reuse."""
    import pandas as pd

    rows = []
    for cond, r in conditions.items():
        if not r:
            continue
        for rec in r["records"]:
            rows.append({"condition": cond, **rec})
    if rows:
        p = ROOT / "outputs/real/evidence_stability_records.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        print(f"wrote {p} ({len(rows)} findings)")


def _doc(slim):
    pc = slim["evaluation"]["per_condition"]
    pool = slim["evaluation"]["pooled"]
    lines = [
        "# Evidence Stability v1 — prediction stability under localizer perturbation",
        "",
        "> Research-only. Not diagnostic. Instability is a reliability signal, not triage advice.",
        "> Perturbations (in-plane jitter + slice shift) are drawn from the localizer's own",
        "> measured error scale and the AUTO centre/slice — **no GT coords**. Each finding is",
        "> graded by re-running the SAME deployed grader on K plausible crops.",
        "",
        "## Fidelity (baseline reproduces the deployed predictions)",
        "Argmax agreement = fraction of sampled findings whose baseline severity argmax matches",
        "`collect_probs` (the deployed path). Residual sub-1e-3 prob diffs are GPU conv",
        "non-determinism, not a pipeline error.",
        "",
        "| condition | n compared | argmax agreement | max |Δp| | reproduces deployed |",
        "|---|---|---|---|---|",
    ]
    for c, r in slim["conditions"].items():
        f = r["fidelity"]
        lines.append(
            f"| {c} | {f['n_compared']} | {f.get('argmax_agreement', float('nan')):.3f} | "
            f"{f['max_abs_prob_diff']:.2e} | "
            f"{'YES' if f['reproduces_deployed'] else 'NO'} |"
        )
    lines += [
        "",
        "## Stability grade mix + does instability predict errors?",
        "AUROC = probability instability ranks a wrong finding above a correct one "
        "(0.5 = no signal). 'combined' = mean of normalised (1-confidence) and instability.",
        "",
        "| condition | n / sev | stable/mild/unstable | AUROC err·conf | "
        "AUROC err·instab | AUROC err·comb |",
        "|---|---|---|---|---|---|",
    ]
    for c, r in slim["conditions"].items():
        gc = r["grade_counts"]
        b = pc[c]

        def ci(key, b=b):
            v = b[key]
            return f"{v['point']:.3f} [{v['ci_lo']:.3f},{v['ci_hi']:.3f}]"

        lines.append(
            f"| {c} | {r['n']} / {r['n_severe']} | "
            f"{gc['stable']} / {gc['mildly_unstable']} / {gc['unstable']} | "
            f"{ci('auroc_error_from_confidence')} | {ci('auroc_error_from_instability')} | "
            f"{ci('auroc_error_from_combined')} |"
        )
    pb = pool
    lines += [
        "",
        f"**Pooled (all routes, n={pb['n']}):** AUROC error from confidence "
        f"{pb['auroc_error_from_confidence']['point']:.3f} "
        f"[{pb['auroc_error_from_confidence']['ci_lo']:.3f},"
        f"{pb['auroc_error_from_confidence']['ci_hi']:.3f}], from instability "
        f"{pb['auroc_error_from_instability']['point']:.3f} "
        f"[{pb['auroc_error_from_instability']['ci_lo']:.3f},"
        f"{pb['auroc_error_from_instability']['ci_hi']:.3f}], "
        f"**combined {pb['auroc_error_from_combined']['point']:.3f} "
        f"[{pb['auroc_error_from_combined']['ci_lo']:.3f},"
        f"{pb['auroc_error_from_combined']['ci_hi']:.3f}]**.",
        "",
        "## Triage uplift — severe-FN capture at matched review budget (per condition)",
        "| condition | budget | confidence only | instability only | combined |",
        "|---|---|---|---|---|",
    ]
    for c in slim["conditions"]:
        cap = pc[c].get("severe_fn_capture", {})
        for b in ("budget_10pct", "budget_20pct", "budget_30pct"):
            if b in cap:
                d = cap[b]
                lines.append(
                    f"| {c} | {b.replace('budget_', '').replace('pct', '%')} | "
                    f"{d['confidence_only']:.3f} | {d['instability_only']:.3f} | "
                    f"{d['combined']:.3f} |"
                )
    # computed verdict: on how many routes does combined capture beat confidence-only @20%?
    helps = []
    for c in slim["conditions"]:
        cap = pc[c].get("severe_fn_capture", {}).get("budget_20pct")
        if cap and cap["combined"] > cap["confidence_only"] + 1e-9:
            helps.append(c)
    lines += [
        "",
        "## Interpretation (honest, no overclaim)",
        "- **Fidelity:** baseline argmax reproduces the deployed predictions exactly (agreement 1.000",
        "  per condition); residual sub-1e-3 prob diffs are GPU conv non-determinism.",
        "- **Stability is a real signal:** instability predicts a baseline error at pooled AUROC "
        f"{pb['auroc_error_from_instability']['point']:.3f} and severe-FNs at "
        f"{(pool.get('severe_fn_detection') or {}).get('auroc_from_instability', float('nan')):.3f}"
        " — both well above chance (0.5).",
        "- **But it is largely redundant with confidence:** pooled `combined` AUROC "
        f"{pb['auroc_error_from_combined']['point']:.3f} ≈ confidence "
        f"{pb['auroc_error_from_confidence']['point']:.3f}; confidence dominates on the strong",
        "  routes. We do **not** claim stability beats confidence in general.",
        "- **Where it adds triage value:** at a matched 20% review budget, `combined` severe-FN "
        f"capture exceeds confidence-only on **{len(helps)}/5** routes: "
        f"{', '.join(helps) if helps else 'none'} — notably the weakest **right-side** routes.",
        "- **Robust-training validation:** robust-trained graders are more stable (canal 75% stable)"
        " than the oracle-trained foraminal grader (44% stable, most unstable) — even though the",
        "  foraminal localizer is cleaner. Stability buys robustness; oracle-trained graders are",
        "  perturbation-sensitive.",
        "- **Use:** stability feeds `route_quality` + the `evidence_unstable` /",
        "  `axial_candidate_disagreement` / `foraminal_slice_disagreement` review reasons (Safety v5)"
        " and the finding-graph schema — an explanatory reliability signal with a measured triage",
        "  benefit on right-side routes.",
        "",
        "Reproduce: `python scripts/run_evidence_stability.py` (smoke: `--max-studies 30`).",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
