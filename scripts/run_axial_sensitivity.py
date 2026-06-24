#!/usr/bin/env python3
"""Axial level-scorer v2 — bounded analysis: would a better scorer improve GRADING?

The current coordinate-supervised axial level scorer reaches ±1-slice hit ~0.43 (vs
geometry 0.275). A natural v1.1 idea is a stack-sequence scorer v2. Before committing to
that training, this script answers the decision-relevant question directly: **how sensitive
is the deployed subarticular grader's prediction to the axial slice/leveling choice?**

We re-run the SAME subarticular grader on the locked-test auto crops under three matched
perturbation regimes (auto coords only, no GT):
  * `slice`   — slice shift only (±2): the axial leveling component a better scorer fixes;
  * `inplane` — in-plane jitter only: the paramedian-offset component;
  * `full`    — both.
If `slice` instability is small relative to `full`, the robust grader already tolerates
leveling noise and a better scorer would yield limited GRADING gains (documented negative
for the grading payoff, even if it improves localization). If `slice` dominates, v2 is
motivated. Either way the conclusion is evidence-backed, not assumed.

Research-only. Not diagnostic. No GT coordinates used to generate perturbations.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.datasets import _load_array
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.data.perturb_crops import SliceDecoder, reextract_25d
from spinescoutx.evaluation import evidence_stability as es
from spinescoutx.training.optim import select_device

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/axial_sensitivity.json"
DOC = ROOT / "docs/run_logs/axial_sensitivity_run.md"  # optional finer attribution run
CONDS = ("left_subarticular_stenosis", "right_subarticular_stenosis")
RUN = ROOT / "runs/v1_subarticular_auto_robust"
CACHE = ROOT / "data/cache/rsna_auto_subarticular"

# reuse the validated grader-load + streaming forward from the evidence-stability runner
_spec = importlib.util.spec_from_file_location("es_run", ROOT / "scripts/run_evidence_stability.py")
_esrun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_esrun)


def _auroc(scores, labels):
    return _esrun.auroc(np.asarray(scores), np.asarray(labels))


def run(mode, device, seed=20260623):
    model, cfg = _esrun.load_grader(RUN, device)
    crop_size = int(getattr(cfg.data, "crop_size", 224))
    split_map = load_splits_v1(SPLITS)
    res = {}
    for cond in CONDS:
        man = read_manifest(CACHE / "manifest.parquet")
        man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
        man["study_id"] = man.study_id.astype(str)
        man = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
        rows = list(man.itertuples())
        n = len(rows)
        lv = np.array([LEVEL_TO_INDEX[r.level] for r in rows])
        cd = np.array([CONDITION_TO_INDEX[r.condition] for r in rows])
        base = _esrun.forward_providers(
            model,
            [(lambda r=r: _load_array(CACHE / r.crop_path)) for r in rows],
            lv,
            cd,
            device,
            crop_size,
        )
        cfgp = es.config_for(cond)
        rng = np.random.default_rng(seed)
        offs = [es.sample_offsets(cfgp, rng, mode=mode) for _ in rows]
        dec = SliceDecoder(max_items=768)
        pert = np.zeros((cfgp.k, n, 3))
        for k in range(cfgp.k):

            def prov(r, o, dec=dec):
                c = reextract_25d(
                    dec,
                    r.dicom_path,
                    r.instance_number,
                    r.x,
                    r.y,
                    dx=o[0],
                    dy=o[1],
                    ds=o[2],
                    crop_size=crop_size,
                )
                return c if c is not None else _load_array(CACHE / r.crop_path)

            pert[k] = _esrun.forward_providers(
                model,
                [(lambda r=rows[i], o=offs[i][k]: prov(r, o)) for i in range(n)],
                lv,
                cd,
                device,
                crop_size,
            )
        inst, wrong = [], []
        for i, r in enumerate(rows):
            st = es.stability_stats(np.vstack([base[i], pert[:, i, :]]))
            inst.append(es.instability_score(st))
            wrong.append(int(st["baseline_pred"] != int(r.severity_index)))
        inst = np.array(inst)
        res[cond] = {
            "n": n,
            "mean_instability": float(inst.mean()),
            "frac_unstable": float((inst >= 0.34).mean()),
            "auroc_error_from_instability": _auroc(inst, wrong),
        }
        print(
            f"  [{mode}] {cond:30s} mean_instab={inst.mean():.3f} "
            f"AUROC(err)={res[cond]['auroc_error_from_instability']:.3f}",
            flush=True,
        )
    return res


def main() -> int:
    device = select_device("auto")
    out = {
        "protocol": "splits_v1 locked-test",
        "grader": "v1_subarticular_auto_robust",
        "modes": {},
    }
    for mode in ("full", "slice", "inplane"):
        print(f"[axial-sensitivity] mode={mode}", flush=True)
        out["modes"][mode] = run(mode, device)
    # attribution: slice instability as a fraction of full, per condition
    attr = {}
    for cond in CONDS:
        full = out["modes"]["full"][cond]["mean_instability"]
        sl = out["modes"]["slice"][cond]["mean_instability"]
        ip = out["modes"]["inplane"][cond]["mean_instability"]
        attr[cond] = {
            "slice_share": float(sl / full) if full > 0 else float("nan"),
            "inplane_share": float(ip / full) if full > 0 else float("nan"),
        }
    out["attribution"] = attr
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _doc(out)
    print(f"\nwrote {OUT}\nwrote {DOC}")
    return 0


def _doc(out):
    m = out["modes"]
    lines = [
        "# Axial level scorer v2 — bounded grading-payoff analysis (locked-test auto)",
        "",
        "> Research-only · not diagnostic. Answers *before training a v2*: how much does the",
        "> deployed subarticular grader's prediction actually depend on the axial slice/leveling",
        "> choice? Same grader re-run under matched slice-only vs in-plane-only perturbations",
        "> (auto coords, no GT). The current per-slice scorer reaches ±1-slice hit ~0.43.",
        "",
        "## Instability by perturbation regime (mean instability; AUROC of predicting an error)",
        "| condition | full | slice-only | in-plane-only | slice share of full |",
        "|---|---|---|---|---|",
    ]
    for cond in CONDS:
        a = out["attribution"][cond]
        lines.append(
            f"| {cond} | {m['full'][cond]['mean_instability']:.3f} | "
            f"{m['slice'][cond]['mean_instability']:.3f} | "
            f"{m['inplane'][cond]['mean_instability']:.3f} | {a['slice_share']:.2f} |"
        )
    # verdict
    shares = [out["attribution"][c]["slice_share"] for c in CONDS]
    mean_share = float(np.nanmean(shares))
    slice_dominant = mean_share >= 0.6
    lines += [
        "",
        "## Conclusion (evidence-backed)",
        f"- The slice/leveling component explains **{mean_share:.0%}** of the subarticular",
        "  grader's instability on average.",
    ]
    if slice_dominant:
        lines += [
            "- Slice/leveling **dominates** instability → a better axial scorer (stack-sequence",
            "  v2) is **motivated**: lower leveling error should translate into steadier grading.",
        ]
    else:
        lines += [
            "- Slice/leveling does **not** dominate (the robust grader tolerates ±2-slice noise) →",
            "  a better axial scorer would improve **localization** but yield **limited grading**",
            "  gains. This is the honest, documented payoff ceiling: robust auto-training already",
            "  absorbs most of the leveling noise (consistent with subarticular auto severe recall",
            "  0.746/0.737 despite ±1-slice hit 0.43).",
        ]
    lines += [
        "",
        "## Stack-sequence v2 design (next step, specified)",
        "- Input: full axial T2 stack (sampled) → lightweight slice CNN encoder.",
        "- Sequence model over slices (1D TCN / BiGRU / small Transformer) with normalized",
        "  z-rank + reliable spacing features → per-level slice distribution + confidence.",
        "- Decode: monotonic-order + minimum-distance DP; top-k pooling; confidence calibration.",
        "- Train on RSNA subarticular axial coordinates (train/dev only); locked-test used once.",
        "- **Keep only if** it improves ±0/±1/±2 slice-hit AND downstream subarticular severe",
        "  recall / recall@FAR; otherwise record as a localization-only improvement per the above.",
        "",
        "Reproduce: `python scripts/run_axial_sensitivity.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
