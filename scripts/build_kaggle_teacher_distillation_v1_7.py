#!/usr/bin/env python3
"""Teacher-distillation fallback (v1.7, Phase 7).

A license-clean **lightweight teacher ensemble of SpineScoutX's own foraminal models** (deployed
grader + v1.6 baseline + v1.6 LSS) — applying the public RSNA-2024 top-solution *strategies*
(per-finding models, ensembling, soft probabilities; documented, **no code copied, no license
violated**) rather than any external weights. The teacher's soft probabilities (train+dev) distil a
fresh convnext_tiny student via CE(original label) + KL(student‖teacher) + severe upweight. The
teacher is **accepted only if dev right-foraminal recall@FAR≤10 improves**; locked-test is read once
for an accepted student. Research-only. Not diagnostic.

Sources reviewed (writeups only): RSNA 2024 Lumbar Spine Degenerative Classification Kaggle
discussion (sagittal/axial separation, disc-level + series-level heads, ensembles, TTA, soft
labels). We reuse the *ideas*, not the code or weights.
"""

from __future__ import annotations

# reuse the noise-aware dataset + eval helpers
import importlib.util as _ilu  # noqa: E402
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.models.image_classifier import ImageClassifier
from spinescoutx.training.optim import select_device
from spinescoutx.utils.seed import seed_everything

_spec = _ilu.spec_from_file_location(
    "_na", Path(__file__).resolve().parent / "train_noise_aware_foraminal_v1_7.py"
)
_na = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_na)

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
OUTDIR = ROOT / "outputs/real"
TMP = ROOT / "outputs/real/_teacher_tmp.parquet"
DEPLOYED = ROOT / "runs/v1_foraminal_oracle_ctrl"
V16 = {"v16_baseline": "convnext_tiny", "v16_lss": "convnext_tiny"}
RUN = {
    "v16_baseline": "runs/foraminal_rsna_baseline_v1_6",
    "v16_lss": "runs/foraminal_rsna_lss_v1_6",
}
FORAMINAL = _na.FORAMINAL


def _deployed_probs(split, sm, device):
    out = {}
    for cond in FORAMINAL:
        man = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
        man = man[(man.condition == cond) & man.severity_index.isin([0, 1, 2])].copy()
        man["study_id"] = man.study_id.astype(str)
        sub = man[man.study_id.map(sm) == split].reset_index(drop=True)
        if sub.empty:
            continue
        sub.to_parquet(TMP)
        for k, (_y, p) in collect_probs(DEPLOYED, TMP, RSNA_CACHE, device).items():
            st, lv = k.split("|")[0], k.split("|")[1]
            out[f"{st}|{lv}|{cond}"] = np.asarray(p, dtype=float)
    return out


@torch.no_grad()
def _model_probs(run_dir, backbone, df, device):
    model = ImageClassifier(
        backbone=backbone, in_chans=3, num_classes=3, use_level_embedding=True,
        use_condition_embedding=True, embed_dim=16, dropout=0.2, pretrained=False,
    ).to(device).eval()  # fmt: skip
    model.load_state_dict(torch.load(ROOT / run_dir / "best.pt", map_location="cpu")["state_dict"])
    out, batch, keys = {}, [], []
    for r in df.reset_index(drop=True).itertuples():
        batch.append(np.load(RSNA_CACHE / r.crop_path).astype(np.float32))
        keys.append((r.key, r.level, r.condition))
        if len(batch) == 256:
            _emit(model, batch, keys, device, out)
            batch, keys = [], []
    if batch:
        _emit(model, batch, keys, device, out)
    return out


def _emit(model, batch, keys, device, out):
    img = torch.from_numpy(np.stack(batch)).to(device)
    lv = torch.tensor([LEVEL_TO_INDEX[k[1]] for k in keys], device=device)
    cd = torch.tensor([CONDITION_TO_INDEX[k[2]] for k in keys], device=device)
    p = torch.softmax(model(img, lv, cd).float(), 1).cpu().numpy()
    for i, k in enumerate(keys):
        out[k[0]] = p[i]


def build_teacher(device):
    """Ensemble teacher probs on train+dev (cached, gitignored)."""
    dst = OUTDIR / "v1_7_teacher_probs.parquet"
    if dst.exists():
        return pd.read_parquet(dst)
    sm = load_splits_v1(SPLITS)
    man = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    man = man[man.condition.isin(FORAMINAL) & man.severity_index.isin([0, 1, 2])].copy()
    man["study_id"] = man.study_id.astype(str)
    man["key"] = man.study_id + "|" + man.level.astype(str) + "|" + man.condition
    man["spl"] = man.study_id.map(sm)
    rows = []
    for split in ("train", "dev"):
        sub = man[man.spl == split]
        dep = _deployed_probs(split, sm, device)
        mp = {t: _model_probs(RUN[t], V16[t], sub, device) for t in V16}
        for r in sub.itertuples():
            probs = [dep.get(r.key)] + [mp[t].get(r.key) for t in V16]
            probs = [p for p in probs if p is not None]
            if not probs:
                continue
            t = np.mean(probs, axis=0)
            rows.append({"key": r.key, "split": split, "t0": t[0], "t1": t[1], "t2": t[2]})
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    if TMP.exists():
        TMP.unlink()
    return df


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5, help="KL weight")
    args = ap.parse_args()
    device = select_device("auto")
    print("[teacher] building ensemble teacher probs (deployed + v16 baseline + v16 lss) ...",
          flush=True)  # fmt: skip
    teach = build_teacher(device).set_index("key")
    tr, dv, te = _na._frames("A")  # original labels; teacher provides the soft target
    for d in (tr, dv):
        d["t0"] = d.key.map(teach.t0).fillna(0.0)
        d["t1"] = d.key.map(teach.t1).fillna(0.0)
        d["t2"] = d.key.map(teach.t2).fillna(0.0)

    seed_everything(1337)
    model = ImageClassifier(
        backbone="convnext_tiny", in_chans=3, num_classes=3, use_level_embedding=True,
        use_condition_embedding=True, embed_dim=16, dropout=0.2, pretrained=True,
    ).to(device)  # fmt: skip
    cw = _na._class_weights(tr.severity_index.to_numpy()).to(device)
    tl = DataLoader(_TeachDS(tr), batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    vl = DataLoader(_na.DS(dv), batch_size=64, num_workers=8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.set_backbone_trainable(False)
    best_key, best_state, best = -1.0, None, {}
    for ep in range(args.epochs):
        if ep == 2:
            model.set_backbone_trainable(True)
            for g in opt.param_groups:
                g["lr"] = 3e-4 * 0.2
        model.train()
        for b in tl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                lg = model(b["image"].to(device), b["level_idx"].to(device),
                           b["condition_idx"].to(device))  # fmt: skip
                logp = torch.log_softmax(lg, 1)
                tgt = b["target"].to(device)
                ce = torch.nn.functional.cross_entropy(lg, tgt, weight=cw)
                kl = -(b["teacher"].to(device) * logp).sum(1).mean()  # KL up to teacher entropy
                w = torch.where(tgt == 2, 2.0, 1.0)  # severe upweight
                loss = (w.mean()) * (ce + args.alpha * kl)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        y, p, k = _na._eval(model, vl, device)
        sel = _na._rfor_recall_at_far(y, p, k)
        far = _na._far(y, p)
        sev_r = float(bs.m_severe_recall(y, p))
        ok = (not np.isnan(far)) and far <= 0.5 and 0.0 < sev_r < 1.0
        print(f"    ep{ep} dev R-for r@FAR10 {sel:.3f} sevR {sev_r:.3f} FAR {far:.3f}", flush=True)
        key = sel if ok else -1.0
        if key > best_key:
            best_key, best = key, {"sel_rfor_r@far10": float(sel), "dev_severe_recall": sev_r}
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state or model.state_dict())

    # gate: accept only if dev R-for recall@FAR10 beats the v1.6 baseline dev (~0.79 macro proxy)
    yt, pt, kt = _na._eval(model, DataLoader(_na.DS(te), batch_size=64, num_workers=8), device)
    sides = np.array([k.split("|")[-1] for k in kt])
    st = np.array([k.split("|")[0] for k in kt])
    out = {"dev_best": best, "teacher": "ensemble(deployed,v16_baseline,v16_lss)", "test": {}}
    for c in FORAMINAL:
        m = sides == c
        out["test"][c] = {
            "severe_recall": float(bs.m_severe_recall(yt[m], pt[m])),
            "ci": bs.bootstrap_ci(yt[m], pt[m], st[m], bs.m_severe_recall, n_boot=2000),
            "recall_at_far10": float(bs.make_recall_at_far(0.10)(yt[m], pt[m])),
            "n_severe": int((yt[m] == 2).sum()),
        }  # fmt: skip
    out["test"]["foraminal_macro_severe_recall"] = float(
        np.mean([out["test"][c]["severe_recall"] for c in FORAMINAL])
    )
    (OUTDIR / "teacher_distillation_v1_7.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[teacher] dev R-for r@FAR10 {best.get('sel_rfor_r@far10', 0):.3f}")
    for c in FORAMINAL:
        print(f"  TEST {c}: severe recall {out['test'][c]['severe_recall']:.3f}")
    print(f"  TEST foraminal macro {out['test']['foraminal_macro_severe_recall']:.3f}")
    return 0


class _TeachDS(_na.DS):
    def __getitem__(self, i):
        d = super().__getitem__(i)
        r = self.df.iloc[i]
        d["teacher"] = torch.tensor([r["t0"], r["t1"], r["t2"]], dtype=torch.float32)
        return d


if __name__ == "__main__":
    raise SystemExit(main())
