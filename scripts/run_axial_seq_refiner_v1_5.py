#!/usr/bin/env python3
"""Phase 5 (v1.5): BiGRU axial level-sequence refiner — does sequence context beat the
independent per-slice scorer at axial level localization?

Pipeline (all real, no GT coords as model input; GT only for the level label/eval):
  1. cache full-stack scorer log-probs per (study, series) [batched fwd], with per-slice level
     labels derived from the labelled-slice manifest (instance -> level); split = manifest split
     (test = locked-test, read once).
  2. train a BiGRU over [scorer_logps(5), norm_z(1)] -> refined 5-level logits; CE on labelled
     slices; select on val +-1 level-hit.
  3. locked-test once: compare localization (+-0/+-1/+-2 hit, median abs err) for raw scorer +
     monotonic decode, raw + positional-prior decode (v2 bar = 0.487 +-1), and BiGRU + decode.

Research-only. Not diagnostic. Reproduce: `python scripts/run_axial_seq_refiner_v1_5.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spinescoutx.data.axial_level import (
    assign_levels_monotonic,
    assign_levels_monotonic_prior,
    level_position_prior,
    load_axial_level_scorer,
)
from spinescoutx.models.axial_seq_refiner import build_axial_seq_refiner
from spinescoutx.training.optim import select_device
from spinescoutx.utils.seed import seed_everything

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
AXCACHE = ROOT / "data/cache/axial_level"
MANIFEST = AXCACHE / "axial_level_manifest.parquet"
SEQCACHE = ROOT / "data/cache/v1_5_axial_seq"
SCORER_RUN = ROOT / "runs/axial_level_scorer"
OUT = ROOT / "outputs/real/axial_seq_refiner_v1_5.json"
DOC = ROOT / "docs/run_logs/axial_seq_refiner_v1_5.md"
NEG = -20.0


def _stack_logps(model, images_dir, study, series, slice_size, device):
    """Batched scorer log-probs over a stack. Returns (zsorted, logps[n,5], norm_z[n])."""
    import cv2

    from spinescoutx.data.axial_match import axial_z_by_instance
    from spinescoutx.data.dicom_io import normalize_intensity, read_dicom

    azs = axial_z_by_instance(images_dir, study, series)
    if len(azs) < 5:
        return None
    zsorted = sorted(azs, key=lambda i: azs[i])
    n = len(zsorted)
    imgs, valid = [], []
    for inst in zsorted:
        try:
            img = normalize_intensity(read_dicom(images_dir / study / series / f"{inst}.dcm"))
        except Exception:  # noqa: BLE001
            imgs.append(None)
            continue
        imgs.append(cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA))
        valid.append(len(imgs) - 1)
    logps = np.full((n, 5), NEG, dtype=np.float32)
    norm_z = np.array([r / (n - 1) for r in range(n)], dtype=np.float32)
    if valid:
        batch = torch.from_numpy(np.stack([imgs[i] for i in valid])[:, None]).float().to(device)
        nz = torch.from_numpy(norm_z[valid][:, None]).to(device)
        with torch.no_grad():
            lp = torch.log_softmax(model(batch, nz), dim=1).cpu().numpy()
        for j, i in enumerate(valid):
            logps[i] = lp[j]
    return zsorted, logps, norm_z


def build_cache(split, model, slice_size, device, series, images_dir):
    """Per (study, series) full-stack sequences with per-slice level labels for one split."""
    dst = SEQCACHE / f"{split}.pt"
    if dst.exists():
        return torch.load(dst, weights_only=False)
    man = pd.read_parquet(MANIFEST)
    man = man[man.split == split].copy()
    man["study_id"] = man.study_id.astype(str)
    seqs, done = [], 0
    for (study, ser), g in man.groupby(["study_id", "series_id"]):
        res = _stack_logps(model, images_dir, str(study), str(ser), slice_size, device)
        if res is None:
            continue
        zsorted, logps, norm_z = res
        rank = {inst: r for r, inst in enumerate(zsorted)}
        n = len(zsorted)
        labels = np.full(n, -1, dtype=np.int64)
        gt = {}
        for r in g.itertuples():
            inst = int(r.instance_number)
            if inst in rank:
                labels[rank[inst]] = int(r.level_idx)
                gt[int(r.level_idx)] = rank[inst]
        if not gt:
            continue
        feats = np.concatenate([logps, norm_z[:, None]], axis=1).astype(np.float32)  # (n,6)
        seqs.append({"study_id": str(study), "feats": feats, "labels": labels, "gt": gt, "n": n})
        done += 1
        if done % 200 == 0:
            print(f"  [{split}] cached {done} stacks", flush=True)
    SEQCACHE.mkdir(parents=True, exist_ok=True)
    torch.save(seqs, dst)
    print(f"[{split}] {len(seqs)} stacks -> {dst}", flush=True)
    return seqs


def _pad_batch(items, device):
    t = max(it["feats"].shape[0] for it in items)
    b = len(items)
    feats = torch.zeros(b, t, items[0]["feats"].shape[1])
    labels = torch.full((b, t), -1, dtype=torch.long)
    lengths = torch.zeros(b, dtype=torch.long)
    for i, it in enumerate(items):
        n = it["feats"].shape[0]
        feats[i, :n] = torch.from_numpy(it["feats"])
        labels[i, :n] = torch.from_numpy(it["labels"])
        lengths[i] = n
    return feats.to(device), labels.to(device), lengths


def _refine_logps(model, feats_np, device):
    """Run the BiGRU on one stack's features -> refined (n,5) log-probs."""
    f = torch.from_numpy(feats_np)[None].to(device)
    length = torch.tensor([feats_np.shape[0]])
    with torch.no_grad():
        logit = model(f, length)[0]
    return torch.log_softmax(logit, dim=1).cpu().numpy()


def _loc_metrics(seqs, decode):
    """Localization vs GT: decode(logps)->{level->rank}; abs err per (study,level)."""
    errs = []
    for s in seqs:
        assign = decode(s)
        for lvl, gt_rank in s["gt"].items():
            errs.append(abs(int(assign[lvl]) - int(gt_rank)))
    errs = np.array(errs)
    return {
        "n": int(len(errs)),
        "hit_0": float((errs == 0).mean()),
        "hit_1": float((errs <= 1).mean()),
        "hit_2": float((errs <= 2).mean()),
        "median_abs_err": float(np.median(errs)),
    }


def main() -> int:
    from spinescoutx.data.rsna_index import RsnaPaths, build_series_index

    device = select_device("auto")
    seed_everything(1337)
    model_sc, slice_size = load_axial_level_scorer(SCORER_RUN, device)
    series = build_series_index(ROOT / "data/raw/rsna")
    series["study_id"] = series.study_id.astype(str)
    images_dir = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    print("[1/3] caching full-stack scorer log-probs ...", flush=True)
    tr = build_cache("train", model_sc, slice_size, device, series, images_dir)
    va = build_cache("val", model_sc, slice_size, device, series, images_dir)
    te = build_cache("test", model_sc, slice_size, device, series, images_dir)
    print(f"stacks: train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    # positional prior (train-derived) for the v2 prior-decode baseline
    prior = level_position_prior(AXCACHE)

    def dec_mono(s):
        return assign_levels_monotonic(s["feats"][:, :5])

    def dec_prior(s):
        return assign_levels_monotonic_prior(s["feats"][:, :5], s["feats"][:, 5], prior, beta=1.0)

    print("[2/3] training BiGRU refiner ...", flush=True)
    ref = build_axial_seq_refiner().to(device)
    opt = torch.optim.AdamW(ref.parameters(), lr=2e-3, weight_decay=1e-4)
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    bs_seq = 32
    best_key, best_state = -1.0, None
    for ep in range(40):
        ref.train()
        perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), bs_seq):
            items = [tr[j] for j in perm[i : i + bs_seq]]
            feats, labels, lengths = _pad_batch(items, device)
            opt.zero_grad(set_to_none=True)
            logit = ref(feats, lengths)
            loss = lossfn(logit.reshape(-1, 5), labels.reshape(-1))
            loss.backward()
            opt.step()
        ref.eval()

        def dec_ref(s):
            return assign_levels_monotonic(_refine_logps(ref, s["feats"], device))

        m = _loc_metrics(va, dec_ref)
        if m["hit_1"] > best_key:
            best_key = m["hit_1"]
            best_state = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
        if ep % 5 == 0 or ep == 39:
            print(f"  ep{ep} val hit_1 {m['hit_1']:.3f} hit_0 {m['hit_0']:.3f}", flush=True)
    ref.load_state_dict(best_state)
    ref.eval()
    ckpt_dir = ROOT / "runs/axial_seq_refiner_v1_5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ref.state_dict(), "val_hit_1": best_key}, ckpt_dir / "best.pt")

    def dec_ref(s):
        return assign_levels_monotonic(_refine_logps(ref, s["feats"], device))

    print("[3/3] locked-test localization (once) ...", flush=True)
    out = {
        "protocol": "manifest split (test=locked-test, once); GT only for label/eval",
        "val": {
            "raw_monotonic": _loc_metrics(va, dec_mono),
            "raw_prior": _loc_metrics(va, dec_prior),
            "bigru_monotonic": _loc_metrics(va, dec_ref),
        },
        "test": {
            "raw_monotonic": _loc_metrics(te, dec_mono),
            "raw_prior": _loc_metrics(te, dec_prior),
            "bigru_monotonic": _loc_metrics(te, dec_ref),
        },
        "v2_reference_test_hit_1": {"current_decode": 0.432, "prior_decode": 0.487},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    for split in ("val", "test"):
        print(f"== {split} ==")
        for k, m in out[split].items():
            print(
                f"  {k:18s} hit_1 {m['hit_1']:.3f} hit_0 {m['hit_0']:.3f} "
                f"hit_2 {m['hit_2']:.3f} medAE {m['median_abs_err']:.1f} (n={m['n']})"
            )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
