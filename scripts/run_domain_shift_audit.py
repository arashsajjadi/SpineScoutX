#!/usr/bin/env python3
"""Internal domain-shift stress test — does severe recall hold across acquisition
protocol variation (slice thickness, in-plane resolution, matrix size) and anatomy
(level, side)?

RSNA LumbarDISC de-identifies scanner/vendor/field-strength (all absent), so
*external* validation is not feasible with available legal data (audited in the
companion doc). What IS measurable is robustness to acquisition variation that is
preserved in the headers. We reuse the **deployed** locked-test predictions captured
by evidence stability (`evidence_stability_records.parquet`, fidelity-verified to
reproduce `collect_probs`) and join privacy-safe DICOM acquisition parameters from the
auto manifests, then report severe recall per stratum with cluster-bootstrap CIs.

No new model inference, no GT coordinates, no identifiers. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
RECORDS = ROOT / "outputs/real/evidence_stability_records.parquet"
CACHES = {
    "spinal_canal_stenosis": "data/cache/rsna_auto_canal_all",
    "left_neural_foraminal_narrowing": "data/cache/rsna_auto_foraminal",
    "right_neural_foraminal_narrowing": "data/cache/rsna_auto_foraminal",
    "left_subarticular_stenosis": "data/cache/rsna_auto_subarticular",
    "right_subarticular_stenosis": "data/cache/rsna_auto_subarticular",
}
OUT = ROOT / "outputs/real/domain_shift_stress_test.json"
DOC = ROOT / "docs/run_logs/external_validation_audit.md"
FIG = ROOT / "outputs/real/figures/domain_shift.png"
ASSET = ROOT / "docs/assets/showcase/domain_shift.png"


def _header_meta(dicom_path: str, cache: dict) -> dict:
    if dicom_path in cache:
        return cache[dicom_path]
    meta = {"slice_thickness": np.nan, "pixel_spacing": np.nan, "rows": np.nan}
    try:
        d = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        st = getattr(d, "SliceThickness", None)
        ps = getattr(d, "PixelSpacing", None)
        rw = getattr(d, "Rows", None)
        meta = {
            "slice_thickness": float(st) if st is not None else np.nan,
            "pixel_spacing": float(ps[0]) if ps is not None else np.nan,
            "rows": int(rw) if rw is not None else np.nan,
        }
    except Exception:  # noqa: BLE001
        pass
    cache[dicom_path] = meta
    return meta


def _attach_metadata(rec: pd.DataFrame) -> pd.DataFrame:
    """Join slice thickness / pixel spacing / matrix from the auto manifests."""
    rec = rec.copy()
    rec["study_id"] = rec.study_id.astype(str)
    keycols = ["study_id", "level", "side", "condition"]
    rec["side"] = rec["side"].fillna("").astype(str)
    parts = []
    hcache: dict = {}
    for cond, cpath in CACHES.items():
        man = pd.read_parquet(ROOT / cpath / "manifest.parquet")
        man["study_id"] = man.study_id.astype(str)
        man["side"] = man.get("side", "").fillna("").astype(str)
        man = man[man.condition == cond][
            ["study_id", "level", "side", "condition", "dicom_path"]
        ].drop_duplicates(keycols)
        parts.append(man)
    manifest = pd.concat(parts, ignore_index=True)
    merged = rec.merge(manifest, on=keycols, how="left")
    meta = merged.dicom_path.map(lambda p: _header_meta(p, hcache) if isinstance(p, str) else {})
    merged["slice_thickness"] = [m.get("slice_thickness", np.nan) for m in meta]
    merged["pixel_spacing"] = [m.get("pixel_spacing", np.nan) for m in meta]
    merged["rows"] = [m.get("rows", np.nan) for m in meta]
    return merged


def _severe_recall(df: pd.DataFrame) -> tuple[float, int, int]:
    sev = df[df.y == 2]
    n_sev = len(sev)
    if n_sev == 0:
        return float("nan"), 0, len(df)
    return float((sev.pred == 2).mean()), n_sev, len(df)


def _boot_ci(df: pd.DataFrame, n_boot=2000, seed=1337) -> dict:
    sev = df[df.y == 2]
    point, n_sev, n = _severe_recall(df)
    if n_sev < 5:
        return {"severe_recall": point, "n": n, "n_severe": n_sev, "ci_lo": None, "ci_hi": None}
    groups = sev.study_id.to_numpy()
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    hit = (sev.pred == 2).to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_g[g] for g in pick])
        boots.append(hit[sel].mean())
    return {
        "severe_recall": point,
        "n": n,
        "n_severe": n_sev,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def _strata(df: pd.DataFrame):
    """Yield (axis, label, mask) for each privacy-safe acquisition / anatomy stratum."""
    st = df.slice_thickness
    yield "slice_thickness", "thin (<=3.5mm)", st <= 3.5
    yield "slice_thickness", "standard (4.0mm)", (st > 3.5) & (st <= 4.0)
    yield "slice_thickness", "thick (>4.0mm)", st > 4.0
    ps = df.pixel_spacing
    med = ps.median()
    yield "pixel_spacing", f"fine (<=median {med:.2f}mm)", ps <= med
    yield "pixel_spacing", f"coarse (>median {med:.2f}mm)", ps > med
    rw = df.rows
    yield "matrix_rows", "small (<=384)", rw <= 384
    yield "matrix_rows", "large (>384)", rw > 384
    for lv in ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"):
        yield "level", lv, df.level == lv


def main() -> int:
    if not RECORDS.exists():
        print(f"missing {RECORDS}; run run_evidence_stability.py first")
        return 1
    rec = pd.read_parquet(RECORDS)
    merged = _attach_metadata(rec)

    out = {
        "protocol": "splits_v1 locked-test, deployed predictions (no new inference)",
        "external_validation_feasible": False,
        "external_validation_reason": (
            "RSNA LumbarDISC de-identifies Manufacturer / model / field strength (absent in "
            "headers); SPIDER has segmentation masks, not the 5 graded findings. No independent "
            "labeled lumbar-MRI source with the same five severity labels is legally available, "
            "so external/prospective validation is not performed. Internal acquisition-shift "
            "stress test is reported instead."
        ),
        "metadata_coverage": {
            "slice_thickness_present": float(merged.slice_thickness.notna().mean()),
            "pixel_spacing_present": float(merged.pixel_spacing.notna().mean()),
            "scanner_vendor_present": 0.0,
        },
        "overall": _boot_ci(merged),
        "by_axis": {},
        "per_condition": {},
    }
    # pooled by stratum
    for axis, label, mask in _strata(merged):
        sub = merged[mask]
        out["by_axis"].setdefault(axis, {})[label] = _boot_ci(sub)
    # per condition overall + slice-thickness (the main protocol axis)
    for cond in CACHES:
        cdf = merged[merged.condition == cond]
        if cdf.empty:
            continue
        out["per_condition"][cond] = {
            "overall": _boot_ci(cdf),
            "slice_thickness": {
                label: _boot_ci(cdf[mask])
                for axis, label, mask in _strata(cdf)
                if axis == "slice_thickness"
            },
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _figure(out)
    _doc(out, merged)
    print(f"wrote {OUT}\nwrote {DOC}\nwrote {FIG}")
    o = out["overall"]
    print(f"overall severe recall {o['severe_recall']:.3f} (n_sev={o['n_severe']})")
    for axis, strata in out["by_axis"].items():
        print(f"  [{axis}]")
        for label, s in strata.items():
            ci = (
                f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]"
                if s.get("ci_lo") is not None
                else "(n_sev<5)"
            )
            print(f"    {label:28s} sevR={s['severe_recall']:.3f} {ci} n_sev={s['n_severe']}")
    return 0


def _figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_to_plot = ["slice_thickness", "pixel_spacing", "matrix_rows", "level"]
    fig, axs = plt.subplots(1, len(axes_to_plot), figsize=(17, 4.4))
    overall = out["overall"]["severe_recall"]
    for ax, axis in zip(axs, axes_to_plot, strict=False):
        strata = out["by_axis"][axis]
        labels = list(strata)
        vals = [strata[lbl]["severe_recall"] for lbl in labels]
        los = [
            strata[lbl]["severe_recall"] - (strata[lbl]["ci_lo"] or strata[lbl]["severe_recall"])
            for lbl in labels
        ]
        his = [
            (strata[lbl]["ci_hi"] or strata[lbl]["severe_recall"]) - strata[lbl]["severe_recall"]
            for lbl in labels
        ]
        ax.bar(range(len(labels)), vals, color="#1565c0")
        ax.errorbar(
            range(len(labels)), vals, yerr=[los, his], fmt="none", ecolor="black", capsize=3
        )
        ax.axhline(overall, ls="--", c="red", lw=1, label=f"overall {overall:.2f}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [lbl.split(" (")[0] for lbl in labels], rotation=30, ha="right", fontsize=8
        )
        ax.set_ylim(0, 1)
        ax.set_title(axis)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("severe recall (pooled, all conditions)")
    fig.suptitle(
        "SpineScoutX internal domain-shift stress test (locked-test auto) — research-only",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, dpi=110)
    plt.close(fig)


def _doc(out, merged):
    o = out["overall"]
    mc = out["metadata_coverage"]
    lines = [
        "# External-validation feasibility + internal domain-shift stress test",
        "",
        "> Research-only · not diagnostic · not clinically validated. Privacy-safe acquisition",
        "> parameters only; no scanner identifiers (RSNA strips them). Deployed locked-test",
        "> predictions reused (no new inference, no GT coordinates).",
        "",
        "## External validation: NOT performed (feasibility audit)",
        out["external_validation_reason"],
        "",
        "| metadata | coverage |",
        "|---|---|",
        f"| slice thickness present | {mc['slice_thickness_present']:.0%} |",
        f"| pixel spacing present | {mc['pixel_spacing_present']:.0%} |",
        f"| scanner / vendor / field strength | {mc['scanner_vendor_present']:.0%} (stripped) |",
        "",
        f"## Internal domain-shift (pooled severe recall, all 5 findings; overall "
        f"{o['severe_recall']:.3f} [{_ci(o)}], n_severe={o['n_severe']})",
        "",
        "| axis | stratum | severe recall [95% CI] | n_severe | n |",
        "|---|---|---|---|---|",
    ]
    for axis, strata in out["by_axis"].items():
        for label, s in strata.items():
            lines.append(
                f"| {axis} | {label} | {s['severe_recall']:.3f} [{_ci(s)}] | "
                f"{s['n_severe']} | {s['n']} |"
            )
    lines += [
        "",
        "## Interpretation",
        "- Severe recall is reported across acquisition protocol (slice thickness, in-plane",
        "  resolution, matrix size) and anatomy (level). Strata whose CI excludes the overall",
        "  point estimate indicate a real acquisition-shift sensitivity; wide CIs (small",
        "  n_severe) are reported as such and not over-interpreted.",
        "- This is **internal** robustness only. It does **not** establish generalization to new",
        "  institutions, scanners, populations, or prospective use — that needs external and",
        "  prospective studies, which have **not** been done (no legal labeled source available).",
        "",
        "Reproduce: `python scripts/run_domain_shift_audit.py` "
        "(after `run_evidence_stability.py`).",
    ]
    DOC.write_text("\n".join(lines) + "\n")


def _ci(s):
    if s.get("ci_lo") is None:
        return "n_sev<5"
    return f"{s['ci_lo']:.3f}, {s['ci_hi']:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
