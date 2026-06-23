"""SpineScoutX command-line interface.

Thin, explicit dispatcher. Heavy modules are imported lazily inside each handler
so ``spinescoutx --help`` and ``spinescoutx doctor`` stay fast and work even when
optional deps (torch/pydicom/monai) are missing. Every command supports
``--help`` and ``--json`` (machine-readable JSON log line on stdout).

This is a research tool. It is NOT diagnostic and NOT for medical decision-making.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .utils.logging import emit_json, get_logger

log = get_logger()

_DISCLAIMER = "Research-only • Not diagnostic • Not for medical decision-making"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ok(args: argparse.Namespace, payload: dict[str, Any], *, status: str = "ok") -> int:
    payload = {"command": getattr(args, "command", "?"), "status": status, **payload}
    if getattr(args, "json", False):
        emit_json(payload)
    return 0 if status == "ok" else 1


def _fail(args: argparse.Namespace, message: str, **extra: Any) -> int:
    log.error(message)
    if getattr(args, "json", False):
        emit_json(
            {
                "command": getattr(args, "command", "?"),
                "status": "error",
                "message": message,
                **extra,
            }
        )
    return 1


def _resolve_run_dir(cfg: Any, run_id: str | None) -> Path:
    from .utils.paths import ensure_dir

    rid = run_id or cfg.name
    return ensure_dir(Path(cfg.output_root) / rid)


# --------------------------------------------------------------------------- #
# command handlers
# --------------------------------------------------------------------------- #
def cmd_doctor(args: argparse.Namespace) -> int:
    """Report environment readiness without requiring any dataset."""
    import importlib.util

    def have(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    info: dict[str, Any] = {"spinescoutx_version": __version__, "python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False

    optional = {
        m: have(m)
        for m in (
            "pydicom",
            "monai",
            "SimpleITK",
            "nibabel",
            "pyarrow",
            "rich",
            "timm",
            "cv2",
            "sklearn",
            "matplotlib",
        )
    }
    info["optional_deps"] = optional
    info["notes"] = []
    if not optional["pydicom"]:
        info["notes"].append(
            "pydicom missing -> cannot decode real RSNA DICOMs (pip install spinescoutx[dicom])."
        )
    if not optional["SimpleITK"] and not optional["nibabel"]:
        info["notes"].append("Neither SimpleITK nor nibabel -> cannot read SPIDER volumes.")

    if getattr(args, "data", False):
        info["datasets"] = _dataset_readiness(args.rsna_root, args.spider_root)

    if not args.json:
        log.info("SpineScoutX %s  (%s)", __version__, _DISCLAIMER)
        log.info(
            "torch=%s cuda=%s device=%s",
            info.get("torch"),
            info.get("cuda_available"),
            info.get("cuda_device"),
        )
        log.info(
            "optional deps: %s", ", ".join(f"{k}={'y' if v else 'n'}" for k, v in optional.items())
        )
        for n in info["notes"]:
            log.info("note: %s", n)
        for name, rep in info.get("datasets", {}).items():
            status = "READY" if rep["exists"] else "MISSING"
            log.info("dataset %s: %s (%s)", name, status, rep.get("root"))
            for miss in rep.get("missing", [])[:6]:
                log.info("    missing: %s", miss)
    return _ok(args, info)


def _dataset_readiness(rsna_root: str, spider_root: str) -> dict[str, Any]:
    """Report RSNA + SPIDER dataset readiness (never raises; no download)."""
    from .data.rsna_index import check_rsna_available
    from .data.spider_index import check_spider_available

    return {
        "rsna": check_rsna_available(rsna_root).to_dict(),
        "spider": check_spider_available(spider_root).to_dict(),
    }


def cmd_prepare_rsna(args: argparse.Namespace) -> int:
    from .data.rsna_index import check_rsna_available

    report = check_rsna_available(args.rsna_root)
    if not report.exists:
        return _fail(
            args,
            f"RSNA data not available under {args.rsna_root}",
            missing=report.missing,
            present=report.present,
        )
    # Real ingestion pipeline (only runs when data is actually present).
    from .data.rsna_index import build_series_index
    from .data.rsna_labels import load_coordinates, load_labels
    from .utils.paths import ensure_dir

    out = ensure_dir(args.out)
    series = build_series_index(args.rsna_root)
    labels = load_labels(args.rsna_root)
    coords = load_coordinates(args.rsna_root)
    summary = {
        "out": str(out),
        "n_series": int(len(series)),
        "n_label_rows": int(len(labels)),
        "n_coord_rows": int(len(coords)),
    }
    log.info("RSNA prepared: %s", summary)
    return _ok(args, summary)


def cmd_prepare_spider(args: argparse.Namespace) -> int:
    from .data.spider_index import check_spider_available

    report = check_spider_available(args.spider_root)
    if not report.exists:
        return _fail(
            args,
            f"SPIDER data not available under {args.spider_root}",
            missing=report.missing,
            present=report.present,
        )
    from .data.spider_index import build_spider_index
    from .utils.paths import ensure_dir

    out = ensure_dir(args.out)
    index = build_spider_index(args.spider_root)
    summary = {"out": str(out), "n_subjects": int(len(index))}
    log.info("SPIDER prepared: %s", summary)
    return _ok(args, summary)


def _load_cfg(path: str) -> Any:
    from .config import load_config

    return load_config(path)


def cmd_train_classifier(args: argparse.Namespace) -> int:
    from .training.train_classifier import train_classifier

    cfg = _load_cfg(args.config)
    run = _resolve_run_dir(cfg, args.run_id)
    result = train_classifier(cfg, run, json_logs=args.json)
    return _ok(
        args,
        {
            "run_dir": str(run),
            "best": result.get("best", {}),
            "checkpoint": result.get("checkpoint"),
        },
    )


def cmd_train_segmenter(args: argparse.Namespace) -> int:
    from .training.train_segmenter import train_segmenter

    cfg = _load_cfg(args.config)
    run = _resolve_run_dir(cfg, args.run_id)
    result = train_segmenter(cfg, run, json_logs=args.json)
    return _ok(
        args,
        {
            "run_dir": str(run),
            "best": result.get("best", {}),
            "checkpoint": result.get("checkpoint"),
        },
    )


def cmd_train_anatomy_guided(args: argparse.Namespace) -> int:
    from .training.train_classifier import train_classifier

    cfg = _load_cfg(args.config)
    if cfg.task != "anatomy_guided":
        log.warning("config task=%s; forcing anatomy_guided for this command", cfg.task)
        cfg.task = "anatomy_guided"
    run = _resolve_run_dir(cfg, args.run_id)
    # train_classifier branches on cfg.task to build the anatomy-guided model + guided loaders.
    result = train_classifier(cfg, run, json_logs=args.json)
    return _ok(
        args,
        {
            "run_dir": str(run),
            "best": result.get("best", {}),
            "checkpoint": result.get("checkpoint"),
        },
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    run = Path(args.run)
    cfg_path = run / "config.json"
    if not cfg_path.exists():
        return _fail(args, f"No config.json in run dir {run}; train first.")
    import json

    from .config import config_from_dict

    cfg = config_from_dict(json.loads(cfg_path.read_text()))
    if cfg.task == "segment":
        from .training.train_segmenter import evaluate_segmenter

        metrics = evaluate_segmenter(cfg, run, split=args.split)
    else:
        from .training.train_classifier import evaluate_classifier

        metrics = evaluate_classifier(cfg, run, split=args.split)
    log.info(
        "evaluation: %s", {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
    )
    return _ok(args, {"run_dir": str(run), "metrics": metrics})


def cmd_ablate(args: argparse.Namespace) -> int:
    from .evaluation.ablation import run_ablation

    cfg = _load_cfg(args.config)
    run = _resolve_run_dir(cfg, args.run_id)
    results = run_ablation(cfg, run, json_logs=args.json)
    from .evaluation.ablation import compare_ablations

    deltas = compare_ablations(results)
    log.info("ablation deltas vs correct: %s", deltas)
    return _ok(args, {"run_dir": str(run), "deltas": deltas})


def cmd_report(args: argparse.Namespace) -> int:
    run = Path(args.run)
    from .reporting.json_report import read_json_report  # noqa: F401  (ensures module present)
    from .utils.paths import ensure_dir

    out_dir = ensure_dir(args.out or "outputs/reports")
    # Build a finding graph for the study from cached run predictions, then write JSON + Markdown.
    import json

    from .reporting.finding_graph import build_finding_graph, finding_graph_to_dict
    from .reporting.json_report import write_json_report
    from .reporting.markdown_report import write_markdown_report

    preds_path = run / "predictions.json"
    if not preds_path.exists():
        return _fail(args, f"No predictions.json in {run}; run evaluate first.")
    preds = json.loads(preds_path.read_text())
    study_preds = [
        p for p in preds.get("predictions", []) if str(p.get("study_id")) == str(args.study_id)
    ]
    if not study_preds:
        return _fail(args, f"No predictions for study {args.study_id} in {preds_path}")
    graph = build_finding_graph(
        args.study_id,
        study_preds,
        run_id=run.name,
        model_version=__version__,
        dataset_source=preds.get("dataset_source", "rsna"),
    )
    json_path = write_json_report(graph, Path(out_dir) / f"{args.study_id}.json")
    md_path = write_markdown_report(graph, Path(out_dir) / f"{args.study_id}.md")
    _ = finding_graph_to_dict(graph)
    log.info("report written: %s, %s", json_path, md_path)
    return _ok(args, {"json": str(json_path), "markdown": str(md_path)})


def cmd_figure(args: argparse.Namespace) -> int:
    from .reporting.json_report import read_json_report
    from .utils.paths import ensure_dir
    from .viz.panels import figures_from_report

    report = read_json_report(args.report)
    out_dir = ensure_dir(args.out or "outputs/figures")
    paths = figures_from_report(report, out_dir)
    log.info("figures: %s", [str(p) for p in paths])
    return _ok(args, {"figures": [str(p) for p in paths]})


def cmd_benchmark(args: argparse.Namespace) -> int:
    import json
    import time

    from .training.optim import select_device

    run = Path(args.run)
    cfg_path = run / "config.json"
    if not cfg_path.exists():
        return _fail(args, f"No config.json in run dir {run}.")
    from .config import config_from_dict

    cfg = config_from_dict(json.loads(cfg_path.read_text()))
    device = select_device(cfg.train.device)
    import torch

    from .models.image_classifier import build_image_classifier

    # Build whatever model the run used and time a forward pass on a synthetic batch.
    model = build_image_classifier(cfg.model).to(device).eval()
    x = torch.randn(
        cfg.train.batch_size,
        cfg.model.in_chans,
        cfg.data.crop_size,
        cfg.data.crop_size,
        device=device,
    )
    lvl = torch.zeros(cfg.train.batch_size, dtype=torch.long, device=device)
    cond = torch.zeros(cfg.train.batch_size, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(x, lvl, cond)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(x, lvl, cond)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    per_batch_ms = 1000.0 * elapsed / max(args.iters, 1)
    mem = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else None
    bench = {
        "device": str(device),
        "batch_size": cfg.train.batch_size,
        "iters": args.iters,
        "per_batch_ms": round(per_batch_ms, 3),
        "per_sample_ms": round(per_batch_ms / max(cfg.train.batch_size, 1), 4),
        "gpu_peak_mem_mb": round(mem, 1) if mem else None,
    }
    log.info("benchmark: %s", bench)
    return _ok(args, bench)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spinescoutx", description=f"SpineScoutX — {_DISCLAIMER}")
    p.add_argument("--version", action="version", version=f"spinescoutx {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text, description=f"{help_text}  [{_DISCLAIMER}]")
        sp.add_argument("--json", action="store_true", help="emit a machine-readable JSON log line")
        sp.set_defaults(func=handler, command=name)
        return sp

    sp = add("doctor", cmd_doctor, "Check environment and optional dependencies")
    sp.add_argument("--data", action="store_true", help="also report RSNA/SPIDER dataset readiness")
    sp.add_argument("--rsna-root", default="data/raw/rsna", help="RSNA root for --data check")
    sp.add_argument("--spider-root", default="data/raw/spider", help="SPIDER root for --data check")

    sp = add("prepare-rsna", cmd_prepare_rsna, "Index/crop RSNA data into a cache")
    sp.add_argument("--rsna-root", required=True)
    sp.add_argument("--out", required=True)

    sp = add("prepare-spider", cmd_prepare_spider, "Index SPIDER segmentation data into a cache")
    sp.add_argument("--spider-root", required=True)
    sp.add_argument("--out", required=True)

    for name, handler, help_text in [
        ("train-classifier", cmd_train_classifier, "Train the image-only baseline (E0)"),
        ("train-segmenter", cmd_train_segmenter, "Train the SPIDER anatomy segmenter (E4)"),
        (
            "train-anatomy-guided",
            cmd_train_anatomy_guided,
            "Train the anatomy-guided classifier (E1)",
        ),
    ]:
        sp = add(name, handler, help_text)
        sp.add_argument("--config", required=True)
        sp.add_argument("--run-id", default=None, help="override run id (default: config name)")

    sp = add("evaluate", cmd_evaluate, "Evaluate a finished run")
    sp.add_argument("--run", required=True)
    sp.add_argument("--split", default="val")

    sp = add("ablate", cmd_ablate, "Counterfactual anatomy ablations (E2/E3)")
    sp.add_argument("--config", required=True)
    sp.add_argument("--run-id", default=None)

    sp = add("report", cmd_report, "Generate finding-graph JSON + Markdown report for a study")
    sp.add_argument("--study-id", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--out", default=None)

    sp = add("figure", cmd_figure, "Render visual panels from a report JSON")
    sp.add_argument("--report", required=True)
    sp.add_argument("--out", default=None)

    sp = add("benchmark", cmd_benchmark, "Benchmark inference latency / memory for a run")
    sp.add_argument("--run", required=True)
    sp.add_argument("--iters", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as e:
        return _fail(args, f"file not found: {e}")
    except (ImportError, ValueError, RuntimeError) as e:
        return _fail(args, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
