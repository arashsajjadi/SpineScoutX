"""Guard against committing data, masks, or model weights."""

from __future__ import annotations

import subprocess
from pathlib import Path

_FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".dcm",
    ".nii",
    ".nii.gz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".parquet",
)

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "data",
        "runs",
        "outputs",
        "__pycache__",
        ".venv",
        "venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".idea",
        ".vscode",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_forbidden(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES)


def _tracked_files_via_git(root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def _walk_files(root: Path) -> list[str]:
    found: list[str] = []
    for path in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            found.append(str(path.relative_to(root)))
    return found


def test_no_forbidden_data_files_tracked() -> None:
    root = _repo_root()
    files = _tracked_files_via_git(root)
    if files is None:
        files = _walk_files(root)
    offenders = [f for f in files if _is_forbidden(Path(f).name)]
    assert offenders == [], f"Forbidden data/weight files present: {offenders}"


def test_gitignore_excludes_data_dirs() -> None:
    root = _repo_root()
    gitignore = root / ".gitignore"
    assert gitignore.exists(), ".gitignore must exist"
    text = gitignore.read_text(encoding="utf-8")
    for needed in ("data/", "runs/", "outputs/"):
        assert needed in text, f".gitignore must exclude {needed}"
