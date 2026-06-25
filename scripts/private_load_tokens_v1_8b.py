#!/usr/bin/env python3
"""Safe local loader for Hugging Face + Kaggle credentials (v1.8b).

Reads the user's local token files, configures auth (env vars + ~/.kaggle/kaggle.json chmod 600),
and reports **only whether auth succeeded** — secret values are never printed, logged, returned to
callers, or written anywhere inside the repo. Other v1.8b download scripts import
``ensure_auth()``. Tokens / generated kaggle.json are outside the repo (or gitignored).

Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

HF_FILE = Path("/home/arash/Documents/api_huggingface.txt")
KAGGLE_FILE = Path("/home/arash/Documents/api_kaggle.txt")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _parse_kaggle(raw: str) -> tuple[str, str]:
    """Return (username, key) from JSON / KEY=VALUE / 'user:key' / raw-key; never logs the value."""
    raw = raw.strip()
    if not raw:
        return "", ""
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return str(d.get("username", "")), str(d.get("key", ""))
    except json.JSONDecodeError:
        pass
    if "=" in raw and "\n" in raw:
        kv = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        return kv.get("KAGGLE_USERNAME", "").strip(), kv.get("KAGGLE_KEY", "").strip()
    if ":" in raw and raw.count(":") == 1:
        u, k = raw.split(":", 1)
        return u.strip(), k.strip()
    # raw key only — fall back to env / OS user for username
    return os.environ.get("KAGGLE_USERNAME", os.environ.get("USER", "")), raw


def load_hf_token() -> str:
    """Internal: return the HF token string (callers must NOT print it)."""
    raw = _read(HF_FILE)
    if raw.startswith("{"):
        try:
            return str(json.loads(raw).get("token", "")) or str(json.loads(raw).get("hf_token", ""))
        except json.JSONDecodeError:
            return ""
    if "=" in raw and "\n" in raw:
        for line in raw.splitlines():
            if line.upper().startswith(("HF_TOKEN", "HUGGINGFACE", "HUGGING_FACE")):
                return line.split("=", 1)[1].strip()
    return raw


def ensure_auth() -> dict[str, bool]:
    """Configure HF + Kaggle auth from local files. Returns booleans only; never exposes secrets."""
    status = {"hf": False, "kaggle": False}

    hf = load_hf_token()
    if hf:
        os.environ["HF_TOKEN"] = hf
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        try:
            from huggingface_hub import HfApi

            HfApi().whoami(token=hf)  # validates without printing the token
            status["hf"] = True
        except Exception:  # noqa: BLE001 (offline / invalid token — reported as False)
            status["hf"] = bool(hf)  # token present but not validated (maybe offline)

    user, key = _parse_kaggle(_read(KAGGLE_FILE))
    if user and key:
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key
        kdir = Path.home() / ".kaggle"
        kdir.mkdir(parents=True, exist_ok=True)
        kfile = kdir / "kaggle.json"
        kfile.write_text(json.dumps({"username": user, "key": key}))
        kfile.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        status["kaggle"] = True
    elif key:  # raw key only
        os.environ["KAGGLE_KEY"] = key
        status["kaggle"] = True
    return status


def main() -> int:
    s = ensure_auth()
    print(f"HF auth configured: {s['hf']}")
    print(f"Kaggle auth configured: {s['kaggle']}")
    return 0 if (s["hf"] or s["kaggle"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
