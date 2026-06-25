#!/usr/bin/env bash
# v1.8c — install the real MedSAM2 stack (official sam2 + checkpoint already local).
# Research-only. Tokens are read locally by private_load_tokens_v1_8b.py; never echoed.
set -euo pipefail
python -c "import importlib.util as u; import sys; sys.exit(0 if u.find_spec('sam2') else 1)" \
  || pip install sam2
python - <<'PY'
from spinescoutx.segmentation.medsam2_runner import available
print("real MedSAM2 available:", available())
PY
