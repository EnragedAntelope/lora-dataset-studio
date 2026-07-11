#!/usr/bin/env bash
# LoRA Dataset Studio - launch the UI (run ./setup.sh once first)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "[ERROR] .venv not found - run ./setup.sh first."
    exit 1
fi
exec .venv/bin/python app.py
