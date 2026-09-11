#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> AVQA setup"

# --- core package (editable) ---
echo "--- installing avqa + dev tools"
python -m pip install -e . --no-deps
python -m pip install pytest pytest-cov pytest-benchmark ruff mypy

echo "==> done. Run 'make test' to verify."
