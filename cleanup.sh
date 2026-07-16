#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> AVQA cleanup"

# build artifacts
rm -rf build/ dist/ *.egg-info src/*.egg-info

# caches
rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/

# Python bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# editable install metadata
python -m pip uninstall avqa -y 2>/dev/null || true

echo "==> done"
