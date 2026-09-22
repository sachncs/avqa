#!/usr/bin/env bash
# Build wheel and sdist for AVQA.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

python -m pip install --upgrade build hatchling
python -m build
