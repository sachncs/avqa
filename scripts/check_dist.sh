#!/usr/bin/env bash
# Verify the built distribution: install wheel + sdist in a temp venv
# and import-test the package.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

python -m pip install --upgrade twine
python -m twine check dist/*
