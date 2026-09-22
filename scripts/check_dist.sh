#!/usr/bin/env bash
# Verify the built distribution: validate metadata, then install wheel + sdist
# in isolated virtual environments and import-test the package.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

python -m pip install --upgrade twine
python -m twine check dist/*

WHEEL="$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)"
SDIST="$(find dist -maxdepth 1 -type f -name '*.tar.gz' -print -quit)"
test -n "$WHEEL"
test -n "$SDIST"

SMOKE_ROOT="$(mktemp -d)"
trap 'rm -rf "$SMOKE_ROOT"' EXIT

for artifact in wheel sdist; do
  python -m venv "$SMOKE_ROOT/$artifact"
  "$SMOKE_ROOT/$artifact/bin/python" -m pip install --upgrade pip
  "$SMOKE_ROOT/$artifact/bin/python" -m pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cpu
done

"$SMOKE_ROOT/wheel/bin/python" -m pip install "$WHEEL"
"$SMOKE_ROOT/wheel/bin/python" -c "import avqa; print(avqa.__version__)"
"$SMOKE_ROOT/sdist/bin/python" -m pip install "$SDIST"
"$SMOKE_ROOT/sdist/bin/python" -c "import avqa; print(avqa.__version__)"
