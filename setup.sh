#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> AVQA environment setup"
echo "Python: $PYTHON_BIN"
echo "Virtual environment: $VENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN was not found" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/check_python_support.py --interpreter "$PYTHON_BIN"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" scripts/check_python_support.py --interpreter "$VENV_PYTHON"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -e ".[dev,viz]"

echo "==> environment ready"
echo "Run: $VENV_PYTHON -m pytest"
echo "Or:  source $VENV_DIR/bin/activate"
