#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> AVQA setup"

# --- core package (editable) ---
echo "--- installing avqa + dev tools"
python -m pip install -e ".[huggingface,viz]" --no-deps 2>/dev/null || \
python -m pip install -e . --no-deps
python -m pip install pytest pytest-cov pytest-benchmark ruff mypy

# --- CUDA-only deps (Linux with CUDA only) ---
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "--- CUDA detected, installing flash-attn + xformers + vllm + triton"
  python -m pip install "flash-attn>=2.5" "xformers>=0.27" "vllm>=0.5" "triton>=2.2"
else
  echo "--- no CUDA (macOS or CPU-only); skipping flash-attn, xformers, vllm, triton"
fi

echo "==> done. Run 'make test' to verify."
