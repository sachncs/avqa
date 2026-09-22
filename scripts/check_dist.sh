#!/usr/bin/env bash
# Verify the built distribution: validate metadata, then install wheel + sdist
# in isolated virtual environments and import-test the package.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

python -m pip install --upgrade twine
python -m twine check dist/*

WHEEL_COUNT="$(find dist -maxdepth 1 -type f -name '*.whl' -print | wc -l | tr -d ' ')"
SDIST_COUNT="$(find dist -maxdepth 1 -type f -name '*.tar.gz' -print | wc -l | tr -d ' ')"
if [ "$WHEEL_COUNT" -ne 1 ] || [ "$SDIST_COUNT" -ne 1 ]; then
  echo "error: expected exactly one wheel and one sdist in dist/" >&2
  echo "found: wheels=$WHEEL_COUNT sdists=$SDIST_COUNT" >&2
  exit 1
fi

WHEEL="$(find dist -maxdepth 1 -type f -name '*.whl' -print)"
SDIST="$(find dist -maxdepth 1 -type f -name '*.tar.gz' -print)"

SMOKE_ROOT="$(mktemp -d)"
trap 'rm -rf "$SMOKE_ROOT"' EXIT

for artifact in wheel sdist; do
  python -m venv "$SMOKE_ROOT/$artifact"
  "$SMOKE_ROOT/$artifact/bin/python" -m pip install --upgrade pip
  "$SMOKE_ROOT/$artifact/bin/python" -m pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cpu
done

"$SMOKE_ROOT/wheel/bin/python" -m pip install "$WHEEL"
"$SMOKE_ROOT/wheel/bin/python" -c "import importlib.resources as r; import avqa; assert r.files('avqa').joinpath('py.typed').is_file(); print(avqa.__version__)"
"$SMOKE_ROOT/sdist/bin/python" -m pip install "$SDIST"
"$SMOKE_ROOT/sdist/bin/python" -c "import importlib.resources as r; import avqa; assert r.files('avqa').joinpath('py.typed').is_file(); print(avqa.__version__)"

for environment in wheel sdist; do
  "$SMOKE_ROOT/$environment/bin/python" - <<'PY'
import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig

config = AVQConfig(
    attention=AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
    codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
    routing=RoutingConfig(refinement_budget=3),
)
module = AVQAttention(config, in_proj=False, out_proj=False)
query = torch.randn(1, 2, 32)
output = module(query, query, query)
assert output.shape == query.shape
assert torch.isfinite(output).all()
print("forward smoke passed")
PY
done
