#!/usr/bin/env bash
# Execute every checked-in example against the installed package.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob
examples=(examples/*.py)
test "${#examples[@]}" -gt 0

for example in "${examples[@]}"; do
  echo "Running $example"
  python "$example"
done
