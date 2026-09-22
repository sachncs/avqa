"""Validate that a release tag matches the canonical package version."""

from __future__ import annotations

from pathlib import Path
import re
import sys


def canonical_version() -> str:
    text = (Path(__file__).resolve().parents[1] / "src/avqa/version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__ = "([0-9]+\.[0-9]+\.[0-9]+)"', text)
    if match is None:
        raise SystemExit("cannot find canonical package version")
    return match.group(1)


def main(tag: str) -> int:
    expected = f"v{canonical_version()}"
    if tag != expected:
        raise SystemExit(f"release tag {tag!r} does not match canonical version {expected!r}")
    print(f"release tag matches canonical version {expected}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_tag_version.py TAG")
    raise SystemExit(main(sys.argv[1]))
