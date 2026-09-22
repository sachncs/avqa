"""Validate release metadata stays synchronized across project surfaces."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    """Read a repository file as UTF-8 text."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def main() -> int:
    """Return a non-zero status when public release metadata has drifted."""
    version_match = re.search(
        r'__version__ = "([0-9]+\.[0-9]+\.[0-9]+)"', read("src/avqa/version.py")
    )
    if version_match is None:
        raise SystemExit("cannot find canonical version in src/avqa/version.py")
    version = version_match.group(1)

    checks = {
        "README status": "public alpha" in read("README.md").lower()
        and version in read("README.md"),
        "release notes": f"## v{version} " in read("RELEASE.md"),
        "site canonical release injection": "RELEASE_VERSION = import.meta.env.VITE_AVQA_VERSION"
        in read("site/src/lib/links.ts")
        and "src/avqa/version.py" in read("site/vite.config.ts"),
        "package dynamic version": 'dynamic = ["version"]' in read("pyproject.toml"),
        "python support range": 'requires-python = ">=3.10,<3.16"' in read("pyproject.toml")
        and "Python 3.10\u20133.15" in read("README.md")
        and "Python 3.10\u20133.15" in read("RELEASE.md"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("release metadata drift: " + ", ".join(failed))
    print(f"release metadata is synchronized for v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
