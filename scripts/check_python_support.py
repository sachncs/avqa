"""Validate that an interpreter is inside AVQA's supported Python range."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
_REQUIRES_PYTHON = re.compile(r'requires-python\s*=\s*"([^\"]+)"')
_SUPPORTED_RANGE = re.compile(r">=(\d+)\.(\d+),<(\d+)\.(\d+)")


def supported_python_range() -> tuple[tuple[int, int], tuple[int, int]]:
    """Read the package's lower-inclusive, upper-exclusive Python range."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requires_match = _REQUIRES_PYTHON.search(pyproject)
    if requires_match is None:
        raise ValueError("pyproject.toml is missing project.requires-python")
    range_match = _SUPPORTED_RANGE.fullmatch(requires_match.group(1))
    if range_match is None:
        raise ValueError(
            "project.requires-python must use the form '>=X.Y,<X.Y' for bootstrap validation"
        )
    lower = (int(range_match.group(1)), int(range_match.group(2)))
    upper = (int(range_match.group(3)), int(range_match.group(4)))
    if lower >= upper:
        raise ValueError("project.requires-python has an empty supported range")
    return lower, upper


def is_supported_python(version: tuple[int, int]) -> bool:
    """Return whether a major/minor Python version is supported by AVQA."""
    lower, upper = supported_python_range()
    return lower <= version < upper


def interpreter_version(executable: str) -> tuple[int, int]:
    """Read the major/minor version from an interpreter executable."""
    output = subprocess.check_output(
        [executable, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
    ).strip()
    major, minor = output.split(".", maxsplit=1)
    return int(major), int(minor)


def validate_interpreter(executable: str) -> None:
    """Raise a clear error when an interpreter is outside the support range."""
    version = interpreter_version(executable)
    lower, upper = supported_python_range()
    if not lower <= version < upper:
        raise SystemExit(
            f"error: {executable} is Python {version[0]}.{version[1]}; "
            f"AVQA supports Python {lower[0]}.{lower[1]} through "
            f"{upper[0]}.{upper[1] - 1}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the interpreter selected by the caller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpreter", required=True)
    args = parser.parse_args(argv)
    validate_interpreter(args.interpreter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
