"""Contract test for the environment bootstrap smoke check."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[2]


def test_environment_smoke_check_passes() -> None:
    """The installed public API can execute a finite minimal forward pass."""
    result = subprocess.run(
        [sys.executable, "scripts/check_environment.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "AVQA environment smoke passed" in result.stdout
