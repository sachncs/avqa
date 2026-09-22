"""Contract tests for tracked benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
RAW_ROOT = ROOT / "benchmarks" / "raw"


def test_tracked_benchmark_records_have_reproducibility_artifacts() -> None:
    """Every published experiment has config, raw output, and interpretation."""
    records = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())
    assert records, "expected at least one tracked benchmark record"

    for record in records:
        for filename in ("config.json", "raw.json", "summary.md"):
            artifact = record / filename
            assert artifact.is_file(), f"missing {artifact}"
            assert artifact.stat().st_size > 0, f"empty {artifact}"

        configuration = json.loads((record / "config.json").read_text(encoding="utf-8"))
        assert isinstance(configuration, dict)
        assert configuration, f"empty benchmark configuration: {record}"
