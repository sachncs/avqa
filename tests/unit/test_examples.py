"""Contract tests for the public example programs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_all_examples_compile() -> None:
    """Every checked-in example must remain syntactically valid Python."""
    examples = sorted((ROOT / "examples").glob("*.py"))
    assert examples, "the repository must contain at least one example"
    for example in examples:
        compile(example.read_text(encoding="utf-8"), str(example), "exec")
