"""Checks that documented Python examples remain syntactically valid."""

from __future__ import annotations

from pathlib import Path
import re

DOCS_PATH = Path(__file__).parents[2] / "docs" / "usage.md"


def test_usage_python_examples_compile() -> None:
    """Every Python code fence in the user guide parses successfully."""
    content = DOCS_PATH.read_text(encoding="utf-8")
    examples = re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL)

    assert examples
    for index, example in enumerate(examples):
        compile(example, f"docs/usage.md code block {index + 1}", "exec")
