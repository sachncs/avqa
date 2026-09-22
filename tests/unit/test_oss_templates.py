"""Contract tests for the repository's OSS intake templates."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_compatibility_issue_form_captures_reproducibility_context() -> None:
    """Compatibility reports request the fields needed to reproduce them."""
    template = (ROOT / ".github" / "ISSUE_TEMPLATE" / "compatibility.yml").read_text(
        encoding="utf-8"
    )

    for field in (
        "AVQA revision or version",
        "Python version",
        "PyTorch version",
        "Operating system and device",
        "Installation details",
        "Minimal reproduction",
        "Expected and actual behavior",
    ):
        assert field in template

    assert "labels:" in template
    assert "compatibility" in template
    assert "required: true" in template
