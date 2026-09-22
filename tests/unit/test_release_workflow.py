"""Contract tests for the release workflow's publication safety boundary."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "release.yml"


def test_pypi_publication_requires_explicit_manual_dispatch() -> None:
    """A version tag must verify/build without publishing to PyPI."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "type: boolean" in workflow
    assert "default: false" in workflow
    assert "if: github.event_name == 'workflow_dispatch' && inputs.publish == true" in workflow


def test_pypi_publication_uses_oidc_without_long_lived_token() -> None:
    """The publish job must use trusted publishing rather than a secret token."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in workflow
    assert "packages-dir: dist/" in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert "password:" not in workflow
