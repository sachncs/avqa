"""Tests for the environment bootstrap Python-version boundary."""

from __future__ import annotations

from scripts.check_python_support import is_supported_python, supported_python_range


def test_supported_python_range_matches_package_metadata() -> None:
    """Bootstrap validation reads the same range declared by the package."""
    assert supported_python_range() == ((3, 10), (3, 13))


def test_supported_python_range_is_lower_inclusive_and_upper_exclusive() -> None:
    """The boundary accepts 3.10--3.12 and rejects adjacent versions."""
    assert not is_supported_python((3, 9))
    assert is_supported_python((3, 10))
    assert is_supported_python((3, 11))
    assert is_supported_python((3, 12))
    assert not is_supported_python((3, 13))
