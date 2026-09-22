"""Contract tests for AVQA's documented top-level Python API."""

from __future__ import annotations

import avqa
from avqa.exceptions import AVQAError


def test_all_public_exports_resolve_and_are_unique() -> None:
    """Every declared export must be importable from the package root."""
    assert len(avqa.__all__) == len(set(avqa.__all__))
    missing = [name for name in avqa.__all__ if not hasattr(avqa, name)]
    assert missing == []


def test_public_aliases_keep_their_documented_identity() -> None:
    """Common aliases must point at the canonical implementation objects."""
    assert avqa.Codebook is avqa.HierarchicalCodebook
    assert avqa.__version_info__ == (0, 1, 0)


def test_public_exception_types_share_the_root_error_contract() -> None:
    """Consumers can catch every documented error through AVQAError."""
    exception_names = (
        "BackendError",
        "CodebookError",
        "ConfigurationError",
        "DeviceError",
        "DtypeError",
        "MergeError",
        "NotInitializedError",
        "RoutingError",
        "ShapeError",
    )
    assert all(issubclass(getattr(avqa, name), AVQAError) for name in exception_names)
