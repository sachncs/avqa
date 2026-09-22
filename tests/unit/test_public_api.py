"""Contract tests for AVQA's documented top-level Python API."""

from __future__ import annotations

import torch

import avqa
from avqa import AVQConfig, attention as public_attention
from avqa.exceptions import AVQAError
from avqa.functional import attention as functional_attention


def test_functional_attention_is_exported_and_matches_module_contract() -> None:
    """The documented functional entry point is available from both paths."""
    assert public_attention is functional_attention

    config = AVQConfig()
    query = torch.randn(1, 2, config.attention.embed_dim)
    key = torch.randn(1, 3, config.attention.embed_dim)
    value = torch.randn(1, 3, config.attention.embed_dim)

    output = public_attention(query, key, value, config)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


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
