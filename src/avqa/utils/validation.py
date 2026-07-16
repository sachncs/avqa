"""Tensor validation utilities for AVQA.

Spec §6.12 requires every public API to validate tensor rank, dtype,
device, shape compatibility, and contiguity (where required). This module
provides the canonical validation helpers used across the codebase.

Validation MAY be disabled in optimized execution modes (e.g., by setting
``AVQAConfig.skip_validation=True``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch

from avqa.exceptions import DeviceError, DtypeError, ShapeError

ShapeLike: TypeAlias = Sequence[int] | torch.Size | torch.Tensor
"""Anything that can be coerced to a tensor shape."""

RankLike: TypeAlias = int | tuple[int, ...] | list[int]
"""A rank (single int) or a set of allowed ranks."""


def coerce_shape(shape: ShapeLike) -> torch.Size:
    """Coerce a shape-like value to a :class:`torch.Size`."""
    if isinstance(shape, torch.Tensor):
        return torch.Size(shape.shape)
    return torch.Size(tuple(int(d) for d in shape))


def shape_to_string(shape: torch.Size) -> str:
    """Format a shape for error messages."""
    return f"[{', '.join(str(d) for d in shape)}]"


def validate_shape(
    tensor: torch.Tensor,
    expected: ShapeLike,
    *,
    name: str = "tensor",
) -> None:
    """Validate that ``tensor`` has the expected shape.

    Supports several match modes:

    - Exact shape: ``expected=[2, 8, 128, 64]``
    - Wildcard dimension: pass ``-1`` in a position to match any size.
    - Symbolic rank: only the rank must match, e.g., ``expected=(2, 8, -1, 64)``.

    Args:
        tensor: The tensor to validate.
        expected: Expected shape. ``-1`` in any position matches any size.
        name: Variable name used in error messages.

    Raises:
        ShapeError: If the tensor shape does not match ``expected``.

    Example:
        >>> import torch
        >>> t = torch.zeros(2, 8, 128, 64)
        >>> validate_shape(t, [2, 8, 128, 64])
        >>> validate_shape(t, [2, 8, -1, 64])
        >>> validate_shape(t, [2, 8, 256, 64])
        Traceback (most recent call last):
            ...
        avqa.exceptions.ShapeError: tensor shape mismatch: expected=[2, 8, 256, 64] actual=[2, 8, 128, 64]
    """
    actual = tensor.shape
    expected_size = coerce_shape(expected)
    if len(actual) != len(expected_size):
        raise ShapeError(
            f"{name} rank mismatch: expected rank {len(expected_size)} but got {len(actual)}",
            expected=shape_to_string(expected_size),
            actual=shape_to_string(actual),
        )
    for dim_idx, (actual_dim, expected_dim) in enumerate(zip(actual, expected_size, strict=False)):
        if expected_dim == -1:
            continue
        if actual_dim != expected_dim:
            raise ShapeError(
                f"{name} shape mismatch at dim {dim_idx}: expected {expected_dim} but got {actual_dim}",
                expected=shape_to_string(expected_size),
                actual=shape_to_string(actual),
            )


def validate_rank(
    tensor: torch.Tensor,
    expected_rank: int,
    *,
    name: str = "tensor",
) -> None:
    """Validate that ``tensor`` has the expected number of dimensions.

    Args:
        tensor: The tensor to validate.
        expected_rank: Required rank (number of dimensions).
        name: Variable name used in error messages.

    Raises:
        ShapeError: If the tensor rank does not match.

    Example:
        >>> import torch
        >>> validate_rank(torch.zeros(2, 8), 2)
        >>> validate_rank(torch.zeros(2, 8), 3)
        Traceback (most recent call last):
            ...
        avqa.exceptions.ShapeError: tensor rank mismatch: expected 3 but got 2
    """
    actual_rank = tensor.ndim
    if actual_rank != expected_rank:
        raise ShapeError(
            f"{name} rank mismatch: expected {expected_rank} but got {actual_rank}",
            expected=f"rank={expected_rank}",
            actual=f"rank={actual_rank}",
        )


def validate_dtype(
    tensor: torch.Tensor,
    expected: torch.dtype | Sequence[torch.dtype],
    *,
    name: str = "tensor",
) -> None:
    """Validate that ``tensor`` has one of the expected dtypes.

    Args:
        tensor: The tensor to validate.
        expected: A single dtype or a sequence of acceptable dtypes.
        name: Variable name used in error messages.

    Raises:
        DtypeError: If the tensor dtype is not in the expected set.

    Example:
        >>> import torch
        >>> validate_dtype(torch.zeros(2), torch.float32)
        >>> validate_dtype(torch.zeros(2), [torch.float32, torch.float16])
        >>> validate_dtype(torch.zeros(2, dtype=torch.float64), torch.float32)
        Traceback (most recent call last):
        ...
        avqa.exceptions.DtypeError: tensor dtype mismatch
    """
    if isinstance(expected, torch.dtype):
        allowed: tuple[torch.dtype, ...] = (expected,)
        expected_str = str(expected)
    else:
        allowed = tuple(expected)
        expected_str = ", ".join(str(d) for d in allowed)
    if tensor.dtype not in allowed:
        raise DtypeError(
            f"{name} dtype mismatch: expected {expected_str} but got {tensor.dtype}",
            expected=expected_str,
            actual=str(tensor.dtype),
        )


def validate_device(
    tensor: torch.Tensor,
    expected: str | torch.device | Sequence[str | torch.device],
    *,
    name: str = "tensor",
) -> None:
    """Validate that ``tensor`` is on an expected device.

    Args:
        tensor: The tensor to validate.
        expected: A device (str or :class:`torch.device`) or a sequence of
            acceptable devices.
        name: Variable name used in error messages.

    Raises:
        DeviceError: If the tensor's device is not in the expected set.

    Example:
        >>> import torch
        >>> t = torch.zeros(2)
        >>> validate_device(t, "cpu")
        >>> validate_device(t, ["cpu", "cuda"])
    """
    if isinstance(expected, (str, torch.device)):
        allowed: tuple[torch.device, ...] = (torch.device(expected),)
        expected_str = str(expected)
    else:
        allowed = tuple(torch.device(d) for d in expected)
        expected_str = ", ".join(str(d) for d in allowed)
    actual_device = tensor.device
    if actual_device not in allowed:
        raise DeviceError(
            f"{name} device mismatch: expected {expected_str} but got {actual_device}",
            expected=expected_str,
            actual=str(actual_device),
        )


def validate_contiguous(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
) -> None:
    """Validate that ``tensor`` is contiguous in memory.

    Args:
        tensor: The tensor to validate.
        name: Variable name used in error messages.

    Raises:
        ShapeError: If the tensor is not contiguous.

    Example:
        >>> import torch
        >>> validate_contiguous(torch.zeros(2, 3))
        >>> t = torch.zeros(2, 3).transpose(0, 1)
        >>> validate_contiguous(t)
        Traceback (most recent call last):
        ...
        avqa.exceptions.ShapeError: tensor must be contiguous
    """
    if not tensor.is_contiguous():
        raise ShapeError(
            f"{name} must be contiguous (actual layout is non-contiguous)",
            expected="contiguous",
            actual=f"non-contiguous (stride={tensor.stride()})",
        )


def validate_embed_dim(
    tensor: torch.Tensor,
    expected_embed_dim: int,
    *,
    name: str = "tensor",
) -> None:
    """Validate that the last dim of ``tensor`` matches ``expected_embed_dim``.

    Args:
        tensor: Tensor whose last dim should equal ``expected_embed_dim``.
        expected_embed_dim: Required embedding dim.
        name: Variable name used in error messages.

    Raises:
        ShapeError: If the last dim does not match.

    Example:
        >>> import torch
        >>> validate_embed_dim(torch.zeros(2, 8, 128), 128)
        >>> validate_embed_dim(torch.zeros(2, 8, 64), 128)
        Traceback (most recent call last):
            ...
        avqa.exceptions.ShapeError: tensor embed_dim mismatch
    """
    actual = tensor.shape[-1]
    if actual != expected_embed_dim:
        raise ShapeError(
            f"{name} embed_dim mismatch: expected {expected_embed_dim} but got {actual}",
            expected=f"embed_dim={expected_embed_dim}",
            actual=f"embed_dim={actual}",
        )


def validate_device_match(
    tensors: Sequence[torch.Tensor],
    *,
    name: str = "tensors",
) -> None:
    """Validate that all tensors in ``tensors`` share the same device.

    Args:
        tensors: Sequence of tensors to check.
        name: Variable name used in error messages.

    Raises:
        DeviceError: If the tensors are on different devices.

    Example:
        >>> import torch
        >>> validate_device_match([torch.zeros(2), torch.zeros(3)])
        >>> validate_device_match([torch.zeros(2), torch.zeros(3, device="meta")])
        Traceback (most recent call last):
            ...
        avqa.exceptions.DeviceError: tensors device mismatch
    """
    if not tensors:
        return
    first_device = tensors[0].device
    for t in tensors[1:]:
        if t.device != first_device:
            raise DeviceError(
                f"{name} device mismatch: expected all on {first_device} but got {t.device}",
                expected=str(first_device),
                actual=str(t.device),
            )


def validate_finite(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
) -> None:
    """Validate that all elements of ``tensor`` are finite (no NaN/Inf).

    Args:
        tensor: The tensor to validate.
        name: Variable name used in error messages.

    Raises:
        ValueError: If any element is non-finite.
    """
    if not torch.isfinite(tensor).all():
        msg = f"{name} contains non-finite values (NaN or Inf)"
        raise ValueError(msg)


__all__ = [
    "RankLike",
    "ShapeLike",
    "coerce_shape",
    "shape_to_string",
    "validate_contiguous",
    "validate_device",
    "validate_device_match",
    "validate_dtype",
    "validate_embed_dim",
    "validate_finite",
    "validate_rank",
    "validate_shape",
]
