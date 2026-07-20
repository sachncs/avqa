"""Triton kernel backend for AVQA (SPEC §11).

This package provides optimized Triton kernels that implement the AVQ
pipeline's four VQ / attention stages. Each kernel module follows the
same pattern:

- ``kernel_fn`` is the @triton.jit-compiled function defined inside the
  module. Importing the module requires ``triton``; the package's
  ``__init__`` lazily imports these modules so a CPU-only environment
  can still load AVQA.
- ``is_available()`` checks both ``triton`` and CUDA availability.
- ``run_*(...)`` Python functions dispatch through ``torch.compile`` /
  ``torch.cuda`` and provide the public entry points used by
  ``TritonBackend``.

Each kernel takes the tensors and tile sizes from SPEC §11.3:

- ``BLOCK_T`` (default 64) — rows of Q processed per query tile.
- ``BLOCK_M`` (default 64) — codebook rows per key tile.
- ``BLOCK_D`` (default 64) — head dimension divisor.

Numerical tolerances (SPEC §11.9) are checked in
``tests/unit/test_triton_kernels.py`` (CPU-only) and
``tests/integration/test_triton_kernels.py`` (CUDA-only, gated by the
``gpu`` marker).
"""
from __future__ import annotations

from dataclasses import dataclass

from avqa.exceptions import BackendError
from avqa.logging import get_logger

_logger = get_logger("triton")

DEFAULT_BLOCK_T = 64
DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_D = 64


@dataclass(frozen=True)
class TritonTileConfig:
    """Block sizes for the AVQA Triton kernels (SPEC §11.3)."""

    block_t: int = DEFAULT_BLOCK_T
    block_m: int = DEFAULT_BLOCK_M
    block_d: int = DEFAULT_BLOCK_D

    def __post_init__(self) -> None:
        if self.block_t & (self.block_t - 1):
            msg = f"block_t must be a power of 2, got {self.block_t}"
            raise BackendError(msg)
        if self.block_m & (self.block_m - 1):
            msg = f"block_m must be a power of 2, got {self.block_m}"
            raise BackendError(msg)
        if self.block_d & (self.block_d - 1):
            msg = f"block_d must be a power of 2, got {self.block_d}"
            raise BackendError(msg)


def is_triton_available() -> bool:
    """Return True iff Triton and CUDA are both available."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch always present
        return False


def has_triton_module() -> bool:
    """Return True iff the local Triton runtime is importable (no CUDA check)."""
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "DEFAULT_BLOCK_D",
    "DEFAULT_BLOCK_M",
    "DEFAULT_BLOCK_T",
    "TritonTileConfig",
    "has_triton_module",
    "is_triton_available",
]
