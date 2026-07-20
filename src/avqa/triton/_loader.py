"""Lazy loader for the Triton kernels.

The kernels live in :mod:`avqa.triton.{vq,parent_attention,child_attention,correction}`
but each module imports :mod:`triton` at module load time. The
runtime may be CPU-only; in that case :func:`is_triton_available`
returns False and we never import the kernel modules.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from avqa.triton import has_triton_module, is_triton_available


@lru_cache(maxsize=1)
def _load_vq() -> Any:
    """Lazy import of the fused VQ kernel module."""
    from avqa.triton.vq import vq_precompute

    return vq_precompute


@lru_cache(maxsize=1)
def _load_parent_attention() -> Any:
    """Lazy import of the parent-attention kernel module."""
    from avqa.triton.parent_attention import parent_attention

    return parent_attention


@lru_cache(maxsize=1)
def _load_child_attention() -> Any:
    """Lazy import of the child-attention kernel module."""
    from avqa.triton.child_attention import child_attention

    return child_attention


@lru_cache(maxsize=1)
def _load_correction() -> Any:
    """Lazy import of the correction kernel module."""
    from avqa.triton.correction import correction

    return correction


def available_kernels() -> tuple[str, ...]:
    """Names of the AVQA Triton kernel functions available on this runtime."""
    return ("vq_precompute", "parent_attention", "child_attention", "correction")


def load_kernel(name: str) -> Any:
    """Return the named Triton kernel function.

    Imports the corresponding module on first call; subsequent calls
    are served from the LRU cache.
    """
    if not has_triton_module():
        msg = f"Triton is not installed; cannot load kernel {name!r}"
        raise RuntimeError(msg)
    if not is_triton_available():
        msg = (
            f"Triton is installed but CUDA is not available; cannot "
            f"load kernel {name!r}. Check torch.cuda.is_available()."
        )
        raise RuntimeError(msg)
    loaders = {
        "vq_precompute": _load_vq,
        "parent_attention": _load_parent_attention,
        "child_attention": _load_child_attention,
        "correction": _load_correction,
    }
    if name not in loaders:
        msg = f"unknown triton kernel name: {name!r}"
        raise KeyError(msg)
    return loaders[name]()


if TYPE_CHECKING:
    # Type-checkers can see the public symbols without triggering the
    # runtime lazy-load guard.
    from avqa.triton.child_attention import child_attention
    from avqa.triton.correction import correction
    from avqa.triton.parent_attention import parent_attention
    from avqa.triton.vq import vq_precompute


__all__ = [
    "available_kernels",
    "child_attention",
    "correction",
    "load_kernel",
    "parent_attention",
    "vq_precompute",
]
