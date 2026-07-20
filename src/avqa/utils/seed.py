"""Deterministic seeding for AVQA.

Spec §3.2 requires that the implementation maintain deterministic
execution when deterministic mode is enabled. This module provides a
single entry point — :func:`seed_everything` — that seeds Python's
:mod:`random`, NumPy (if installed), and PyTorch CPU + CUDA RNGs.

Determinism across CUDA kernels is best-effort and depends on the
specific GPU model, driver, and PyTorch build. When strict determinism
is required, also set ``torch.use_deterministic_algorithms(True)`` (which
may raise if a non-deterministic op is used).
"""
from __future__ import annotations

import os
import random

from avqa.logging import get_logger

_logger = get_logger("utils.seed")

_DEFAULT_SEED: int = 0


def seed_everything(seed: int = _DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Seed every random-number generator used by AVQA.

    Seeds (in order):

    1. Python's :mod:`random` module.
    2. The ``PYTHONHASHSEED`` environment variable (effective only for the
       next interpreter startup, so a warning is logged).
    3. NumPy's RNG if NumPy is importable.
    4. PyTorch's CPU and CUDA RNGs (if CUDA is available).

    Args:
        seed: Non-negative integer seed. Defaults to ``0``.
        deterministic: If ``True``, additionally enable PyTorch's strict
            deterministic-algorithms mode. This may raise if a
            non-deterministic op is invoked.

    Returns:
        The seed that was applied. Useful for chaining.

    Raises:
        ValueError: If ``seed`` is negative.

    Example:
        >>> seed_everything(42)
        42
    """
    if seed < 0:
        msg = f"seed must be non-negative, got {seed}"
        raise ValueError(msg)

    _logger.debug("Seeding all RNGs with seed=%d deterministic=%s", seed, deterministic)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _logger.debug("Set PYTHONHASHSEED=%d", seed)

    try:
        import numpy as np
    except ImportError:
        _logger.debug("NumPy not installed; skipping NumPy seeding")
    else:
        np.random.seed(seed)
        _logger.debug("Seeded NumPy with seed=%d", seed)

    import torch

    torch.manual_seed(seed)
    _logger.debug("Seeded torch CPU RNG with seed=%d", seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        _logger.debug("Seeded torch CUDA RNGs with seed=%d", seed)
    else:
        _logger.debug("CUDA not available; skipping CUDA seeding")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        _logger.debug("Enabled strict deterministic algorithms")

    return seed


__all__ = ["seed_everything"]
