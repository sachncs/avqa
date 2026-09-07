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

import importlib.util
import os
import random

import torch

from avqa.exceptions import ConfigurationError
from avqa.logging import get_logger

logger = get_logger("utils.seed")

DEFAULT_SEED: int = 0


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
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
        raise ConfigurationError(msg, {"seed": seed})

    logger.debug("Seeding all RNGs with seed=%d deterministic=%s", seed, deterministic)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Set PYTHONHASHSEED=%d", seed)

    if importlib.util.find_spec("numpy") is not None:
        np_module = importlib.import_module("numpy")
        random_state = np_module.random
        random_state.seed(seed)
        logger.debug("Seeded NumPy with seed=%d", seed)
    else:
        logger.debug("NumPy not installed; skipping NumPy seeding")


    torch.manual_seed(seed)
    logger.debug("Seeded torch CPU RNG with seed=%d", seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.debug("Seeded torch CUDA RNGs with seed=%d", seed)
    else:
        logger.debug("CUDA not available; skipping CUDA seeding")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.debug("Enabled strict deterministic algorithms")

    return seed


__all__ = ["seed_everything"]
