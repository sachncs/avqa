"""Logging integration for AVQA.

AVQA integrates with Python's standard :mod:`logging` module (spec §5.14).
No library component SHALL print directly to stdout during normal operation.
Users control verbosity through the standard Python logging configuration.

The module exposes:

- :func:`configure_logger`: idempotently configure the AVQA logger.
- :func:`get_logger`: retrieve the AVQA logger (lazy).
- :class:`LogLevel`: type alias for the standard logging level names.
"""
from __future__ import annotations

import logging
from typing import Final

AVQA_LOGGER_NAME: Final[str] = "avqa"
"""Canonical logger name used throughout AVQA."""


_DEFAULT_FORMAT: Final[str] = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LEVEL: Final[int] = logging.WARNING

LogLevel = int
"""Standard Python logging level (e.g., ``logging.DEBUG``, ``logging.INFO``)."""

_CONFIGURED: list[bool] = [False]


def set_configured(value: bool) -> None:
    """Mark the logger as configured (internal use)."""
    _CONFIGURED[0] = value


def is_internal_configured() -> bool:
    """Return whether the logger has been configured (internal use)."""
    return _CONFIGURED[0]


def get_logger(name: str | None = None) -> logging.Logger:
    """Retrieve the AVQA logger or a child logger.

    Args:
        name: Optional child logger name (relative to ``avqa``). If ``None``,
            the root AVQA logger is returned.

    Returns:
        A :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger()
        >>> child = get_logger("attention.module")
        >>> isinstance(child, logging.Logger)
        True
    """
    if name is None:
        return logging.getLogger(AVQA_LOGGER_NAME)
    full_name = f"{AVQA_LOGGER_NAME}.{name}" if not name.startswith(AVQA_LOGGER_NAME) else name
    return logging.getLogger(full_name)


def configure_logger(
    level: LogLevel = _DEFAULT_LEVEL,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the AVQA logger.

    Idempotent: repeated calls without ``force=True`` are no-ops once the
    logger has been configured. Use ``force=True`` to reconfigure (e.g., in
    tests).

    Args:
        level: Standard Python logging level. Defaults to ``logging.WARNING``.
        fmt: Log record format string.
        datefmt: Date format for log records.
        force: If ``True``, remove existing handlers before reconfiguring.

    Returns:
        The configured AVQA logger.

    Example:
        >>> import logging
        >>> logger = configure_logger(level=logging.DEBUG)
        >>> logger.getEffectiveLevel() == logging.DEBUG
        True
    """
    logger = get_logger()
    if is_internal_configured() and not force:
        return logger

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    set_configured(True)
    return logger


def is_configured() -> bool:
    """Return whether :func:`configure_logger` has been called.

    Example:
        >>> configure_logger()
        >>> is_configured()
        True
    """
    return is_internal_configured()


__all__ = [
    "AVQA_LOGGER_NAME",
    "LogLevel",
    "configure_logger",
    "get_logger",
    "is_configured",
    "is_internal_configured",
    "set_configured",
]
