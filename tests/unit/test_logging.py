"""Tests for avqa.logging module."""

from __future__ import annotations

import logging

import pytest

from avqa import logging as avqa_logging
from avqa.logging import (
    AVQA_LOGGER_NAME,
    configure_logger,
    get_logger,
    is_configured,
)


@pytest.fixture(autouse=True)
def reset_logger_state() -> None:
    """Reset AVQA logger state around each test."""
    logger = logging.getLogger(AVQA_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    avqa_logging.set_configured(False)
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    avqa_logging.set_configured(False)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_avqa_root_logger_by_default(self) -> None:
        """Default returns the root AVQA logger."""
        logger = get_logger()
        assert logger.name == AVQA_LOGGER_NAME

    def test_returns_child_logger(self) -> None:
        """Passing a name returns a child logger."""
        logger = get_logger("attention.module")
        assert logger.name == f"{AVQA_LOGGER_NAME}.attention.module"

    def test_full_name_returned_as_is(self) -> None:
        """A name already prefixed with 'avqa' is returned as-is."""
        logger = get_logger(f"{AVQA_LOGGER_NAME}.test")
        assert logger.name == f"{AVQA_LOGGER_NAME}.test"

    def test_returns_logger_instance(self) -> None:
        """Returned object is a stdlib logger."""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)


class TestConfigureLogger:
    """Tests for configure_logger function."""

    def test_configures_logger_once(self) -> None:
        """Subsequent calls without force are idempotent."""
        logger_first = configure_logger(level=logging.DEBUG)
        handler_count = len(logger_first.handlers)
        logger_second = configure_logger(level=logging.INFO)
        assert logger_first is logger_second
        assert len(logger_second.handlers) == handler_count

    def test_force_reconfigures(self) -> None:
        """force=True removes handlers and adds a fresh one."""
        logger = configure_logger(level=logging.DEBUG)
        first_handler = logger.handlers[0]
        logger_reconfigured = configure_logger(level=logging.WARNING, force=True)
        assert logger_reconfigured.handlers[0] is not first_handler
        assert logger_reconfigured.level == logging.WARNING

    def test_sets_level(self) -> None:
        """Configured level is applied."""
        logger = configure_logger(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_does_not_propagate(self) -> None:
        """Logger does not propagate to root after configuration."""
        logger = configure_logger()
        assert logger.propagate is False

    def test_returns_the_logger(self) -> None:
        """Returns the configured logger."""
        logger = configure_logger()
        assert logger.name == AVQA_LOGGER_NAME


class TestIsConfigured:
    """Tests for is_configured function."""

    def test_false_initially(self) -> None:
        """is_configured returns False before configure_logger is called."""
        assert is_configured() is False

    def test_true_after_configure(self) -> None:
        """is_configured returns True after configure_logger is called."""
        configure_logger()
        assert is_configured() is True

    def test_false_after_force_reset(self) -> None:
        """force=True re-sets the flag."""
        configure_logger(force=True)
        assert is_configured() is True


class TestLogRecordOutput:
    """Tests for actual log record emission."""

    def test_handler_emits_records(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Configured handler emits formatted log records."""
        logger = configure_logger(level=logging.INFO)
        logger.info("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.err or "test message" in captured.out

    def test_child_logger_uses_root_handlers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Child loggers emit through the root handler."""
        configure_logger(level=logging.INFO)
        child = get_logger("child")
        child.warning("warning from child")
        captured = capsys.readouterr()
        assert "warning from child" in captured.err or "warning from child" in captured.out

    def test_respects_level_threshold(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Records below the configured level are not emitted."""
        logger = configure_logger(level=logging.WARNING)
        logger.debug("debug message should not appear")
        logger.warning("warning message should appear")
        captured = capsys.readouterr()
        combined = captured.err + captured.out
        assert "debug message should not appear" not in combined
        assert "warning message should appear" in combined
