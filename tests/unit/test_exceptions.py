"""Tests for avqa.exceptions module."""

from __future__ import annotations

import pytest

from avqa.exceptions import (
    AVQAError,
    BackendError,
    CodebookError,
    ConfigurationError,
    DeviceError,
    DtypeError,
    NotInitializedError,
    RoutingError,
    ShapeError,
)


class TestAVQAErrorBase:
    """Tests for the AVQAError base exception."""

    def test_inherits_from_exception(self) -> None:
        """AVQAError inherits from Exception."""
        assert issubclass(AVQAError, Exception)

    def test_simple_message(self) -> None:
        """Error stores the message."""
        error = AVQAError("simple error")
        assert error.message == "simple error"
        assert error.context == {}

    def test_with_context(self) -> None:
        """Error stores the context dict."""
        error = AVQAError("with context", {"key": "value"})
        assert error.context == {"key": "value"}

    def test_str_no_context(self) -> None:
        """str() returns just the message when no context."""
        error = AVQAError("just a message")
        assert str(error) == "just a message"

    def test_str_with_context(self) -> None:
        """str() includes the context when present."""
        error = AVQAError("with context", {"a": 1})
        assert "with context" in str(error)
        assert "a" in str(error)
        assert "1" in str(error)

    def test_repr_includes_class_name(self) -> None:
        """repr() includes the class name."""
        error = AVQAError("oops")
        assert "AVQAError" in repr(error)

    def test_context_is_copied(self) -> None:
        """Context dict is defensively copied."""
        ctx = {"x": 1}
        error = AVQAError("msg", ctx)
        ctx["x"] = 999
        assert error.context["x"] == 1

    def test_can_be_raised_and_caught(self) -> None:
        """Errors can be raised and caught as AVQAError."""
        with pytest.raises(AVQAError) as excinfo:
            raise AVQAError("test")
        assert excinfo.value.message == "test"


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            BackendError,
            RoutingError,
            CodebookError,
            NotInitializedError,
        ],
    )
    def test_subclasses_inherit_from_avqa_error(self, exc_cls: type[AVQAError]) -> None:
        """All public exception subclasses inherit from AVQAError."""
        assert issubclass(exc_cls, AVQAError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            BackendError,
            RoutingError,
            CodebookError,
            NotInitializedError,
        ],
    )
    def test_subclasses_can_be_caught_as_avqa_error(self, exc_cls: type[AVQAError]) -> None:
        """Subclass instances are catchable as AVQAError."""
        with pytest.raises(AVQAError):
            raise exc_cls("test error")


class TestShapeError:
    """Tests for ShapeError."""

    def test_inherits_from_avqa_error(self) -> None:
        """ShapeError inherits from AVQAError."""
        assert issubclass(ShapeError, AVQAError)

    def test_stores_expected_and_actual(self) -> None:
        """Context contains expected and actual fields."""
        error = ShapeError(
            "shape mismatch",
            expected=[2, 8, 128, 64],
            actual=[2, 8, 256, 64],
        )
        assert error.context["expected"] == [2, 8, 128, 64]
        assert error.context["actual"] == [2, 8, 256, 64]

    def test_extra_context_merged(self) -> None:
        """Extra context is merged with expected/actual."""
        error = ShapeError(
            "shape mismatch",
            expected=[2, 8, 128, 64],
            actual=[2, 8, 256, 64],
            context={"op": "attention"},
        )
        assert error.context["op"] == "attention"
        assert error.context["expected"] == [2, 8, 128, 64]


class TestDtypeError:
    """Tests for DtypeError."""

    def test_inherits_from_avqa_error(self) -> None:
        """DtypeError inherits from AVQAError."""
        assert issubclass(DtypeError, AVQAError)

    def test_stores_expected_and_actual(self) -> None:
        """Context contains expected and actual fields."""
        error = DtypeError(
            "dtype mismatch",
            expected="float32",
            actual="float64",
        )
        assert error.context["expected"] == "float32"
        assert error.context["actual"] == "float64"


class TestDeviceError:
    """Tests for DeviceError."""

    def test_inherits_from_avqa_error(self) -> None:
        """DeviceError inherits from AVQAError."""
        assert issubclass(DeviceError, AVQAError)

    def test_stores_expected_and_actual(self) -> None:
        """Context contains expected and actual fields."""
        error = DeviceError(
            "device mismatch",
            expected="cuda:0",
            actual="cpu",
        )
        assert error.context["expected"] == "cuda:0"
        assert error.context["actual"] == "cpu"


class TestExceptionMessages:
    """Tests for informative error messages (spec §3.22)."""

    def test_no_internal_implementation_leaked(self) -> None:
        """Messages do not contain implementation-specific stack info."""
        error = ConfigurationError("invalid refinement_budget")
        msg = str(error)
        assert "Traceback" not in msg
        assert "File" not in msg

    def test_subclass_distinct_from_each_other(self) -> None:
        """Each subclass is its own type (no overlap)."""
        assert ConfigurationError is not BackendError
        assert RoutingError is not CodebookError
        assert ShapeError is not DtypeError
        assert ShapeError is not DeviceError


class TestContextIsMutable:
    """Tests that the context dict remains mutable for callers."""

    def test_can_add_to_context(self) -> None:
        """Context can be extended after construction."""
        error = AVQAError("initial")
        error.context["late"] = "value"
        assert error.context["late"] == "value"

    def test_can_remove_from_context(self) -> None:
        """Context entries can be removed."""
        error = AVQAError("initial", {"x": 1})
        del error.context["x"]
        assert "x" not in error.context
