"""Exception hierarchy for AVQA.

All exceptions raised by AVQA inherit from :class:`AVQAError`. Public methods
SHOULD raise documented exceptions from this hierarchy. Recoverable errors
provide informative messages without exposing internal implementation details
(spec §3.22, §5.13).
"""
from __future__ import annotations


class AVQAError(Exception):
    """Base class for all exceptions raised by AVQA.

    All other AVQA exceptions inherit from this class. Catching
    :class:`AVQAError` catches every exception the library may raise.

    Args:
        message: Human-readable description of the error.
        context: Optional mapping of additional context for debugging.

    Example:
        >>> try:
        ...     raise AVQAError("something went wrong", {"x": 1})
        ... except AVQAError as exc:
        ...     print(exc.context)
        {'x': 1}
    """

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = dict(context) if context else {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


class ConfigurationError(AVQAError):
    """Raised when configuration is invalid, missing, or violates a constraint.

    Args:
        message: Description of the configuration error.
        config: Optional reference to the offending configuration object.

    Example:
        >>> raise ConfigurationError(
        ...     "refinement_budget must be > 0",
        ...     {"refinement_budget": 0},
        ... )
    """


class BackendError(AVQAError):
    """Raised when a backend operation fails or is unsupported.

    Args:
        message: Description of the backend error.
        backend: Optional backend identifier.
    """


class RoutingError(AVQAError):
    """Raised when routing decisions cannot be produced.

    Examples include: empty importance scores, invalid refinement budget,
    selector failure.
    """


class CodebookError(AVQAError):
    """Raised when codebook operations fail.

    Examples include: mean-constraint violation, invalid initialization,
    empty codeword, mismatched parent-child structure.
    """


class MergeError(AVQAError):
    """Raised when a merge strategy fails or receives bad input."""


class ShapeError(AVQAError):
    """Raised when tensor shapes violate documented contracts.

    Args:
        message: Description of the shape mismatch.
        expected: Expected shape or shape description.
        actual: Actual shape or shape description.
    """

    def __init__(
        self,
        message: str,
        expected: object = None,
        actual: object = None,
        context: dict[str, object] | None = None,
    ) -> None:
        merged_context: dict[str, object] = dict(context) if context else {}
        if expected is not None:
            merged_context["expected"] = expected
        if actual is not None:
            merged_context["actual"] = actual
        super().__init__(message, merged_context)


class DtypeError(AVQAError):
    """Raised when tensor dtype is unsupported or mismatched.

    Args:
        message: Description of the dtype error.
        expected: Expected dtype or set of dtypes.
        actual: Actual dtype.
    """

    def __init__(
        self,
        message: str,
        expected: object = None,
        actual: object = None,
        context: dict[str, object] | None = None,
    ) -> None:
        merged_context: dict[str, object] = dict(context) if context else {}
        if expected is not None:
            merged_context["expected"] = expected
        if actual is not None:
            merged_context["actual"] = actual
        super().__init__(message, merged_context)


class DeviceError(AVQAError):
    """Raised when tensor device is unsupported or mismatched.

    Args:
        message: Description of the device error.
        expected: Expected device.
        actual: Actual device.
    """

    def __init__(
        self,
        message: str,
        expected: object = None,
        actual: object = None,
        context: dict[str, object] | None = None,
    ) -> None:
        merged_context: dict[str, object] = dict(context) if context else {}
        if expected is not None:
            merged_context["expected"] = expected
        if actual is not None:
            merged_context["actual"] = actual
        super().__init__(message, merged_context)


class NotInitializedError(AVQAError):
    """Raised when a required component has not been initialized.

    Examples include: attempting forward pass before codebook training,
    using a KVCache before append, calling a backend method before its
    prerequisite state is set.
    """


__all__ = [
    "AVQAError",
    "BackendError",
    "CodebookError",
    "ConfigurationError",
    "DeviceError",
    "DtypeError",
    "MergeError",
    "NotInitializedError",
    "RoutingError",
    "ShapeError",
]
