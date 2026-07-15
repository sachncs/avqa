"""Tests for avqa.utils.validation module."""

from __future__ import annotations

import pytest
import torch

from avqa.exceptions import DeviceError, DtypeError, ShapeError
from avqa.utils.validation import (
    ShapeLike,
    validate_contiguous,
    validate_device,
    validate_dtype,
    validate_finite,
    validate_rank,
    validate_shape,
)


class TestValidateShape:
    """Tests for validate_shape function."""

    def test_exact_match(self) -> None:
        """Exact shape matches without error."""
        t = torch.zeros(2, 8, 128, 64)
        validate_shape(t, [2, 8, 128, 64])

    def test_tuple_expected(self) -> None:
        """Tuple shape is accepted."""
        t = torch.zeros(2, 8, 128, 64)
        validate_shape(t, (2, 8, 128, 64))

    def test_torch_size_expected(self) -> None:
        """torch.Size is accepted."""
        t = torch.zeros(2, 8, 128, 64)
        validate_shape(t, t.shape)

    def test_wildcard_dimension(self) -> None:
        """Wildcard (-1) matches any dimension size."""
        t = torch.zeros(2, 8, 128, 64)
        validate_shape(t, [2, 8, -1, 64])
        validate_shape(t, [-1, -1, -1, -1])

    def test_rank_mismatch_raises(self) -> None:
        """Different rank raises ShapeError."""
        t = torch.zeros(2, 8, 128, 64)
        with pytest.raises(ShapeError, match="rank mismatch"):
            validate_shape(t, [2, 8, 128])

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched dimension raises ShapeError."""
        t = torch.zeros(2, 8, 128, 64)
        with pytest.raises(ShapeError, match="shape mismatch"):
            validate_shape(t, [2, 8, 256, 64])

    def test_error_includes_actual_shape(self) -> None:
        """Error message includes actual shape."""
        t = torch.zeros(2, 8, 128, 64)
        with pytest.raises(ShapeError) as excinfo:
            validate_shape(t, [2, 8, 256, 64])
        assert excinfo.value.context["actual"] == "[2, 8, 128, 64]"

    def test_error_includes_expected_shape(self) -> None:
        """Error message includes expected shape."""
        t = torch.zeros(2, 8, 128, 64)
        with pytest.raises(ShapeError) as excinfo:
            validate_shape(t, [2, 8, 256, 64])
        assert excinfo.value.context["expected"] == "[2, 8, 256, 64]"

    def test_name_in_error_message(self) -> None:
        """Custom name appears in error message."""
        t = torch.zeros(2, 8, 128, 64)
        with pytest.raises(ShapeError, match="query"):
            validate_shape(t, [1, 1, 1, 1], name="query")


class TestValidateRank:
    """Tests for validate_rank function."""

    def test_exact_match(self) -> None:
        """Correct rank passes."""
        validate_rank(torch.zeros(2, 8), 2)
        validate_rank(torch.zeros(2, 8, 128, 64), 4)

    def test_mismatch_raises(self) -> None:
        """Wrong rank raises ShapeError."""
        with pytest.raises(ShapeError, match="rank mismatch"):
            validate_rank(torch.zeros(2, 8), 3)

    def test_zero_rank(self) -> None:
        """Zero-rank tensor validates against rank=0."""
        validate_rank(torch.tensor(1.0), 0)

    def test_name_in_error(self) -> None:
        """Custom name appears in error."""
        with pytest.raises(ShapeError, match="my_tensor"):
            validate_rank(torch.zeros(2, 8), 5, name="my_tensor")


class TestValidateDtype:
    """Tests for validate_dtype function."""

    def test_single_dtype_match(self) -> None:
        """Single matching dtype passes."""
        validate_dtype(torch.zeros(2, dtype=torch.float32), torch.float32)

    def test_single_dtype_mismatch(self) -> None:
        """Single non-matching dtype raises."""
        with pytest.raises(DtypeError, match="dtype mismatch"):
            validate_dtype(torch.zeros(2, dtype=torch.float64), torch.float32)

    def test_sequence_of_dtypes(self) -> None:
        """Sequence of acceptable dtypes passes for any in the set."""
        t = torch.zeros(2, dtype=torch.float16)
        validate_dtype(t, [torch.float32, torch.float16, torch.bfloat16])

    def test_sequence_mismatch(self) -> None:
        """Sequence mismatch raises DtypeError."""
        with pytest.raises(DtypeError):
            validate_dtype(torch.zeros(2, dtype=torch.float64), [torch.float32, torch.float16])

    def test_error_includes_expected_and_actual(self) -> None:
        """Error context includes expected and actual dtypes."""
        with pytest.raises(DtypeError) as excinfo:
            validate_dtype(torch.zeros(2, dtype=torch.float64), torch.float32)
        assert "float32" in str(excinfo.value.context["expected"])
        assert "float64" in str(excinfo.value.context["actual"])


class TestValidateDevice:
    """Tests for validate_device function."""

    def test_cpu_string(self) -> None:
        """String 'cpu' is accepted."""
        t = torch.zeros(2)
        validate_device(t, "cpu")

    def test_cpu_torch_device(self) -> None:
        """torch.device is accepted."""
        t = torch.zeros(2)
        validate_device(t, torch.device("cpu"))

    def test_sequence_of_devices(self) -> None:
        """Sequence of devices is accepted."""
        t = torch.zeros(2)
        validate_device(t, ["cpu", "cuda"])

    def test_mismatch_raises(self) -> None:
        """Mismatched device raises DeviceError."""
        t = torch.zeros(2)
        with pytest.raises(DeviceError, match="device mismatch"):
            validate_device(t, "cuda:0")


class TestValidateContiguous:
    """Tests for validate_contiguous function."""

    def test_contiguous_passes(self) -> None:
        """Freshly-created tensor is contiguous."""
        validate_contiguous(torch.zeros(2, 3))

    def test_non_contiguous_raises(self) -> None:
        """Transposed tensor is not contiguous."""
        t = torch.zeros(2, 3).transpose(0, 1)
        with pytest.raises(ShapeError, match="contiguous"):
            validate_contiguous(t)

    def test_contiguous_after_reshape(self) -> None:
        """Reshaped tensor is contiguous."""
        t = torch.zeros(6).reshape(2, 3)
        validate_contiguous(t)


class TestValidateFinite:
    """Tests for validate_finite function."""

    def test_all_finite_passes(self) -> None:
        """Tensor with finite values passes."""
        validate_finite(torch.zeros(3))

    def test_nan_raises(self) -> None:
        """Tensor with NaN raises ValueError."""
        t = torch.zeros(3)
        t[0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(t)

    def test_inf_raises(self) -> None:
        """Tensor with Inf raises ValueError."""
        t = torch.zeros(3)
        t[0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(t)

    def test_neg_inf_raises(self) -> None:
        """Tensor with -Inf raises ValueError."""
        t = torch.zeros(3)
        t[0] = float("-inf")
        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(t)


class TestShapeLikeAlias:
    """Tests for the ShapeLike type alias."""

    def test_accepts_list(self) -> None:
        """List is accepted."""
        t = torch.zeros(2, 3)
        shape: ShapeLike = [2, 3]
        validate_shape(t, shape)

    def test_accepts_tuple(self) -> None:
        """Tuple is accepted."""
        t = torch.zeros(2, 3)
        shape: ShapeLike = (2, 3)
        validate_shape(t, shape)
