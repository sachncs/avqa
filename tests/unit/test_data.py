"""Tests for avqa.data module."""

from __future__ import annotations

import pytest
import torch

from avqa.data import (
    EXTENDED_DTYPES,
    SHAPE_ACTIVE_CODEWORDS,
    SHAPE_ASSIGNMENT,
    SHAPE_CODEBOOK,
    SHAPE_OUTPUT,
    SHAPE_QUERY,
    SUPPORTED_DEVICES,
    SUPPORTED_DTYPES,
    TensorContract,
    is_supported_device,
    is_supported_dtype,
    make_default_contract,
)


class TestShapeSymbols:
    """Tests for the canonical shape symbol constants (spec §6.3)."""

    def test_query_shape_symbols(self) -> None:
        """Query shape is (B, H, T, D)."""
        assert SHAPE_QUERY == ("B", "H", "T", "D")

    def test_output_shape_matches_input(self) -> None:
        """Output shape matches query shape (spec §6.5)."""
        assert SHAPE_OUTPUT == SHAPE_QUERY

    def test_codebook_shape_is_per_head(self) -> None:
        """Codebook shape is (H, C, D) (per-head)."""
        assert SHAPE_CODEBOOK == ("H", "C", "D")

    def test_assignment_shape_is_per_token(self) -> None:
        """Assignment matrix shape is (B, H, T)."""
        assert SHAPE_ASSIGNMENT == ("B", "H", "T")

    def test_active_codewords_shape(self) -> None:
        """Active codewords shape is (B, H, K)."""
        assert SHAPE_ACTIVE_CODEWORDS == ("B", "H", "K")


class TestSupportedDtypes:
    """Tests for the supported dtype registry (spec §6.9)."""

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16],
    )
    def test_reference_dtypes_supported(self, dtype: torch.dtype) -> None:
        """The three reference dtypes are supported."""
        assert is_supported_dtype(dtype)

    def test_float64_not_in_reference_set(self) -> None:
        """float64 is in EXTENDED but not in the reference SUPPORTED set."""
        assert not is_supported_dtype(torch.float64)
        assert torch.float64 in EXTENDED_DTYPES

    def test_supported_dtypes_is_frozenset(self) -> None:
        """SUPPORTED_DTYPES is immutable."""
        assert isinstance(SUPPORTED_DTYPES, frozenset)

    def test_supported_set_size(self) -> None:
        """Three dtypes in reference set."""
        assert len(SUPPORTED_DTYPES) == 3


class TestSupportedDevices:
    """Tests for the supported device registry (spec §6.10)."""

    def test_cpu_is_supported(self) -> None:
        """CPU is supported."""
        assert is_supported_device("cpu")
        assert is_supported_device(torch.device("cpu"))

    def test_cuda_is_supported(self) -> None:
        """CUDA device type is supported (even if not present)."""
        assert is_supported_device("cuda")
        assert is_supported_device(torch.device("cuda:0"))

    def test_unknown_device_rejected(self) -> None:
        """Unknown/invalid device strings return False without raising."""
        assert is_supported_device("invalid_device_xyz") is False
        assert is_supported_device("") is False

    def test_supported_devices_is_frozenset(self) -> None:
        """SUPPORTED_DEVICES is immutable."""
        assert isinstance(SUPPORTED_DEVICES, frozenset)


class TestTensorContract:
    """Tests for the TensorContract dataclass."""

    def test_immutable(self) -> None:
        """Contract is frozen — assignment raises."""
        c = make_default_contract("q", ("B", "H", "T", "D"), "attention")
        with pytest.raises((AttributeError, Exception)):
            setattr(c, "name", "other")

    def test_supports_matching_tensor(self) -> None:
        """Tensor with matching shape/dtype/device is supported."""
        c = make_default_contract("q", ("B", "H", "T", "D"), "attention")
        t = torch.zeros(2, 8, 128, 64)
        assert c.supports(t)

    def test_rejects_wrong_rank(self) -> None:
        """Tensor with wrong rank is rejected."""
        c = make_default_contract("q", ("B", "H", "T", "D"), "attention")
        t = torch.zeros(2, 8, 128)
        assert not c.supports(t)

    def test_rejects_wrong_dtype(self) -> None:
        """Tensor with unsupported dtype is rejected."""
        c = make_default_contract("q", ("B", "H", "T", "D"), "attention")
        t = torch.zeros(2, 8, 128, 64, dtype=torch.float64)
        assert not c.supports(t)

    def test_defaults(self) -> None:
        """make_default_contract uses the documented defaults."""
        c = make_default_contract("x", ("B", "T"), "x")
        assert c.dtype == SUPPORTED_DTYPES
        assert c.device == SUPPORTED_DEVICES
        assert c.mutable is False
        assert c.complexity == "O(B*H*T*D)"

    def test_supports_method(self) -> None:
        """Contract.supports() checks all three structural fields."""
        c = TensorContract(
            name="v",
            shape=("B", "H", "T", "D"),
            dtype=frozenset({torch.float32}),
            device=frozenset({"cpu"}),
            owner="attention",
            mutable=False,
            complexity="O(B*H*T*D)",
        )
        assert c.supports(torch.zeros(1, 1, 1, 1, dtype=torch.float32))
        assert not c.supports(torch.zeros(1, 1, 1, 1, dtype=torch.float16))
        assert not c.supports(torch.zeros(1, 1, 1))


class TestExtendedDtypes:
    """Tests for EXTENDED_DTYPES (spec §6.9 optional)."""

    def test_includes_reference_set(self) -> None:
        """Extended set is a superset of reference set."""
        assert SUPPORTED_DTYPES.issubset(EXTENDED_DTYPES)

    def test_includes_float64(self) -> None:
        """float64 is in extended set."""
        assert torch.float64 in EXTENDED_DTYPES
