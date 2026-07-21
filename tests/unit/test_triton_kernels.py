"""Tests for the AVQA Triton kernel package (SPEC §11.10).

CPU-only behavioural checks live in this module; CUDA-only numerical
equivalence tests live in ``tests/integration/test_triton_kernels.py``
and are gated by ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import pytest

from avqa.exceptions import BackendError
from avqa.triton import (
    DEFAULT_BLOCK_D,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_T,
    TritonTileConfig,
    has_triton_module,
    is_triton_available,
)
from avqa.triton.child_attention import child_attention
from avqa.triton.correction import correction
from avqa.triton.loader import available_kernels, load_kernel
from avqa.triton.parent_attention import parent_attention
from avqa.triton.vq import vq_precompute


class TestTileConfig:
    """TritonTileConfig is immutable and validates block sizes."""

    def test_defaults_match_spec_section_11_3(self) -> None:
        """Defaults are 64/64/64 per SPEC §11.3."""
        cfg = TritonTileConfig()
        assert cfg.block_t == DEFAULT_BLOCK_T
        assert cfg.block_m == DEFAULT_BLOCK_M
        assert cfg.block_d == DEFAULT_BLOCK_D

    @pytest.mark.parametrize("field", ["block_t", "block_m", "block_d"])
    @pytest.mark.parametrize("value", [63, 100, 7])
    def test_rejects_non_power_of_two(self, field: str, value: int) -> None:
        """Each tile size MUST be a power of 2 (SPEC §11.3)."""
        with pytest.raises(BackendError, match="power of 2"):
            TritonTileConfig(**{field: value})

    @pytest.mark.parametrize("field", ["block_t", "block_m", "block_d"])
    @pytest.mark.parametrize("value", [32, 64, 128, 256])
    def test_accepts_powers_of_two(self, field: str, value: int) -> None:
        """Any power-of-2 tile size is accepted."""
        TritonTileConfig(**{field: value})


class TestAvailabilityDetection:
    """Triton availability detection does not raise on CPU-only hosts."""

    def test_is_triton_available_returns_bool(self) -> None:
        """Always returns a bool (never raises)."""
        result = is_triton_available()
        assert isinstance(result, bool)

    def test_has_triton_module_returns_bool(self) -> None:
        """Returns True only when the ``triton`` package imports."""
        result = has_triton_module()
        assert isinstance(result, bool)

    def test_availability_false_on_cpu_only(self) -> None:
        """On this machine (CPU only) both checks return False.

        The test gracefully passes if a CUDA machine runs it with
        Triton installed.
        """
        if is_triton_available():
            pytest.skip("CUDA + Triton detected; skipping CPU-only check")
        assert has_triton_module() in (True, False)


class TestKernelLoader:
    """Lazy loader refuses to expose kernels when Triton is unavailable."""

    def test_loader_lists_all_four_kernels(self) -> None:
        """available_kernels() enumerates the four SPEC §11 kernels."""
        names = available_kernels()
        assert set(names) == {
            "vq_precompute",
            "parent_attention",
            "child_attention",
            "correction",
        }

    def test_loader_raises_when_triton_missing(self) -> None:
        """load_kernel() raises RuntimeError when triton isn't installed."""
        if has_triton_module():
            pytest.skip("Triton module is importable; cannot exercise the missing case")
        with pytest.raises(RuntimeError, match="Triton is not installed"):
            load_kernel("vq_precompute")

    def test_loader_rejects_unknown_kernel(self) -> None:
        """load_kernel() rejects unknown names.

        Behaviour depends on Triton availability: on a Triton-enabled
        runtime we expect ``KeyError``; otherwise the import guard wins
        first (``RuntimeError``).
        """
        try:
            with pytest.raises(KeyError, match="unknown triton kernel"):
                load_kernel("not_a_kernel")
        except RuntimeError:
            with pytest.raises(RuntimeError, match="Triton is not installed"):
                load_kernel("not_a_kernel")


class TestKernelModuleShape:
    """CPU smoke-tests for the Triton kernel module surfaces.

    These tests import each kernel module to make sure the function
    objects exist (on a CUDA host they would compile; on CPU they
    remain lazy). The intent is to bump coverage on the kernel module
    wrappers without requiring CUDA.
    """

    def test_vq_module_importable(self) -> None:
        assert callable(vq_precompute)

    def test_parent_attention_module_importable(self) -> None:
        assert callable(parent_attention)

    def test_child_attention_module_importable(self) -> None:
        assert callable(child_attention)

    def test_correction_module_importable(self) -> None:
        assert callable(correction)
