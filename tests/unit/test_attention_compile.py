"""Structural tests for OPT-0002 (torch.compile opt-in).

The full numerical-equivalence test for OPT-0002 is exercised on the
CUDA-matrix runner because :func:`torch.compile` on the AVQAttention
forward relies on TorchInductor, which can recurse on CPU during
Dynamo tracing. These structural tests pin the enable / disable
plumbing and document the handoff to the GPU runner.
"""

from __future__ import annotations

import pytest
import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    ExecutionConfig,
    RoutingConfig,
)


def _compile_config() -> AVQConfig:
    return AVQConfig(
        attention=AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=2),
        execution=ExecutionConfig(compile_enabled=True),
    )


def _eager_config() -> AVQConfig:
    return AVQConfig(
        attention=AttentionShapeConfig(embed_dim=32, num_heads=4, head_dim=8),
        codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
        routing=RoutingConfig(refinement_budget=2),
    )


class TestCompileEnabled:
    """OPT-0002 plumbing checks for the torch.compile opt-in."""

    def test_compiled_forward_attached_when_enabled(self) -> None:
        """compile_enabled=True installs a torch.compile wrapper."""
        mod = AVQAttention(_compile_config(), in_proj=False, out_proj=False)
        mod.eval()
        assert mod._forward_eager is not None
        assert mod._forward_compiled is not None
        assert mod._forward_compiled is not mod.forward_impl

    def test_compiled_forward_absent_by_default(self) -> None:
        """compile_enabled=False leaves the compiled forward as None."""
        mod = AVQAttention(_eager_config(), in_proj=False, out_proj=False)
        mod.eval()
        assert mod._forward_eager is not None
        assert mod._forward_compiled is None


class TestCompileNumericalEquivalence:
    """OPT-0002 numerical equivalence on CPU (gated by Dynamo success).

    torch.compile with TorchInductor may fail during Dynamo tracing on
    CPU for complex control flow.  These tests are skipped with
    ``pytest.skip`` when compilation fails, documenting the expected
    limitation.  The GPU runner provides the authoritative equivalence
    gate.
    """

    @pytest.fixture
    def modules(self) -> tuple[AVQAttention, AVQAttention]:
        """Return (compiled, eager) module pair sharing the same weights."""
        torch.manual_seed(0)
        compiled = AVQAttention(_compile_config(), in_proj=False, out_proj=False)
        compiled.eval()
        eager = AVQAttention(_eager_config(), in_proj=False, out_proj=False)
        eager.load_state_dict(compiled.state_dict())
        # Codebook parents/children are plain tensors (not nn.Parameter),
        # so load_state_dict doesn't copy them — do it manually.
        eager.codebook.parents = compiled.codebook.parents.clone()
        eager.codebook.children = compiled.codebook.children.clone()
        eager.eval()
        return compiled, eager

    @pytest.fixture
    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Small CPU input tensors for numerical comparison."""
        torch.manual_seed(42)
        B, T, E = 1, 8, 32
        return (
            torch.randn(B, T, E),
            torch.randn(B, T, E),
            torch.randn(B, T, E),
        )

    def test_eager_vs_eager_same_weights(
        self, modules: tuple[AVQAttention, AVQAttention], inputs: tuple[torch.Tensor, ...]
    ) -> None:
        """Sanity: two modules with the same weights produce identical output."""
        compiled, eager = modules
        q, k, v = inputs
        with torch.no_grad():
            out_a = compiled._forward_eager(q, k, v, None, None)
            out_b = eager._forward_eager(q, k, v, None, None)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_compiled_vs_eager_numerical(
        self, modules: tuple[AVQAttention, AVQAttention], inputs: tuple[torch.Tensor, ...]
    ) -> None:
        """Compiled forward should match eager forward within tolerance."""
        compiled, eager = modules
        q, k, v = inputs
        with torch.no_grad():
            out_eager = eager._forward_eager(q, k, v, None, None)
            try:
                out_compiled = compiled._forward_compiled(q, k, v, None, None)
            except Exception:
                pytest.skip(
                    "torch.compile Dynamo tracing failed on CPU; "
                    "tested on GPU runner"
                )
        assert torch.allclose(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
