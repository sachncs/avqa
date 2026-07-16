"""Structural tests for OPT-0002 (torch.compile opt-in).

The full numerical-equivalence test for OPT-0002 is exercised on the
CUDA-matrix runner because :func:`torch.compile` on the AVQAttention
forward relies on TorchInductor, which can recurse on CPU during
Dynamo tracing. These structural tests pin the enable / disable
plumbing and document the handoff to the GPU runner.
"""

from __future__ import annotations

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
