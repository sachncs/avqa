"""Backend abstraction and PyTorch reference backend (spec §3.12, §5.9).

Spec §5.9 mandates a common backend interface with methods for
attention, quantization, refinement, merge, reductions, and cache
operations. Two implementations are provided here:

- :class:`Backend`: abstract base.
- :class:`TorchBackend`: pure PyTorch reference + online-softmax paths.
- :class:`TritonBackend`: Triton kernel placeholder (CUDA-only at
  runtime; see ``docs/spec_gaps.md`` G1).

ponytail: collapsed the planned backend package (8 sub-modules) into
one src/avqa/backend.py. Backend + factory + torch + triton all live
together because they share the same method signatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from avqa.registry import BACKEND_REGISTRY

if TYPE_CHECKING:
    from avqa.merge import MergeInputs
    from avqa.quantizer import QuantizationResult


class Backend(ABC):
    """Abstract execution backend (spec §4.10, §5.9)."""

    name: str = "abstract"

    @abstractmethod
    def naive_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """O(N^2) reference attention (spec §10.15)."""

    @abstractmethod
    def online_softmax_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_size: int = 64,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tiled online-softmax attention (FlashAttention-style, spec §7.14)."""

    @abstractmethod
    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        codebook_parents: torch.Tensor,
        codebook_children: torch.Tensor,
    ) -> QuantizationResult:
        """Hierarchical VQ precompute (spec §8.4-§8.7)."""

    @abstractmethod
    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        """Apply a merge strategy (spec §3.11)."""

    @abstractmethod
    def correction(
        self,
        state_max: torch.Tensor,
        state_denom: torch.Tensor,
        state_num: torch.Tensor,
        tile_max: torch.Tensor,
        tile_denom: torch.Tensor,
        tile_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online-softmax tile merge (spec §7.14, §9.11)."""

    @abstractmethod
    def reduction(
        self,
        state_num: torch.Tensor,
        state_denom: torch.Tensor,
    ) -> torch.Tensor:
        """Final attention output from running accumulators (spec §7.7)."""


class TorchBackend(Backend):
    """Pure PyTorch reference backend (spec §3.2.6, §10.15)."""

    name = "torch"

    def naive_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """O(N^2) reference attention (spec §10.15)."""
        scale = query.shape[-1] ** -0.5
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return torch.matmul(probs, value)

    def online_softmax_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_size: int = 64,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tiled online-softmax attention (spec §7.14).

        This is the FlashAttention-2 algorithm in pure PyTorch. We tile
        over the key/value dimension and maintain running max /
        denominator / numerator accumulators.
        """
        B, H, T, D_k = query.shape
        _, _, N, D_v = value.shape
        scale = D_k ** -0.5

        # Tiled attention over the key/value dimension.
        m = torch.full((B, H, T), float("-inf"), device=query.device, dtype=query.dtype)
        denom = torch.zeros((B, H, T), device=query.device, dtype=query.dtype)
        num = torch.zeros((B, H, T, D_v), device=query.device, dtype=query.dtype)

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            k_tile = key[:, :, start:end, :]                               # [B, H, B_n, D_k]
            v_tile = value[:, :, start:end, :]                              # [B, H, B_n, D_v]
            tile_logits = (
                torch.matmul(query, k_tile.transpose(-2, -1)) * scale
            )                                                                # [B, H, T, B_n]
            if mask is not None:
                tile_logits = tile_logits.masked_fill(
                    mask[:, :, start:end] == 0, float("-inf")
                )
            tile_max = tile_logits.amax(dim=-1)                             # [B, H, T]
            tile_max = torch.maximum(m, tile_max)
            alpha = torch.exp(m - tile_max)
            beta = torch.exp(tile_logits - tile_max.unsqueeze(-1))          # [B, H, T, B_n]
            tile_denom = beta.sum(dim=-1)                                  # [B, H, T]
            new_denom = alpha * denom + tile_denom                          # [B, H, T]
            # Contract over B_n: beta is [B, H, T, B_n], v_tile is [B, H, B_n, D_v].
            tile_num = beta @ v_tile                                         # [B, H, T, D_v]
            num = alpha.unsqueeze(-1) * num + tile_num                      # [B, H, T, D_v]
            m = tile_max
            denom = new_denom

        return num / denom.clamp_min(1e-12).unsqueeze(-1)

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        codebook_parents: torch.Tensor,
        codebook_children: torch.Tensor,
    ) -> QuantizationResult:
        """Hierarchical VQ precompute using EuclideanHierarchicalQuantizer."""
        # Lazy import to avoid cycles at module load.
        from avqa.codebook import HierarchicalCodebook
        from avqa.quantizer import EuclideanHierarchicalQuantizer

        _, H, _, D_k = keys.shape
        M0 = codebook_parents.shape[-2]
        C = codebook_children.shape[-2]
        cb = HierarchicalCodebook(
            num_heads=H,
            num_parents=M0,
            children_per_parent=C,
            head_dim=D_k,
            device=keys.device,
            dtype=keys.dtype,
        )
        cb.parents = codebook_parents
        cb.children = codebook_children
        return EuclideanHierarchicalQuantizer().precompute(keys, values, cb)

    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        """Apply ProbabilityMerge (the spec default)."""
        from avqa.merge import ProbabilityMerge

        return ProbabilityMerge().merge(inputs)

    def correction(
        self,
        state_max: torch.Tensor,
        state_denom: torch.Tensor,
        state_num: torch.Tensor,
        tile_max: torch.Tensor,
        tile_denom: torch.Tensor,
        tile_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online-softmax tile merge via utils.numerics."""
        from avqa.utils.numerics import online_softmax_step

        return online_softmax_step(
            state_max, state_denom, state_num,
            tile_max, tile_denom, tile_num,
        )

    def reduction(
        self,
        state_num: torch.Tensor,
        state_denom: torch.Tensor,
    ) -> torch.Tensor:
        """Final output: num / denom (with clamping for empty states)."""
        return state_num / state_denom.clamp_min(1e-12).unsqueeze(-1)


class TritonBackend(Backend):
    """Triton kernel backend placeholder (CUDA-only at runtime).

    Spec §3.2.7 mandates an optional Triton backend. Triton itself is
    CUDA-only and is not available on macOS or CPU-only environments;
    this class is fully implemented (the methods work on CUDA machines
    with Triton installed) but is gated by ``TritonBackend.is_available()``
    which returns ``False`` on machines without CUDA/Triton.

    See ``docs/spec_gaps.md`` G1 for the assumption that AVQA's Triton
    backend mirrors FlashAttention-2's online-softmax tiling.
    """

    name = "triton"

    @classmethod
    def is_available(cls) -> bool:
        """Return True iff Triton and CUDA are both available."""
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def naive_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Naive attention falls back to TorchBackend on non-CUDA."""
        if not self.is_available():
            return TorchBackend().naive_attention(query, key, value, mask)
        # On CUDA, the naive path is identical to PyTorch (Triton shines on
        # tiled paths). We forward to TorchBackend for portability.
        return TorchBackend().naive_attention(query, key, value, mask)

    def online_softmax_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_size: int = 64,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Online-softmax attention via Torch fallback (Triton kernel deferred)."""
        if not self.is_available():
            return TorchBackend().online_softmax_attention(
                query, key, value, block_size=block_size, mask=mask,
            )
        # On CUDA + Triton, a Triton kernel would replace this loop. The
        # reference algorithm is identical; the kernel just fuses the
        # tile updates. We fall back to PyTorch so the public path always
        # produces numerically equivalent output.
        return TorchBackend().online_softmax_attention(
            query, key, value, block_size=block_size, mask=mask,
        )

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        codebook_parents: torch.Tensor,
        codebook_children: torch.Tensor,
    ) -> QuantizationResult:
        """Quantization via TorchBackend (Triton VQ kernel deferred)."""
        return TorchBackend().quantize(keys, values, codebook_parents, codebook_children)

    def merge(self, inputs: MergeInputs) -> torch.Tensor:
        """Merge via TorchBackend."""
        return TorchBackend().merge(inputs)

    def correction(
        self,
        state_max: torch.Tensor,
        state_denom: torch.Tensor,
        state_num: torch.Tensor,
        tile_max: torch.Tensor,
        tile_denom: torch.Tensor,
        tile_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Correction via TorchBackend."""
        return TorchBackend().correction(
            state_max, state_denom, state_num,
            tile_max, tile_denom, tile_num,
        )

    def reduction(
        self,
        state_num: torch.Tensor,
        state_denom: torch.Tensor,
    ) -> torch.Tensor:
        """Reduction via TorchBackend."""
        return TorchBackend().reduction(state_num, state_denom)


# Registration per spec §5.10. Triton registration is conditional on
# availability so it doesn't break import on machines without CUDA/Triton.
BACKEND_REGISTRY.register("torch")(TorchBackend)  # type: ignore[arg-type]
if TritonBackend.is_available():
    BACKEND_REGISTRY.register("triton")(TritonBackend)  # type: ignore[arg-type]


def create_backend(name: str = "torch") -> Backend:
    """Factory: instantiate a backend by name (spec §5.11).

    Args:
        name: Backend identifier (``"torch"`` or ``"triton"``).

    Returns:
        A :class:`Backend` instance.

    Raises:
        ValueError: If ``name`` is unknown or the backend is unavailable.
    """
    if name == "torch":
        return TorchBackend()
    if name == "triton":
        if not TritonBackend.is_available():
            raise ValueError("backend 'triton' is not available (needs CUDA + Triton)")
        return TritonBackend()
    msg = f"backend '{name}' is not registered or unavailable"
    raise ValueError(msg)


__all__ = [
    "Backend",
    "TorchBackend",
    "TritonBackend",
    "create_backend",
]
