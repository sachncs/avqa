"""AVQAttention: the primary nn.Module entry point (spec §3.4, §5.6).

AVQAttention orchestrates the full 9-stage AVQ pipeline (spec §10.4):

1. Q/K/V projection (optional — accepts pre-projected tensors).
2. Vector-quantization precompute (parent + child aggregates).
3. Parent attention (online softmax over the codebook).
4. Importance estimation from attention statistics.
5. Adaptive parent selection (top-P).
6. Child attention over selected parents.
7. Correcting attention (replace parent contribution with children).
8. Weighted value reduction.
9. Output projection (optional).

ponytail: the nn.Module wrapper lives in the existing
src/avqa/attention.py namespace; the pipeline class itself is small
because every stage is delegated to the corresponding subsystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from avqa.attention import OnlineSoftmaxState
from avqa.backend import create_backend
from avqa.codebook import HierarchicalCodebook
from avqa.config import AVQConfig
from avqa.logging import get_logger
from avqa.refinement import refine as refine_step
from avqa.routing import TopPRouter, compute_importance

if TYPE_CHECKING:
    from avqa.cache import KVCache


_logger = get_logger("attention.module")


class AVQAttention(nn.Module):
    """AVQ-Attention module (spec §3.4, §5.6, §10.4).

    Args:
        config: Immutable :class:`AVQConfig`.
        in_proj: If ``True``, apply a learnable ``E -> E`` linear layer to
            Q/K/V inputs. Default ``True``.
        out_proj: If ``True``, apply a learnable ``E -> E`` linear layer to
            the output. Default ``True``.

    Example:
        >>> config = AVQConfig()
        >>> module = AVQAttention(config)
        >>> q = torch.randn(2, 8, 64, 64)
        >>> k = torch.randn(2, 8, 128, 64)
        >>> v = torch.randn(2, 8, 128, 64)
        >>> out = module(q, k, v)
        >>> out.shape
        torch.Size([2, 8, 64, 64])
    """

    def __init__(self, config: AVQConfig, *, in_proj: bool = True, out_proj: bool = True) -> None:
        super().__init__()
        self.config = config
        self.backend = create_backend(config.backend.name)
        E = config.attention.embed_dim
        # Use Module so we can mix Linear and Identity branches uniformly.
        self.q_proj: nn.Module
        self.k_proj: nn.Module
        self.v_proj: nn.Module
        self.out_proj: nn.Module
        if in_proj:
            self.q_proj = nn.Linear(E, E, bias=False)
            self.k_proj = nn.Linear(E, E, bias=False)
            self.v_proj = nn.Linear(E, E, bias=False)
        else:
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
        if out_proj:
            self.out_proj = nn.Linear(E, E, bias=False)
        else:
            self.out_proj = nn.Identity()

        self.codebook = HierarchicalCodebook(
            num_heads=config.attention.num_heads,
            num_parents=config.codebook.num_codewords,
            children_per_parent=config.codebook.children_per_codeword,
            head_dim=E // config.attention.num_heads,
            perturbation_scale=config.codebook.perturbation_scale,
            device="cpu",
            dtype=torch.float32,
        )
        # Initialize children near parents so the mean constraint holds.
        self.codebook.initialize_children_around_parents()

        self.router = TopPRouter()

        if config.refinement.enabled:
            from avqa.scheduler import DefaultScheduler
            self.scheduler: DefaultScheduler | None = DefaultScheduler(
                budget=config.routing.refinement_budget,
            )
        else:
            self.scheduler = None

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_project(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the input projections if ``in_proj=True``."""
        return self.q_proj(query), self.k_proj(key), self.v_proj(value)

    @staticmethod
    def _split_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """``[B, T, E]`` → ``[B, H, T, D]``."""
        B, T, E = tensor.shape
        D = E // num_heads
        return tensor.view(B, T, num_heads, D).transpose(1, 2)

    @staticmethod
    def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
        """``[B, H, T, D]`` → ``[B, T, E]``."""
        B, H, T, D = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(B, T, H * D)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular mask for causal attention (1 = keep, 0 = mask)."""
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # noqa: PLR0915
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Run the AVQ-Attention forward pass.

        Args:
            query: ``[B, T_q, E]`` queries.
            key: ``[B, T_k, E]`` keys.
            value: ``[B, T_k, E]`` values.
            mask: Optional ``[T_q, T_k]`` boolean mask (True = attend). When
                ``config.causal`` is True and ``mask`` is None, a causal
                mask is built automatically.
            kv_cache: Optional cache to extend with the new K/V tensors.

        Returns:
            ``[B, T_q, E]`` attention output.
        """
        Q = self._maybe_project(query, key, value)
        q_proj, k_proj, v_proj = Q
        H = self.config.attention.num_heads
        q = self._split_heads(q_proj, H)
        k = self._split_heads(k_proj, H)
        v = self._split_heads(v_proj, H)

        # Ensure codebook dtype matches the input dtype (the codebook is
        # initialized in FP32 at construction time).
        if self.codebook.parents.dtype != q.dtype:
            self.codebook.parents = self.codebook.parents.to(dtype=q.dtype)
            self.codebook.children = self.codebook.children.to(dtype=q.dtype)

        # Optionally extend the KV cache.
        if kv_cache is not None:
            kv_cache.append(k, v)

        # Resolve mask.
        if mask is None and self.config.causal:
            mask = self._causal_mask(q.shape[-2], q.device)

        # Branch 1: refinement disabled — plain attention via the backend.
        if self.scheduler is None:
            attn_out = self.backend.naive_attention(q, k, v, mask=mask)
            out = self._merge_heads(attn_out)
            out = self.out_proj(out)
            out = self.dropout(out)
            return out  # type: ignore[no-any-return]

        # Branch 2: full AVQ pipeline.
        B, _, T_q, D = q.shape
        D_v = v.shape[-1]
        M0 = self.config.codebook.num_codewords
        C = self.config.codebook.children_per_codeword

        # Stage 2 (spec §10.6): VQ precompute.
        result = self.backend.quantize(k, v, self.codebook.parents, self.codebook.children)
        expected_parents = (H, M0, D)
        if tuple(self.codebook.parents.shape) != expected_parents:
            self.codebook = HierarchicalCodebook(
                num_heads=H,
                num_parents=M0,
                children_per_parent=C,
                head_dim=D,
                device=q.device,
                dtype=q.dtype,
            )
            self.codebook.initialize_children_around_parents()
            result = self.backend.quantize(k, v, self.codebook.parents, self.codebook.children)

        # Stage 3 (spec §10.7): parent attention via online-softmax tiled approach.
        parent_keys = self.codebook.parents.unsqueeze(0).expand(B, H, M0, D)
        parent_values = result.parent_aggregates                      # [B, H, M_0, D_v]
        valid = result.parent_counts > 0                              # [B, H, M_0]

        # Parent logits: Q · C_p^T / sqrt(D). Shape [B, H, T_q, M_0].
        parent_logits = torch.matmul(q, parent_keys.transpose(-2, -1)) / (D ** 0.5)
        if mask is not None:
            codeword_mask = torch.ones(T_q, M0, dtype=torch.bool, device=q.device)
            parent_logits = parent_logits.masked_fill(~codeword_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        # Empty codewords excluded from softmax (spec §7.15).
        parent_logits = parent_logits.masked_fill(~valid.unsqueeze(2), float("-inf"))

        # Build running online-softmax state from parent attention.
        # tile_max: amax over M_0 parents → [B, H, T_q, 1].
        tile_max = parent_logits.amax(dim=-1, keepdim=True)
        tile_exp = torch.exp(parent_logits - tile_max)               # [B, H, T_q, M_0]
        tile_denom = tile_exp.sum(dim=-1, keepdim=True)              # [B, H, T_q, 1]
        # Numerator: contract exp(S) over M_0 with parent values.
        # tile_exp: [B, H, T_q, M_0], parent_values: [B, H, M_0, D_v]
        tile_num = torch.einsum("bhta,bhad->bhtd", tile_exp, parent_values)
        # [B, H, T_q, D_v] → unsqueeze to [B, H, T_q, 1, D_v] for state.
        tile_num = tile_num.unsqueeze(-2)
        state = OnlineSoftmaxState.empty(B, H, T_q, D, D_v)
        state = state.merge(tile_max, tile_denom, tile_num)

        # Parent attention probabilities (normalized).
        parent_attention_probs = tile_exp / tile_exp.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Stage 4 (spec §10.8): importance from attention statistics.
        importance = compute_importance(parent_attention_probs, result.parent_counts)

        # Stage 5 (spec §10.9): parent selection.
        budget = self.scheduler.budget_for(importance)
        # Cap budget at the minimum number of parents with valid children
        # across all (B, H) pairs (spec §9.12).
        num_valid_per_bh = (result.parent_counts > 0).sum(dim=-1)    # [B, H]
        min_valid = int(num_valid_per_bh.min().item())
        budget = min(budget, min_valid)
        if budget <= 0:
            # No valid parents — fall back to naive attention.
            attn_out = self.backend.naive_attention(q, k, v, mask=mask) if mask is not None else self.backend.naive_attention(q, k, v)
            out = self._merge_heads(attn_out)
            out = self.out_proj(out)
            out = self.dropout(out)
            return out  # type: ignore[no-any-return]
        self.router.select(importance, budget)

        # Stage 6 (spec §10.10): child attention — compute real Q · C_c^T
        # for the children of all parents (selected ones will be used).
        child_keys = self.codebook.children.unsqueeze(0).expand(B, H, M0, C, D)
        # [B, H, T_q, M_0, C] child attention logits.
        child_logits = torch.einsum("bhtd,bhmcd->bhtmc", q, child_keys) / (D ** 0.5)
        # Mask empty children.
        child_valid = result.child_counts > 0                         # [B, H, M_0, C]
        child_logits = child_logits.masked_fill(~child_valid.unsqueeze(2), float("-inf"))

        # Stage 6 + 7: refine with real child logits.
        parent_value_per_parent = (
            parent_attention_probs.unsqueeze(-1) * parent_values.unsqueeze(2)
        )
        refinement = refine_step(
            state=state,
            parent_probs=parent_attention_probs,
            parent_value=parent_value_per_parent,
            parent_aggregates=parent_values,
            child_aggregates=result.child_aggregates,
            children_per_parent=C,
            budget=budget,
            attention_probs=parent_attention_probs,
            parent_counts=result.parent_counts,
            child_logits=child_logits,
        )
        attn_out = refinement.merge_value

        out = self._merge_heads(attn_out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out  # type: ignore[no-any-return]


__all__ = ["AVQAttention"]
