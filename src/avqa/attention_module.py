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
from avqa.merge import ProbabilityMerge
from avqa.quantizer import EuclideanHierarchicalQuantizer
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

        self.quantizer = EuclideanHierarchicalQuantizer()
        self.router = TopPRouter()
        self.merge = ProbabilityMerge()

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

    def forward(
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

        # Stage 2 (spec §10.6): VQ precompute.
        result = self.backend.quantize(k, v, self.codebook.parents, self.codebook.children)
        expected = (*self.codebook.children.shape[:3], D)
        if tuple(self.codebook.parents.shape) != expected:
            # Resize codebook if D changed (rare).
            self.codebook = HierarchicalCodebook(
                num_heads=H,
                num_parents=self.config.codebook.num_codewords,
                children_per_parent=self.config.codebook.children_per_codeword,
                head_dim=D,
                device=q.device,
                dtype=q.dtype,
            )
            self.codebook.initialize_children_around_parents()
            result = self.backend.quantize(k, v, self.codebook.parents, self.codebook.children)

        # Stage 3 (spec §10.7): parent attention via online softmax.
        # Use the codeword parents as keys and parent_aggregates as values.
        # Expand codebook parents to [B, H, M_0, D] for the batch dimension.
        parent_keys = self.codebook.parents.unsqueeze(0).expand(B, H, -1, -1)
        # parent_aggregates is [B, H, M_0, D_v]; we want [B, H, M_0, D_v] as
        # "values" for codeword attention.
        parent_values = result.parent_aggregates                  # [B, H, M_0, D_v]
        # The "key/value" lengths are M_0; query is T_q.
        attention_logits = torch.matmul(q, parent_keys.transpose(-2, -1)) / (D ** 0.5)
        if mask is not None:
            codeword_mask = torch.ones(
                T_q, parent_keys.shape[-2], dtype=torch.bool, device=q.device,
            )
            attention_logits = attention_logits.masked_fill(
                ~codeword_mask.unsqueeze(0).unsqueeze(0), float("-inf"),
            )
        # Spec §7.15: empty codewords (count=0) excluded from running max.
        valid = result.parent_counts > 0
        # Replace logits with -inf for empty codewords (excluded from softmax).
        attention_logits = attention_logits.masked_fill(
            ~valid.unsqueeze(2), float("-inf"),
        )
        # VQ attention (spec §7.7):
        #   y_i = sum_a exp(q . C_a) V_bar_a / sum_a exp(q . C_a) n_a
        # Use the unnormalized exp to avoid double-normalization.
        exp_logits = torch.exp(attention_logits - attention_logits.amax(dim=-1, keepdim=True))
        weighted_value = torch.einsum(
            "bhta,bhvd->bhtv", exp_logits, parent_values,
        )
        denom = torch.einsum(
            "bhta,bhv->bht", exp_logits, result.parent_counts,
        ).clamp_min(1e-12)
        _parent_vq_out = weighted_value / denom.unsqueeze(-1)
        # Parent attention probabilities for importance + downstream stages.
        parent_attention_probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Stage 4 (spec §10.8): importance from attention statistics.
        importance = compute_importance(parent_attention_probs, result.parent_counts)

        # Stage 5 (spec §10.9): parent selection.
        budget = self.scheduler.budget_for(importance)
        _ = self.router.select(importance, budget)

        # Stage 6 + 7 (spec §10.10-§10.11): child attention + correction.
        # Build the inputs to the refine() orchestrator.
        # We treat parent_attention_probs * parent_aggregates as parent_value.
        # Shape: [B, H, T_q, M_0, D_v] (one weighted value per parent).
        parent_value_per_parent = (
            parent_attention_probs.unsqueeze(-1) * parent_values.unsqueeze(2)
        )
        # Set up the running online-softmax state.
        state = OnlineSoftmaxState.empty(B, H, T_q, D, D_v)
        # Drive the running state from the parent attention.
        tile_logits = parent_attention_logits_for_state(
            q, parent_keys, D, valid, mask,
        )                                                          # [B, H, T, M_0]
        tile_max = tile_logits.amax(dim=-1, keepdim=True)        # [B, H, T, 1]
        tile_exp = torch.exp(tile_logits - tile_max)             # [B, H, T, M_0]
        tile_denom = tile_exp.sum(dim=-1, keepdim=True)          # [B, H, T, 1]
        # parent_values has shape [B, H, M_0, D_v]; contract over M_0.
        tile_num = torch.einsum(
            "bhta,bhvd->bhtv", tile_exp, parent_values,
        ).unsqueeze(-1)                                          # [B, H, T, 1, D_v]
        state = state.merge(tile_max, tile_denom, tile_num)

        # Stage 6 + 7: refine.
        # Build the parent_value for refine(): per-query weighted value, shape
        # [B, H, T, M_0, D_v]. The refine function expects per-parent values.
        parent_value_per_parent = (
            parent_attention_probs.unsqueeze(-1) * parent_values.unsqueeze(2)
        )
        refinement = refine_step(
            state=state,
            parent_probs=parent_attention_probs,
            parent_value=parent_value_per_parent,
            parent_aggregates=parent_values,
            child_aggregates=result.child_aggregates,
            children_per_parent=self.config.codebook.children_per_codeword,
            budget=budget,
            attention_probs=parent_attention_probs,
            parent_counts=result.parent_counts,
        )
        attn_out = refinement.merge_value

        out = self._merge_heads(attn_out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out  # type: ignore[no-any-return]


def parent_attention_logits_for_state(
    query: torch.Tensor,
    parent_keys: torch.Tensor,
    head_dim: int,
    valid: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute parent attention logits with empty-codeword masking (spec §7.15).

    Helper used by :class:`AVQAttention` for the running-state tile update.
    """
    logits = torch.matmul(query, parent_keys.transpose(-2, -1)) / (head_dim ** 0.5)
    if mask is not None:
        codeword_mask = torch.ones(
            query.shape[-2], parent_keys.shape[-2], dtype=torch.bool, device=query.device,
        )
        logits = logits.masked_fill(~codeword_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    return logits  # type: ignore[no-any-return]


__all__ = ["AVQAttention", "parent_attention_logits_for_state"]
