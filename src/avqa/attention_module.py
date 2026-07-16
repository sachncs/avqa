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

        # M6: Select router based on configured strategy.
        router_name = config.routing.strategy.lower()
        if router_name == "topp":
            self.router = TopPRouter()
        elif router_name == "threshold":
            from avqa.routing import ThresholdRouter

            self.router = ThresholdRouter()
        elif router_name == "budget":
            from avqa.routing import BudgetRouter

            self.router = BudgetRouter()
        else:
            msg = f"unknown routing strategy: {router_name!r}"
            raise ValueError(msg)

        if config.refinement.enabled:
            from avqa.scheduler import AdaptiveScheduler, DefaultScheduler

            # M6: Use AdaptiveScheduler when adaptive_budget is True.
            if config.refinement.adaptive_budget:
                self.scheduler: DefaultScheduler | AdaptiveScheduler | None = AdaptiveScheduler(
                    budget=config.routing.refinement_budget,
                )
            else:
                self.scheduler = DefaultScheduler(
                    budget=config.routing.refinement_budget,
                )
        else:
            self.scheduler = None

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # H4: Store last forward pass data for commitment loss computation.
        self.last_keys: torch.Tensor | None = None
        self.last_parent_assignments: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def maybe_project(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the input projections if ``in_proj=True``."""
        return self.q_proj(query), self.k_proj(key), self.v_proj(value)

    @staticmethod
    def split_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """``[B, T, E]`` → ``[B, H, T, D]``."""
        B, T, E = tensor.shape
        D = E // num_heads
        return tensor.view(B, T, num_heads, D).transpose(1, 2)

    @staticmethod
    def merge_heads(tensor: torch.Tensor) -> torch.Tensor:
        """``[B, H, T, D]`` → ``[B, T, E]``."""
        B, H, T, D = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(B, T, H * D)

    def causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular mask for causal attention (1 = keep, 0 = mask)."""
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Commitment loss (spec §8.9)
    # ------------------------------------------------------------------

    def commitment_loss(self) -> torch.Tensor:
        """Compute the commitment (encoding) loss from the last forward pass.

        Returns the squared distance between each key and its assigned
        parent codeword, weighted by ``config.codebook.commitment_loss_weight``.

        Returns:
            Scalar weighted commitment loss. Returns ``0.0`` if
            ``commitment_loss_weight`` is set to ``0``.

        Raises:
            RuntimeError: If called before any forward pass has executed.
        """
        if self.last_keys is None or self.last_parent_assignments is None:
            msg = "commitment_loss() requires at least one prior forward pass"
            raise RuntimeError(msg)
        raw_loss = self.codebook.commitment_loss(
            self.last_keys,
            self.last_parent_assignments,
        )
        return raw_loss * self.config.codebook.commitment_loss_weight

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
        # ISSUE-0017: wrap in autocast when enabled (spec §3.4).
        autocast_enabled = self.config.precision.autocast
        autocast_dtype = getattr(torch, self.config.precision.dtype, torch.float32)
        with torch.autocast(
            device_type=query.device.type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            return self.forward_impl(query, key, value, mask, kv_cache)

    def forward_impl(  # noqa: PLR0915
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: KVCache | None,
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
        # H5: Input validation (spec §6.12, §10.5).
        if not self.config.backend.skip_validation:
            from avqa.utils.validation import (
                validate_device_match,
                validate_dtype,
                validate_embed_dim,
                validate_rank,
            )

            supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
            for name, tensor in [("query", query), ("key", key), ("value", value)]:
                validate_rank(tensor, 3, name=name)
                validate_dtype(tensor, supported_dtypes, name=name)
            validate_embed_dim(query, self.config.attention.embed_dim, name="query")
            validate_embed_dim(key, self.config.attention.embed_dim, name="key")
            validate_embed_dim(value, self.config.attention.embed_dim, name="value")
            validate_device_match([query, key, value], name="query/key/value")
            if query.shape[-1] != key.shape[-1]:
                from avqa.utils.validation import validate_shape

                validate_shape(key, query.shape, name="key")
            if key.shape != value.shape:
                from avqa.utils.validation import validate_shape

                validate_shape(value, key.shape, name="value")

        q_proj, k_proj, v_proj = self.maybe_project(query, key, value)
        H = self.config.attention.num_heads
        q = self.split_heads(q_proj, H)
        k = self.split_heads(k_proj, H)
        v = self.split_heads(v_proj, H)

        # Ensure codebook dtype and device match the input.
        # M5: codebook is initialized on CPU in FP32; move it to the input's
        # device and dtype when the inputs are on a different device/dtype.
        if self.codebook.parents.device != q.device or self.codebook.parents.dtype != q.dtype:
            self.codebook.parents = self.codebook.parents.to(device=q.device, dtype=q.dtype)
            self.codebook.children = self.codebook.children.to(device=q.device, dtype=q.dtype)

        # M4: Look up cached K/V and concatenate with current K/V before
        # running attention. Previously the cache was appended but never
        # consumed; only the current K/V participated in attention.
        if kv_cache is not None:
            cached_k, cached_v = kv_cache.lookup()
            if cached_k.shape[-2] > 0:
                # Concatenate along the sequence dim (assumed axis -2).
                k_full = torch.cat([cached_k, k], dim=-2)
                v_full = torch.cat([cached_v, v], dim=-2)
            else:
                k_full, v_full = k, v
            kv_cache.append(k, v)
        else:
            k_full, v_full = k, v

        # Resolve mask.
        if mask is None and self.config.causal:
            mask = self.causal_mask(q.shape[-2], q.device)

        # ISSUE-0018: execution mode controls pipeline selection (spec §4.13, §10.15).
        # "reference" always uses naive attention; "optimized"/"research" use AVQ.
        use_naive = self.scheduler is None or self.config.execution.mode == "reference"
        if use_naive:
            attn_out = self.backend.naive_attention(q, k_full, v_full, mask=mask)
            out = self.merge_heads(attn_out)
            out = self.out_proj(out)
            out = self.dropout(out)
            return out  # type: ignore[no-any-return]

        # Branch 2: full AVQ pipeline (use cached+current K/V).
        k_for_vq = k_full
        v_for_vq = v_full

        # Branch 2: full AVQ pipeline.
        B, _, T_q, D = q.shape
        D_v = v_for_vq.shape[-1]
        M0 = self.config.codebook.num_codewords
        C = self.config.codebook.children_per_codeword

        # Stage 2 (spec §10.6): VQ precompute over cached+current K/V.
        result = self.backend.quantize(
            k_for_vq, v_for_vq, self.codebook.parents, self.codebook.children
        )

        # H4: Store keys and assignments for commitment loss computation.
        self.last_keys = k_for_vq.detach()
        self.last_parent_assignments = result.parent_assignments.detach()
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
            result = self.backend.quantize(
                k_for_vq, v_for_vq, self.codebook.parents, self.codebook.children
            )

        # Stage 3 (spec §10.7): parent attention via online-softmax tiled approach.
        parent_keys = self.codebook.parents.unsqueeze(0).expand(B, H, M0, D)
        parent_values = result.parent_aggregates  # [B, H, M_0, D_v]
        valid = result.parent_counts > 0  # [B, H, M_0]

        # Parent logits: Q · C_p^T / sqrt(D). Shape [B, H, T_q, M_0].
        parent_logits = torch.matmul(q, parent_keys.transpose(-2, -1)) / (D**0.5)
        # M3: Apply the supplied mask (e.g., causal). A mask of shape
        # [T_q, T_k] (bool, True = keep) is broadcast to [B, H, T_q, T_k].
        # For AVQ, we collapse to [T_q, 1] — if any key position is masked
        # for a given query, mask the entire parent logit for that query.
        if mask is not None:
            if mask.ndim == 2:
                # Reduce [T_q, T_k] to [T_q, 1]: True if any key is visible.
                codeword_mask = mask.any(dim=-1, keepdim=True)  # [T_q, 1]
            elif mask.ndim == 4:
                codeword_mask = mask.any(dim=-1).any(dim=1, keepdim=True)  # [B, 1, T_q, 1]
            else:
                codeword_mask = torch.ones(T_q, 1, dtype=torch.bool, device=q.device)
            parent_logits = parent_logits.masked_fill(
                ~codeword_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        # Empty codewords excluded from softmax (spec §7.15).
        parent_logits = parent_logits.masked_fill(~valid.unsqueeze(2), float("-inf"))

        # Build running online-softmax state from parent attention.
        # tile_max: amax over M_0 parents → [B, H, T_q, 1].
        tile_max = parent_logits.amax(dim=-1, keepdim=True)
        tile_exp = torch.exp(parent_logits - tile_max)  # [B, H, T_q, M_0]
        # VQ attention denominator: Σ_a exp(S_ia) n_a (spec §7.7).
        tile_denom = (tile_exp * result.parent_counts.unsqueeze(2)).sum(
            dim=-1, keepdim=True
        )  # [B, H, T_q, 1]
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
        num_valid_per_bh = (result.parent_counts > 0).sum(dim=-1)  # [B, H]
        min_valid = int(num_valid_per_bh.min().item())
        budget = min(budget, min_valid)
        if budget <= 0:
            # No valid parents — fall back to naive attention.
            attn_out = (
                self.backend.naive_attention(q, k, v, mask=mask)
                if mask is not None
                else self.backend.naive_attention(q, k, v)
            )
            out = self.merge_heads(attn_out)
            out = self.out_proj(out)
            out = self.dropout(out)
            return out  # type: ignore[no-any-return]
        decision = self.router.select(importance, budget)

        # Stage 6 (spec §10.10): child attention — compute real Q · C_c^T
        # ONLY for the selected parents' children (spec §9.8).
        selected = decision.selected_indices  # [B, H, P]
        P = selected.shape[-1]
        # Gather selected parents' children keys: [B, H, P, C, D].
        parent_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(B, H, P, C, D)
        selected_child_keys = torch.gather(
            self.codebook.children.unsqueeze(0).expand(B, H, M0, C, D),
            2,
            parent_idx,
        )
        # [B, H, T_q, P, C] child attention logits for selected parents only.
        child_logits = torch.einsum("bhtd,bhpcd->bhtpc", q, selected_child_keys) / (D**0.5)
        # Mask empty children.
        selected_child_valid = (
            torch.gather(
                result.child_counts,
                2,
                selected.unsqueeze(-1).expand(B, H, P, C),
            )
            > 0
        )  # [B, H, P, C] bool
        child_logits = child_logits.masked_fill(~selected_child_valid.unsqueeze(2), float("-inf"))

        # Stage 6 + 7: refine with real child logits.
        parent_value_per_parent = parent_attention_probs.unsqueeze(-1) * parent_values.unsqueeze(2)
        refinement = refine_step(
            state=state,
            parent_probs=parent_attention_probs,
            parent_value=parent_value_per_parent,
            parent_aggregates=parent_values,
            child_aggregates=result.child_aggregates,
            children_per_parent=C,
            decision=decision,
            attention_probs=parent_attention_probs,
            parent_counts=result.parent_counts,
            child_logits=child_logits,
            child_counts=result.child_counts,
            merge_strategy=self.config.merge.strategy,
        )
        # C2+C4 + M1: Use the REFINED state from refinement.state (which
        # includes the correction term), not the original state (spec §7.7,
        # §7.13, §7.14). The corrected state has all parents updated: the
        # P selected parents have their coarse contribution replaced by the
        # child contribution; the unselected ones retain the parent state.
        refined_state = refinement.state
        attn_out = refined_state.running_numerator[
            :, :, :, 0, :
        ] / refined_state.running_denominator[:, :, :, 0:1].clamp_min(1e-12)

        # ISSUE-0018: research mode emits diagnostics (spec §10.15).
        if self.config.execution.mode == "research":
            _logger.debug(
                "avq_pipeline: budget=%d selected=%s utilization=%.2f",
                budget,
                refinement.selected_parents.shape[-1],
                (result.parent_counts > 0).float().mean().item(),
            )

        out = self.merge_heads(attn_out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out  # type: ignore[no-any-return]


__all__ = ["AVQAttention"]
