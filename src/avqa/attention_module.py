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
    from avqa.quantizer import QuantizationResult
    from avqa.routing import RoutingDecision


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
        # OPT-0002: optional torch.compile wrapping. We attach the
        # eager function as ``self._forward_eager`` so the original
        # path remains the source of truth and so the compiled forward
        # can be reverted at runtime.
        self._forward_eager = self.forward_impl
        if config.execution.compile_enabled:
            # OPT-0002: route the forward through a torch.compile graph.
            # `dynamic=None` lets Dynamo adapt to mask / kv-cache variants
            # while still collapsing the Python overhead per call.
            self._forward_compiled = torch.compile(
                self.forward_impl,
                dynamic=None,
            )
        else:
            self._forward_compiled = None
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

        # OPT-0005 (HVAQ): learnable parameters for per-parent β_p
        # and per-head α. When disabled these are absent from
        # parameters() and have zero overhead.
        hopfield_active = config.backend.hopfield and config.hopfield.adaptive != "none"
        if hopfield_active and config.hopfield.learnable_parent_beta:
            M0 = config.codebook.num_codewords
            self._parent_beta = nn.Parameter(torch.ones(1, 1, 1, M0))
        if hopfield_active and config.hopfield.learnable_alpha:
            H = config.attention.num_heads
            self._alpha = nn.Parameter(
                torch.full((H,), config.hopfield.alpha)
            )

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
        # OPT-0002: when torch.compile is enabled we route through the
        # compiled forward; otherwise we keep the eager path identical
        # to the prior behaviour.
        target = self._forward_compiled or self.forward_impl
        with torch.autocast(
            device_type=query.device.type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            return target(query, key, value, mask, kv_cache)

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """Run the AVQ-Attention forward pass (eager path).

        Args:
            query: ``[B, T_q, E]`` queries.
            key: ``[B, T_k, E]`` keys.
            value: ``[B, T_k, E]`` values.
            mask: Optional ``[T_q, T_k]`` boolean mask (True = attend).
            kv_cache: Optional cache to extend with the new K/V tensors.

        Returns:
            ``[B, T_q, E]`` attention output.
        """
        self._validate_inputs(query, key, value)

        q_proj, k_proj, v_proj = self.maybe_project(query, key, value)
        H = self.config.attention.num_heads
        q = self.split_heads(q_proj, H)
        k = self.split_heads(k_proj, H)
        v = self.split_heads(v_proj, H)

        self._sync_codebook_device(q)
        k_full, v_full = self._resolve_kv_cache(k, v, kv_cache)
        mask = self._resolve_mask(mask, q)

        use_naive = self.scheduler is None or self.config.execution.mode == "reference"
        if use_naive:
            return self._run_naive(q, k_full, v_full, mask)

        B, _, T_q, D = q.shape
        D_v = v_full.shape[-1]
        M0 = self.config.codebook.num_codewords
        C = self.config.codebook.children_per_codeword

        result = self._run_vq_precompute(k_full, v_full, q, H, M0, C, D, kv_cache)

        parent_logits, valid, parent_values = self._compute_parent_logits(
            q, result, mask, B, H, T_q, M0, D,
        )
        parent_logits = self._apply_hopfield(parent_logits, valid, D)

        state, parent_attention_probs = self._compute_online_softmax(
            parent_logits, parent_values, result, B, H, T_q, D, D_v,
        )

        _importance, budget, decision, result = self._compute_routing(
            parent_attention_probs, result, q, k, v, mask, B, H, M0,
        )
        if budget <= 0:
            return self._run_naive(q, k, v, mask)
        assert decision is not None

        child_logits = self._compute_child_logits(
            q, result, decision.selected_indices, H,
            decision.selected_indices.shape[-1], C, D, M0,
        )

        attn_out = self._refine_and_output(
            state, parent_attention_probs, parent_values,
            child_logits, result, decision, budget, C, D_v,
        )

        out = self.merge_heads(attn_out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Pipeline stage helpers (extracted from forward_impl)
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Validate input tensors (spec §6.12, §10.5)."""
        if self.config.backend.skip_validation:
            return
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

    def _sync_codebook_device(self, q: torch.Tensor) -> None:
        """Move codebook to match input device and dtype (M5)."""
        if self.codebook.parents.device != q.device or self.codebook.parents.dtype != q.dtype:
            self.codebook.parents = self.codebook.parents.to(device=q.device, dtype=q.dtype)
            self.codebook.children = self.codebook.children.to(device=q.device, dtype=q.dtype)

    def _resolve_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up cached K/V and concatenate with current K/V (M4)."""
        if kv_cache is not None:
            cached_k, cached_v = kv_cache.lookup()
            if cached_k.shape[-2] > 0:
                k_full = torch.cat([cached_k, k], dim=-2)
                v_full = torch.cat([cached_v, v], dim=-2)
            else:
                k_full, v_full = k, v
            kv_cache.append(k, v)
        else:
            k_full, v_full = k, v
        return k_full, v_full

    def _resolve_mask(
        self,
        mask: torch.Tensor | None,
        q: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build causal mask when needed."""
        if mask is None and self.config.causal:
            return self.causal_mask(q.shape[-2], q.device)
        return mask

    def _run_naive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run naive (non-AVQ) attention and project output."""
        attn_out = self.backend.naive_attention(q, k, v, mask=mask)
        out = self.merge_heads(attn_out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out  # type: ignore[no-any-return]

    def _run_vq_precompute(
        self,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        q: torch.Tensor,
        H: int,
        M0: int,
        C: int,
        D: int,
        kv_cache: KVCache | None,
    ) -> QuantizationResult:
        """Run VQ precompute (streaming or batched) and BCAR adaptation.

        Returns:
            QuantizationResult with parent/child aggregates, assignments,
            and counts.
        """
        if (
            self.config.execution.causal_incremental
            and kv_cache is not None
            and hasattr(kv_cache, "causal_extend")
        ):
            from avqa.streaming_vq import StreamingVQBuffer

            new_tokens = k_full[..., -1:, :]
            if not hasattr(self, "_streaming_buffer"):
                self._streaming_buffer = StreamingVQBuffer(
                    num_heads=H,
                    num_parents=M0,
                    children_per_parent=C,
                    head_dim=D,
                    device=k_full.device,
                    dtype=self.codebook.parents.dtype,
                )
            else:
                self._streaming_buffer.reset()
            for h in range(H):
                self._streaming_buffer.extend(
                    new_tokens[:, h, 0, :].unsqueeze(0),
                    self.codebook.parents,
                    self.codebook.children,
                )
            result = self._streaming_buffer.realize()
        else:
            result = self.backend.quantize(
                k_full, v_full, self.codebook.parents, self.codebook.children
            )

        self.last_keys = k_full.detach()
        self.last_parent_assignments = result.parent_assignments.detach()

        if self.config.codebook.bcar_enabled:
            from avqa.online_adaptation import online_codebook_adaptation
            online_codebook_adaptation(
                keys=k_full.detach(),
                parents=self.codebook.parents,
                children=self.codebook.children,
                parent_assignments=result.parent_assignments.detach(),
                child_assignments=result.child_assignments.detach(),
                decay=self.config.codebook.bcar_decay,
            )

        expected_parents = (H, M0, D)
        if tuple(self.codebook.parents.shape) != expected_parents:
            self.codebook = HierarchicalCodebook(
                num_heads=H, num_parents=M0, children_per_parent=C,
                head_dim=D, device=q.device, dtype=q.dtype,
            )
            self.codebook.initialize_children_around_parents()
            result = self.backend.quantize(
                k_full, v_full, self.codebook.parents, self.codebook.children
            )
        return result

    def _compute_parent_logits(
        self,
        q: torch.Tensor,
        result: QuantizationResult,
        mask: torch.Tensor | None,
        B: int,
        H: int,
        T_q: int,
        M0: int,
        D: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute parent attention logits with mask applied.

        Returns:
            Tuple of (parent_logits, valid, parent_values).
        """
        parent_keys = self.codebook.parents.unsqueeze(0).expand(B, H, M0, D)
        parent_values = result.parent_aggregates
        valid = result.parent_counts > 0

        parent_logits = torch.matmul(q, parent_keys.transpose(-2, -1)) / (D**0.5)
        if mask is not None:
            if mask.ndim == 2:
                codeword_mask = mask.any(dim=-1, keepdim=True)
            elif mask.ndim == 4:
                codeword_mask = mask.any(dim=-1).any(dim=1, keepdim=True)
            else:
                codeword_mask = torch.ones(T_q, 1, dtype=torch.bool, device=q.device)
            parent_logits = parent_logits.masked_fill(
                ~codeword_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        parent_logits = parent_logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
        return parent_logits, valid, parent_values

    def _apply_hopfield(
        self,
        parent_logits: torch.Tensor,
        valid: torch.Tensor,
        D: int,
    ) -> torch.Tensor:
        """Apply HVAQ per-query temperature schedule (OPT-0005, SPEC §16).

        ``parent_logits`` arrives as ``Q · C_p^T / sqrt(D)``.  We undo
        the ``1/sqrt(D)`` factor, apply the schedule, and return the
        rescaled logits.

        When disabled, returns ``parent_logits`` unchanged.
        """
        if not (self.config.backend.hopfield and self.config.hopfield.adaptive != "none"):
            return parent_logits
        from avqa.hopfield import hopfield_logits, paper_beta, per_query_beta

        beta_init = self.config.hopfield.beta_init
        if beta_init <= 0.0:
            beta_init = paper_beta(D)
        base_for_entropy = parent_logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
        paper_probs = base_for_entropy.softmax(dim=-1)
        paper_probs = paper_probs * valid.unsqueeze(2).to(paper_probs.dtype)
        denom = paper_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        paper_probs = paper_probs / denom
        beta_q = per_query_beta(
            paper_probs,
            beta_init=beta_init,
            adaptive=self.config.hopfield.adaptive,
            alpha=self._alpha.view(1, -1, 1) if hasattr(self, "_alpha") else self.config.hopfield.alpha,
        )
        # hopfield_logits expects raw (unscaled) base logits; undo /sqrt(D).
        raw_logits = parent_logits * (D**0.5)
        parent_beta = self._parent_beta if hasattr(self, "_parent_beta") else 1.0
        return hopfield_logits(raw_logits, beta_q, parent_beta=parent_beta)

    def _compute_online_softmax(
        self,
        parent_logits: torch.Tensor,
        parent_values: torch.Tensor,
        result: QuantizationResult,
        B: int,
        H: int,
        T_q: int,
        D: int,
        D_v: int,
    ) -> tuple[OnlineSoftmaxState, torch.Tensor]:
        """Build running online-softmax state and parent attention probs."""
        tile_max = parent_logits.amax(dim=-1, keepdim=True)
        tile_exp = torch.exp(parent_logits - tile_max)
        tile_denom = (tile_exp * result.parent_counts.unsqueeze(2)).sum(
            dim=-1, keepdim=True
        )
        tile_num = torch.einsum("bhta,bhad->bhtd", tile_exp, parent_values)
        tile_num = tile_num.unsqueeze(-2)
        state = OnlineSoftmaxState.empty(B, H, T_q, D, D_v)
        state = state.merge(tile_max, tile_denom, tile_num)
        parent_attention_probs = tile_exp / tile_exp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return state, parent_attention_probs

    def _compute_routing(
        self,
        parent_attention_probs: torch.Tensor,
        result: QuantizationResult,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        B: int,
        H: int,
        M0: int,
    ) -> tuple[torch.Tensor, int, RoutingDecision | None, QuantizationResult]:
        """Compute importance, budget, routing decision.

        Returns:
            Tuple of (importance, budget, decision, result).
            ``budget <= 0`` means fall back to naive attention.
        """
        importance = compute_importance(parent_attention_probs, result.parent_counts)
        budget = self.scheduler.budget_for(importance)
        num_valid_per_bh = (result.parent_counts > 0).sum(dim=-1)
        min_valid = int(num_valid_per_bh.min().item())
        budget = min(budget, min_valid)
        if budget <= 0:
            return importance, budget, None, result
        decision = self.router.select(importance, budget)
        return importance, budget, decision, result

    def _compute_child_logits(
        self,
        q: torch.Tensor,
        result: QuantizationResult,
        selected: torch.Tensor,
        H: int,
        P: int,
        C: int,
        D: int,
        M0: int,
    ) -> torch.Tensor:
        """Compute child attention logits for selected parents (spec §10.10)."""
        B = q.shape[0]
        parent_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(B, H, P, C, D)
        selected_child_keys = torch.gather(
            self.codebook.children.unsqueeze(0).expand(B, H, M0, C, D),
            2,
            parent_idx,
        )
        child_logits: torch.Tensor = (
            torch.einsum("bhtd,bhpcd->bhtpc", q, selected_child_keys) / (D**0.5)
        )
        selected_child_valid = (
            torch.gather(
                result.child_counts, 2,
                selected.unsqueeze(-1).expand(B, H, P, C),
            )
            > 0
        )
        child_logits = child_logits.masked_fill(
            ~selected_child_valid.unsqueeze(2), float("-inf")
        )
        return child_logits

    def _refine_and_output(
        self,
        state: OnlineSoftmaxState,
        parent_attention_probs: torch.Tensor,
        parent_values: torch.Tensor,
        child_logits: torch.Tensor,
        result: QuantizationResult,
        decision: RoutingDecision,
        budget: int,
        C: int,
        D_v: int,
    ) -> torch.Tensor:
        """Run refinement and extract the final attention output (spec §7.7)."""
        if self.config.refinement.passes > 1:
            _logger.debug(
                "refinement.passes=%d > 1; falling back to single-pass",
                self.config.refinement.passes,
            )
        refinement = refine_step(
            state=state,
            parent_probs=parent_attention_probs,
            parent_value=(
                parent_attention_probs.unsqueeze(-1) * parent_values.unsqueeze(2)
            ),
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
        refined_state = refinement.state
        attn_out = (
            refined_state.running_numerator[:, :, :, 0, :]
            / refined_state.running_denominator[:, :, :, 0:1].clamp_min(1e-12)
        )

        if self.config.execution.mode == "research":
            _logger.debug(
                "avq_pipeline: budget=%d selected=%s utilization=%.2f",
                budget,
                decision.selected_indices.shape[-1],
                (result.parent_counts > 0).float().mean().item(),
            )
        return attn_out


__all__ = ["AVQAttention"]
