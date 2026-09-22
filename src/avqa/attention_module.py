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
from typing_extensions import Protocol

from avqa.attention import OnlineSoftmaxState
from avqa.backend import Backend
from avqa.codebook import HierarchicalCodebook
from avqa.config import AVQConfig
from avqa.exceptions import DeviceError, DtypeError, NotInitializedError, ShapeError
from avqa.logging import get_logger
from avqa.multipass import MultiPassRefiner
from avqa.online_adaptation import online_codebook_adaptation
from avqa.pipeline import run_pipeline
from avqa.refinement import refine as refine_step
from avqa.routing import Router, compute_importance
from avqa.scheduler import Scheduler
from avqa.utils.validation import (
    validate_device_match,
    validate_dtype,
    validate_embed_dim,
    validate_rank,
    validate_shape,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from avqa.cache import KVCache
    from avqa.quantizer import QuantizationResult
    from avqa.routing import RoutingDecision


logger = get_logger("attention.module")


class TensorModule(Protocol):
    """Protocol for any :class:`nn.Module` whose forward returns a tensor.

    PyTorch's :class:`nn.Module.__call__` is untyped; this protocol
    captures the subset we actually use so the type checker can resolve
    ``module(x) -> Tensor`` without suppressions.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class AVQAttention(nn.Module):
    """AVQ-Attention module (spec §3.4, §5.6, §10.4).

    Owns Q/K/V projections, the hierarchical codebook, the router, the
    scheduler, and the optional HVAQ parameters. The 12-stage AVQ
    forward (validate → split heads → VQ precompute → parent logits
    → HVAQ → online softmax → routing → child logits → refinement →
    output) is orchestrated by :func:`avqa._pipeline.run_pipeline`.

    Args:
        config: Immutable :class:`AVQConfig`.
        in_proj: If ``True``, apply a learnable ``E -> E`` linear layer to
            Q/K/V inputs. Default ``True``.
        out_proj: If ``True``, apply a learnable ``E -> E`` linear layer to
            the output. Default ``True``.

    Example:
        >>> config = AVQConfig()
        >>> module = AVQAttention(config)
        >>> q = torch.randn(2, 64, 512)
        >>> k = torch.randn(2, 128, 512)
        >>> v = torch.randn(2, 128, 512)
        >>> out = module(q, k, v)
        >>> out.shape
        torch.Size([2, 64, 512])
    """

    def __init__(self, config: AVQConfig, *, in_proj: bool = True, out_proj: bool = True) -> None:
        super().__init__()
        self.config = config
        self.backend = Backend.create(config.backend.name)
        # OPT-0002: optional torch.compile wrapping. We attach the
        # eager function as ``self.forward_eager`` so the original path
        # remains the source of truth and so the compiled forward can
        # be reverted at runtime.
        self.forward_eager: Callable[..., torch.Tensor] = self.forward_impl
        self.forward_compiled: Callable[..., torch.Tensor] | None
        if config.execution.compile_enabled:
            # OPT-0002: route the forward through a torch.compile graph.
            # `dynamic=None` lets Dynamo adapt to mask / kv-cache variants
            # while still collapsing the Python overhead per call.
            self.forward_compiled = torch.compile(
                self.forward_impl,
                dynamic=None,
            )
        else:
            self.forward_compiled = None
        E = config.attention.embed_dim
        # Use Module so we can mix Linear and Identity branches uniformly.
        self.q_proj: TensorModule
        self.k_proj: TensorModule
        self.v_proj: TensorModule
        self.out_proj: TensorModule
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
        # HierarchicalCodebook intentionally keeps its established lightweight
        # API (its public child tensor is named ``children``). Mirror its
        # tensors as module buffers so AVQAttention checkpoints persist and
        # restore the full attention state without changing that API.
        self.codebook_parents: torch.Tensor
        self.codebook_children: torch.Tensor
        self.register_buffer("codebook_parents", self.codebook.parents)
        self.register_buffer("codebook_children", self.codebook.children)

        # Resolve the configured router via the classmethod factory.
        self.router = Router.create(config.routing.strategy)

        # Scheduler (None when refinement is disabled).
        self.scheduler: Scheduler | None
        if config.refinement.enabled:
            # adaptive_budget picks between the two scheduler families.
            scheduler_strategy = "adaptive" if config.refinement.adaptive_budget else "default"
            self.scheduler = Scheduler.create(
                scheduler_strategy,
                budget=config.routing.refinement_budget,
            )
        else:
            self.scheduler = None

        self.dropout: TensorModule = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # OPT-0005 (HVAQ): learnable parameters for per-parent beta_p
        # and per-head alpha. Always register the parameters so
        # state_dict keys are stable across flag toggles; gradients
        # are gated by requires_grad on the underlying tensor.
        hopfield_active = config.backend.hopfield and config.hopfield.adaptive != "none"
        M0 = config.codebook.num_codewords
        H = config.attention.num_heads
        parent_beta_train = hopfield_active and config.hopfield.learnable_parent_beta
        alpha_train = hopfield_active and config.hopfield.learnable_alpha
        self.parent_beta = nn.Parameter(torch.ones(1, 1, 1, M0), requires_grad=parent_beta_train)
        self.alpha = nn.Parameter(
            torch.full((H,), config.hopfield.alpha), requires_grad=alpha_train
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

    def causal_mask(self, t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask ``[T_q, T_k]`` (1 = keep, 0 = mask).

        ``mask[i, j] = 1`` iff query position ``i`` may attend to key
        position ``j``. With a KV cache (T_q smaller than T_k), cached
        tokens at positions ``0 .. T_k - T_q - 1`` are entirely visible
        to every query (the offset accounts for the cache prefix).
        """
        m = torch.zeros(t_q, t_k, device=device, dtype=torch.bool)
        offset = t_k - t_q
        for i in range(t_q):
            j_max = i + offset
            m[i, : j_max + 1] = True
        return m

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
            NotInitializedError: If called before any forward pass has executed.
        """
        if self.last_keys is None or self.last_parent_assignments is None:
            msg = "commitment_loss() requires at least one prior forward pass"
            raise NotInitializedError(msg)
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
        with torch.autocast(
            device_type=query.device.type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            if self.forward_compiled is not None:
                return self.forward_compiled(query, key, value, mask, kv_cache)
            return self.forward_impl(query, key, value, mask, kv_cache)

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """Run the AVQ-Attention forward pass (eager path).

        The orchestration lives in :func:`avqa._pipeline.run_pipeline` so
        this method stays focused on parameter state and dispatch.

        Args:
            query: ``[B, T_q, E]`` queries.
            key: ``[B, T_k, E]`` keys.
            value: ``[B, T_k, E]`` values.
            mask: Optional ``[T_q, T_k]`` boolean mask (True = attend).
            kv_cache: Optional cache to extend with the new K/V tensors.

        Returns:
            ``[B, T_q, E]`` attention output.
        """
        return run_pipeline(self, query, key, value, mask, kv_cache)

    # ------------------------------------------------------------------
    # Pipeline stage helpers (extracted from forward_impl)
    # ------------------------------------------------------------------

    def validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Validate input tensors (spec §6.12, §10.5)."""
        if self.config.backend.skip_validation:
            return

        supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        for name, tensor in [("query", query), ("key", key), ("value", value)]:
            validate_rank(tensor, 3, name=name)
            validate_dtype(tensor, supported_dtypes, name=name)
        validate_embed_dim(query, self.config.attention.embed_dim, name="query")
        validate_embed_dim(key, self.config.attention.embed_dim, name="key")
        validate_embed_dim(value, self.config.attention.embed_dim, name="value")
        validate_device_match([query, key, value], name="query/key/value")
        if query.shape[-1] != key.shape[-1]:
            validate_shape(key, query.shape, name="key")
        if key.shape != value.shape:
            validate_shape(value, key.shape, name="value")

    def validate_mask(
        self,
        mask: torch.Tensor | None,
        query: torch.Tensor,
        key_length: int,
    ) -> None:
        """Validate a user mask against resolved query/key dimensions."""
        if mask is None or self.config.backend.skip_validation:
            return
        if mask.ndim not in (2, 4):
            raise ShapeError(
                "mask must be rank 2 [T_q, T_k] or rank 4 [B, H, T_q, T_k]",
                expected="rank=2 or rank=4",
                actual=f"rank={mask.ndim}",
            )
        if mask.dtype is not torch.bool:
            raise DtypeError(
                "mask dtype mismatch: expected torch.bool",
                expected=torch.bool,
                actual=mask.dtype,
            )
        if mask.device != query.device:
            raise DeviceError(
                "mask device mismatch",
                expected=query.device,
                actual=mask.device,
            )

        expected_query = int(query.shape[-2])
        expected_key = key_length
        if mask.ndim == 2:
            expected_shape = (expected_query, expected_key)
            if tuple(mask.shape) != expected_shape:
                raise ShapeError(
                    "mask shape must match resolved query/key lengths",
                    expected=expected_shape,
                    actual=tuple(mask.shape),
                )
            return

        batch, heads = int(query.shape[0]), int(query.shape[1])
        if mask.shape[0] not in (1, batch) or mask.shape[1] not in (1, heads):
            raise ShapeError(
                "rank-4 mask batch/head dimensions must be broadcastable",
                expected=f"batch in (1, {batch}), heads in (1, {heads})",
                actual=tuple(mask.shape[:2]),
            )
        expected_tail = (expected_query, expected_key)
        if tuple(mask.shape[-2:]) != expected_tail:
            raise ShapeError(
                "mask shape must match resolved query/key lengths",
                expected=f"[..., {expected_query}, {expected_key}]",
                actual=tuple(mask.shape),
            )

    def sync_codebook_device(self, q: torch.Tensor) -> None:
        """Move codebook to match input device and dtype (M5)."""
        if self.codebook_parents.device != q.device or self.codebook_parents.dtype != q.dtype:
            self.codebook_parents = self.codebook_parents.to(device=q.device, dtype=q.dtype)
            self.codebook_children = self.codebook_children.to(device=q.device, dtype=q.dtype)
            self.codebook.parents = self.codebook_parents
            self.codebook.children = self.codebook_children

    def resolve_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up cached K/V and concatenate with current K/V (M4).

        This method is intentionally read-only. The pipeline commits the
        current K/V only after the complete forward pass succeeds, keeping
        caller-owned cache state failure-atomic.
        """
        if kv_cache is not None:
            cached_k, cached_v = kv_cache.lookup()
            if cached_k.device != k.device or cached_v.device != v.device:
                raise DeviceError(
                    "KV cache device must match projected key/value tensors",
                    expected=k.device,
                    actual=f"key={cached_k.device}, value={cached_v.device}",
                )
            if cached_k.ndim != 4 or cached_v.ndim != 4:
                raise ShapeError(
                    "KV cache lookup must return rank-4 key/value tensors",
                    expected="rank=4",
                    actual=f"key_rank={cached_k.ndim}, value_rank={cached_v.ndim}",
                )
            if cached_k.shape[0] != k.shape[0] or cached_v.shape[0] != v.shape[0]:
                raise ShapeError(
                    "KV cache batch size must match the current request",
                    expected=f"batch={k.shape[0]}",
                    actual=f"key_batch={cached_k.shape[0]}, value_batch={cached_v.shape[0]}",
                )
            if cached_k.shape[1] != k.shape[1] or cached_v.shape[1] != v.shape[1]:
                raise ShapeError(
                    "KV cache head count must match the current request",
                    expected=f"heads={k.shape[1]}",
                    actual=f"key_heads={cached_k.shape[1]}, value_heads={cached_v.shape[1]}",
                )
            if cached_k.shape[-1] != k.shape[-1] or cached_v.shape[-1] != v.shape[-1]:
                raise ShapeError(
                    "KV cache head dimensions must match the current request",
                    expected=f"key_dim={k.shape[-1]}, value_dim={v.shape[-1]}",
                    actual=f"key_dim={cached_k.shape[-1]}, value_dim={cached_v.shape[-1]}",
                )
            # Cache storage dtype is configurable; normalize it at the
            # attention boundary so the query/key/value computation remains
            # dtype-consistent without rewriting the stored checkpoint.
            cached_k = cached_k.to(dtype=k.dtype)
            cached_v = cached_v.to(dtype=v.dtype)
            if cached_k.shape[-2] > 0:
                k_full = torch.cat([cached_k, k], dim=-2)
                v_full = torch.cat([cached_v, v], dim=-2)
            else:
                k_full, v_full = k, v
        else:
            k_full, v_full = k, v
        return k_full, v_full

    @staticmethod
    def commit_kv_cache(
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> None:
        """Append current K/V after a successful forward pass."""
        if kv_cache is not None:
            kv_cache.append(k, v)

    def resolve_mask(
        self,
        mask: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build a causal mask keyed by the resolved ``[T_q, T_k]`` pair.

        A user-supplied ``mask`` is returned as-is. When ``self.config.causal``
        is ``True`` and the caller did not supply a mask, build a
        ``[T_q, T_k_total]`` mask — the ``T_q`` query dim and the
        ``T_k_total = T_k_cached + current`` key dim are independent (the
        KV cache splits them). For training-style equal-length inputs,
        the mask is plain lower-triangular. For cached decoding
        (``T_q=1, T_k_total = num_cached + 1``), all cached keys are
        visible to the current query.
        """
        if mask is None and self.config.causal:
            t_q = q.shape[-2]
            t_k = kv.shape[-2] if kv is not None else t_q
            return self.causal_mask(t_q, t_k, q.device)
        return mask

    def run_vq_precompute(
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
        # The streaming-VQ path (StreamingVQBuffer) is researcher-driven;
        # the orchestrator always uses the batched backend.quantize here.
        # Callers that need the streaming path should instantiate and
        # drive StreamingVQBuffer directly (see streaming_vq docs).
        result = self.backend.quantize(
            k_full, v_full, self.codebook.parents, self.codebook.children
        )

        self.last_keys = k_full.detach()
        self.last_parent_assignments = result.parent_assignments.detach()

        if self.config.codebook.bcar_enabled:
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
                num_heads=H,
                num_parents=M0,
                children_per_parent=C,
                head_dim=D,
                device=q.device,
                dtype=q.dtype,
            )
            self.codebook.initialize_children_around_parents()
            self.codebook_parents = self.codebook.parents
            self.codebook_children = self.codebook.children
            result = self.backend.quantize(
                k_full, v_full, self.codebook.parents, self.codebook.children
            )
        return result

    def compute_routing(
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
        assert self.scheduler is not None
        raw_budget = self.scheduler.budget_for(importance)
        num_valid_per_bh = (result.parent_counts > 0).sum(dim=-1)
        if isinstance(raw_budget, torch.Tensor):
            budget_per_bh = torch.minimum(raw_budget, num_valid_per_bh.to(raw_budget.dtype))
            budget = int(budget_per_bh.min().item())
        else:
            budget = min(raw_budget, int(num_valid_per_bh.min().item()))
        if budget <= 0:
            return importance, budget, None, result
        decision = self.router.select(importance, budget)
        return importance, budget, decision, result

    def refine_and_output(
        self,
        state: OnlineSoftmaxState,
        parent_logits: torch.Tensor,
        parent_values: torch.Tensor,
        child_logits: torch.Tensor,
        result: QuantizationResult,
        decision: RoutingDecision,
        budget: int,
        C: int,
        D_v: int,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run refinement and extract the final attention output (spec §7.7)."""
        if self.config.refinement.passes > 1:
            refiner = MultiPassRefiner(
                passes=self.config.refinement.passes,
                decay=self.config.refinement.pass_decay,
            )
            current_state, residual_norms = refiner.refine(
                state=state,
                parent_logits=parent_logits,
                parent_value=(
                    parent_logits.softmax(dim=-1).unsqueeze(-1) * parent_values.unsqueeze(2)
                ),
                parent_aggregates=parent_values,
                child_aggregates=result.child_aggregates,
                children_per_parent=C,
                decision=decision,
                attention_probs=parent_logits.softmax(dim=-1),
                parent_counts=result.parent_counts,
                child_logits=child_logits,
                child_counts=result.child_counts,
                merge_strategy=self.config.merge.strategy,
                query=q,
                child_keys=self.codebook.children,
            )
            logger.debug(
                "multi-pass refinement: %d passes, residual norms=%s",
                len(residual_norms),
                [f"{r:.6f}" for r in residual_norms],
            )
        else:
            refinement = refine_step(
                state=state,
                parent_logits=parent_logits,
                parent_value=(
                    parent_logits.softmax(dim=-1).unsqueeze(-1) * parent_values.unsqueeze(2)
                ),
                parent_aggregates=parent_values,
                child_aggregates=result.child_aggregates,
                children_per_parent=C,
                decision=decision,
                attention_probs=parent_logits.softmax(dim=-1),
                parent_counts=result.parent_counts,
                child_logits=child_logits,
                child_counts=result.child_counts,
                merge_strategy=self.config.merge.strategy,
            )
            current_state = refinement.state

        attn_out = current_state.running_numerator[
            :, :, :, 0, :
        ] / current_state.running_denominator[:, :, :, 0:1].clamp_min(1e-12)

        if self.config.execution.mode == "research":
            logger.debug(
                "avq_pipeline: budget=%d selected=%s utilization=%.2f",
                budget,
                decision.selected_indices.shape[-1],
                (result.parent_counts > 0).float().mean().item(),
            )
        return attn_out


__all__ = ["AVQAttention"]
