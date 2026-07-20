"""AVQAttention pipeline helpers (private).

The :class:`avqa.attention_module.AVQAttention` orchestrator delegates
the entire forward pass to :func:`run_pipeline`; the algebraic stages
(scaling, parent/child logits, online softmax, HVAQ) live as free
functions in this module. Splitting them out:

1. Keeps the orchestrator readable — one stage = one function call.
2. Lets tests exercise individual stages in isolation.
3. Removes a 700-line "god class" that knew how VQ, routing,
   refinement and HVAQ all work.

The orchestrator (``AVQAttention.forward_impl``) is a thin wrapper that
calls :func:`run_pipeline(self, q, k, v, mask, kv_cache)`.

Only lifecycle hooks that need module state (``apply_hopfield``) take
the :class:`AVQAttention` instance explicitly; everything else accepts
its parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from avqa.attention import OnlineSoftmaxState
from avqa.routing import compute_importance

if TYPE_CHECKING:
    from avqa.attention_module import AVQAttention
    from avqa.cache import KVCache
    from avqa.quantizer import QuantizationResult

EPS: float = 1e-12
"""Default epsilon for safe softmax denominators."""


def attention_scale(head_dim: int) -> float:
    """Return the attention scale ``1 / sqrt(d)``."""
    return head_dim**-0.5


def parent_logits(
    q: torch.Tensor,
    codebook_parents: torch.Tensor,
    valid_mask: torch.Tensor,
    mask: torch.Tensor | None,
    head_dim: int,
) -> torch.Tensor:
    """Compute parent attention logits with mask + validity applied.

    Args:
        q: ``[B, H, T, D]`` queries.
        codebook_parents: ``[H, M_0, D]``.
        valid_mask: ``[H, M_0]`` (broadcast to queries).
        mask: Optional ``[T_q, T_k]`` boolean mask (True = attend).
            Rank-2 only; the caller's contract is 2-D.
        head_dim: ``D``.

    Returns:
        ``[B, H, T, M_0]`` logits ready for softmax.

    Raises:
        ValueError: If ``mask`` is not rank-2 when provided.
    """
    B, H, T_q, D = q.shape
    M_0 = codebook_parents.shape[-2]
    parent_keys = codebook_parents.unsqueeze(0).expand(B, H, M_0, D)
    logits = torch.matmul(q, parent_keys.transpose(-2, -1)) / math.sqrt(head_dim)
    if mask is not None:
        if mask.ndim != 2:
            msg = f"mask must be rank 2 [T_q, T_k], got rank {mask.ndim}"
            raise ValueError(msg)
        codeword_mask = mask.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(
            ~codeword_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    logits = logits.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))
    return logits


def online_softmax(
    parent_logits: torch.Tensor,
    parent_values: torch.Tensor,
    parent_counts: torch.Tensor,
    head_dim: int,
    head_dim_v: int,
) -> tuple[OnlineSoftmaxState, torch.Tensor]:
    """Build a running online-softmax state and per-(B,H,N) parent probabilities.

    Args:
        parent_logits: ``[B, H, T, M_0]`` from :func:`parent_logits`.
        parent_values: ``[B, H, M_0, D_v]``.
        parent_counts: ``[B, H, M_0]``; zero entries contribute 0.
        head_dim: ``D`` (kept for symmetry with the orchestrator's arg list).
        head_dim_v: ``D_v``.

    Returns:
        Tuple of (state, parent_attention_probs).
    """
    del head_dim  # accepted for stable call-site; unused mathematically
    B, H, T_q, _ = parent_logits.shape
    D_v = head_dim_v
    tile_max = parent_logits.amax(dim=-1, keepdim=True)
    tile_exp = torch.exp(parent_logits - tile_max)
    tile_denom = (tile_exp * parent_counts.unsqueeze(2)).sum(dim=-1, keepdim=True)
    tile_num = torch.einsum("bhta,bhad->bhtd", tile_exp, parent_values).unsqueeze(-2)
    state = OnlineSoftmaxState.empty(B, H, T_q, head_dim_v, D_v)
    state = state.merge(tile_max, tile_denom, tile_num)
    probs = tile_exp / tile_exp.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return state, probs


def child_logits(
    q: torch.Tensor,
    codebook_children: torch.Tensor,
    selected_indices: torch.Tensor,
    child_counts: torch.Tensor,
    head_dim: int,
    children_per_parent: int,
) -> torch.Tensor:
    """Compute child attention logits for the selected parents.

    Args:
        q: ``[B, H, T, D]`` queries.
        codebook_children: ``[H, M_0, C, D]`` full child codebook.
        selected_indices: ``[B, H, P]`` indices into the parent axis.
        child_counts: ``[B, H, M_0, C]`` from the quantizer.
        head_dim: ``D``.
        children_per_parent: ``C``.

    Returns:
        ``[B, H, T, P, C]`` logits.
    """
    B, H, T_q, D = q.shape
    P = selected_indices.shape[-1]
    C = children_per_parent
    M_0 = codebook_children.shape[1]
    expanded_children = codebook_children.unsqueeze(0).expand(B, H, M_0, C, D)
    parent_idx = (
        selected_indices.unsqueeze(-1).unsqueeze(-1).expand(B, H, P, C, D)
    )
    selected_keys = torch.gather(expanded_children, 2, parent_idx)
    logits = torch.einsum("bhtd,bhpcd->bhtpc", q, selected_keys) / math.sqrt(head_dim)
    selected_counts = (
        torch.gather(child_counts, 2, selected_indices.unsqueeze(-1).expand(B, H, P, C))
        > 0
    )
    return logits.masked_fill(~selected_counts.unsqueeze(2), float("-inf"))


def apply_hopfield(
    state: AVQAttention,
    parent_logits: torch.Tensor,
    valid: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Apply the HVAQ per-query temperature schedule (SPEC §16).

    Returns ``parent_logits`` unchanged if HVAQ is disabled.
    """
    if not (state.config.backend.hopfield and state.config.hopfield.adaptive != "none"):
        return parent_logits

    from avqa.hopfield import hopfield_logits, paper_beta, per_query_beta

    beta_init = state.config.hopfield.beta_init
    if beta_init <= 0.0:
        beta_init = paper_beta(head_dim)

    base = parent_logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    probs = base.softmax(dim=-1) * valid.unsqueeze(2).to(base.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)

    alpha_param = getattr(state, "_alpha", state.config.hopfield.alpha)
    alpha_source = (
        alpha_param.view(1, -1, 1)
        if isinstance(alpha_param, torch.Tensor)
        else alpha_param
    )
    beta_q = per_query_beta(
        probs,
        beta_init=beta_init,
        adaptive=state.config.hopfield.adaptive,
        alpha=alpha_source,
    )

    raw_logits = parent_logits * math.sqrt(head_dim)
    parent_beta = getattr(state, "_parent_beta", 1.0)
    return hopfield_logits(raw_logits, beta_q, parent_beta=parent_beta)


def naive_fallback(
    state: AVQAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run naive attention + output projection + dropout."""
    attn_out = state.backend.naive_attention(q, k, v, mask=mask)
    out = state.merge_heads(attn_out)
    out = state.out_proj(out)
    out = state.dropout(out)
    return out


def run_pipeline(
    state: AVQAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    kv_cache: KVCache | None,
) -> torch.Tensor:
    """Compose the 12-stage AVQ forward pass.

    Args:
        state: The :class:`AVQAttention` module (parameter & projection
            source).
        query: ``[B, T_q, E]``.
        key: ``[B, T_k, E]``.
        value: ``[B, T_k, E]``.
        mask: Optional boolean mask.
        kv_cache: Optional KV-cache to extend.

    Returns:
        ``[B, T_q, E]`` attention output.
    """
    state._validate_inputs(query, key, value)

    q_proj, k_proj, v_proj = state.maybe_project(query, key, value)
    H = state.config.attention.num_heads
    q = state.split_heads(q_proj, H)
    k = state.split_heads(k_proj, H)
    v = state.split_heads(v_proj, H)

    state._sync_codebook_device(q)
    k_full, v_full = state._resolve_kv_cache(k, v, kv_cache)
    mask = state._resolve_mask(mask, q)

    use_naive = state.scheduler is None or state.config.execution.mode == "reference"
    if use_naive:
        return naive_fallback(state, q, k_full, v_full, mask)

    B, _, T_q, D = q.shape
    D_v = v_full.shape[-1]
    M0 = state.config.codebook.num_codewords
    C = state.config.codebook.children_per_codeword

    # Stage 5: VQ precompute + BCAR (kept on the module because it mutates state).
    result = state._run_vq_precompute(k_full, v_full, q, H, M0, C, D, kv_cache)

    # Stage 6: parent logits with mask + codeword validity.
    valid = result.parent_counts > 0
    parent_logits_t = parent_logits(q, state.codebook.parents, valid, mask, D)

    # Stage 7: HVAQ schedule — needs module state for `_parent_beta`/`_alpha`.
    parent_logits_t = apply_hopfield(state, parent_logits_t, valid, D)

    # Stage 8: online softmax + parent attention probabilities.
    softmax_state, parent_attention_probs = online_softmax(
        parent_logits_t, result.parent_aggregates, result.parent_counts, D, D_v,
    )

    # Stage 9: routing — compute importance, ask the scheduler for the
    # budget, ask the router for the selected parent indices.
    importance = compute_importance(parent_attention_probs, result.parent_counts)
    budget = state.scheduler.budget_for(importance)
    num_valid_per_bh = (result.parent_counts > 0).sum(dim=-1)
    min_valid = int(num_valid_per_bh.min().item())
    budget = min(budget, min_valid)
    if budget <= 0:
        return naive_fallback(state, q, k, v, mask)
    decision = state.router.select(importance, budget)
    assert decision is not None

    # Stage 10: child logits over the selected parents.
    child_logits_t = child_logits(
        q,
        state.codebook.children,
        decision.selected_indices,
        result.child_counts,
        D,
        C,
    )

    # Stage 11: refinement + final reduction.
    attn_out = state._refine_and_output(
        softmax_state,
        parent_attention_probs,
        result.parent_aggregates,
        child_logits_t,
        result,
        decision,
        budget,
        C,
        D_v,
        q=q,
    )

    # Stage 12: output projection + dropout.
    out = state.merge_heads(attn_out)
    out = state.out_proj(out)
    out = state.dropout(out)
    return out


__all__ = [
    "EPS",
    "apply_hopfield",
    "attention_scale",
    "child_logits",
    "naive_fallback",
    "online_softmax",
    "parent_logits",
    "run_pipeline",
] 
