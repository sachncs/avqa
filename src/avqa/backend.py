"""Backend abstraction and PyTorch reference backend (spec §3.12, §5.9).

The backend is the only seam between AVQA's algorithm and the underlying
hardware strategy. ``Backend.create()`` is the canonical factory; a
third party subclasses :class:`Backend` and overrides the two required
methods (:meth:`quantize` and :meth:`naive_attention`) to extend the
package.

The :class:`TorchBackend` ships a *broad* implementation: it provides
the two abstract methods plus a tiled online-softmax path, the merge
helper, the online-softmax tile-correction helper, and a final
reduction. These five non-abstract methods are useful for tests and
ad-hoc callers, but production code paths do not invoke them through
the :class:`Backend` interface — only ``quantize`` and ``naive_attention``
are reached by :class:`AVQAttention`.

ponytail: collapsed the planned backend package (8 sub-modules) into
one ``src/avqa/backend.py``. The class-based factory
``Backend.create("torch")`` returns ``TorchBackend()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from avqa.codebook import HierarchicalCodebook
from avqa.exceptions import BackendError
from avqa.logging import get_logger
from avqa.merge import MergeInputs, ProbabilityMerge
from avqa.quantizer import EuclideanHierarchicalQuantizer, QuantizationResult
from avqa.utils.numerics import online_softmax_step

logger = get_logger("backend")


# Numerical constants reused by both TorchBackend and free-function helpers.
EPS: float = 1e-12
"""Default epsilon for safe softmax denominators."""


def scale_for(head_dim: int) -> float:
    """Attention scale ``1 / sqrt(d)`` for the given head dimension."""
    return float(head_dim**-0.5)


class Backend(ABC):
    """Abstract execution backend.

    A backend only needs to implement two methods:

    - :meth:`quantize` — fused VQ precompute over a single streaming pass.
    - :meth:`naive_attention` — full ``O(N^2)`` reference attention used as
      a fallback when the AVQ pipeline decides to skip refinement.

    Third-party subclasses MAY add more methods; the orchestrator only
    calls these two. See :class:`TorchBackend` for the full set of
    supported methods on the reference implementation.
    """

    name: str = "abstract"

    @classmethod
    def create(cls, name: str = "torch") -> Backend:
        """Factory: resolve ``name`` to a concrete backend.

        Args:
            name: ``"torch"`` (default).

        Returns:
            A :class:`Backend` instance ready for :meth:`quantize` and
            :meth:`naive_attention` calls.

        Raises:
            BackendError: If ``name`` is unknown or unavailable in this
                environment.
        """
        if name == "torch":
            return TorchBackend()
        msg = f"backend '{name}' is not a known backend"
        raise BackendError(msg)

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
    def naive_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """O(N^2) reference attention (spec §10.15)."""


def online_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int = 64,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tiled online-softmax attention (FlashAttention-2 in pure PyTorch, spec §7.14).

    Module-level helper. The :class:`TorchBackend` keeps a thin method
    shim that delegates here so existing callers continue to work;
    orchestrator code does not depend on it.
    """
    B, H, T, D_k = query.shape
    _, _, N, D_v = value.shape
    scale = scale_for(D_k)

    m = torch.full((B, H, T), float("-inf"), device=query.device, dtype=query.dtype)
    denom = torch.zeros((B, H, T), device=query.device, dtype=query.dtype)
    num = torch.zeros((B, H, T, D_v), device=query.device, dtype=query.dtype)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        k_tile = key[:, :, start:end, :]
        v_tile = value[:, :, start:end, :]
        tile_logits = torch.matmul(query, k_tile.transpose(-2, -1)) * scale
        if mask is not None:
            tile_logits = tile_logits.masked_fill(
                mask[..., start:end] == 0,
                float("-inf"),
            )
        tile_max_raw = tile_logits.amax(dim=-1)
        tile_max = torch.maximum(m, tile_max_raw)
        # NaN-safe exp: when both m and tile_max are -inf (fully-masked tile
        # on the first iteration), exp(-inf - -inf) = NaN. Clamp the
        # subtraction result to 0 before exp so fully-masked tiles do not
        # poison the running state. Same for tile_logits - tile_max when
        # tile_logits is fully -inf (the resulting beta is 0 anyway).
        diff = m - tile_max
        diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
        alpha = torch.exp(diff)
        beta_in = tile_logits - tile_max.unsqueeze(-1)
        beta_in = torch.where(torch.isnan(beta_in), torch.full_like(beta_in, float("-inf")), beta_in)
        beta = torch.exp(beta_in)
        tile_denom = beta.sum(dim=-1)
        new_denom = alpha * denom + tile_denom
        tile_num = beta @ v_tile
        num = alpha.unsqueeze(-1) * num + tile_num
        m = tile_max
        denom = new_denom

    return num / denom.clamp_min(EPS).unsqueeze(-1)


def correction(
    state_max: torch.Tensor,
    state_denom: torch.Tensor,
    state_num: torch.Tensor,
    tile_max: torch.Tensor,
    tile_denom: torch.Tensor,
    tile_num: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax tile merge (spec §7.14, §9.11).

    Module-level helper; thin re-export of :func:`avqa.utils.numerics.online_softmax_step`.
    """
    return online_softmax_step(
        state_max,
        state_denom,
        state_num,
        tile_max,
        tile_denom,
        tile_num,
    )


class TorchBackend(Backend):
    """Pure PyTorch reference backend (spec §3.2.6, §10.15).

    In addition to the two abstract methods, this class exposes
    four concrete helpers used by tests and ad-hoc callers.
    """

    name = "torch"

    def naive_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """O(N^2) reference attention (spec §10.15)."""
        scale = scale_for(query.shape[-1])
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
        """Method shim that delegates to the module-level helper."""
        return online_softmax_attention(query, key, value, block_size=block_size, mask=mask)

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        codebook_parents: torch.Tensor,
        codebook_children: torch.Tensor,
    ) -> QuantizationResult:
        """Hierarchical VQ precompute using :class:`EuclideanHierarchicalQuantizer`."""
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
        """Apply :class:`ProbabilityMerge` (the spec default)."""
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
        """Method shim that delegates to the module-level helper."""
        return correction(state_max, state_denom, state_num, tile_max, tile_denom, tile_num)

    def reduction(
        self,
        state_num: torch.Tensor,
        state_denom: torch.Tensor,
    ) -> torch.Tensor:
        """Final output: ``num / clamp_min(denom)`` (with epsilon for empty states)."""
        return state_num / state_denom.clamp_min(EPS).unsqueeze(-1)


__all__ = [
    "EPS",
    "Backend",
    "BackendError",
    "EuclideanHierarchicalQuantizer",
    "HierarchicalCodebook",
    "TorchBackend",
    "correction",
    "online_softmax_attention",
    "scale_for",
]
