"""Causal incremental VQ primitive (SPEC \u00a714).

Streams keys and values through a :class:`StreamingVQBuffer` one
token (or one chunk) at a time, maintaining running parent/child
assignments, counts, and aggregates at ``O(D)`` per new key. After
``T`` streaming updates the aggregate equals the batched paper
quantizer within ``O(1/\u221aT)`` variance under a stationary
distribution (Theorem 14.1).

This is the second algorithmic contribution of the project after
BCAR / Chapter 13.

Wire-up
-------

By default :class:`avqa.attention_module.AVQAttention` uses the
batched :class:`EuclideanHierarchicalQuantizer`. The streaming buffer
is researcher-driven: instantiate :class:`StreamingVQBuffer`
directly, call :meth:`StreamingVQBuffer.extend` once per token (or
per chunk), then read :meth:`StreamingVQBuffer.realize`. The
``ExecutionConfig.causal_incremental`` flag is reserved for a future
integration; ``AVQAttention`` does not currently route through this
module. See ``tests/unit/test_streaming_vq.py`` for the convergence
test.
"""
from __future__ import annotations



import torch

from avqa.exceptions import ShapeError
from avqa.quantizer import QuantizationResult


class StreamingVQBuffer:
    """Causal incremental VQ state (SPEC \u00a714.2).

    Maintains per-(b, h) parent/child assignments, counts, and
    aggregates for the cached key/value prefix. ``extend`` adds a
    single token at a time and runs in ``O(D)`` where ``D`` is the
    head dimension. ``realize`` snapshots the buffers into a
    :class:`QuantizationResult` for the standard attention pipeline.
    """

    def __init__(
        self,
        num_heads: int,
        num_parents: int,
        children_per_parent: int,
        head_dim: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_heads <= 0 or num_parents <= 0 or children_per_parent <= 0:
            msg = f"invalid shape: H={num_heads}, M_0={num_parents}, C={children_per_parent}"
            raise ValueError(msg)
        self.num_heads = num_heads
        self.num_parents = num_parents
        self.children_per_parent = children_per_parent
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        # Empty-state accumulators. Sizes are fixed at the codebook
        # dimensions; length-N dims grow lazily through ``extend``.
        self._parent_assignments: torch.Tensor | None = None
        self._child_assignments: torch.Tensor | None = None
        self._parent_counts: torch.Tensor = torch.zeros(
            1, num_heads, num_parents, dtype=torch.long, device=device
        )
        self._child_counts: torch.Tensor = torch.zeros(
            1,
            num_heads,
            num_parents,
            children_per_parent,
            dtype=torch.long,
            device=device,
        )
        self._parent_aggregates: torch.Tensor = torch.zeros(
            1, num_heads, num_parents, head_dim, dtype=dtype, device=device
        )
        self._child_aggregates: torch.Tensor = torch.zeros(
            1,
            num_heads,
            num_parents,
            children_per_parent,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def reset(self) -> None:
        """Drop all cached state."""
        self._parent_assignments = None
        self._child_assignments = None
        self._parent_counts.zero_()
        self._child_counts.zero_()
        self._parent_aggregates.zero_()
        self._child_aggregates.zero_()
        self._size = 0

    def extend(
        self,
        keys: torch.Tensor,
        parents: torch.Tensor,
        children: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add one (B, D) batch of new keys; return (parent, child) assignments.

        Args:
            keys: ``[B, D]`` new keys.
            parents: ``[H, M_0, D]`` parent codebook (per-head).
            children: ``[H, M_0, C, D]`` child codebook (per-head).

        Returns:
            (parent_assignments, child_assignments) — both
            ``[B, H]`` int64 tensors representing the new tokens' VQ
            targets.
        """
        if keys.dim() != 2 or keys.shape[1] != self.head_dim:
            raise ShapeError(
                "keys must be [B, D]",
                expected=f"[B, {self.head_dim}]",
                actual=tuple(keys.shape),
            )
        if parents.shape != (self.num_heads, self.num_parents, self.head_dim):
            raise ShapeError(
                "parents shape mismatch",
                expected=(
                    self.num_heads,
                    self.num_parents,
                    self.head_dim,
                ),
                actual=tuple(parents.shape),
            )
        if children.shape != (
            self.num_heads,
            self.num_parents,
            self.children_per_parent,
            self.head_dim,
        ):
            raise ShapeError(
                "children shape mismatch",
                expected=(
                    self.num_heads,
                    self.num_parents,
                    self.children_per_parent,
                    self.head_dim,
                ),
                actual=tuple(children.shape),
            )
        H, M0, D = parents.shape
        B = keys.shape[0]
        device, dtype = keys.device, keys.dtype
        keys = keys.to(dtype)

        # Stage 1: assign parent (Euclidean nearest neighbour).
        # ``keys`` [B, D], ``parents`` [H, M_0, D] -> distance [B, H, M_0]
        # via the squared-norm expansion ``||k||^2 - 2 k·p + ||p||^2``,
        # computed with ``torch.matmul`` (NOT ``einsum``: the latter
        # misinterprets the shared ``b`` label across the two operands
        # and silently sums over the whole B dimension, off by a factor
        # of B). The matmul form is bit-equivalent to ``torch.cdist``
        # under argmin.
        flat_parents = parents.reshape(-1, D)  # [H*M_0, D]
        k_sq = (keys * keys).sum(dim=1)  # [B]
        cross = torch.matmul(keys, flat_parents.T)  # [B, H*M_0]
        cross = cross.reshape(B, H, M0)
        p_sq = (parents * parents).sum(dim=-1)  # [H, M_0]
        dist_sq = k_sq[:, None, None] - 2.0 * cross + p_sq[None, :, :]
        parent_assn = dist_sq.argmin(dim=-1)  # [B, H]
        # Stage 2: assign child within the chosen parent. We index
        # ``children`` with shape ``[H, M_0, C, D]`` via a 2-D index
        # ``[B, H]`` that selects (h, parent_assn[b, h]) for each
        # token. Result has shape ``[B, H, C, D]``.
        h_idx = torch.arange(H, device=device).unsqueeze(0).expand(B, H)
        chosen_children = children[h_idx, parent_assn, :, :]  # type: ignore[index]  # fmt: skip
        # Broadcast ``keys [B, D]`` against ``chosen_children [B, H, C, D]``
        # by inserting singleton axes at positions (1, 2) only.
        diff = keys[:, None, None, :] - chosen_children  # type: ignore[index]
        child_assn = (diff * diff).sum(dim=-1).argmin(dim=-1)  # [B, H]
        child_assn = child_assn.to(torch.int64)

        # Update running aggregators in place. ``index_put_`` is the
        # right primitive: it broadcasts the indices across the
        # head dim and only writes the touched codewords.
        flat_parent = parent_assn.reshape(-1)  # [B*H]
        idx_p = torch.arange(H, device=device).repeat_interleave(B) * M0 + flat_parent
        keys_flat = keys.unsqueeze(1).expand(B, H, D).reshape(-1, D).to(dtype)
        self._parent_counts.view(-1).index_add_(
            0,
            idx_p,
            torch.ones_like(idx_p, dtype=torch.long),
        )
        self._parent_aggregates.view(-1, D).index_add_(0, idx_p, keys_flat)
        flat_pc = parent_assn.reshape(-1) * self.children_per_parent + child_assn.reshape(-1)
        idx_pc = (
            torch.arange(H, device=device).repeat_interleave(B) * (M0 * self.children_per_parent)
            + flat_pc
        )
        self._child_counts.view(-1).index_add_(
            0,
            idx_pc,
            torch.ones_like(idx_pc, dtype=torch.long),
        )
        self._child_aggregates.view(-1, D).index_add_(0, idx_pc, keys_flat)

        # Append the new assignments to the running record.
        if self._parent_assignments is None:
            self._parent_assignments = parent_assn.to(torch.int64).reshape(1, H, B)
            self._child_assignments = child_assn.to(torch.int64).reshape(1, H, B)
        else:
            self._parent_assignments = torch.cat(
                [self._parent_assignments, parent_assn.to(torch.int64).reshape(1, H, B)],
                dim=-1,
            )
            self._child_assignments = torch.cat(
                [self._child_assignments, child_assn.to(torch.int64).reshape(1, H, B)],
                dim=-1,
            )
        self._size += B
        return parent_assn, child_assn

    def realize(self) -> QuantizationResult:
        """Snapshot the buffers into a :class:`QuantizationResult`."""
        if self._parent_assignments is None:
            # No keys seen yet; emit zero-tensors that downstream code
            # can still call ``validate_shapes`` against.
            return _empty_realisation(
                self.num_heads,
                self.num_parents,
                self.children_per_parent,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
        # The buffer stores ``[1, H, N_total]`` (batch dim is
        # implicit; CI-VQ accepts one-batch streams as documented in
        # SPEC \u00a714.5). Transpose is a no-op since the spec shape is
        # ``[B, H, N]`` with ``B = 1``.
        return QuantizationResult(
            parent_assignments=self._parent_assignments,
            child_assignments=self._child_assignments,
            parent_aggregates=self._parent_aggregates,
            child_aggregates=self._child_aggregates,
            parent_counts=self._parent_counts.to(dtype=self.dtype),
            child_counts=self._child_counts.to(dtype=self.dtype),
        )


def _empty_realisation(
    num_heads: int,
    num_parents: int,
    children_per_parent: int,
    head_dim: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> QuantizationResult:
    """Build a zero-tensor QuantizationResult for an empty buffer."""
    return QuantizationResult(
        parent_assignments=torch.zeros(1, num_heads, 0, dtype=torch.int64, device=device),
        child_assignments=torch.zeros(1, num_heads, 0, dtype=torch.int64, device=device),
        parent_aggregates=torch.zeros(
            1, num_heads, num_parents, head_dim, dtype=dtype, device=device
        ),
        child_aggregates=torch.zeros(
            1,
            num_heads,
            num_parents,
            children_per_parent,
            head_dim,
            dtype=dtype,
            device=device,
        ),
        parent_counts=torch.zeros(1, num_heads, num_parents, dtype=dtype, device=device),
        child_counts=torch.zeros(
            1,
            num_heads,
            num_parents,
            children_per_parent,
            dtype=dtype,
            device=device,
        ),
    )


__all__ = ["StreamingVQBuffer"]
