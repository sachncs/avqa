"""KV cache implementations for AVQA (spec §3.13, §3.15).

Spec §3.13 requires the cache to support incremental updates, efficient
lookup, configurable storage, cache reset, and serialization. Spec
§3.15 mentions paged attention as required for vLLM integration.

ponytail: collapsed the planned cache package (4 sub-modules) into one
src/avqa/cache.py. The in-memory cache is the reference; paged is the
vLLM-compatible layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class CacheEntry:
    """One cache slot (spec §3.13).

    Attributes:
        key: Cached key tensor.
        value: Cached value tensor.
        positions: Original sequence positions (for paged layouts).
    """

    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor


class KVCache(ABC):
    """Abstract KV cache (spec §3.13)."""

    @abstractmethod
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append new key/value tokens to the cache (spec §3.13)."""

    @abstractmethod
    def lookup(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full cached (key, value) pair (spec §3.13)."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the cache (spec §3.13)."""

    @abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the cache (spec §3.13)."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the cache from :meth:`state_dict`."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of cached tokens (spec §3.13)."""


class InMemoryKVCache(KVCache):
    """Contiguous in-memory cache (spec §3.13 reference implementation).

    Stacks keys/values along the sequence dimension. Suitable for
    single-stream inference and unit tests.

    Args:
        num_heads: Number of attention heads.
        head_dim_k: Key head dimension.
        head_dim_v: Value head dimension.
        max_size: Maximum cache size (``0`` means unbounded).
        device: Device for the cache tensors.
        dtype: Dtype for the cache tensors.
    """

    def __init__(
        self,
        num_heads: int = 1,
        head_dim_k: int = 64,
        head_dim_v: int = 64,
        max_size: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.max_size = max_size
        self.device = device
        self.dtype = dtype
        self._key: torch.Tensor | None = None
        self._value: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append new tokens to the cache.

        Args:
            key: ``[B, H, T_new, D_k]`` new keys.
            value: ``[B, H, T_new, D_v]`` new values.
        """
        if tuple(key.shape) != (*value.shape[:-1], self.head_dim_k):
            raise ValueError(
                f"key/value shape mismatch: key={tuple(key.shape)}, value={tuple(value.shape)}",
            )
        if self._key is None:
            self._key = key.to(device=self.device, dtype=self.dtype)
            self._value = value.to(device=self.device, dtype=self.dtype)
        else:
            existing_key = self._key
            assert existing_key is not None
            self._key = torch.cat([existing_key, key.to(existing_key.dtype)], dim=-2)
            existing_value = self._value
            assert existing_value is not None
            self._value = torch.cat(
                [existing_value, value.to(existing_value.dtype)], dim=-2,
            )
        if self.max_size > 0 and self.size > self.max_size:
            # Drop the oldest tokens.
            excess = self.size - self.max_size
            assert self._key is not None and self._value is not None
            self._key = self._key[..., excess:, :]
            self._value = self._value[..., excess:, :]

    def lookup(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached (key, value); empty cache returns empty tensors."""
        if self._key is None or self._value is None:
            empty_k = torch.zeros(
                1, self.num_heads, 0, self.head_dim_k, dtype=self.dtype, device=self.device,
            )
            empty_v = torch.zeros(
                1, self.num_heads, 0, self.head_dim_v, dtype=self.dtype, device=self.device,
            )
            return empty_k, empty_v
        return self._key, self._value

    def reset(self) -> None:
        """Drop all cached entries."""
        self._key = None
        self._value = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize empty cache as zeros (full state too large for default)."""
        return {
            "schema_version": torch.tensor(1, dtype=torch.int32),
            "num_heads": torch.tensor(self.num_heads, dtype=torch.int32),
            "head_dim_k": torch.tensor(self.head_dim_k, dtype=torch.int32),
            "head_dim_v": torch.tensor(self.head_dim_v, dtype=torch.int32),
            "size": torch.tensor(self.size, dtype=torch.int32),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore metadata from a state dict (data is not persisted here)."""
        if "num_heads" in state and int(state["num_heads"]) != self.num_heads:
            raise ValueError("num_heads mismatch")
        self.reset()

    @property
    def size(self) -> int:
        """Number of cached tokens."""
        if self._key is None:
            return 0
        return int(self._key.shape[-2])


class PagedKVCache(KVCache):
    """Paged KV cache for vLLM-style inference (spec §3.15).

    Keys and values are stored in fixed-size pages indexed by a
    block-table. This matches vLLM's paged-attention layout and enables
    non-contiguous memory allocation.

    Args:
        page_size: Tokens per page (default 16).
        num_heads: Number of attention heads.
        head_dim_k: Key head dimension.
        head_dim_v: Value head dimension.
        max_pages: Maximum number of pages (``0`` = unbounded).
    """

    def __init__(
        self,
        page_size: int = 16,
        num_heads: int = 1,
        head_dim_k: int = 64,
        head_dim_v: int = 64,
        max_pages: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.max_pages = max_pages
        self.device = device
        self.dtype = dtype
        self._pages: list[CacheEntry] = []

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append tokens; allocate a new page when the current one fills."""
        T = key.shape[-2]
        cursor = 0
        while cursor < T:
            current_page = self._current_page()
            free = self.page_size - current_page.key.shape[-2]
            take = min(free, T - cursor)
            k_chunk = key[..., cursor : cursor + take, :]
            v_chunk = value[..., cursor : cursor + take, :]
            current_page.key = torch.cat([current_page.key, k_chunk], dim=-2)
            current_page.value = torch.cat([current_page.value, v_chunk], dim=-2)
            cursor += take
            if current_page.key.shape[-2] == self.page_size and cursor < T:
                self._allocate_page()

    def _current_page(self) -> CacheEntry:
        """Return the most recent page, allocating one if none exists."""
        if not self._pages or self._pages[-1].key.shape[-2] == self.page_size:
            self._allocate_page()
        return self._pages[-1]

    def _allocate_page(self) -> None:
        """Allocate a new empty page."""
        if self.max_pages > 0 and len(self._pages) >= self.max_pages:
            raise RuntimeError("paged KV cache is full")
        self._pages.append(
            CacheEntry(
                key=torch.zeros(1, self.num_heads, 0, self.head_dim_k, dtype=self.dtype, device=self.device),
                value=torch.zeros(1, self.num_heads, 0, self.head_dim_v, dtype=self.dtype, device=self.device),
                positions=torch.zeros(0, dtype=torch.long, device=self.device),
            )
        )

    def lookup(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all page contents into a single (key, value) pair."""
        if not self._pages:
            return (
                torch.zeros(1, self.num_heads, 0, self.head_dim_k, dtype=self.dtype, device=self.device),
                torch.zeros(1, self.num_heads, 0, self.head_dim_v, dtype=self.dtype, device=self.device),
            )
        keys = torch.cat([p.key for p in self._pages], dim=-2)
        values = torch.cat([p.value for p in self._pages], dim=-2)
        return keys, values

    def reset(self) -> None:
        """Drop all pages."""
        self._pages = []

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize page table only (data lives on GPU in production)."""
        return {
            "schema_version": torch.tensor(1, dtype=torch.int32),
            "page_size": torch.tensor(self.page_size, dtype=torch.int32),
            "num_pages": torch.tensor(len(self._pages), dtype=torch.int32),
            "size": torch.tensor(self.size, dtype=torch.int32),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore metadata from a state dict (data not persisted)."""
        if "page_size" in state and int(state["page_size"]) != self.page_size:
            raise ValueError("page_size mismatch")
        self.reset()

    @property
    def size(self) -> int:
        """Number of cached tokens across all pages."""
        return sum(int(p.key.shape[-2]) for p in self._pages)

    @property
    def num_pages(self) -> int:
        """Number of allocated pages."""
        return len(self._pages)


__all__ = ["CacheEntry", "InMemoryKVCache", "KVCache", "PagedKVCache"]

