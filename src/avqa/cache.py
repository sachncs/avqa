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

from avqa.exceptions import ConfigurationError, NotInitializedError, ShapeError
from avqa.logging import get_logger

logger = get_logger("cache")


def validate_cache_tensors(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    name: str = "cache entry",
) -> None:
    """Validate one cache append before it can mutate cache state."""
    if key.ndim != 4 or value.ndim != 4:
        raise ShapeError(
            f"{name} key/value must be rank 4 [B, H, T, D]",
            expected="rank=4",
            actual=f"key_rank={key.ndim}, value_rank={value.ndim}",
        )
    if key.shape[:3] != value.shape[:3]:
        raise ShapeError(
            f"{name} key/value batch, head, and token dimensions must match",
            expected=str(tuple(key.shape[:3])),
            actual=str(tuple(value.shape[:3])),
        )
    if key.shape[1] != num_heads:
        raise ShapeError(
            f"{name} head count mismatch",
            expected=num_heads,
            actual=int(key.shape[1]),
        )
    if key.shape[-1] != head_dim_k or value.shape[-1] != head_dim_v:
        raise ShapeError(
            f"{name} head dimension mismatch",
            expected=f"key={head_dim_k}, value={head_dim_v}",
            actual=f"key={key.shape[-1]}, value={value.shape[-1]}",
        )


@dataclass
class CacheEntry:
    """One cache slot (spec §3.13).

    Attributes:
        key: Cached key tensor.
        value: Cached value tensor.
        positions: Original sequence positions (for paged layouts).
            L7: populated on append in PagedKVCache; InMemoryKVCache
            always sets an empty tensor since positions are implicit
            in the contiguous key layout.
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
        if num_heads <= 0 or head_dim_k <= 0 or head_dim_v <= 0 or max_size < 0:
            raise ConfigurationError(
                "cache dimensions must be positive and max_size must be non-negative",
                {
                    "num_heads": num_heads,
                    "head_dim_k": head_dim_k,
                    "head_dim_v": head_dim_v,
                    "max_size": max_size,
                },
            )
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.max_size = max_size
        self.device = device
        self.dtype = dtype
        self.cache_key: torch.Tensor = torch.empty(
            0, num_heads, 0, head_dim_k, device=device, dtype=dtype
        )
        self.cache_value: torch.Tensor = torch.empty(
            0, num_heads, 0, head_dim_v, device=device, dtype=dtype
        )
        self.eviction_count: int = 0
        self.hit_count: int = 0
        self.miss_count: int = 0
        self.batch_size: int | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append new tokens to the cache.

        Args:
            key: ``[B, H, T_new, D_k]`` new keys.
            value: ``[B, H, T_new, D_v]`` new values.
        """
        validate_cache_tensors(
            key,
            value,
            num_heads=self.num_heads,
            head_dim_k=self.head_dim_k,
            head_dim_v=self.head_dim_v,
        )
        if self.batch_size is not None and key.shape[0] != self.batch_size:
            raise ShapeError(
                "cache batch size mismatch",
                expected=self.batch_size,
                actual=int(key.shape[0]),
            )
        if self.batch_size is None:
            self.batch_size = int(key.shape[0])
        if self.cache_key.numel() == 0:
            self.cache_key = key.to(device=self.device, dtype=self.dtype)
            self.cache_value = value.to(device=self.device, dtype=self.dtype)
        else:
            existing_key = self.cache_key
            self.cache_key = torch.cat([existing_key, key.to(existing_key.dtype)], dim=-2)
            existing_value = self.cache_value
            self.cache_value = torch.cat(
                [existing_value, value.to(existing_value.dtype)],
                dim=-2,
            )
        if self.max_size > 0 and self.size > self.max_size:
            # Drop the oldest tokens. .contiguous() to keep downstream
            # reshape/tracing paths from incurring hidden copies.
            excess = self.size - self.max_size
            self.cache_key = self.cache_key[..., excess:, :].contiguous()
            self.cache_value = self.cache_value[..., excess:, :].contiguous()
            self.eviction_count += 1
            logger.debug(
                "evicted %d tokens from KV cache (max_size=%d)",
                excess,
                self.max_size,
            )

    def lookup(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached (key, value); empty cache returns empty tensors."""
        if self.cache_key.numel() == 0 or self.cache_value.numel() == 0:
            self.miss_count += 1
            empty_k = torch.zeros(
                1,
                self.num_heads,
                0,
                self.head_dim_k,
                dtype=self.dtype,
                device=self.device,
            )
            empty_v = torch.zeros(
                1,
                self.num_heads,
                0,
                self.head_dim_v,
                dtype=self.dtype,
                device=self.device,
            )
            return empty_k, empty_v
        self.hit_count += 1
        return self.cache_key, self.cache_value

    def cache_stats(self) -> dict[str, int]:
        """Return eviction/hit/miss counters for observability."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "eviction_count": self.eviction_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
        }

    def reset(self) -> None:
        """Drop all cached entries."""
        self.cache_key = torch.empty(
            0, self.num_heads, 0, self.head_dim_k, device=self.device, dtype=self.dtype
        )
        self.cache_value = torch.empty(
            0, self.num_heads, 0, self.head_dim_v, device=self.device, dtype=self.dtype
        )
        self.batch_size = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize cache contents (metadata + tensors)."""
        return {
            "schema_version": torch.tensor(1, dtype=torch.int32),
            "num_heads": torch.tensor(self.num_heads, dtype=torch.int32),
            "head_dim_k": torch.tensor(self.head_dim_k, dtype=torch.int32),
            "head_dim_v": torch.tensor(self.head_dim_v, dtype=torch.int32),
            "size": torch.tensor(self.size, dtype=torch.int32),
            "batch_size": torch.tensor(self.batch_size or 0, dtype=torch.int32),
            "cache_key": self.cache_key.detach().clone(),
            "cache_value": self.cache_value.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore cache from :meth:`state_dict` output."""
        schema_version = state.get("schema_version")
        if schema_version is not None and int(schema_version) != 1:
            raise ShapeError(
                "unsupported cache schema_version",
                expected=1,
                actual=int(schema_version),
            )
        if "num_heads" in state and int(state["num_heads"]) != self.num_heads:
            raise ShapeError(
                "num_heads mismatch", expected=self.num_heads, actual=int(state["num_heads"])
            )
        if "head_dim_k" in state and int(state["head_dim_k"]) != self.head_dim_k:
            raise ShapeError(
                "head_dim_k mismatch",
                expected=self.head_dim_k,
                actual=int(state["head_dim_k"]),
            )
        if "head_dim_v" in state and int(state["head_dim_v"]) != self.head_dim_v:
            raise ShapeError(
                "head_dim_v mismatch",
                expected=self.head_dim_v,
                actual=int(state["head_dim_v"]),
            )
        cache_key = state.get("cache_key")
        cache_value = state.get("cache_value")
        if (cache_key is None) != (cache_value is None):
            raise ShapeError(
                "serialized cache must contain both cache_key and cache_value",
                expected="both tensors",
                actual="one tensor",
            )
        if cache_key is not None and cache_value is not None:
            validate_cache_tensors(
                cache_key,
                cache_value,
                num_heads=self.num_heads,
                head_dim_k=self.head_dim_k,
                head_dim_v=self.head_dim_v,
                name="serialized cache",
            )
            serialized_size = int(state.get("size", torch.tensor(cache_key.shape[-2])))
            serialized_batch = int(state.get("batch_size", torch.tensor(cache_key.shape[0])))
            if serialized_size != int(cache_key.shape[-2]):
                raise ShapeError(
                    "serialized cache size metadata mismatch",
                    expected=int(cache_key.shape[-2]),
                    actual=serialized_size,
                )
            expected_batch = int(cache_key.shape[0]) if cache_key.numel() > 0 else 0
            if serialized_batch != expected_batch:
                raise ShapeError(
                    "serialized cache batch metadata mismatch",
                    expected=expected_batch,
                    actual=serialized_batch,
                )
            if self.max_size > 0 and serialized_size > self.max_size:
                raise ShapeError(
                    "serialized cache exceeds max_size",
                    expected=f"<= {self.max_size}",
                    actual=serialized_size,
                )
            self.cache_key = cache_key.to(device=self.device, dtype=self.dtype)
            self.cache_value = cache_value.to(device=self.device, dtype=self.dtype)
            self.batch_size = serialized_batch or None
        else:
            self.reset()

    @property
    def size(self) -> int:
        """Number of cached tokens."""
        if self.cache_key.numel() == 0:
            return 0
        return int(self.cache_key.shape[-2])


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
        if page_size <= 0 or num_heads <= 0 or head_dim_k <= 0 or head_dim_v <= 0 or max_pages < 0:
            raise ConfigurationError(
                "page size and dimensions must be positive; max_pages must be non-negative",
                {
                    "page_size": page_size,
                    "num_heads": num_heads,
                    "head_dim_k": head_dim_k,
                    "head_dim_v": head_dim_v,
                    "max_pages": max_pages,
                },
            )
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.max_pages = max_pages
        self.device = device
        self.dtype = dtype
        self.pages: list[CacheEntry] = []

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append tokens; allocate a new page when the current one fills."""
        validate_cache_tensors(
            key,
            value,
            num_heads=self.num_heads,
            head_dim_k=self.head_dim_k,
            head_dim_v=self.head_dim_v,
        )
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        T = key.shape[-2]
        if self.pages and int(self.pages[-1].key.shape[0]) != int(key.shape[0]):
            raise ShapeError(
                "paged cache batch size mismatch",
                expected=int(self.pages[-1].key.shape[0]),
                actual=int(key.shape[0]),
            )
        if self.max_pages > 0:
            required_pages = (self.size + T + self.page_size - 1) // self.page_size
            if required_pages > self.max_pages:
                raise NotInitializedError(
                    f"paged KV cache is full ({self.max_pages} pages); "
                    f"append requires {required_pages} pages"
                )
        cursor = 0
        position_start = self.size
        while cursor < T:
            current_page = self.current_page(batch_size=int(key.shape[0]))
            free = self.page_size - current_page.key.shape[-2]
            take = min(free, T - cursor)
            k_chunk = key[..., cursor : cursor + take, :]
            v_chunk = value[..., cursor : cursor + take, :]
            current_page.key = torch.cat([current_page.key, k_chunk], dim=-2)
            current_page.value = torch.cat([current_page.value, v_chunk], dim=-2)
            current_page.positions = torch.cat(
                [
                    current_page.positions,
                    torch.arange(
                        position_start + cursor,
                        position_start + cursor + take,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ]
            )
            cursor += take
            if current_page.key.shape[-2] == self.page_size and cursor < T:
                self.allocate_page(batch_size=int(key.shape[0]))

    def current_page(self, batch_size: int = 1) -> CacheEntry:
        """Return the most recent page, allocating one if none exists."""
        if not self.pages or self.pages[-1].key.shape[-2] == self.page_size:
            self.allocate_page(batch_size=batch_size)
        return self.pages[-1]

    def allocate_page(self, batch_size: int = 1) -> None:
        """Allocate a new empty page."""
        if self.max_pages > 0 and len(self.pages) >= self.max_pages:
            raise NotInitializedError(f"paged KV cache is full ({self.max_pages} pages)")
        self.pages.append(
            CacheEntry(
                key=torch.zeros(
                    batch_size,
                    self.num_heads,
                    0,
                    self.head_dim_k,
                    dtype=self.dtype,
                    device=self.device,
                ),
                value=torch.zeros(
                    batch_size,
                    self.num_heads,
                    0,
                    self.head_dim_v,
                    dtype=self.dtype,
                    device=self.device,
                ),
                positions=torch.zeros(0, dtype=torch.long, device=self.device),
            )
        )

    def lookup(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all page contents into a single (key, value) pair."""
        if not self.pages:
            return (
                torch.zeros(
                    1, self.num_heads, 0, self.head_dim_k, dtype=self.dtype, device=self.device
                ),
                torch.zeros(
                    1, self.num_heads, 0, self.head_dim_v, dtype=self.dtype, device=self.device
                ),
            )
        keys = torch.cat([p.key for p in self.pages], dim=-2)
        values = torch.cat([p.value for p in self.pages], dim=-2)
        return keys, values

    def reset(self) -> None:
        """Drop all pages."""
        self.pages = []

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize page metadata and tensor contents for restart safety."""
        state: dict[str, torch.Tensor] = {
            "schema_version": torch.tensor(1, dtype=torch.int32),
            "page_size": torch.tensor(self.page_size, dtype=torch.int32),
            "num_pages": torch.tensor(len(self.pages), dtype=torch.int32),
            "size": torch.tensor(self.size, dtype=torch.int32),
        }
        for index, page in enumerate(self.pages):
            state[f"page_{index}_key"] = page.key.detach().clone()
            state[f"page_{index}_value"] = page.value.detach().clone()
            state[f"page_{index}_positions"] = page.positions.detach().clone()
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore page metadata and tensor contents from a checkpoint."""
        schema_version = state.get("schema_version")
        if schema_version is not None and int(schema_version) != 1:
            raise ShapeError(
                "unsupported cache schema_version",
                expected=1,
                actual=int(schema_version),
            )
        if "page_size" in state and int(state["page_size"]) != self.page_size:
            raise ShapeError(
                "page_size mismatch",
                expected=self.page_size,
                actual=int(state["page_size"]),
            )
        num_pages = int(state.get("num_pages", torch.tensor(0)))
        if num_pages < 0:
            raise ShapeError("num_pages must be non-negative", expected=">= 0", actual=num_pages)
        if self.max_pages > 0 and num_pages > self.max_pages:
            raise ShapeError(
                "serialized cache exceeds max_pages",
                expected=f"<= {self.max_pages}",
                actual=num_pages,
            )
        restored: list[CacheEntry] = []
        expected_position = 0
        expected_batch: int | None = None
        for index in range(num_pages):
            key = state.get(f"page_{index}_key")
            value = state.get(f"page_{index}_value")
            positions = state.get(f"page_{index}_positions")
            if key is None or value is None:
                raise ShapeError(
                    "paged cache checkpoint is missing page tensors",
                    expected=f"page_{index}_key/page_{index}_value",
                    actual="missing",
                )
            validate_cache_tensors(
                key,
                value,
                num_heads=self.num_heads,
                head_dim_k=self.head_dim_k,
                head_dim_v=self.head_dim_v,
                name=f"serialized page {index}",
            )
            if key.shape[-2] > self.page_size:
                raise ShapeError(
                    "serialized page exceeds page_size",
                    expected=f"<= {self.page_size}",
                    actual=int(key.shape[-2]),
                )
            if positions is None:
                positions = torch.arange(key.shape[-2], device=self.device, dtype=torch.long)
            if positions.ndim != 1 or positions.shape[0] != key.shape[-2]:
                raise ShapeError(
                    "serialized page positions must match token count",
                    expected=int(key.shape[-2]),
                    actual=tuple(positions.shape),
                )
            batch_size = int(key.shape[0])
            if expected_batch is None:
                expected_batch = batch_size
            elif batch_size != expected_batch:
                raise ShapeError(
                    "serialized pages must use one batch size",
                    expected=expected_batch,
                    actual=batch_size,
                )
            expected_positions = torch.arange(
                expected_position,
                expected_position + key.shape[-2],
                device=positions.device,
                dtype=torch.long,
            )
            if not torch.equal(positions.to(dtype=torch.long), expected_positions):
                raise ShapeError(
                    "serialized page positions must be contiguous",
                    expected=f"{expected_position}:{expected_position + key.shape[-2]}",
                    actual=positions.detach().cpu().tolist(),
                )
            restored.append(
                CacheEntry(
                    key=key.to(device=self.device, dtype=self.dtype),
                    value=value.to(device=self.device, dtype=self.dtype),
                    positions=positions.to(device=self.device, dtype=torch.long),
                )
            )
            expected_position += int(key.shape[-2])
        serialized_size = int(
            state.get("size", torch.tensor(sum(int(p.key.shape[-2]) for p in restored)))
        )
        restored_size = sum(int(page.key.shape[-2]) for page in restored)
        if serialized_size != restored_size:
            raise ShapeError(
                "serialized paged cache size metadata mismatch",
                expected=restored_size,
                actual=serialized_size,
            )
        self.pages = restored

    @property
    def size(self) -> int:
        """Number of cached tokens across all pages."""
        return sum(int(p.key.shape[-2]) for p in self.pages)

    @property
    def num_pages(self) -> int:
        """Number of allocated pages."""
        return len(self.pages)


__all__ = ["CacheEntry", "InMemoryKVCache", "KVCache", "PagedKVCache"]
