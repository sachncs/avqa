"""Tests for avqa.cache module."""

from __future__ import annotations

import pytest
import torch

from avqa.cache import InMemoryKVCache, KVCache, PagedKVCache
from avqa.exceptions import (
    AVQAError,
    ConfigurationError,
    NotInitializedError,
    ShapeError,
)


class TestInMemoryKVCache:
    """Tests for the in-memory KV cache (spec §3.13)."""

    def test_empty_cache_size_zero(self) -> None:
        """Empty cache has size 0."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        assert cache.size == 0

    def test_append_grows(self) -> None:
        """Append increases cache size."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        cache.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
        assert cache.size == 3
        cache.append(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))
        assert cache.size == 5

    def test_lookup_returns_concatenated(self) -> None:
        """lookup() concatenates appended chunks."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        k1 = torch.randn(1, 2, 3, 8)
        v1 = torch.randn(1, 2, 3, 8)
        k2 = torch.randn(1, 2, 2, 8)
        v2 = torch.randn(1, 2, 2, 8)
        cache.append(k1, v1)
        cache.append(k2, v2)
        k_full, v_full = cache.lookup()
        assert k_full.shape == (1, 2, 5, 8)
        assert v_full.shape == (1, 2, 5, 8)
        assert torch.equal(k_full[..., :3, :], k1)
        assert torch.equal(k_full[..., 3:, :], k2)

    def test_max_size_evicts_oldest(self) -> None:
        """max_size evicts the oldest tokens when exceeded."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4, max_size=4)
        cache.append(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
        cache.append(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
        assert cache.size == 4  # evicted oldest 2

    def test_reset(self) -> None:
        """reset() drops the cache."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
        cache.reset()
        assert cache.size == 0

    def test_state_dict_round_trip(self) -> None:
        """state_dict contains the documented fields."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        cache.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
        state = cache.state_dict()
        assert state["num_heads"].item() == 2
        assert state["size"].item() == 3
        assert "schema_version" in state

    def test_load_state_dict_rejects_mismatch(self) -> None:
        """load_state_dict rejects mismatched num_heads."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        with pytest.raises(ShapeError, match="num_heads"):
            cache.load_state_dict({"num_heads": torch.tensor(4, dtype=torch.int32)})

    def test_load_state_dict_mismatch_is_avqaerror(self) -> None:
        """All cache raises are AVQAError subclasses."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        with pytest.raises(AVQAError, match="num_heads"):
            cache.load_state_dict({"num_heads": torch.tensor(4, dtype=torch.int32)})


class TestPagedKVCache:
    """Tests for the paged KV cache (spec §3.15)."""

    def test_empty_size_zero(self) -> None:
        """Empty paged cache has size 0."""
        cache = PagedKVCache(page_size=4, num_heads=2, head_dim_k=8, head_dim_v=8)
        assert cache.size == 0
        assert cache.num_pages == 0

    def test_allocates_pages_on_append(self) -> None:
        """Appending beyond page_size allocates a new page."""
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4))
        assert cache.num_pages == 2
        assert cache.size == 5

    def test_page_boundary(self) -> None:
        """Exactly page_size tokens fit in one page."""
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        assert cache.num_pages == 1
        assert cache.size == 4

    def test_lookup_concatenates_pages(self) -> None:
        """lookup() concatenates all pages."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        k = torch.randn(1, 1, 5, 4)
        v = torch.randn(1, 1, 5, 4)
        cache.append(k, v)
        k_full, v_full = cache.lookup()
        assert k_full.shape == (1, 1, 5, 4)
        assert v_full.shape == (1, 1, 5, 4)
        assert torch.equal(k_full, k)
        assert torch.equal(v_full, v)

    def test_max_pages_limit(self) -> None:
        """max_pages raises NotInitializedError when exceeded."""
        cache = PagedKVCache(page_size=1, num_heads=1, head_dim_k=4, head_dim_v=4, max_pages=2)
        cache.append(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
        with pytest.raises(NotInitializedError, match="full"):
            cache.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))

    def test_reset(self) -> None:
        """reset() drops all pages."""
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4))
        cache.reset()
        assert cache.size == 0
        assert cache.num_pages == 0

    def test_invalid_page_size(self) -> None:
        """page_size <= 0 raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            PagedKVCache(page_size=0)


class TestKVCacheInterface:
    """Tests for the KVCache abstract base."""

    def test_cannot_instantiate(self) -> None:
        """KVCache cannot be instantiated directly."""
        with pytest.raises(TypeError):
            KVCache.__new__(KVCache)

    def test_subclass_relationship(self) -> None:
        """Concrete caches inherit from KVCache."""
        assert issubclass(InMemoryKVCache, KVCache)
        assert issubclass(PagedKVCache, KVCache)
