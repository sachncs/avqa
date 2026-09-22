"""Tests for avqa.cache module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from avqa.cache import InMemoryKVCache, KVCache, PagedKVCache
from avqa.exceptions import (
    AVQAError,
    ConfigurationError,
    NotInitializedError,
    ShapeError,
)
from avqa.utils.validation import NonFiniteTensorError


class TestInMemoryKVCache:
    """Tests for the in-memory KV cache (spec §3.13)."""

    def test_empty_cache_size_zero(self) -> None:
        """Empty cache has size 0."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        assert cache.size == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_heads": 0},
            {"head_dim_k": 0},
            {"head_dim_v": 0},
            {"max_size": -1},
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, int]) -> None:
        """Invalid cache capacities fail at construction time."""
        with pytest.raises(ConfigurationError):
            InMemoryKVCache(**kwargs)

    def test_append_grows(self) -> None:
        """Append increases cache size."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        cache.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
        assert cache.size == 3
        cache.append(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))
        assert cache.size == 5

    def test_append_rejects_non_finite_values(self) -> None:
        """NaN/Inf values cannot enter persistent cache state."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        key = torch.zeros(1, 1, 1, 4)
        value = torch.zeros(1, 1, 1, 4)
        value[0, 0, 0, 0] = float("nan")
        with pytest.raises(NonFiniteTensorError, match="non-finite"):
            cache.append(key, value)
        assert cache.size == 0

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

    @pytest.mark.parametrize("field", ["head_dim_k", "head_dim_v"])
    def test_load_state_dict_rejects_dimension_mismatch(self, field: str) -> None:
        """Checkpoint dimensions are part of the cache contract."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=6)
        with pytest.raises(ShapeError, match=field):
            cache.load_state_dict({field: torch.tensor(99, dtype=torch.int32)})

    def test_load_state_dict_restores_tensor_contents(self) -> None:
        """Contiguous cache contents survive a checkpoint round trip."""
        source = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=6)
        key = torch.randn(2, 1, 3, 4)
        value = torch.randn(2, 1, 3, 6)
        source.append(key, value)
        restored = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=6)
        restored.load_state_dict(source.state_dict())
        actual_key, actual_value = restored.lookup()
        assert torch.equal(actual_key, key)
        assert torch.equal(actual_value, value)

    def test_load_state_dict_mismatch_is_avqaerror(self) -> None:
        """All cache raises are AVQAError subclasses."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        with pytest.raises(AVQAError, match="num_heads"):
            cache.load_state_dict({"num_heads": torch.tensor(4, dtype=torch.int32)})

    def test_load_state_dict_rejects_partial_payload(self) -> None:
        """A checkpoint cannot silently reset when one tensor is missing."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        with pytest.raises(ShapeError, match="both cache_key"):
            cache.load_state_dict({"cache_key": torch.empty(0, 1, 0, 4)})

    def test_load_state_dict_rejects_metadata_drift(self) -> None:
        """Serialized size and schema metadata are part of the contract."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        key = torch.randn(1, 1, 2, 4)
        value = torch.randn(1, 1, 2, 4)
        state = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        state.append(key, value)
        payload = state.state_dict()
        payload["size"] = torch.tensor(99)
        with pytest.raises(ShapeError, match="size metadata"):
            cache.load_state_dict(payload)

        payload = state.state_dict()
        payload["schema_version"] = torch.tensor(2)
        with pytest.raises(ShapeError, match="schema_version"):
            cache.load_state_dict(payload)

    @pytest.mark.parametrize(
        ("key_shape", "value_shape", "message"),
        [
            ((1, 2, 3), (1, 2, 3), "rank 4"),
            ((1, 2, 3, 8), (1, 2, 3, 7), "head dimension"),
            ((1, 3, 3, 8), (1, 3, 3, 8), "head count"),
        ],
    )
    def test_append_rejects_malformed_tensors(
        self,
        key_shape: tuple[int, ...],
        value_shape: tuple[int, ...],
        message: str,
    ) -> None:
        """Invalid appends fail with domain errors, not native torch errors."""
        cache = InMemoryKVCache(num_heads=2, head_dim_k=8, head_dim_v=8)
        with pytest.raises(ShapeError, match=message):
            cache.append(torch.randn(*key_shape), torch.randn(*value_shape))

    def test_append_rejects_batch_size_change(self) -> None:
        """A cache cannot silently combine independent streams."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=6)
        cache.append(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 6))
        with pytest.raises(ShapeError, match="batch size"):
            cache.append(torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 6))

    def test_append_rejects_token_count_change(self) -> None:
        """Key and value sequences must stay aligned."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=6)
        with pytest.raises(ShapeError, match="token dimensions"):
            cache.append(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 3, 6))

    def test_failed_first_append_does_not_set_batch_state(self) -> None:
        """A conversion failure cannot poison an otherwise empty cache."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        with pytest.raises(NotImplementedError, match="meta"):
            cache.append(
                torch.empty(1, 1, 1, 4, device="meta"),
                torch.empty(1, 1, 1, 4, device="meta"),
            )
        assert cache.batch_size is None
        assert cache.size == 0

    def test_failed_restore_preserves_existing_state(self) -> None:
        """A value conversion failure cannot partially replace cached keys."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        original_key = torch.ones(1, 1, 1, 4)
        original_value = torch.ones(1, 1, 1, 4) * 2
        cache.append(original_key, original_value)
        payload = cache.state_dict()
        payload["cache_key"] = torch.zeros(1, 1, 1, 4)
        payload["cache_value"] = torch.empty(1, 1, 1, 4, device="meta")
        with pytest.raises(NotImplementedError, match="meta"):
            cache.load_state_dict(payload)
        actual_key, actual_value = cache.lookup()
        assert torch.equal(actual_key, original_key)
        assert torch.equal(actual_value, original_value)

    def test_restore_rejects_non_finite_values(self) -> None:
        """Corrupted serialized tensors cannot replace valid cache state."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        original_key = torch.zeros(1, 1, 1, 4)
        original_value = torch.zeros(1, 1, 1, 4)
        cache.append(original_key, original_value)
        state = cache.state_dict()
        state["cache_key"][0, 0, 0, 0] = float("inf")
        with pytest.raises(NonFiniteTensorError, match="non-finite"):
            cache.load_state_dict(state)
        actual_key, actual_value = cache.lookup()
        assert torch.equal(actual_key, original_key)
        assert torch.equal(actual_value, original_value)

    def test_cache_stats_expose_hits_and_misses(self) -> None:
        """Cache counters provide stable observability for serving code."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.lookup()
        cache.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))
        cache.lookup()
        stats = cache.cache_stats()
        assert stats["miss_count"] == 1
        assert stats["hit_count"] == 1

    def test_concurrent_appends_publish_consistent_state(self) -> None:
        """Concurrent appends are serialized without losing tokens."""
        cache = InMemoryKVCache(num_heads=1, head_dim_k=4, head_dim_v=4)
        key = torch.ones(1, 1, 1, 4)
        value = torch.ones(1, 1, 1, 4)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: cache.append(key, value), range(32)))

        cached_key, cached_value = cache.lookup()
        assert cached_key.shape[-2] == 32
        assert cached_value.shape[-2] == 32


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

    def test_concurrent_appends_publish_complete_pages(self) -> None:
        """Concurrent paged appends do not expose partial page publication."""
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim_k=4, head_dim_v=4)
        key = torch.ones(1, 1, 1, 4)
        value = torch.ones(1, 1, 1, 4)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: cache.append(key, value), range(16)))

        cached_key, cached_value = cache.lookup()
        assert cached_key.shape[-2] == 16
        assert cached_value.shape[-2] == 16

    def test_max_pages_limit(self) -> None:
        """max_pages raises NotInitializedError when exceeded."""
        cache = PagedKVCache(page_size=1, num_heads=1, head_dim_k=4, head_dim_v=4, max_pages=2)
        cache.append(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
        with pytest.raises(NotInitializedError, match="full"):
            cache.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))

    def test_capacity_failure_is_atomic(self) -> None:
        """An over-capacity append cannot leave partially written pages."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4, max_pages=2)
        with pytest.raises(NotInitializedError, match="requires 3 pages"):
            cache.append(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4))
        assert cache.size == 0
        assert cache.num_pages == 0

    def test_conversion_failure_is_atomic(self) -> None:
        """A tensor conversion failure cannot partially append a page."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        original_key = torch.randn(1, 1, 2, 4)
        original_value = torch.randn(1, 1, 2, 4)
        cache.append(original_key, original_value)

        with pytest.raises((NotImplementedError, RuntimeError)):
            cache.append(
                torch.ones(1, 1, 1, 4),
                torch.ones(1, 1, 1, 4, device="meta"),
            )

        actual_key, actual_value = cache.lookup()
        assert cache.size == 2
        assert cache.num_pages == 1
        assert torch.equal(actual_key, original_key)
        assert torch.equal(actual_value, original_value)

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

    def test_preserves_batch_dimension_across_pages(self) -> None:
        """Paged storage supports batched decoding without shape drift."""
        cache = PagedKVCache(page_size=2, num_heads=2, head_dim_k=4, head_dim_v=6)
        key = torch.randn(3, 2, 5, 4)
        value = torch.randn(3, 2, 5, 6)
        cache.append(key, value)
        actual_key, actual_value = cache.lookup()
        assert actual_key.shape == key.shape
        assert actual_value.shape == value.shape
        assert torch.equal(actual_key, key)
        assert torch.equal(actual_value, value)

    def test_rejects_batch_change_on_existing_page(self) -> None:
        """Appending another stream cannot reuse an existing page."""
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))
        with pytest.raises(ShapeError, match="batch size"):
            cache.append(torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4))

    def test_rejects_invalid_configuration(self) -> None:
        """Paged cache validates dimensions and capacity."""
        with pytest.raises(ConfigurationError):
            PagedKVCache(num_heads=0)
        with pytest.raises(ConfigurationError):
            PagedKVCache(max_pages=-1)

    def test_load_state_dict_rejects_missing_page(self) -> None:
        """Partial page checkpoints fail clearly."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        with pytest.raises(ShapeError, match="missing page"):
            cache.load_state_dict({"num_pages": torch.tensor(1)})

    def test_rejects_wrong_value_dimension(self) -> None:
        """Paged cache validates value dimensions before allocation."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=6)
        with pytest.raises(ShapeError, match="head dimension"):
            cache.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))

    def test_state_dict_round_trip_restores_pages(self) -> None:
        """Paged tensor contents survive a checkpoint round trip."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        key = torch.randn(1, 1, 3, 4)
        value = torch.randn(1, 1, 3, 4)
        cache.append(key, value)
        state = cache.state_dict()
        restored = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        restored.load_state_dict(state)
        actual_key, actual_value = restored.lookup()
        assert restored.size == 3
        assert restored.num_pages == 2
        assert torch.equal(actual_key, key)
        assert torch.equal(actual_value, value)
        assert torch.equal(restored.pages[0].positions, torch.tensor([0, 1]))
        assert torch.equal(restored.pages[1].positions, torch.tensor([2]))

    def test_load_state_dict_rejects_page_size_metadata_drift(self) -> None:
        """Paged checkpoints cannot claim a different token count."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
        payload = cache.state_dict()
        payload["size"] = torch.tensor(99)
        restored = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        with pytest.raises(ShapeError, match="size metadata"):
            restored.load_state_dict(payload)

    def test_load_state_dict_rejects_noncontiguous_positions(self) -> None:
        """A checkpoint cannot silently skip or reorder cached tokens."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
        payload = cache.state_dict()
        payload["page_1_positions"] = torch.tensor([7])
        restored = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        with pytest.raises(ShapeError, match="contiguous"):
            restored.load_state_dict(payload)

    def test_load_state_dict_rejects_non_integer_positions(self) -> None:
        """Checkpoint metadata must not be silently narrowed to integers."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))
        payload = cache.state_dict()
        payload["page_0_positions"] = torch.tensor([0.5])
        restored = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        with pytest.raises(ShapeError, match=r"torch\.long"):
            restored.load_state_dict(payload)

    def test_load_state_dict_rejects_page_batch_drift(self) -> None:
        """A checkpoint cannot combine pages from different request batches."""
        cache = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        cache.append(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
        payload = cache.state_dict()
        payload["page_1_key"] = torch.randn(2, 1, 1, 4)
        payload["page_1_value"] = torch.randn(2, 1, 1, 4)
        restored = PagedKVCache(page_size=2, num_heads=1, head_dim_k=4, head_dim_v=4)
        with pytest.raises(ShapeError, match="one batch size"):
            restored.load_state_dict(payload)


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
