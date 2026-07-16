"""Tests for avqa.integrations module."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModel

from avqa import AVQConfig
from avqa.integrations import (
    HFReplaceReport,
    detect_compatible,
    flash_attention_interop,
    is_flash_attention_available,
    is_huggingface_available,
    is_vllm_available,
    is_xformers_available,
    replace_attention,
    vllm_attention_backend,
    xformers_interop,
)


class TestHuggingFaceAvailability:
    """Tests for Hugging Face availability detection."""

    def test_is_huggingface_available(self) -> None:
        """is_huggingface_available returns True when transformers is installed."""
        assert isinstance(is_huggingface_available(), bool)


class TestFlashAttentionAvailability:
    """Tests for FlashAttention availability detection."""

    def test_is_flash_attention_available(self) -> None:
        """Returns True iff flash_attn is importable."""
        assert isinstance(is_flash_attention_available(), bool)


class TestXFormersAvailability:
    """Tests for xFormers availability detection."""

    def test_is_xformers_available(self) -> None:
        """Returns True iff xformers is importable."""
        assert isinstance(is_xformers_available(), bool)


class TestVLLMAvailability:
    """Tests for vLLM availability detection."""

    def test_is_vllm_available(self) -> None:
        """Returns True iff vllm is importable."""
        assert isinstance(is_vllm_available(), bool)


@pytest.mark.skipif(not is_huggingface_available(), reason="transformers not installed")
class TestHFReplacement:
    """Tests for Hugging Face attention replacement (spec §3.14)."""

    def test_detect_compatible(self) -> None:
        """detect_compatible returns True for models with attention layers."""


        model = AutoModel.from_pretrained("hf-internal-testing/tiny-bert")
        assert detect_compatible(model) is True

    def test_replace_attention_runs(self) -> None:
        """replace_attention successfully swaps attention modules."""




        model = AutoModel.from_pretrained("hf-internal-testing/tiny-bert")
        report = replace_attention(model, AVQConfig())
        assert isinstance(report, HFReplaceReport)
        assert report.modules_replaced > 0

    def test_replaced_model_forward(self) -> None:
        """The replaced model still runs forward."""




        model = AutoModel.from_pretrained("hf-internal-testing/tiny-bert")
        replace_attention(model, AVQConfig())
        # Tiny BERT has hidden_size=128, num_heads=2.
        out = model(torch.zeros(1, 4, dtype=torch.long)).last_hidden_state
        assert out.shape == (1, 4, 128)

    def test_pretrained_weights_preserved(self) -> None:
        """Other (non-attention) parameters are unchanged after replacement.

        We compare a non-replaced parameter against the original. Spec
        §3.14.3 requires pretrained weights to be preserved.
        """




        model = AutoModel.from_pretrained("hf-internal-testing/tiny-bert")
        # Capture an unrelated parameter (embedding).
        original = model.embeddings.word_embeddings.weight.detach().clone()
        replace_attention(model, AVQConfig())
        new = model.embeddings.word_embeddings.weight.detach().clone()
        assert torch.equal(original, new)

    def test_predicate_filters(self) -> None:
        """Custom predicate can skip modules."""




        model = AutoModel.from_pretrained("hf-internal-testing/tiny-bert")
        report = replace_attention(model, AVQConfig(), predicate=lambda _n, _m: False)
        assert report.modules_replaced == 0
        assert report.modules_skipped >= 1


class TestVLLMBackend:
    """Tests for the vLLM backend selector (spec §3.15)."""

    def test_torch_backend(self) -> None:
        """torch backend is always available."""
        backend = vllm_attention_backend("torch")
        assert backend.name == "torch"

    def test_avqa_backend(self) -> None:
        """avqa backend is always available."""
        backend = vllm_attention_backend("avqa")
        assert backend.name == "avqa"

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend name raises ValueError."""
        with pytest.raises(ValueError, match="unknown"):
            vllm_attention_backend("nonsense")


class TestFlashAttentionInterop:
    """Tests for the FlashAttention interop (spec §3.16)."""

    def test_falls_back_to_torch_when_unavailable(self) -> None:
        """Without flash-attn or CUDA, falls back to TorchBackend."""
        if is_flash_attention_available() and torch.cuda.is_available():
            pytest.skip("flash-attn with CUDA is available; skip fallback test")
        q = torch.randn(1, 4, 2, 8)  # [B, T, H, D] HF layout
        k = torch.randn(1, 4, 2, 8)
        v = torch.randn(1, 4, 2, 8)
        out = flash_attention_interop(q, k, v)
        assert out.shape == (1, 4, 2, 8)


class TestXFormersInterop:
    """Tests for the xFormers interop."""

    def test_falls_back_to_torch_when_unavailable(self) -> None:
        """Without xformers or CUDA, falls back to TorchBackend."""
        if is_xformers_available() and torch.cuda.is_available():
            pytest.skip("xformers with CUDA is available; skip fallback test")
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 6, 8)
        v = torch.randn(1, 2, 6, 8)
        out = xformers_interop(q, k, v)
        assert out.shape == (1, 2, 4, 8)
