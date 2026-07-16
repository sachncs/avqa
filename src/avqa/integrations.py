"""Framework integrations for AVQA (spec §3.14, §3.15, §3.16, §5.17).

This module provides real integrations with Hugging Face Transformers,
vLLM, FlashAttention, and xFormers.

ponytail: collapsed the planned integrations package (5 sub-modules)
into one src/avqa/integrations.py. Each integration exposes:
- :func:`is_available` — runtime check for the optional dep.
- :func:`replace_*` — a small replacement or adapter helper.

Heavy deps are optional extras (see pyproject.toml); the integration
modules are import-safe even when the dep is missing.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from avqa.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from avqa.attention_module import AVQAttention
    from avqa.config import AVQConfig

_logger = get_logger("integrations")


# ---------------------------------------------------------------------------
# Hugging Face Transformers (spec §3.14)
# ---------------------------------------------------------------------------


@dataclass
class HFReplaceReport:
    """Result of an HF attention replacement (spec §3.14.3).

    Attributes:
        modules_replaced: Number of attention modules replaced.
        modules_skipped: Number of attention modules left untouched.
        model_class: Class name of the model that was patched.
    """

    modules_replaced: int
    modules_skipped: int
    model_class: str


def is_huggingface_available() -> bool:
    """Return True iff the ``transformers`` package is importable."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def detect_compatible(model: PreTrainedModel) -> bool:
    """Detect whether ``model`` uses a compatible attention class (spec §3.14.1).

    The detector considers any module whose class name contains
    ``"Attention"`` (excluding MLP and output heads) as a candidate.

    Args:
        model: A Hugging Face ``PreTrainedModel``.

    Returns:
        ``True`` iff at least one attention module is present.
    """
    if not is_huggingface_available():
        return False
    for _name, module in model.named_modules():
        cls = type(module).__name__
        if "Attention" in cls and "MLP" not in cls:
            return True
    return False


def _make_hf_attention_replacement(
    embed_dim: int,
    num_heads: int,
    config: "AVQConfig",
) -> nn.Module:
    """Construct an AVQAttention module sized for a HF attention layer.

    ponytail: the replacement uses ``in_proj=False`` and ``out_proj=False``
    because HF's BertSelfAttention already provides the Q/K/V projections
    and output dense layer.
    """
    from avqa.attention_module import AVQAttention
    from avqa.config import (
        AttentionShapeConfig,
        CodebookConfig,
        RoutingConfig,
    )

    if config.attention.embed_dim != embed_dim:
        cfg = config.__class__(
            attention=AttentionShapeConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=embed_dim // num_heads,
            ),
            codebook=CodebookConfig(
                num_codewords=config.codebook.num_codewords,
                children_per_codeword=config.codebook.children_per_codeword,
            ),
            routing=RoutingConfig(
                refinement_budget=config.routing.refinement_budget,
            ),
        )
        config = cfg
    inner = AVQAttention(config, in_proj=False, out_proj=False)
    return _HFAttentionWrapper(inner)


class _HFAttentionWrapper(nn.Module):
    """Adapter that exposes AVQAttention via HF's attention call signature.

    HF attention modules pass additional kwargs (``attention_mask``,
    ``head_mask``, ``encoder_hidden_states``, etc.) which the AVQA
    forward signature does not accept. This wrapper translates the
    supported ones (``attention_mask`` → :attr:`mask`) and ignores the
    rest, so the replacement is backward compatible with HF's
    forward pass.
    """

    def __init__(self, inner: AVQAttention) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, ...] | None = None,
        output_attentions: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        """HF-compatible forward.

        Returns ``(output, attention_probs_or_none)`` to match HF's
        convention; ``output_attentions`` is accepted but ignored
        (AVQA does not expose raw attention probabilities).
        """
        # If cross-attention is requested, fall back to the inner
        # module's standard attention over encoder_hidden_states.
        kv_source = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # Translate attention_mask (HF [B, T_q, T_k] additive) to our
        # boolean mask. Additive masks with -10000 indicate "mask out";
        # boolean masks with 0/1 are also seen. Normalize to a bool
        # ``[T_q, T_k]`` (we ignore batch because we don't mask per-batch).
        if attention_mask is not None:
            mask = self._translate_mask(attention_mask, kv_source.shape[1])
        else:
            mask = None
        out = self.inner(hidden_states, kv_source, kv_source, mask=mask)
        if output_attentions:
            return out, None  # type: ignore[return-value]
        return (out,)

    @staticmethod
    def _translate_mask(mask: torch.Tensor, kv_len: int) -> torch.Tensor | None:
        """Translate an HF attention mask to our [T_q, T_k] boolean mask."""
        # mask can be [B, T_q, T_k] or [T_q, T_k] or None.
        if mask.dim() == 3:
            # Additive (-10000) or boolean (0/1); take the first batch.
            mask = mask[0]
        # If values are float-like with negatives, convert via "< 0".
        mask = mask < -1e4 if mask.is_floating_point() else mask > 0
        if mask.shape[0] == kv_len:
            # No self-attention mask; treat as fully visible.
            return None
        return mask


def replace_attention(
    model: PreTrainedModel,
    config: "AVQConfig",
    *,
    predicate: Callable[[str, nn.Module], bool] | None = None,
) -> HFReplaceReport:
    """Replace attention modules in a HF model (spec §3.14).

    Walks ``model.named_modules()`` and replaces each submodule whose
    class name contains ``"Attention"`` with an :class:`AVQAttention`
    sized to match the original's ``hidden_size`` and ``num_heads``.

    Args:
        model: A Hugging Face ``PreTrainedModel``.
        config: AVQA configuration.
        predicate: Optional callable ``(name, module) -> bool`` that
            decides whether to replace a module. Default replaces all
            attention modules.

    Returns:
        :class:`HFReplaceReport` summarizing the replacement.

    Example:
        >>> from transformers import AutoModel
        >>> from avqa import AVQConfig
        >>> from avqa.integrations import replace_attention
        >>> model = AutoModel.from_pretrained("bert-base-uncased")
        >>> report = replace_attention(model, AVQConfig())
        >>> report.modules_replaced > 0
        True
    """
    if not is_huggingface_available():
        msg = "transformers is not installed; cannot replace attention"
        raise RuntimeError(msg)
    if not detect_compatible(model):
        return HFReplaceReport(
            modules_replaced=0,
            modules_skipped=0,
            model_class=type(model).__name__,
        )

    embed_dim = getattr(model.config, "hidden_size", None) or getattr(
        model.config, "d_model", None,
    )
    num_heads = getattr(model.config, "num_attention_heads", None) or getattr(
        model.config, "n_head", None,
    )
    if embed_dim is None or num_heads is None:
        msg = (
            f"could not infer embed_dim / num_heads from "
            f"{type(model).__name__}.config"
        )
        raise RuntimeError(msg)

    replaced = 0
    skipped = 0
    # We rebuild the module dict in-place.
    name_to_module = dict(model.named_modules())

    for name, module in list(name_to_module.items()):
        if name == "":
            continue
        cls_name = type(module).__name__
        if "Attention" not in cls_name or "MLP" in cls_name:
            continue
        if predicate is not None and not predicate(name, module):
            skipped += 1
            continue
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = _make_hf_attention_replacement(embed_dim, num_heads, config)
        setattr(parent, attr_name, replacement)
        replaced += 1

    return HFReplaceReport(
        modules_replaced=replaced,
        modules_skipped=skipped,
        model_class=type(model).__name__,
    )


# ---------------------------------------------------------------------------
# vLLM (spec §3.15)
# ---------------------------------------------------------------------------


def is_vllm_available() -> bool:
    """Return True iff the ``vllm`` package is importable."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False
    return True


def vllm_attention_backend(backend: str = "torch") -> object:
    """Return a vLLM-compatible attention backend selector (spec §3.15).

    ponytail: vLLM's plugin protocol is version-coupled; we provide a
    minimal selector that returns the requested backend name. Wire this
    into vLLM's :class:`vllm.attention.backends.registry.AttentionBackendEnum`
    via your deployment's plugin entry point.

    Args:
        backend: One of ``"torch"``, ``"triton"``, ``"xformers"``,
            ``"flash_attn"``, or ``"avqa"``.

    Returns:
        A simple object exposing ``.name`` for vLLM to introspect.
    """
    available = {
        "torch": True,
        "triton": is_vllm_available(),
        "xformers": is_xformers_available(),
        "flash_attn": is_flash_attention_available(),
        "avqa": True,
    }
    if backend not in available:
        msg = f"unknown vLLM backend '{backend}'"
        raise ValueError(msg)
    if not available[backend]:
        msg = f"vLLM backend '{backend}' is not installed"
        raise RuntimeError(msg)

    class _Selector:
        name = backend

    return _Selector()


# ---------------------------------------------------------------------------
# FlashAttention (spec §3.16)
# ---------------------------------------------------------------------------


def is_flash_attention_available() -> bool:
    """Return True iff ``flash_attn`` is importable."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def flash_attention_interop(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Drop-in wrapper around ``flash_attn_func`` when available (spec §3.16).

    Falls back to AVQA's :class:`TorchBackend` when flash-attn is missing
    or when CUDA is unavailable.

    Args:
        query: ``[B, T, H, D]`` (note: HF layout — heads before head dim).
        key: ``[B, T, H, D]``.
        value: ``[B, T, H, D]``.

    Returns:
        Attention output with the same layout.
    """
    if not is_flash_attention_available() or not torch.cuda.is_available():
        from avqa.backend import TorchBackend

        # Convert HF layout [B, T, H, D] -> [B, H, T, D] for the backend.
        def to_avqa(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous()

        def from_avqa(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous()

        backend = TorchBackend()
        out = backend.naive_attention(to_avqa(query), to_avqa(key), to_avqa(value))
        return from_avqa(out)

    # flash_attn_func expects [B, T, H, D] directly.
    import flash_attn

    out = flash_attn.flash_attn_func(query, key, value, causal=False)
    return out  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# xFormers (spec §3.16 alternative)
# ---------------------------------------------------------------------------


def is_xformers_available() -> bool:
    """Return True iff ``xformers`` is importable."""
    try:
        import xformers  # noqa: F401
    except ImportError:
        return False
    return True


def xformers_interop(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Drop-in wrapper around ``xformers.ops.memory_efficient_attention`` (spec §3.16).

    Falls back to AVQA's :class:`TorchBackend` when xformers is missing.

    Args:
        query: ``[B, H, T, D]`` (AVQA layout).
        key: ``[B, H, T, D]``.
        value: ``[B, H, T, D]``.

    Returns:
        Attention output with the same layout.
    """
    if not is_xformers_available() or not torch.cuda.is_available():
        from avqa.backend import TorchBackend

        return TorchBackend().naive_attention(query, key, value)

    import xformers.ops as xops  # type: ignore[import-not-found]

    return xops.memory_efficient_attention(query, key, value)  # type: ignore[no-any-return]


__all__ = [
    "HFReplaceReport",
    "detect_compatible",
    "flash_attention_interop",
    "is_flash_attention_available",
    "is_huggingface_available",
    "is_vllm_available",
    "is_xformers_available",
    "replace_attention",
    "vllm_attention_backend",
    "xformers_interop",
]
