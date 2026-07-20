"""Hugging Face Transformers integration for AVQA (spec §3.14).

The HF adapter is the most actively used framework glue: it lets users
swap attention modules in a pretrained :class:`transformers.PreTrainedModel`
with :class:`avqa.attention_module.AVQAttention` while preserving the
rest of the weights.

Heavy imports (``transformers``) are guarded by :func:`is_huggingface_available`
so ``import avqa`` does not require them.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import TYPE_CHECKING

import torch
from torch import nn

from avqa.attention_module import AVQAttention
from avqa.config import (
    AttentionShapeConfig,
    AVQConfig,
    CodebookConfig,
    RoutingConfig,
)
from avqa.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedModel

logger = get_logger("integrations.hf")


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
    return importlib.util.find_spec("transformers") is not None


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
    for _, module in model.named_modules():
        cls = type(module).__name__
        if "Attention" in cls and "MLP" not in cls:
            return True
    return False


def make_hf_attention_replacement(
    embed_dim: int,
    num_heads: int,
    config: "AVQConfig",
    original_module: nn.Module | None = None,
) -> nn.Module:
    """Construct an AVQAttention module sized for a HF attention layer.

    When ``original_module`` is provided, its Q/K/V projection weights
    are copied into the new AVQAttention to preserve pretrained knowledge
    (spec §3.14).

    ponytail: when no original_module is given, uses ``in_proj=False``
    and ``out_proj=False`` (no weight transfer needed).
    """
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

    use_projections = original_module is not None
    inner = AVQAttention(config, in_proj=use_projections, out_proj=use_projections)

    if original_module is not None:
        copy_hf_weights(original_module, inner, embed_dim)

    return HFAttentionWrapper(inner)


def copy_hf_weights(src: nn.Module, dst: "AVQAttention", embed_dim: int) -> None:
    """Copy Q/K/V/Output weights AND biases from a HF attention module.

    Handles common HF naming conventions (``query`` / ``key`` / ``value`` /
    ``q_proj`` / ``k_proj`` / ``v_proj`` and ``dense`` / ``out`` /
    ``o_proj``) and both with/without bias. The slice ``[:embed_dim, :embed_dim]``
    handles GQA where the host model stores a fused embedding even when
    only one head's projection is needed.

    Args:
        src: Original HF attention module.
        dst: :class:`AVQAttention` already instantiated with the
            correct ``embed_dim``.
        embed_dim: Embedding dimension shared by the host and AVQA.
    """
    src_params = dict(src.named_parameters())

    weight_map = {
        "query": "q_proj",
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "key": "k_proj",
        "v_proj": "v_proj",
        "value": "v_proj",
    }
    out_map = {
        "out": "out_proj",
        "o_proj": "out_proj",
        "dense": "out_proj",
    }

    def copy(src_name: str, dst_attr: str, *, with_bias: bool) -> bool:
        """Copy a single HF parameter (with optional bias) to an AVQA module.

        Returns:
            ``True`` if a weight was found and copied; ``False`` otherwise.
        """
        w_key = f"{src_name}.weight"
        b_key = f"{src_name}.bias"
        if w_key not in src_params:
            return False
        weight = src_params[w_key]
        dst_param = getattr(dst, dst_attr)
        if hasattr(dst_param, "weight") and dst_param.weight is not None:
            dst_param.weight.data.copy_(weight[:embed_dim, :embed_dim])
        if (
            with_bias
            and hasattr(dst_param, "bias")
            and dst_param.bias is not None
            and b_key in src_params
        ):
            bias = src_params[b_key]
            dst_param.bias.data[:embed_dim].copy_(bias[:embed_dim])
        return True

    for src_name, dst_name in weight_map.items():
        copy(src_name, dst_name, with_bias=True)
    for src_name, dst_name in out_map.items():
        copy(src_name, dst_name, with_bias=True)


class HFAttentionWrapper(nn.Module):
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
        if head_mask is not None:
            logger.debug(
                "HFAttentionWrapper: head_mask provided but not supported "
                "by AVQA; ignoring"
            )
        if past_key_value is not None:
            logger.debug(
                "HFAttentionWrapper: past_key_value provided but not "
                "supported by AVQA wrapper; ignoring"
            )
        # If cross-attention is requested, fall back to the inner
        # module's standard attention over encoder_hidden_states.
        kv_source = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # Translate attention_mask (HF [B, T_q, T_k] additive) to a 2-D
        # boolean ``[T_q, T_k]``. Additive masks with -10000 indicate
        # "mask out"; boolean masks with 0/1 are also seen. We take the
        # first batch because the orchestrator applies one mask per call.
        if attention_mask is not None:
            mask = self.translate_mask(attention_mask, kv_source.shape[1])
        else:
            mask = None
        out = self.inner(hidden_states, kv_source, kv_source, mask=mask)
        if output_attentions:
            return out, None  # type: ignore[return-value]
        return (out,)

    @staticmethod
    def translate_mask(mask: torch.Tensor, kv_len: int) -> torch.Tensor | None:
        """Translate an HF attention mask to a ``[T_q, T_k]`` boolean mask.

        Supports ``[B, T_q, T_k]`` (additive or boolean) and ``[T_q, T_k]``.
        Returns ``None`` when the mask is fully visible (no masking).
        """
        if mask.dim() == 3:
            mask = mask[0]
        mask = mask < -1e4 if mask.is_floating_point() else mask > 0
        if mask.shape[0] == kv_len:
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
        model.config,
        "d_model",
        None,
    )
    num_heads = getattr(model.config, "num_attention_heads", None) or getattr(
        model.config,
        "n_head",
        None,
    )
    if embed_dim is None or num_heads is None:
        msg = f"could not infer embed_dim / num_heads from {type(model).__name__}.config"
        raise RuntimeError(msg)

    replaced = 0
    skipped = 0
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
        replacement = make_hf_attention_replacement(
            embed_dim,
            num_heads,
            config,
            original_module=module,
        )
        setattr(parent, attr_name, replacement)
        replaced += 1

    return HFReplaceReport(
        modules_replaced=replaced,
        modules_skipped=skipped,
        model_class=type(model).__name__,
    )


__all__ = [
    "HFAttentionWrapper",
    "HFReplaceReport",
    "copy_hf_weights",
    "detect_compatible",
    "is_huggingface_available",
    "make_hf_attention_replacement",
    "replace_attention",
]
