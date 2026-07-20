"""Framework integrations for AVQA (spec §3.14, §3.15, §3.16, §5.17).

Sub-package that bridges AVQA with Hugging Face Transformers,
vLLM, FlashAttention, and xFormers. Each framework is a small
adapter; the four are split into separate modules under this
package:

- :mod:`avqa.integrations.hf` — HuggingFace replacement + weight copy
- :mod:`avqa.integrations.vllm` — vLLM attention backend selector
- :mod:`avqa.integrations.flashattn` — flash-attn interop
- :mod:`avqa.integrations.xformers` — xformers interop

Heavy third-party imports are guarded by per-module
``is_*_available()`` helpers; importing the package is safe even
when none are installed.
"""

from __future__ import annotations

from avqa.integrations.flashattn import (
    flash_attention_interop,
    is_flash_attention_available,
)
from avqa.integrations.hf import (
    HFAttentionWrapper,
    HFReplaceReport,
    copy_hf_weights,
    detect_compatible,
    is_huggingface_available,
    make_hf_attention_replacement,
    replace_attention,
)
from avqa.integrations.vllm import (
    AVQvLLMBackend,
    is_vllm_available,
    vllm_attention_backend,
)
from avqa.integrations.xformers import (
    is_xformers_available,
    xformers_interop,
)

__all__ = [
    "AVQvLLMBackend",
    "HFAttentionWrapper",
    "HFReplaceReport",
    "copy_hf_weights",
    "detect_compatible",
    "flash_attention_interop",
    "is_flash_attention_available",
    "is_huggingface_available",
    "is_vllm_available",
    "is_xformers_available",
    "make_hf_attention_replacement",
    "replace_attention",
    "vllm_attention_backend",
    "xformers_interop",
]
