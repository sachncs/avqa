"""Core data-model primitives for AVQA.

Spec §6.5 defines canonical tensor shapes; §6.9 enumerates supported
dtypes; §6.10 enumerates supported devices; §6.19 requires every public
function to publish an explicit input/output contract.

ponytail: collapsed four planned submodules (shapes, dtypes, devices,
contracts) into one file. Each concept is small (a frozenset, a few
constants, a dataclass). Splitting would have added import overhead
without clarity benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

# Canonical shape symbols (spec §6.3). Kept short to match math notation.
BATCH: Final[str] = "B"
SEQUENCE: Final[str] = "T"
HEAD: Final[str] = "H"
HEAD_DIM: Final[str] = "D"
EMBED: Final[str] = "E"
PARENT_CODEWORDS: Final[str] = "C"
REFINED_CODEWORDS: Final[str] = "R"
TOKENS: Final[str] = "N"
REFINEMENT_BUDGET: Final[str] = "K"
HIERARCHY_DEPTH: Final[str] = "L"

# Canonical tensor shapes (spec §6.5). The comments reference the spec
# sections each shape is required by.
SHAPE_QUERY: Final[tuple[str, ...]] = (BATCH, HEAD, SEQUENCE, HEAD_DIM)
SHAPE_KEY: Final[tuple[str, ...]] = (BATCH, HEAD, SEQUENCE, HEAD_DIM)
SHAPE_VALUE: Final[tuple[str, ...]] = (BATCH, HEAD, SEQUENCE, HEAD_DIM)
SHAPE_CODEBOOK: Final[tuple[str, ...]] = (HEAD, PARENT_CODEWORDS, HEAD_DIM)
SHAPE_ASSIGNMENT: Final[tuple[str, ...]] = (BATCH, HEAD, SEQUENCE)
SHAPE_ROUTING_SCORES: Final[tuple[str, ...]] = (BATCH, HEAD, PARENT_CODEWORDS)
SHAPE_ACTIVE_CODEWORDS: Final[tuple[str, ...]] = (BATCH, HEAD, REFINEMENT_BUDGET)
SHAPE_REFINED_CODEBOOK: Final[tuple[str, ...]] = (BATCH, HEAD, REFINED_CODEWORDS, HEAD_DIM)
SHAPE_REFINED_ATTENTION: Final[tuple[str, ...]] = (BATCH, HEAD, SEQUENCE, REFINED_CODEWORDS)
SHAPE_FINAL_ATTENTION: Final[tuple[str, ...]] = (
    BATCH,
    HEAD,
    SEQUENCE,
    f"{PARENT_CODEWORDS}+{REFINED_CODEWORDS}",
)
SHAPE_OUTPUT: Final[tuple[str, ...]] = (BATCH, HEAD, SEQUENCE, HEAD_DIM)

# Supported dtypes (spec §6.9). Reference implementation MUST support
# float32, float16, bfloat16. float64 / FP8 / INT8 are optional and
# ponytail: FP8/INT8 support deferred (optional per spec §6.9).
SUPPORTED_DTYPES: Final[frozenset[torch.dtype]] = frozenset(
    {torch.float32, torch.float16, torch.bfloat16}
)
EXTENDED_DTYPES: Final[frozenset[torch.dtype]] = frozenset(
    {torch.float32, torch.float16, torch.bfloat16, torch.float64}
)


def is_supported_dtype(dtype: torch.dtype) -> bool:
    """Return ``True`` if ``dtype`` is in the reference supported set."""
    return dtype in SUPPORTED_DTYPES


# Supported devices (spec §6.10). CPU + CUDA are baseline; ROCm / Metal /
# XPU are future. ROCm uses the cuda device type in PyTorch.
SUPPORTED_DEVICES: Final[frozenset[str]] = frozenset({"cpu", "cuda", "mps"})


def is_supported_device(device: str | torch.device) -> bool:
    """Return ``True`` if ``device`` type is supported.

    Invalid device strings return ``False`` rather than raising.
    """
    try:
        device_type = torch.device(device).type
    except (RuntimeError, TypeError, ValueError):
        return False
    return device_type in SUPPORTED_DEVICES


@dataclass(frozen=True, slots=True)
class TensorContract:
    """Documented input/output contract for a public function.

    Spec §6.19 requires every public function to publish: input tensors,
    output tensors, shape, dtype, device, ownership, mutability, complexity.

    This dataclass captures the structural metadata (shape, dtype, device)
    for one tensor. It is a documentation artifact, not an enforcement
    mechanism — see :mod:`avqa.utils.validation` for runtime checks.

    Attributes:
        name: Tensor name (e.g., ``"query"``, ``"codebook"``).
        shape: Canonical shape symbol tuple (e.g., ``("B","H","T","D")``).
        dtype: Set of acceptable dtypes.
        device: Set of acceptable device types (e.g., ``{"cpu","cuda"}``).
        owner: Logical owner subsystem (e.g., ``"attention"``, ``"codebook"``).
        mutable: Whether the tensor may be modified during forward execution.
        complexity: Computational or memory complexity annotation.

    Example:
        >>> from avqa.data import TensorContract
        >>> contract = TensorContract(
        ...     name="query",
        ...     shape=("B", "H", "T", "D"),
        ...     dtype=frozenset({torch.float32, torch.float16, torch.bfloat16}),
        ...     device=frozenset({"cpu", "cuda"}),
        ...     owner="attention",
        ...     mutable=False,
        ...     complexity="O(B*H*T*D)",
        ... )
        >>> contract.name
        'query'
    """

    name: str
    shape: tuple[str, ...]
    dtype: frozenset[torch.dtype]
    device: frozenset[str]
    owner: str
    mutable: bool
    complexity: str

    def supports(self, tensor: torch.Tensor) -> bool:
        """Return ``True`` if ``tensor`` is structurally compatible."""
        if len(tensor.shape) != len(self.shape):
            return False
        if tensor.dtype not in self.dtype:
            return False
        return tensor.device.type in self.device


def make_default_contract(name: str, shape: Sequence[str], owner: str) -> TensorContract:
    """Construct a :class:`TensorContract` with library defaults.

    Defaults match the reference implementation: supports FP32/FP16/BF16,
    CPU/CUDA, immutable, O(B*H*T*D) cost. Override individual fields
    after construction if needed.

    Args:
        name: Tensor name.
        shape: Canonical shape symbol tuple.
        owner: Logical owner subsystem.

    Returns:
        A populated :class:`TensorContract`.

    Example:
        >>> from avqa.data import make_default_contract
        >>> c = make_default_contract("query", ("B", "H", "T", "D"), "attention")
        >>> c.dtype == frozenset({torch.float32, torch.float16, torch.bfloat16})
        True
    """
    return TensorContract(
        name=name,
        shape=tuple(shape),
        dtype=SUPPORTED_DTYPES,
        device=SUPPORTED_DEVICES,
        owner=owner,
        mutable=False,
        complexity="O(B*H*T*D)",
    )


__all__ = [
    "BATCH",
    "EMBED",
    "EXTENDED_DTYPES",
    "HEAD",
    "HEAD_DIM",
    "HIERARCHY_DEPTH",
    "PARENT_CODEWORDS",
    "REFINED_CODEWORDS",
    "REFINEMENT_BUDGET",
    "SEQUENCE",
    "SHAPE_ACTIVE_CODEWORDS",
    "SHAPE_ASSIGNMENT",
    "SHAPE_CODEBOOK",
    "SHAPE_FINAL_ATTENTION",
    "SHAPE_KEY",
    "SHAPE_OUTPUT",
    "SHAPE_QUERY",
    "SHAPE_REFINED_ATTENTION",
    "SHAPE_REFINED_CODEBOOK",
    "SHAPE_ROUTING_SCORES",
    "SHAPE_VALUE",
    "SUPPORTED_DEVICES",
    "SUPPORTED_DTYPES",
    "TOKENS",
    "TensorContract",
    "is_supported_device",
    "is_supported_dtype",
    "make_default_contract",
]
