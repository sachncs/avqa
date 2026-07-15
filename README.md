# AVQA — Adaptive Vector Quantized Attention

> **Independent Implementation Disclaimer**
>
> This is an **independent, community-driven open-source implementation** of
> the Adaptive Vector Quantized Attention (AVQ-Attention) algorithm. The
> author of this codebase is **not** an author of the reference paper and is
> not affiliated with the paper's authors or their institutions. This project
> is offered to the community under the Apache 2.0 License and follows the
> publicly-available specification only.
>
> Reference paper: [arXiv:2607.12789v1](https://arxiv.org/html/2607.12789v1)
>
> Any deviation from the paper is documented in
> [`docs/spec_gaps.md`](docs/spec_gaps.md).

---

## Overview

**AVQA** is a production-grade Python library implementing Adaptive Vector
Quantized Attention (AVQ-Attention) as a drop-in attention backend for
PyTorch-based Transformer architectures. It transforms the research paper
into reusable, testable, benchmarked, and framework-integrated
infrastructure.

Key features:

- **Pure PyTorch reference implementation** with the canonical online-softmax
  algorithm from FlashAttention-2.
- **Optimized Triton backend** for CUDA GPUs.
- **Hierarchical codebook** with mean-constrained parent-child structure.
- **Adaptive refinement** that expands only the most-attended codewords.
- **Correcting attention** that replaces — not augments — parent
  contributions with child contributions while preserving normalization.
- **Framework integrations** for Hugging Face Transformers, vLLM,
  FlashAttention, and xFormers.
- **Profiling, visualization, and benchmarking** tools.
- **Strict typing, zero-warning lint, ≥ 90% test coverage** on the core
  package.

---

## Installation

The base install pulls in only PyTorch:

```bash
pip install avqa
```

Optional extras for framework integrations:

```bash
pip install "avqa[huggingface]"     # Hugging Face Transformers
pip install "avqa[vllm]"            # vLLM integration
pip install "avqa[flash-attn]"      # FlashAttention interop
pip install "avqa[xformers]"        # xFormers interop
pip install "avqa[triton]"          # Triton kernels (requires CUDA)
pip install "avqa[viz]"             # Visualization backends
pip install "avqa[all]"             # All of the above
```

For local development:

```bash
git clone https://github.com/sachncs/avqa.git
cd avqa
pip install -e ".[all]"
pip install pytest pytest-cov pytest-benchmark hypothesis ruff mypy matplotlib
```

---

## Quick Start

```python
import torch
from avqa import AVQAttention, AVQConfig

config = AVQConfig(
    embed_dim=512,
    num_heads=8,
    num_codewords=64,
    children_per_codeword=4,
    refinement_budget=8,
    backend="torch",
)

attention = AVQAttention(config)

query = torch.randn(2, 8, 128, 64)  # [B, H, T, D]
key = torch.randn(2, 8, 128, 64)
value = torch.randn(2, 8, 128, 64)

output = attention(query, key, value)  # [B, H, T, D]
```

Functional API:

```python
from avqa import AVQConfig
from avqa.functional import attention

config = AVQConfig(...)
output = attention(query=query, key=key, value=value, config=config)
```

---

## Documentation

- [`docs/implementation_plan.md`](docs/implementation_plan.md) — overall
  implementation plan, scope, and acceptance criteria.
- [`docs/dependency_graph.md`](docs/dependency_graph.md) — subsystem-level
  dependency DAG.
- [`docs/milestone_plan.md`](docs/milestone_plan.md) — milestone breakdown
  with exit criteria.
- [`docs/checklist.md`](docs/checklist.md) — every normative requirement from
  the specification.
- [`docs/spec_gaps.md`](docs/spec_gaps.md) — implementation assumptions for
  specification gaps.
- [`docs/spec_compliance.md`](docs/spec_compliance.md) — compliance matrix
  tracking implementation status.
- [`TODO.md`](TODO.md) — atomic task tracker.

The authoritative source is [`spec.md`](spec.md).

---

## Status

This project is in **alpha** (v0.1.0). The implementation is being built
incrementally from the ground up, one atomic commit per task. See
[`TODO.md`](TODO.md) for the current task status.

---

## Development

All commands run from the repository root.

```bash
# Lint (zero warnings required)
ruff check .
ruff format --check .

# Format (applies changes in place)
ruff format .

# Type-check (strict mode on src/avqa/)
mypy src/avqa

# Tests
pytest tests/unit -q
pytest tests/integration -q
pytest tests/reference -q
pytest tests/performance -q

# With coverage
pytest tests/unit tests/reference --cov=avqa --cov-report=term --cov-fail-under=90
```

CI:

- `.github/workflows/ci-cpu.yml` — runs on every push and PR (lint, type,
  CPU tests, coverage gate at 90%).
- `.github/workflows/ci-gpu.yml` — runs on self-hosted GPU runners, gated by
  `gpu` PR labels or manual dispatch.

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (forthcoming).

## Citation

This is an independent implementation. If you use AVQA in research, please
cite the original paper:

```bibtex
@misc{avq-attention,
  title  = {Adaptive Vector Quantized Attention (AVQ-Attention)},
  url    = {https://arxiv.org/html/2607.12789v1},
  year   = {2025},
}
```
