<p align="center">
  <h1 align="center">AVQA</h1>
  <p align="center">Adaptive Vector Quantized Attention for PyTorch.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
    <a href="https://github.com/sachncs/avqa/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/avqa/ci.yml?branch=main" alt="CI"></a>
    <a href="https://github.com/sachncs/avqa/stargazers"><img src="https://img.shields.io/github/stars/sachncs/avqa" alt="Stars"></a>
  </p>
</p>

**AVQA** is a production-grade Python library implementing Adaptive Vector
Quantized Attention (AVQ-Attention) as a drop-in attention backend for
PyTorch-based Transformer architectures.

> **Disclaimer**
>
> This is an independent, community-driven implementation. The author of this
> codebase is **not** an author of the reference paper and is not affiliated
> with the paper's authors or their institutions. See [Citation](#citation).

---

## Features

- **Pure PyTorch reference implementation** with the canonical online-softmax
  algorithm from FlashAttention-2.
- **Triton backend** for CUDA GPUs (delegates to Torch until kernel spec
  is finalized).
- **Hierarchical codebook** with mean-constrained parent-child structure.
- **Adaptive refinement** that expands only the most-attended codewords.
- **Correcting attention** that replaces — not augments — parent
  contributions with child contributions while preserving normalization.
- **Framework integrations** for Hugging Face Transformers, vLLM,
  FlashAttention, and xFormers.
- **Profiling, visualization, and benchmarking** tools.
- **Strict typing, zero-warning lint, ≥90% test coverage** on the core
  package.

---

## Installation

### From source

```bash
git clone https://github.com/sachncs/avqa.git
cd avqa
pip install -e .
```

### With optional extras

```bash
pip install "avqa[huggingface]"     # Hugging Face Transformers
pip install "avqa[vllm]"            # vLLM integration
pip install "avqa[flash-attn]"      # FlashAttention interop
pip install "avqa[xformers]"        # xFormers interop
pip install "avqa[triton]"          # Triton kernels (requires CUDA)
pip install "avqa[viz]"             # Visualization backends
pip install "avqa[all]"             # All of the above
```

### With dev dependencies

```bash
pip install -e ".[all]"
pip install pytest pytest-cov pytest-benchmark hypothesis ruff mypy matplotlib
```

---

## Quick Start

### Module API

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

### Functional API

```python
from avqa import AVQConfig
from avqa.functional import attention

config = AVQConfig(...)
output = attention(query=query, key=key, value=value, config=config)
```

---

## API Reference

| Symbol | Type | Description |
|--------|------|-------------|
| `AVQAttention` | class | Primary `nn.Module` attention wrapper |
| `AVQConfig` | dataclass | Immutable configuration (codebook, routing, merge, backend, cache, precision) |
| `VectorQuantizer` | class | Hierarchical vector quantization engine |
| `HierarchicalCodebook` | class | Parent-child codebook with mean constraint |
| `Router` | class | Routing strategy interface (TopP, Threshold, Budget) |
| `AdaptiveRefinement` | class | Refinement orchestrator |
| `Scheduler` | class | Refinement budget scheduler (Default, Adaptive) |
| `KVCache` | class | Autoregressive KV cache (InMemory, Paged) |
| `Backend` | class | Execution backend (Torch, Triton) |
| `Profiler` | class | Runtime profiler with JSON export |
| `attention` | function | Stateless functional entry point |

---

## Project Structure

```
avqa/
├── src/avqa/                  # Package source
│   ├── __init__.py            # Public API exports
│   ├── attention_module.py    # AVQAttention nn.Module
│   ├── attention.py           # Online softmax state + correction
│   ├── codebook.py            # HierarchicalCodebook
│   ├── quantizer.py           # EuclideanHierarchicalQuantizer
│   ├── routing.py             # Router + importance + selectors
│   ├── merge.py               # Merge strategies
│   ├── refinement.py          # AdaptiveRefinement orchestrator
│   ├── backend.py             # TorchBackend / TritonBackend
│   ├── cache.py               # KVCache (InMemory, Paged)
│   ├── scheduler.py           # Default + Adaptive schedulers
│   ├── config.py              # AVQConfig + sub-configs
│   ├── data.py                # Shapes, dtypes, devices, contracts
│   ├── functional.py          # Stateless functional API
│   ├── integrations.py        # HF, vLLM, FlashAttention, xFormers
│   ├── profiling.py           # Profiler + metrics + report
│   ├── visualization.py       # Visualizer (tree, heatmap, timeline)
│   ├── exceptions.py          # Exception hierarchy
│   ├── logging.py             # Logging configuration
│   ├── registry.py            # Extension registry
│   └── utils/                 # seed, validation, numerics
├── tests/
│   ├── unit/                  # Unit tests
│   ├── reference/             # Hand-computed reference tests
│   ├── integration/           # Integration tests (HF, vLLM, FA, xF)
│   └── performance/           # pytest-benchmark suite
├── docs/                      # Architecture + compliance docs
├── examples/                  # Usage examples
├── pyproject.toml             # Build & tool config
└── .github/                   # CI workflow
```

---

## Development

```bash
# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type-check
mypy src/avqa

# Tests
pytest tests/unit -q
pytest tests/reference -q
pytest tests/integration -q
pytest tests/performance -q

# With coverage
pytest tests/unit tests/reference --cov=avqa --cov-report=term --cov-fail-under=90
```

### Code Style

- Line length: 100
- Quotes: double
- Formatter/linter: ruff
- Type hints: required on all public signatures
- Docstrings: Google-style

### Commit Conventions

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add online-softmax tiled attention
fix: correct einsum dimension mapping in parent attention
docs: update compliance matrix
test: add hand-computed reference tests
```

---

## Testing

```bash
pytest                                          # full suite
pytest --cov=avqa tests/unit tests/reference    # with coverage
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Framework | [PyTorch](https://pytorch.org/) 2.1+ |
| Build | [Hatchling](https://hatch.pypa.io/) |
| Lint/Format | [ruff](https://docs.astral.sh/ruff/) |
| Type Check | [mypy](https://mypy-lang.org/) (strict) |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov + pytest-benchmark |

---

## Roadmap

- **v0.1.0** — Current: reference implementation, 402 tests, ≥90% coverage
- **v0.2.0** — Triton kernel implementation, FAISS quantizer, k-means init
- **v1.0.0** — Stable API, PyPI release, full spec compliance

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, PR process,
and coding standards.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## License

[Apache License 2.0](LICENSE)

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
