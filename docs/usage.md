# Build with AVQA

AVQA is a public-alpha research implementation. This guide covers a CPU-first
install, the module and functional APIs, tensor contracts, runtime boundaries,
and issue diagnostics. Read [`compatibility.md`](compatibility.md) before
choosing Python/PyTorch versions.

## Install

The package currently installs from a Git checkout; it is not published on
PyPI. Python 3.10–3.15 is declared and tested on the CPU CI matrix. Python 3.15
is still a prerelease interpreter, so the CI lane uses upstream prerelease
PyTorch wheels when available. Check the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for wheel availability for your Python version and platform.
Current upstream PyTorch Python 3.15 wheels reject `torch.compile`; if compile
is enabled in the AVQA config, AVQA warns and uses eager execution instead.

```bash
git clone https://github.com/sachncs/avqa.git
cd avqa
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The `viz` extra installs optional visualization dependencies. AVQA does not
install a CUDA build on your behalf; follow PyTorch's installation selector for
your own environment. CUDA execution, CUDA/Triton numerical equivalence, and
GPU performance have not been tested in a CUDA environment.

## First forward pass

`AVQAttention` accepts query, key, and value tensors in `[batch, tokens,
embedding]` layout. The example uses a 64-wide embedding and returns a tensor
with the query's shape:

```python
import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig

config = AVQConfig(
    attention=AttentionShapeConfig(embed_dim=64, num_heads=4, head_dim=16),
    codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),
    routing=RoutingConfig(refinement_budget=3),
)
attention = AVQAttention(config, in_proj=False, out_proj=False)
query = torch.randn(2, 8, 64)
key = torch.randn(2, 16, 64)
value = torch.randn(2, 16, 64)
output = attention(query, key, value)
assert output.shape == query.shape
```

Keep batch, device, and dtype consistent across inputs. `embed_dim` must match
the last tensor dimension; `num_heads * head_dim` must match that embedding.
The module API can optionally project inputs when constructed with projections
enabled; consult the constructor docstring for its exact initialization and
projection behavior.

## Functional API

Use the functional API for stateless calls. It accepts the same rank-3 layout
and configuration as the module:

```python
from avqa.functional import attention

output = attention(query=query, key=key, value=value, config=config)
```

The functional API accepts an optional mutable KV cache. If supplied, cache
state is updated in place; do not share that cache across independent
sequences. See [KV cache and decoding](#kv-cache-and-decoding).

## Configuration

`AVQConfig` contains nested configuration objects. Start with defaults and
change a small number of options at a time. `AttentionShapeConfig` defines the
embedding/head dimensions; `CodebookConfig` defines parent and child counts;
`RoutingConfig` controls the refinement budget and strategy. Further options
are defined in `src/avqa/config.py` and validated during construction. Use
`to_dict()` / `from_dict()` for JSON-compatible configuration persistence;
invalid or unknown fields raise `ConfigurationError` rather than being
silently ignored.

Keep a serialized configuration with every experiment. A changed codebook,
budget, precision, or adaptation setting changes either algorithm behavior or
the amount of work performed.

## KV cache and decoding

`InMemoryKVCache` and `PagedKVCache` store mutable rank-4 tensors shaped
`[batch, heads, tokens, head_dim]`. For example, a 4-head key/value append with
head dimension 16 and 4 new tokens has shape `[1, 4, 4, 16]`. Cache structural
dimensions, device, and dtype must remain consistent. Call the cache's
documented reset method when starting a new sequence. `state_dict()` and
`load_state_dict()` allow checkpoint persistence; restore into a cache created
with matching structural settings.

The precise cache method signatures and failure behavior are documented in
`src/avqa/cache.py`. Paged caching changes storage organization, not the
supported framework integration surface.

## Troubleshooting

- **Shape errors:** verify rank-3 `[B,T,E]` attention inputs and the configured
  embedding/head dimensions. Cache append inputs use rank 4 `[B,H,T,D]`.
- **Configuration errors:** inspect the raised `ConfigurationError` and
  preserve the complete config when filing an issue.
- **Unexpected numerics:** reproduce with CPU, a fixed seed, finite inputs, and
  the same dtype and configuration as the reference tests.
- **Slow execution:** capture sequence lengths, batch, heads, dtype, backend,
  warmups, repetitions, and AVQA/PyTorch versions. The reference backend is not
  an optimized serving kernel; timing depends on workload and hardware.
- **CUDA problems:** GPU behavior has not been validated in a CUDA environment.
  Report the GPU model, driver, CUDA runtime, PyTorch build, shapes, dtype, and
  a minimal reproduction; do not interpret CPU checks as CUDA evidence.

For support channels, see [`SUPPORT.md`](../SUPPORT.md). For benchmark records,
see [`BENCHMARKS.md`](../BENCHMARKS.md) and the [reproducibility guide](../BENCHMARKS.md#reproducibility-record).
