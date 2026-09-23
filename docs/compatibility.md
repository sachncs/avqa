# Compatibility and support matrix

AVQA is a public alpha. This matrix describes the tested baseline, not a
promise that every model or deployment environment will behave identically.

| Area | Baseline | Notes |
|------|----------|-------|
| Python | 3.10–3.15 | CPU CI runs every listed minor. Python 3.15 is prerelease and uses the upstream prerelease wheel lane. |
| PyTorch | 2.1+ | CPU CI installs the newest compatible wheel available for each interpreter. Verify newer versions against the reference tests. |
| Execution | Pure PyTorch TorchBackend | Vendor kernels are not part of the core alpha package. |
| `torch.compile` | Optional, depends on the installed PyTorch build | Current upstream Python 3.15 prerelease wheels reject compilation. With `compile_enabled=True`, AVQA emits a `RuntimeWarning` and uses the eager path. |
| Devices | CPU validated; CUDA code paths unvalidated in CI | GPU correctness and performance require validation on the target environment. |
| Optional visualization | matplotlib, graphviz | Install with the `viz` extra. |
| Framework adapters | Not bundled | Write and pin adapters for Hugging Face, vLLM, FlashAttention, or xFormers separately. |
| Distribution | Source install | PyPI publication is a future release milestone. |

When reporting a compatibility issue, include the AVQA revision, Python,
PyTorch, operating system, device, dtype, tensor shapes, configuration, and a
minimal reproduction.
