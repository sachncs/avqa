# Compatibility and support matrix

AVQA is a public alpha. This matrix describes the tested baseline, not a
promise that every model or deployment environment will behave identically.

| Area | Baseline | Notes |
|------|----------|-------|
| Python | 3.10–3.15 | CI validates 3.10–3.14. Python 3.15 is declared compatible, but has not been runtime-tested because compatible PyTorch wheels are not available. |
| PyTorch | 2.1+ | Verify newer releases against the reference tests. PyTorch currently limits which Python versions have installable wheels; Python 3.15 is not in the validated runtime matrix. |
| Execution | Pure PyTorch TorchBackend | Vendor kernels are not part of the core alpha package. |
| Devices | CPU validated; CUDA code paths unvalidated in CI | GPU correctness and performance require validation on the target environment. |
| Optional visualization | matplotlib, graphviz | Install with the `viz` extra. |
| Framework adapters | Not bundled | Write and pin adapters for Hugging Face, vLLM, FlashAttention, or xFormers separately. |
| Distribution | Source install | PyPI publication is a future release milestone. |

When reporting a compatibility issue, include the AVQA revision, Python,
PyTorch, operating system, device, dtype, tensor shapes, configuration, and a
minimal reproduction.
