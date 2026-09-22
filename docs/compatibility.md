# Compatibility and support matrix

AVQA is a public alpha. This matrix describes the tested baseline, not a
promise that every model or deployment environment will behave identically.

| Area | Baseline | Notes |
|------|----------|-------|
| Python | 3.10–3.12 | CI runs the supported Python matrix. |
| PyTorch | 2.1+ | Verify newer releases against the reference tests. |
| Execution | Pure PyTorch TorchBackend | Vendor kernels are not part of the core alpha package. |
| Devices | CPU; CUDA where the installed PyTorch path supports it | GPU performance evidence is not implied by CPU correctness. |
| Optional visualization | matplotlib, graphviz | Install with the `viz` extra. |
| Framework adapters | Not bundled | Write and pin adapters for Hugging Face, vLLM, FlashAttention, or xFormers separately. |
| Distribution | Source install | PyPI publication is a future release milestone. |

When reporting a compatibility issue, include the AVQA revision, Python,
PyTorch, operating system, device, dtype, tensor shapes, configuration, and a
minimal reproduction.
