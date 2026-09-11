# AVQA Integrations

This subpackage is intentionally empty in the core AVQA distribution.

Earlier revisions shipped Hugging Face, vLLM, FlashAttention, and
xFormers adapters here; those have been removed because the upstream
packages have version-pinned dependencies that conflict with the AVQA
core toolchain.

## Writing your own adapter

Re-introduce specific adapters as separate distribution extras if you
need them. A minimal skeleton wraps `AVQAttention` to match a target
framework's attention-call signature:

```python
from avqa import AVQAttention, AVQConfig


class FrameworkAttentionWrapper:
    """Adapter template for embedding AVQAttention in another framework.

    Map the framework's attention args (q, k, v, mask, is_causal, ...)
    onto AVQAttention.forward and convert the framework's output
    projection back to the expected container type.
    """

    def __init__(self, config: AVQConfig) -> None:
        self._attention = AVQAttention(config, in_proj=False, out_proj=False)

    def __call__(self, q, k, v, mask=None):
        return self._attention(q, k, v, mask)
```

Place concrete adapters in sibling packages (e.g. `avqa-hf`,
`avqa-vllm`) so each can pin its own upstream dependencies without
constraining the AVQA core.