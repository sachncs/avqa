"""Replace Hugging Face BERT attention with AVQA on a tiny model.

Usage::

    python examples/03_hf_replacement.py
"""

from __future__ import annotations

import torch
from transformers import AutoModel

from avqa import AVQConfig
from avqa.integrations import replace_attention


def main() -> None:
    """Load a tiny BERT and swap its attention with AVQA."""
    model = AutoModel.from_pretrained("hf-internal-testing/tiny-bert")
    print(f"Loaded {type(model).__name__}")

    report = replace_attention(model, AVQConfig())
    print(
        f"Replaced {report.modules_replaced} attention modules, "
        f"skipped {report.modules_skipped}.",
    )

    token_ids = torch.randint(0, 1000, (1, 8))
    output = model(token_ids).last_hidden_state
    print(f"Forward output shape: {tuple(output.shape)}")
    print(f"Sample output mean:  {output.mean().item():.6f}")


if __name__ == "__main__":
    main()