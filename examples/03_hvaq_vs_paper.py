"""HVAQ on/off equivalence comparison.

Usage::

    python examples/03_hvaq_vs_paper.py
"""

from __future__ import annotations

import torch

from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    BackendConfig,
    CodebookConfig,
    HopfieldConfig,
    RoutingConfig,
)


def main() -> None:
    """Compare HVAQ (adaptive='entropy') to the paper-equivalent ('none')."""
    torch.manual_seed(0)
    base = {
        "attention": AttentionShapeConfig(embed_dim=128, num_heads=4, head_dim=32),
        "codebook": CodebookConfig(num_codewords=32, children_per_codeword=4),
        "routing": RoutingConfig(refinement_budget=4),
        "backend": BackendConfig(hopfield=True),
    }
    cfg_paper = AVQConfig(**base, hopfield=HopfieldConfig(enabled=True, adaptive="none"))
    cfg_hvaq = AVQConfig(**base, hopfield=HopfieldConfig(enabled=True, adaptive="entropy"))

    mod_paper = AVQAttention(cfg_paper, in_proj=False, out_proj=False).eval()
    mod_hvaq = AVQAttention(cfg_hvaq, in_proj=False, out_proj=False).eval()
    mod_hvaq.load_state_dict(mod_paper.state_dict(), strict=False)

    q = torch.randn(2, 16, 128)
    k = torch.randn(2, 16, 128)
    v = torch.randn(2, 16, 128)
    with torch.no_grad():
        out_paper = mod_paper(q, k, v)
        out_hvaq = mod_hvaq(q, k, v)
    print(f"paper output shape: {tuple(out_paper.shape)}")
    print(f"hvaq  output shape: {tuple(out_hvaq.shape)}")
    print(f"|paper - hvaq|.max(): {(out_paper - out_hvaq).abs().max().item():.6f}")


if __name__ == "__main__":
    main()
