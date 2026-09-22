import { FadeIn } from "./FadeIn";
import { CodeBlock } from "./CodeBlock";

const MODULE_API = `import torch
from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    RoutingConfig,
)

config = AVQConfig(
    attention=AttentionShapeConfig(embed_dim=512, num_heads=8, head_dim=64),
    codebook=CodebookConfig(num_codewords=64, children_per_codeword=4),
    routing=RoutingConfig(refinement_budget=8),
)

attention = AVQAttention(config, in_proj=False, out_proj=False)

query = torch.randn(2, 64, 512)   # [B, T, E]
key   = torch.randn(2, 128, 512)
value = torch.randn(2, 128, 512)

output = attention(query, key, value)   # [B, T, E]`;

const FUNCTIONAL_API = `from avqa import AVQConfig
from avqa.functional import attention

config = AVQConfig(...)
out = attention(query=q, key=k, value=v, config=config)`;

const HVAQ = `from avqa import AVQConfig
from avqa import HopfieldConfig, RoutingConfig

config = AVQConfig(
    hopfield=HopfieldConfig(
        schedule="entropy",     # or "linear"
        learnable=True,
    ),
    routing=RoutingConfig(refinement_budget=4),
)`;

export function CodeSection() {
  return (
    <section className="relative py-28 sm:py-36">
      <div className="container-edge">
        <FadeIn className="max-w-3xl">
          <span className="eyebrow">In code</span>
          <h2 className="mt-5 font-display text-[36px] font-semibold leading-[1.05] tracking-tightest text-white sm:text-[52px]">
            A reference path.{" "}
            <span className="text-gradient-accent">Pure PyTorch.</span>
          </h2>
          <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-ink-300">
            AVQA exposes an{" "}
            <span className="font-mono text-accent-300">nn.Module</span> and a
            stateless functional API. Use it for research, ablation, and
            correctness comparisons; it is not a drop-in replacement for
            optimized attention kernels.
          </p>
        </FadeIn>

        <div className="mt-14 grid gap-6 lg:grid-cols-12">
          <FadeIn delay={0.05} className="lg:col-span-8">
            <CodeBlock code={MODULE_API} filename="attention.py" />
            <p className="mt-3 text-[13px] text-ink-500">
              Module API — the path most users will take.
            </p>
          </FadeIn>
          <FadeIn delay={0.1} className="space-y-6 lg:col-span-4">
            <div>
              <CodeBlock code={FUNCTIONAL_API} filename="functional.py" />
              <p className="mt-3 text-[13px] text-ink-500">
                Functional API — for stateless calls and tests.
              </p>
            </div>
            <div>
              <CodeBlock
                code={HVAQ}
                filename="hopfield.py"
                showLineNumbers={false}
              />
              <p className="mt-3 text-[13px] text-ink-500">
                HVAQ — per-query temperature, learnable.
              </p>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
