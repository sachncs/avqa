import {
  Layers,
  GitBranch,
  Thermometer,
  Repeat,
  Cpu,
  BookOpen,
} from "lucide-react";
import { FadeIn } from "./FadeIn";

const FEATURES = [
  {
    icon: Layers,
    eyebrow: "Quantization",
    title: "Hierarchical codebook",
    desc: "Mean-constrained parent–child structure with disjoint refinement — coarse at the root, surgical at the leaves.",
    accent: "from-accent-300/20",
  },
  {
    icon: GitBranch,
    eyebrow: "Routing",
    title: "Adaptive refinement",
    desc: "Expand only the most-attended codewords. Top-p, threshold, and budget-aware selectors — all pluggable.",
    accent: "from-glow/20",
  },
  {
    icon: Thermometer,
    eyebrow: "HVAQ",
    title: "Hopfield temperature schedules",
    desc: "Per-query entropy or linear temperature, with learnable parameters — control sharpness exactly where you need it.",
    accent: "from-accent-200/20",
  },
  {
    icon: Repeat,
    eyebrow: "Multi-pass",
    title: "Disjoint-set re-routing",
    desc: "Converging residual norms with budget decay. Multiple passes, never the same codeword twice.",
    accent: "from-glow/15",
  },
  {
    icon: Cpu,
    eyebrow: "Performance",
    title: "Pure PyTorch · torch.compile",
    desc: "Online softmax from FlashAttention-2. torch.compile opt-in for reduced Python overhead on CPU and GPU.",
    accent: "from-accent-300/15",
  },
  {
    icon: BookOpen,
    eyebrow: "Engineering",
    title: "Strict typing · full coverage",
    desc: "mypy clean across the core package. 461 tests. ≥90% coverage. Production discipline from day one.",
    accent: "from-glow/10",
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="relative py-28 sm:py-36">
      <div className="container-edge">
        <FadeIn className="max-w-3xl">
          <span className="eyebrow">What's inside</span>
          <h2 className="mt-5 font-display text-[36px] font-semibold leading-[1.05] tracking-tightest text-white sm:text-[52px]">
            Six pieces.{" "}
            <span className="text-gradient-accent">One attention.</span>
          </h2>
          <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-ink-300">
            AVQA is not one algorithm — it's a layered stack of contributions
            from the AVQ-Attention paper, each with its own clear extension
            point.
          </p>
        </FadeIn>

        <div className="mt-14 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <FadeIn key={f.title} delay={i * 0.05}>
                <div className="group relative h-full overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-b from-white/[0.02] to-white/[0.01] p-7 transition-all duration-500 hover:border-white/15 hover:from-white/[0.05] hover:to-white/[0.02]">
                  <div
                    className={`pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-gradient-to-br ${f.accent} to-transparent opacity-0 blur-2xl transition-opacity duration-700 group-hover:opacity-100`}
                  />
                  <div className="relative flex items-center justify-between">
                    <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-400">
                      {f.eyebrow}
                    </span>
                    <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-ink-200 transition group-hover:border-accent-300/40 group-hover:text-white">
                      <Icon className="h-4 w-4" />
                    </span>
                  </div>
                  <h3 className="mt-12 font-display text-xl font-semibold text-white">
                    {f.title}
                  </h3>
                  <p className="mt-3 text-[14.5px] leading-relaxed text-ink-300">
                    {f.desc}
                  </p>
                </div>
              </FadeIn>
            );
          })}
        </div>
      </div>
    </section>
  );
}