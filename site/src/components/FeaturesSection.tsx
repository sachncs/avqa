import { ArrowUpRight } from "lucide-react";
import { FadeIn } from "./FadeIn";

const CAPABILITIES = [
  {
    group: "Core method",
    title: "Hierarchical key codebook",
    description:
      "Keys are assigned to child codewords beneath parent representations, enabling a coarse pass before selected groups are expanded.",
  },
  {
    group: "Core method",
    title: "Configurable parent routing",
    description:
      "Top-p, threshold, and budget routers expose the selection policy. Implementations can register additional router strategies.",
  },
  {
    group: "Optional research paths",
    title: "HVAQ temperature schedules",
    description:
      "Entropy- or linear-based temperature scaling modifies parent attention logits; it is a research option, not a default guarantee.",
  },
  {
    group: "Optional research paths",
    title: "Multi-pass correction",
    description:
      "ACMPR can refine disjoint parent groups over successive passes using a decaying budget, as specified in the repository method notes.",
  },
  {
    group: "Implementation",
    title: "Readable PyTorch reference",
    description:
      "The online-softmax path is intended for inspection, experiments, and CPU correctness comparisons—not as a tuned production kernel.",
  },
  {
    group: "Implementation",
    title: "Experimental adaptation and compile",
    description:
      "BCAR and torch.compile are optional paths with workload and environment limits. Neither is presented as a validated performance win.",
  },
];

export function FeaturesSection() {
  return (
    <section
      id="implementation"
      className="relative scroll-mt-24 py-20 sm:py-28"
      aria-labelledby="implementation-heading"
    >
      <div className="container-edge">
        <div className="grid gap-10 border-t border-white/15 pt-6 lg:grid-cols-12 lg:gap-12">
          <FadeIn className="lg:col-span-4">
            <span className="eyebrow">Implementation map</span>
            <h2
              id="implementation-heading"
              className="mt-5 max-w-[13ch] font-display text-3xl font-semibold leading-tight tracking-[-0.045em] text-white sm:text-4xl"
            >
              Separate the method from the experiments.
            </h2>
            <p className="mt-5 max-w-lg text-sm leading-6 text-ink-300">
              The core path, configurable choices, and optional research
              components have different maturity. The labels here are intended
              to make that boundary legible.
            </p>
            <a
              href="/avqa/docs/#api"
              className="mt-5 inline-flex min-h-10 items-center gap-1 text-sm text-ink-100 underline decoration-white/30 underline-offset-4 hover:decoration-accent-300"
            >
              Browse API and configuration
              <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
            </a>
          </FadeIn>

          <FadeIn delay={0.06} className="lg:col-span-8">
            <ol className="border-y border-white/10">
              {CAPABILITIES.map((item, index) => (
                <li
                  key={item.title}
                  className="grid gap-x-5 gap-y-2 border-b border-white/10 py-5 last:border-b-0 sm:grid-cols-[42px_1fr_1.45fr] sm:items-start sm:py-6"
                >
                  <span className="pt-0.5 font-mono text-[11px] text-accent-300">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <div>
                    <span className="font-mono text-[10px] uppercase tracking-[0.11em] text-ink-500">
                      {item.group}
                    </span>
                    <h3 className="mt-1 text-sm font-semibold leading-5 text-ink-100">
                      {item.title}
                    </h3>
                  </div>
                  <p className="text-sm leading-6 text-ink-300">
                    {item.description}
                  </p>
                </li>
              ))}
            </ol>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
