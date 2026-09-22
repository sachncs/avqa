import { FadeIn } from "./FadeIn";

const COMPARISON = [
  {
    step: "Dense attention",
    work: "Scores query–key pairs across the sequence.",
    consequence: "Direct, dense reference computation.",
  },
  {
    step: "AVQA hypothesis",
    work: "Scores coarse key groups, then refines routed groups.",
    consequence: "Trades a smaller candidate set for approximation error.",
  },
  {
    step: "What to measure",
    work: "Quality, end-to-end latency, and memory on the target workload.",
    consequence: "No general speed or memory advantage is implied.",
  },
];

export function WhySection() {
  return (
    <section id="method" className="relative scroll-mt-24 py-20 sm:py-28">
      <div className="container-edge">
        <div className="grid gap-10 border-t border-white/15 pt-6 lg:grid-cols-12 lg:gap-12">
          <FadeIn className="lg:col-span-4">
            <span className="eyebrow">The research question</span>
            <h2 className="mt-5 max-w-[14ch] font-display text-3xl font-semibold leading-tight tracking-[-0.045em] text-white sm:text-4xl">
              Can coarse routing preserve useful attention?
            </h2>
            <p className="mt-5 max-w-lg text-sm leading-6 text-ink-300">
              Dense attention evaluates query–key interactions across the
              sequence. AVQA explores a hierarchical alternative: allocate finer
              work only after a coarse pass identifies candidate groups.
            </p>
          </FadeIn>

          <FadeIn delay={0.06} className="lg:col-span-8">
            <div className="border-y border-white/10">
              <div className="hidden grid-cols-[1fr_1.35fr_1.2fr] gap-5 border-b border-white/10 py-3 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-500 sm:grid">
                <span>Approach</span>
                <span>Compute path</span>
                <span>Interpretation</span>
              </div>
              {COMPARISON.map((item, index) => (
                <div
                  key={item.step}
                  className="grid gap-2 border-b border-white/10 py-5 last:border-b-0 sm:grid-cols-[1fr_1.35fr_1.2fr] sm:gap-5"
                >
                  <div className="flex items-start gap-3">
                    <span className="pt-0.5 font-mono text-[10px] text-accent-300">
                      0{index + 1}
                    </span>
                    <span className="text-sm font-semibold text-ink-100">
                      {item.step}
                    </span>
                  </div>
                  <p className="text-sm leading-6 text-ink-200">{item.work}</p>
                  <p className="text-sm leading-6 text-ink-400">
                    {item.consequence}
                  </p>
                </div>
              ))}
            </div>
            <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.08em] text-ink-500">
              Attention scores depend on the query, key, model, and routing
              configuration.
            </p>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
