import { FadeIn } from "./FadeIn";

export function WhySection() {
  return (
    <section id="why" className="relative py-28 sm:py-36">
      <div className="container-edge">
        <FadeIn className="max-w-3xl">
          <span className="eyebrow">The problem</span>
          <h2 className="mt-5 font-display text-[36px] font-semibold leading-[1.05] tracking-tightest text-white sm:text-[52px]">
            Long context has an{" "}
            <span className="text-gradient-accent">attention wall.</span>
          </h2>
          <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-ink-300">
            Dense attention evaluates query–key interactions across the input
            sequence. AVQA explores grouping keys first, then spending extra
            work on routed groups. That trades compute for approximation error;
            the result depends on the model, sequence, configuration, and
            hardware.
          </p>
        </FadeIn>

        <div className="mt-16 grid gap-6 lg:grid-cols-3">
          {[
            {
              label: "Compute",
              k: "O(N²)",
              title: "Pairwise interactions",
              desc: "Dense attention computes scores across query and key positions. AVQA's codebook path changes which interactions are evaluated at each refinement stage.",
            },
            {
              label: "Memory",
              k: "Implementation-dependent",
              title: "Memory depends on the kernel",
              desc: "A materialized score matrix grows with query and key lengths; memory-efficient kernels can avoid storing it. KV caches grow with stored context length.",
            },
            {
              label: "Entropy",
              k: "Research question",
              title: "Route, then measure error",
              desc: "The method tests whether coarse grouping and selective refinement can preserve useful attention. Quality and speed must be measured for each workload.",
            },
          ].map((c, i) => (
            <FadeIn key={c.label} delay={i * 0.07}>
              <div className="surface surface-hover group h-full rounded-2xl p-7">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-medium uppercase tracking-[0.2em] text-ink-500">
                    {c.label}
                  </span>
                  <span className="font-mono text-xs text-ink-400">{c.k}</span>
                </div>
                <h3 className="mt-10 font-display text-xl font-semibold text-white">
                  {c.title}
                </h3>
                <p className="mt-3 text-[15px] leading-relaxed text-ink-300">
                  {c.desc}
                </p>
                <div className="mt-8 h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
