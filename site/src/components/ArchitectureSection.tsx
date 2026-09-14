import { FadeIn } from "./FadeIn";

const LAYERS = [
  {
    name: "Applications",
    items: ["Your code", "nn.Module wrappers"],
    accent: "rgba(124,140,255,0.4)",
  },
  {
    name: "Public API",
    items: ["AVQAttention", "attention()", "AVQConfig"],
    accent: "rgba(124,140,255,0.55)",
  },
  {
    name: "Pipeline",
    items: ["Refinement", "Backend", "Routing", "Merge", "Quantizer"],
    accent: "rgba(125,249,255,0.5)",
  },
  {
    name: "Codebook",
    items: ["HierarchicalCodebook"],
    accent: "rgba(125,249,255,0.55)",
  },
  {
    name: "Foundations",
    items: ["Config", "Data", "Utils", "Cache", "Scheduler", "Profiling"],
    accent: "rgba(255,255,255,0.4)",
  },
];

export function ArchitectureSection() {
  return (
    <section id="architecture" className="relative py-28 sm:py-36">
      <div className="absolute inset-x-0 top-0 -z-10 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
      <div className="container-edge">
        <FadeIn className="max-w-3xl">
          <span className="eyebrow">Architecture</span>
          <h2 className="mt-5 font-display text-[36px] font-semibold leading-[1.05] tracking-tightest text-white sm:text-[52px]">
            A layered stack with{" "}
            <span className="text-gradient-accent">clean boundaries.</span>
          </h2>
          <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-ink-300">
            Every backend implements the same interface. Every router
            implements the same selector contract. The core algorithm never
            imports profiling or visualization — so you can observe, replace,
            or extend without touching the math.
          </p>
        </FadeIn>

        <FadeIn delay={0.1} className="mt-14">
          <div className="surface relative overflow-hidden rounded-3xl p-6 sm:p-10">
            <div className="flex flex-col gap-3">
              {LAYERS.map((layer, i) => (
                <div
                  key={layer.name}
                  className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-ink-950/40 p-4 sm:p-5"
                  style={{
                    borderLeft: `2px solid ${layer.accent}`,
                  }}
                >
                  <div
                    className="pointer-events-none absolute inset-y-0 left-0 -z-10 opacity-30"
                    style={{
                      width: "60%",
                      background: `linear-gradient(90deg, ${layer.accent} 0%, transparent 100%)`,
                    }}
                  />
                  <div className="flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-ink-500">
                        L{i}
                      </span>
                      <span className="font-display text-base font-semibold text-white sm:text-lg">
                        {layer.name}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {layer.items.map((item) => (
                        <span
                          key={item}
                          className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1 font-mono text-[11px] text-ink-200"
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-8 flex flex-col gap-2 text-[12px] text-ink-400 sm:flex-row sm:items-center sm:gap-6">
              <div className="inline-flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-accent-300" />
                Dependency rule: core never imports profiling or visualization.
              </div>
              <div className="inline-flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-glow" />
                Extension points: Backend, Router, Scheduler, Merge.
              </div>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}