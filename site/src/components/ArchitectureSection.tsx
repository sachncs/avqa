import { ArrowRight, ArrowUpRight } from "lucide-react";
import { FadeIn } from "./FadeIn";

const LAYERS = [
  {
    number: "01",
    title: "Public surface",
    pieces: ["AVQAttention", "attention()", "AVQConfig"],
    detail: "Module and functional entry points",
  },
  {
    number: "02",
    title: "Attention pipeline",
    pieces: ["Quantize", "Score", "Route", "Refine", "Merge"],
    detail: "Online-softmax state and parent correction",
  },
  {
    number: "03",
    title: "Core structures",
    pieces: ["Codebook", "Shapes", "Validation", "Cache"],
    detail: "Tensor data, configuration, and invariants",
  },
];

const EXTENSIONS = ["Backend", "Router", "Merge strategy", "Scheduler"];

export function ArchitectureSection() {
  return (
    <section
      id="architecture"
      className="relative scroll-mt-24 py-20 sm:py-28"
      aria-labelledby="architecture-heading"
    >
      <div className="container-edge">
        <div className="grid gap-10 border-t border-white/15 pt-6 lg:grid-cols-12 lg:gap-12">
          <FadeIn className="lg:col-span-4">
            <span className="eyebrow">Architecture</span>
            <h2
              id="architecture-heading"
              className="mt-5 max-w-[13ch] font-display text-3xl font-semibold leading-tight tracking-[-0.045em] text-white sm:text-4xl"
            >
              A small public surface; explicit seams.
            </h2>
            <p className="mt-5 max-w-lg text-sm leading-6 text-ink-300">
              The implementation is organized around the call path. Extension
              registries exist for selected strategies; the current CPU
              reference remains the primary validated execution path.
            </p>
            <a
              href="/avqa/docs/#architecture"
              className="mt-5 inline-flex min-h-10 items-center gap-1 text-sm text-ink-100 underline decoration-white/30 underline-offset-4 hover:decoration-accent-300"
            >
              Open the architecture guide
              <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
            </a>
          </FadeIn>

          <FadeIn delay={0.06} className="lg:col-span-8">
            <div className="grid border-y border-white/10 md:grid-cols-[1fr_auto_1.25fr_auto_1fr] md:items-stretch">
              {LAYERS.map((layer, index) => (
                <div key={layer.number} className="contents">
                  <section className="py-5 md:px-4 md:py-6 first:md:pl-0 last:md:pr-0">
                    <p className="font-mono text-[10px] text-accent-300">
                      {layer.number} <span className="text-ink-500">/</span>
                    </p>
                    <h3 className="mt-2 font-display text-base font-semibold text-white">
                      {layer.title}
                    </h3>
                    <p className="mt-1 text-xs leading-5 text-ink-400">
                      {layer.detail}
                    </p>
                    <ul className="mt-4 space-y-2">
                      {layer.pieces.map((piece) => (
                        <li
                          key={piece}
                          className="border-l border-white/15 pl-2 font-mono text-[10px] text-ink-200"
                        >
                          {piece}
                        </li>
                      ))}
                    </ul>
                  </section>
                  {index < LAYERS.length - 1 && (
                    <div
                      className="flex items-center justify-center border-y border-white/10 py-1 text-ink-500 md:border-y-0"
                      aria-hidden="true"
                    >
                      <ArrowRight className="h-4 w-4 rotate-90 md:rotate-0" />
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-[140px_1fr]">
              <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-500">
                Registered strategies
              </p>
              <div className="flex flex-wrap gap-x-5 gap-y-2">
                {EXTENSIONS.map((extension) => (
                  <span
                    key={extension}
                    className="inline-flex items-center gap-2 text-xs text-ink-200"
                  >
                    <span className="h-1.5 w-1.5 bg-accent-400" />
                    {extension}
                  </span>
                ))}
              </div>
            </div>
            <p className="mt-5 border-l border-glow/60 pl-3 text-xs leading-5 text-ink-400">
              Core attention does not import profiling or visualization; those
              modules observe the execution path. See the
              <a
                href="https://github.com/sachncs/avqa/blob/main/docs/architecture.md"
                target="_blank"
                rel="noreferrer"
                className="ml-1 text-ink-100 underline decoration-white/30 underline-offset-4 hover:decoration-accent-300"
              >
                source architecture notes
              </a>
              .
            </p>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
