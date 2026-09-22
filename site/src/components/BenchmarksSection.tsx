import { ArrowUpRight } from "lucide-react";
import { FadeIn } from "./FadeIn";

const LOSS = [
  {
    label: "Static codebook",
    value: 15.8983,
    width: "100%",
    tone: "bg-ink-500",
  },
  {
    label: "BCAR · adapted",
    value: 4.8974,
    width: "30.8%",
    tone: "bg-accent-400",
  },
  { label: "Oracle codebook", value: 0.01, width: "1px", tone: "bg-signal" },
];

export function BenchmarksSection() {
  return (
    <section
      id="evidence"
      className="relative scroll-mt-24 py-20 sm:py-28"
      aria-labelledby="evidence-heading"
    >
      <div className="container-edge">
        <div className="grid gap-10 border-t border-white/15 pt-6 lg:grid-cols-12 lg:gap-12">
          <FadeIn className="lg:col-span-4">
            <span className="eyebrow">Evidence and limits</span>
            <h2
              id="evidence-heading"
              className="mt-5 max-w-[13ch] font-display text-3xl font-semibold leading-tight tracking-[-0.045em] text-white sm:text-4xl"
            >
              Read the experiment, not a promise.
            </h2>
            <p className="mt-5 max-w-lg text-sm leading-6 text-ink-300">
              The checked-in EXP-0004 CPU artifact measures codebook
              reconstruction loss during synthetic online adaptation. It is not
              an end-to-end attention latency or quality benchmark.
            </p>
            <p className="mt-4 max-w-lg text-sm leading-6 text-ink-400">
              CUDA execution, CUDA/Triton numerical equivalence, and GPU
              performance have not been tested in a CUDA environment.
            </p>
            <a
              href="https://github.com/sachncs/avqa/tree/main/benchmarks/raw/EXP-0004"
              target="_blank"
              rel="noreferrer"
              className="mt-5 inline-flex min-h-10 items-center gap-1 text-sm text-ink-100 underline decoration-white/30 underline-offset-4 hover:decoration-accent-300"
            >
              Inspect raw data and configuration
              <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
            </a>
          </FadeIn>

          <FadeIn delay={0.06} className="lg:col-span-8">
            <div className="grid gap-8 xl:grid-cols-[1.3fr_0.7fr]">
              <figure
                aria-labelledby="bcar-chart-title"
                className="border-y border-white/15 py-5 sm:py-6"
              >
                <figcaption id="bcar-chart-title">
                  <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-400">
                    EXP-0004 · VQ reconstruction loss
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    After 1,024 adaptation updates
                  </p>
                  <p className="mt-1 text-xs text-ink-400">
                    Lower is better · synthetic stream · dimensionless loss
                  </p>
                </figcaption>

                <div
                  className="mt-7 space-y-5"
                  role="img"
                  aria-label="At 1,024 updates, static codebook loss is 15.8983, BCAR loss is 4.8974, and oracle codebook loss is 0.0100."
                >
                  {LOSS.map((item) => (
                    <div key={item.label}>
                      <div className="flex items-baseline justify-between gap-4 text-xs">
                        <span className="text-ink-200">{item.label}</span>
                        <span className="font-mono tabular-nums text-white">
                          {item.value.toFixed(4)}
                        </span>
                      </div>
                      <div className="mt-2 h-2 bg-ink-800">
                        <div
                          className={`h-full ${item.tone}`}
                          style={{ width: item.width }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div
                  aria-hidden="true"
                  className="mt-3 flex justify-between border-t border-white/15 pt-2 font-mono text-[10px] tabular-nums text-ink-500"
                >
                  <span>0</span>
                  <span>4</span>
                  <span>8</span>
                  <span>12</span>
                  <span>16</span>
                </div>
                <p className="mt-1 text-right font-mono text-[10px] uppercase tracking-[0.08em] text-ink-500">
                  Loss
                </p>

                <div className="mt-6 flex flex-wrap items-baseline justify-between gap-3 border-t border-white/10 pt-4">
                  <p className="font-display text-2xl font-semibold tracking-tight text-white">
                    69.2%
                  </p>
                  <p className="text-right text-xs leading-5 text-ink-400">
                    lower VQ loss than static at this checkpoint; not a speedup
                    measurement
                  </p>
                </div>
              </figure>

              <div className="py-5 sm:py-6">
                <h3 className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-400">
                  Experiment scope
                </h3>
                <dl className="mt-4 divide-y divide-white/10 border-y border-white/10 text-xs">
                  {[
                    ["Codewords", "4 parents × 2 children"],
                    ["Attention heads", "1"],
                    ["Head dimension", "8"],
                    ["Tokens per update", "16"],
                    ["Runtime", "CPU"],
                    ["Artifact", "EXP-0004 / raw.json"],
                  ].map(([term, value]) => (
                    <div
                      key={term}
                      className="flex items-start justify-between gap-3 py-3"
                    >
                      <dt className="text-ink-400">{term}</dt>
                      <dd className="text-right font-mono text-ink-100">
                        {value}
                      </dd>
                    </div>
                  ))}
                </dl>
                <p className="mt-4 text-xs leading-5 text-ink-400">
                  Reproduce the adaptation task before interpreting the loss
                  change. The oracle is a reference codebook, not a deployable
                  runtime baseline.
                </p>
              </div>
            </div>

            <div className="mt-7 grid gap-4 border-t border-white/10 pt-5 sm:grid-cols-[1fr_1fr]">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-500">
                  Dense attention
                </p>
                <p className="mt-2 font-mono text-sm text-ink-100">O(N²D)</p>
                <p className="mt-1 text-xs leading-5 text-ink-400">
                  Reference score work across N tokens at dimension D.
                </p>
              </div>
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-500">
                  AVQA method model
                </p>
                <p className="mt-2 font-mono text-sm text-ink-100">
                  O(N(M₀ + PC)D)
                </p>
                <p className="mt-1 text-xs leading-5 text-ink-400">
                  M₀ parent groups, P routed parents, C children per parent.
                  Formal cost is not measured latency.
                </p>
              </div>
            </div>
            <a
              href="/avqa/docs/#reproducibility"
              className="mt-4 inline-flex min-h-10 items-center gap-1 text-xs text-ink-200 underline decoration-white/25 underline-offset-4 hover:decoration-accent-300"
            >
              Benchmark protocol and interpretation
              <ArrowUpRight aria-hidden="true" className="h-3 w-3" />
            </a>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
