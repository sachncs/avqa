import { FadeIn } from "./FadeIn";

const ROWS = [
  {
    method: "vanilla attention",
    desc: "O(N²) · dense softmax",
    cost: "O(N²)",
    scaling: "Quadratic",
    highlight: false,
  },
  {
    method: "AVQA — single pass",
    desc: "Quantize → route → refine",
    cost: "≈ O(N · K · log C)",
    scaling: "Sub-quadratic",
    highlight: true,
  },
  {
    method: "AVQA — multi-pass (ACMPR)",
    desc: "Disjoint-set re-routing",
    cost: "≈ O(N · K · log C · P)",
    scaling: "Bounded by budget",
    highlight: true,
  },
];

export function BenchmarksSection() {
  return (
    <section id="benchmarks" className="relative py-28 sm:py-36">
      <div className="absolute inset-0 -z-10 bg-radial-fade opacity-50" />
      <div className="container-edge">
        <FadeIn className="max-w-3xl">
          <span className="eyebrow">Benchmarks</span>
          <h2 className="mt-5 font-display text-[36px] font-semibold leading-[1.05] tracking-tightest text-white sm:text-[52px]">
            Correctness first.{" "}
            <span className="text-gradient-accent">Always.</span>
          </h2>
          <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-ink-300">
            AVQA ships a strict benchmarking protocol. Every result published
            below comes from a deterministic run — warm-up, repetitions,
            platform metadata, raw JSON, and a hand-written summary. No
            cherry-picking.
          </p>
        </FadeIn>

        <div className="mt-14 grid gap-6 lg:grid-cols-12">
          <FadeIn delay={0.05} className="lg:col-span-7">
            <div className="surface overflow-hidden rounded-2xl">
              <div className="border-b border-white/5 px-5 py-4">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] uppercase tracking-widest text-ink-400">
                    Complexity comparison
                  </span>
                  <span className="font-mono text-[10px] text-ink-500">theoretical</span>
                </div>
              </div>
              <div className="divide-y divide-white/5">
                {ROWS.map((row) => (
                  <div
                    key={row.method}
                    className={`flex items-center justify-between gap-4 px-5 py-4 ${
                      row.highlight ? "bg-accent-400/[0.04]" : ""
                    }`}
                  >
                    <div>
                      <div
                        className={`font-display text-sm font-semibold ${
                          row.highlight ? "text-white" : "text-ink-200"
                        }`}
                      >
                        {row.method}
                      </div>
                      <div className="font-mono text-[11px] text-ink-500">{row.desc}</div>
                    </div>
                    <div className="hidden sm:block text-right">
                      <div className="font-mono text-[12px] text-ink-300">{row.cost}</div>
                      <div className="text-[10px] uppercase tracking-widest text-ink-500">
                        {row.scaling}
                      </div>
                    </div>
                    {row.highlight && (
                      <span className="hidden md:inline-flex items-center gap-1 rounded-full border border-accent-300/30 bg-accent-400/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-accent-200">
                        AVQA
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>

          <FadeIn delay={0.1} className="lg:col-span-5">
            <div className="surface flex h-full flex-col justify-between rounded-2xl p-6">
              <div>
                <div className="font-mono text-[11px] uppercase tracking-widest text-ink-400">
                  BCAR — online adaptation
                </div>
                <p className="mt-3 text-[15px] leading-relaxed text-ink-300">
                  Mean residual reduction against a static codebook on a
                  4k-step toy problem with <span className="font-mono text-accent-300">num_codewords=4</span>,
                  <span className="font-mono text-accent-300"> children_per_codeword=2</span>.
                </p>
              </div>
              <div className="mt-6 grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-white/[0.06] bg-ink-950/40 p-4">
                  <div className="font-display text-3xl font-semibold text-white">61.8%</div>
                  <div className="mt-1 text-[11px] uppercase tracking-widest text-ink-500">
                    at step 1
                  </div>
                </div>
                <div className="rounded-xl border border-accent-300/20 bg-accent-400/[0.05] p-4">
                  <div className="font-display text-3xl font-semibold text-white">69.2%</div>
                  <div className="mt-1 text-[11px] uppercase tracking-widest text-ink-400">
                    steady state
                  </div>
                </div>
              </div>
              <div className="mt-6 font-mono text-[10px] uppercase tracking-widest text-ink-500">
                Source · EXP-0004 (raw.json)
              </div>
            </div>
          </FadeIn>
        </div>

        <FadeIn delay={0.15} className="mt-8">
          <div className="surface rounded-2xl px-5 py-4">
            <div className="flex flex-col gap-3 text-[12px] text-ink-400 sm:flex-row sm:items-center sm:justify-between">
              <div className="inline-flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                ACMPR vs paper · max abs diff ={" "}
                <span className="font-mono text-white">0.0000</span>
              </div>
              <div className="inline-flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                HVAQ-LIN vs paper · max abs diff ={" "}
                <span className="font-mono text-white">0.0000</span>
              </div>
              <a
                href="https://github.com/sachncs/avqa/tree/main/benchmarks"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-ink-200 transition hover:text-white"
              >
                Full benchmark methodology →
              </a>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}