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
            Vanilla attention scales with the square of sequence length. At 4k
            tokens the compute graph dominates every other op. At 32k it's the
            only thing in your trace. Memory grows the same way — and your
            GPUs stop being useful long before your model stops being capable.
          </p>
        </FadeIn>

        <div className="mt-16 grid gap-6 lg:grid-cols-3">
          {[
            {
              label: "Compute",
              k: "O(N²)",
              title: "Quadratic FLOPs",
              desc: "Every query attends to every key. Long context isn't just slower — it's mathematically wasteful when most interactions are low-entropy.",
            },
            {
              label: "Memory",
              k: "N²",
              title: "Attention maps dominate",
              desc: "KV caches and intermediate scores balloon with sequence length. Paged caches help, but the wall-clock cost remains.",
            },
            {
              label: "Entropy",
              k: "≈ 0",
              title: "Most attention is noise",
              desc: "Empirically, only a handful of codewords carry the signal. The rest is precision you pay for and never use.",
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
                <h3 className="mt-10 font-display text-xl font-semibold text-white">{c.title}</h3>
                <p className="mt-3 text-[15px] leading-relaxed text-ink-300">{c.desc}</p>
                <div className="mt-8 h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}