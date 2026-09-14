import { FadeIn } from "./FadeIn";
import { PipelineVisual } from "./visuals/PipelineVisual";

export function SolutionSection() {
  return (
    <section className="relative py-28 sm:py-36">
      <div className="container-edge">
        <div className="grid gap-12 lg:grid-cols-12 lg:gap-16">
          <FadeIn className="lg:col-span-5">
            <span className="eyebrow">The solution</span>
            <h2 className="mt-5 font-display text-[36px] font-semibold leading-[1.05] tracking-tightest text-white sm:text-[52px]">
              Route attention{" "}
              <span className="text-gradient-accent">through a codebook.</span>
            </h2>
            <p className="mt-6 text-[17px] leading-relaxed text-ink-300">
              Quantize keys into a hierarchical codebook. Compute attention at
              the codebook level first. Then refine — only the most-attended
              codewords, only the children that actually matter.
            </p>
            <p className="mt-4 text-[17px] leading-relaxed text-ink-300">
              You stop paying for the long tail of low-entropy interactions
              and start paying for signal.
            </p>
          </FadeIn>

          <FadeIn delay={0.15} className="lg:col-span-7">
            <div className="surface relative overflow-hidden rounded-3xl p-2 ring-glow">
              <div className="absolute inset-x-0 top-0 -z-10 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
              <div className="rounded-2xl bg-ink-950/60 p-6 sm:p-10">
                <PipelineVisual className="h-auto w-full" />
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}