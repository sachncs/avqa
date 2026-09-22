import { ArrowDownRight, ArrowRight, Github } from "lucide-react";
import { HeroVisual } from "./visuals/HeroVisual";
import { DOCS_URL, GITHUB_URL, RELEASE_LABEL } from "../lib/links";

export function Hero() {
  return (
    <section
      id="top"
      className="relative isolate overflow-hidden pb-16 pt-28 sm:pb-20 sm:pt-36 lg:pb-24"
    >
      <div className="absolute inset-x-0 top-0 -z-10 h-[38rem] bg-grid-faint [background-size:56px_56px] [mask-image:linear-gradient(to_bottom,black,transparent)]" />

      <div className="container-edge">
        <div className="grid grid-cols-1 gap-12 xl:grid-cols-12 xl:items-center xl:gap-14">
          <div className="xl:col-span-5">
            <div className="eyebrow">
              <span className="h-1.5 w-1.5 bg-accent-400" />
              Adaptive vector-quantized attention
            </div>

            <h1 className="mt-7 max-w-[12ch] font-display text-[clamp(2.8rem,6.5vw,5.7rem)] font-semibold leading-[0.98] tracking-[-0.065em] text-white">
              Spend compute where attention goes.
            </h1>

            <p className="prose-measure mt-7 text-base leading-7 text-ink-200 sm:text-lg sm:leading-8">
              AVQA is an inspectable PyTorch implementation of hierarchical,
              codebook-routed attention. It groups keys, scores groups at a
              coarse level, then refines selected regions. The result is an
              approximation to study—not a claim of faster attention.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <a href={`${DOCS_URL}#quickstart`} className="btn-primary">
                Start with the CPU quickstart <ArrowRight className="h-4 w-4" />
              </a>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                className="btn-ghost"
              >
                <Github aria-hidden="true" className="h-4 w-4" /> Source code
              </a>
            </div>

            <div className="mt-9 border-t border-white/10 pt-4">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-400">
                {RELEASE_LABEL} <span className="px-1.5 text-ink-600">/</span>
                Python 3.10–3.15
                <span className="px-1.5 text-ink-600">/</span> CPU validation
                <span className="px-1.5 text-ink-600">/</span> Apache 2.0
              </p>
              <p className="mt-3 max-w-[60ch] text-xs leading-5 text-ink-300">
                CUDA execution, CUDA/Triton numerical equivalence, and GPU
                performance have not been tested in a CUDA environment. Python
                3.15 is prerelease; see the support matrix for details.
              </p>
            </div>
          </div>

          <div className="xl:col-span-7">
            <div className="border-y border-white/15 bg-ink-900/45 px-4 py-4 sm:px-6 sm:py-6">
              <div className="mb-5 flex items-center justify-between gap-4">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-400">
                    The routing idea
                  </p>
                  <p className="mt-1 text-sm text-ink-100">
                    From grouped keys to selective refinement
                  </p>
                </div>
                <a
                  href="#method"
                  className="inline-flex min-h-10 items-center gap-2 font-mono text-[10px] uppercase tracking-[0.08em] text-ink-300 transition-colors hover:text-white"
                >
                  Read the method <ArrowDownRight className="h-4 w-4" />
                </a>
              </div>
              <HeroVisual />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
