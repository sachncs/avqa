import { ArrowRight, Github, Terminal } from "lucide-react";
import { motion } from "framer-motion";
import { HeroVisual } from "./visuals/HeroVisual";
import { DOCS_URL, GITHUB_URL, RELEASE_LABEL } from "../lib/links";

export function Hero() {
  return (
    <section
      id="top"
      className="relative isolate overflow-hidden pb-24 pt-32 sm:pb-32 sm:pt-40"
    >
      <div className="absolute inset-0 -z-10 bg-radial-fade" />
      <div className="absolute inset-x-0 top-0 -z-10 h-[42rem] bg-grid-faint [background-size:48px_48px] [mask-image:linear-gradient(to_bottom,black,transparent)]" />

      <div className="container-edge">
        <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-12 lg:gap-12">
          <div className="lg:col-span-7">
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="eyebrow"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-glow" />
              Research software <span aria-hidden="true">·</span>{" "}
              {RELEASE_LABEL}
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.8,
                delay: 0.15,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="mt-6 font-display text-[44px] font-semibold leading-[1.02] tracking-tightest text-white sm:text-[64px] lg:text-[78px]"
            >
              Spend compute{" "}
              <span className="text-gradient-accent">
                where attention goes.
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.25 }}
              className="mt-7 max-w-xl text-[17px] leading-relaxed text-ink-300 sm:text-[19px]"
            >
              AVQA is an open-source PyTorch research implementation of
              hierarchical, codebook-routed attention. It explores coarse
              attention followed by selective refinement—with a transparent
              reference path you can inspect and test.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.35 }}
              className="mt-10 flex flex-wrap items-center gap-3"
            >
              <a href={`${DOCS_URL}#quickstart`} className="btn-primary">
                Run the quick start <ArrowRight className="h-4 w-4" />
              </a>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                className="btn-ghost"
              >
                <Github className="h-4 w-4" /> View on GitHub
              </a>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.55 }}
              className="mt-10 flex flex-wrap items-center gap-x-6 gap-y-3 text-[12px] uppercase tracking-[0.12em] text-ink-300"
            >
              <span className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-ink-400" />
                PyTorch 2.1+
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-ink-400" />
                Python 3.10–3.15 · CPU CI
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-ink-400" />
                Apache 2.0
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-ink-400" /> Independent
                research implementation
              </span>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="relative lg:col-span-5"
          >
            <div className="relative mx-auto max-w-lg">
              <div className="surface overflow-hidden rounded-3xl border-white/10 shadow-card">
                <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
                  <div className="flex items-center gap-3">
                    <img src="/avqa/avqa-mark.svg" alt="" className="h-9 w-9" />
                    <div>
                      <p className="text-sm font-semibold text-white">
                        AVQA pipeline
                      </p>
                      <p className="text-xs text-ink-400">
                        coarse-to-fine attention
                      </p>
                    </div>
                  </div>
                  <span className="rounded-full border border-glow/20 bg-glow/10 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-glow">
                    reference
                  </span>
                </div>
                <div className="flex items-center justify-center px-4 py-5 sm:px-8 sm:py-8">
                  <HeroVisual size={390} />
                </div>
                <div className="grid grid-cols-3 border-t border-white/10 text-center">
                  {[
                    { label: "Group", value: "Keys" },
                    { label: "Rank", value: "Parents" },
                    { label: "Refine", value: "Children" },
                  ].map((item, index) => (
                    <div
                      key={item.label}
                      className={`px-2 py-4 ${index ? "border-l border-white/10" : ""}`}
                    >
                      <p className="font-mono text-[10px] uppercase tracking-widest text-ink-500">
                        0{index + 1} · {item.label}
                      </p>
                      <p className="mt-1 text-sm font-medium text-ink-100">
                        {item.value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-4 flex items-start gap-3 rounded-2xl border border-amber-200/15 bg-amber-100/[0.04] p-4 text-sm leading-6 text-ink-200">
                <Terminal className="mt-1 h-4 w-4 shrink-0 text-amber-200" />
                <p>
                  <span className="font-semibold text-white">
                    CPU reference path.
                  </span>{" "}
                  CUDA execution, CUDA/Triton equivalence, and GPU performance
                  have not been tested in a CUDA environment.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
