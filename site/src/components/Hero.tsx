import { ArrowRight, Github, Star } from "lucide-react";
import { motion } from "framer-motion";
import { HeroVisual } from "./visuals/HeroVisual";
import { DOCS_URL, GITHUB_URL, RELEASE_LABEL, RELEASE_VERSION } from "../lib/links";

export function Hero() {
  return (
    <section id="top" className="relative isolate overflow-hidden pt-36 pb-24 sm:pt-44 sm:pb-32">
      <div className="absolute inset-0 -z-10 bg-radial-fade" />
      <div className="absolute inset-0 -z-10 bg-grid-faint [background-size:64px_64px] [mask-image:radial-gradient(ellipse_at_top,black_30%,transparent_70%)]" />

      <div className="container-edge">
        <div className="grid grid-cols-1 items-center gap-14 lg:grid-cols-12 lg:gap-10">
          <div className="lg:col-span-7">
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="eyebrow"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-accent-300 animate-pulse-soft" />
              {RELEASE_LABEL} · PyTorch reference backend
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
              className="mt-6 font-display text-[44px] font-semibold leading-[1.02] tracking-tightest text-white sm:text-[64px] lg:text-[78px]"
            >
              Attention at the{" "}
              <span className="text-gradient-accent">speed of thought.</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.25 }}
              className="mt-7 max-w-xl text-[17px] leading-relaxed text-ink-300 sm:text-[19px]"
            >
              AVQA is an open-source alpha attention backend for PyTorch. It
                turns the O(N²) wall that long-context Transformers hit into a
                hierarchical, codebook-routed flow — preserving precision where
                it matters, and skipping what doesn't.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.35 }}
              className="mt-10 flex flex-wrap items-center gap-3"
            >
              <a
                href={DOCS_URL}
                className="btn-primary"
              >
                Read the docs <ArrowRight className="h-4 w-4" />
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
              className="mt-12 flex flex-wrap items-center gap-x-6 gap-y-3 text-[12px] uppercase tracking-[0.18em] text-ink-500"
            >
              <span className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-ink-400" />
                PyTorch 2.1+
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-ink-400" />
                Python 3.10–3.15*
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-ink-400" />
                Apache 2.0
              </span>
              <span className="inline-flex items-center gap-2">
                <Star className="h-3 w-3 text-amber-300" /> Independent implementation
              </span>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="relative lg:col-span-5"
          >
            <div className="relative mx-auto flex max-w-md items-center justify-center">
              <div className="absolute -inset-8 -z-10 rounded-full bg-gradient-to-tr from-accent-500/15 via-transparent to-glow/10 blur-3xl" />
              <HeroVisual size={460} className="animate-float" />
            </div>
            <div className="mt-6 grid grid-cols-3 gap-3 text-center">
              {[
                { k: "K × D", v: "routed work", sub: "fixed budget" },
                { k: `v${RELEASE_VERSION}`, v: "release", sub: "public alpha" },
                { k: "90%", v: "coverage gate", sub: "CI enforced" },
              ].map((m) => (
                <div key={m.v} className="surface rounded-xl p-3">
                  <div className="font-display text-lg font-semibold text-white sm:text-xl">{m.k}</div>
                  <div className="mt-0.5 text-[11px] uppercase tracking-widest text-ink-400">{m.v}</div>
                  <div className="text-[10px] text-ink-500">{m.sub}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
