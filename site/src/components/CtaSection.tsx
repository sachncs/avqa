import { ArrowRight, Github, Terminal } from "lucide-react";
import { FadeIn } from "./FadeIn";
import { DOCS_URL, GITHUB_URL } from "../lib/links";

export function CtaSection() {
  return (
    <section className="relative py-32 sm:py-40">
      <div className="container-edge">
        <FadeIn>
          <div className="relative overflow-hidden rounded-3xl border border-white/[0.06] bg-gradient-to-br from-accent-500/10 via-ink-900/40 to-glow/[0.06] p-10 sm:p-16">
            <div className="pointer-events-none absolute -inset-1 -z-10 bg-[radial-gradient(circle_at_30%_20%,rgba(124,140,255,0.25),transparent_60%)]" />
            <div className="pointer-events-none absolute -inset-1 -z-10 bg-[radial-gradient(circle_at_80%_70%,rgba(125,249,255,0.15),transparent_60%)]" />

            <div className="grid items-center gap-10 lg:grid-cols-12">
              <div className="lg:col-span-7">
                <span className="eyebrow">Get started</span>
                <h2 className="mt-5 font-display text-[40px] font-semibold leading-[1.02] tracking-tightest text-white sm:text-[56px]">
                  Start with a clear{" "}
                  <span className="text-gradient-accent">
                    reference implementation.
                  </span>
                </h2>
                <p className="mt-6 max-w-xl text-[16px] leading-relaxed text-ink-300">
                  Install from source, start on CPU, and inspect every step of
                  the routed-attention pipeline. AVQA is Apache 2.0 research
                  software in public alpha.
                </p>
              </div>

              <div className="lg:col-span-5">
                <div className="rounded-2xl border border-white/10 bg-ink-950/80 p-5 backdrop-blur">
                  <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-ink-500">
                    <Terminal className="h-3.5 w-3.5" />
                    Terminal
                  </div>
                  <pre className="mt-3 overflow-x-auto font-mono text-[13px] leading-relaxed text-ink-100">
                    <code>
                      <span className="text-ink-500">$</span>{" "}
                      <span className="text-accent-300">git clone</span>{" "}
                      https://github.com/sachncs/avqa.git{"\n"}
                      <span className="text-ink-500">$</span>{" "}
                      <span className="text-accent-300">cd</span> avqa{"\n"}
                      <span className="text-ink-500">$</span>{" "}
                      <span className="text-accent-300">
                        python -m pip install
                      </span>{" "}
                      -e &quot;.[dev]&quot;
                    </code>
                  </pre>
                </div>
                <div className="mt-5 flex flex-wrap gap-3">
                  <a href={`${DOCS_URL}#quickstart`} className="btn-primary">
                    Run the quick start <ArrowRight className="h-4 w-4" />
                  </a>
                  <a
                    href={GITHUB_URL}
                    target="_blank"
                    rel="noreferrer"
                    className="btn-ghost"
                  >
                    <Github className="h-4 w-4" /> Star on GitHub
                  </a>
                </div>
              </div>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
