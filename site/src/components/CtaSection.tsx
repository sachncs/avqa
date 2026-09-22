import { ArrowRight, Github } from "lucide-react";
import { FadeIn } from "./FadeIn";
import { DOCS_URL, GITHUB_URL } from "../lib/links";

export function CtaSection() {
  return (
    <section
      className="relative py-20 sm:py-28"
      aria-labelledby="start-heading"
    >
      <div className="container-edge">
        <FadeIn>
          <div className="grid gap-9 border-y border-white/15 py-8 sm:py-10 lg:grid-cols-12 lg:items-center lg:gap-12">
            <div className="min-w-0 lg:col-span-5">
              <span className="eyebrow">Start with the reference</span>
              <h2
                id="start-heading"
                className="mt-5 max-w-[13ch] font-display text-3xl font-semibold leading-tight tracking-[-0.045em] text-white sm:text-4xl"
              >
                Inspect it on CPU first.
              </h2>
              <p className="mt-4 max-w-lg text-sm leading-6 text-ink-300">
                AVQA is source-first research software in public alpha. Begin
                with the documented CPU path, then validate any changes against
                the reference tests and your own workload.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <a href={`${DOCS_URL}#quickstart`} className="btn-primary">
                  Open quick start{" "}
                  <ArrowRight aria-hidden="true" className="h-4 w-4" />
                </a>
                <a
                  href={GITHUB_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="btn-ghost"
                >
                  <Github aria-hidden="true" className="h-4 w-4" /> Repository
                </a>
              </div>
            </div>

            <div className="min-w-0 lg:col-span-7">
              <div className="border border-white/15 bg-ink-950 px-4 py-4 sm:px-6 sm:py-5">
                <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-500">
                  Install from the repository
                </p>
                <pre className="mt-4 overflow-x-auto font-mono text-xs leading-6 text-ink-100 sm:text-sm">
                  <code>
                    <span className="text-ink-500">$</span> git clone
                    https://github.com/sachncs/avqa.git{"\n"}
                    <span className="text-ink-500">$</span> cd avqa{"\n"}
                    <span className="text-ink-500">$</span> python -m pip
                    install -e &quot;.[dev]&quot;
                  </code>
                </pre>
              </div>
              <p className="mt-3 text-xs leading-5 text-ink-500">
                This editable install is for contributors and local experiments.
                Follow the docs for the minimal CPU installation and first
                forward pass.
              </p>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
