import { useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { FadeIn } from "./FadeIn";
import { PipelineVisual, type PipelineStage } from "./visuals/PipelineVisual";

const STAGES = [
  {
    title: "Group keys",
    description:
      "Assign each key to a leaf in the hierarchical codebook; parent and child representations summarize progressively finer groups.",
  },
  {
    title: "Score parents",
    description:
      "Compute a coarse attention pass over parent codewords and estimate each group's importance for the query.",
  },
  {
    title: "Route groups",
    description:
      "A configured router selects parent groups within the refinement budget. The selection policy is explicit and replaceable.",
  },
  {
    title: "Refine children",
    description:
      "Recompute the selected groups at child level and replace their coarse contributions in the running attention result.",
  },
] as const;

export function SolutionSection() {
  const [activeStage, setActiveStage] = useState<PipelineStage>(0);
  const stage = STAGES[activeStage];

  return (
    <section
      className="relative py-16 sm:py-24"
      aria-labelledby="pipeline-heading"
    >
      <div className="container-edge">
        <div className="grid gap-9 border-t border-white/15 pt-6 xl:grid-cols-12 xl:gap-12">
          <FadeIn className="xl:col-span-4">
            <span className="eyebrow">How the method works</span>
            <h2
              id="pipeline-heading"
              className="mt-5 max-w-[13ch] font-display text-3xl font-semibold leading-tight tracking-[-0.045em] text-white sm:text-4xl"
            >
              A coarse pass sets the next resolution.
            </h2>
            <p className="mt-5 max-w-lg text-sm leading-6 text-ink-300">
              Select a stage to trace the idea. This schematic explains the
              method; it does not depict measured attention weights or promise a
              speedup.
            </p>

            <div className="mt-8 grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-2">
              {STAGES.map((item, index) => {
                const selected = index === activeStage;
                return (
                  <button
                    key={item.title}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => setActiveStage(index as PipelineStage)}
                    className={`flex min-h-12 items-center gap-2 border px-3 text-left text-xs transition-colors ${
                      selected
                        ? "border-glow/60 bg-white/[0.045] text-white"
                        : "border-white/10 text-ink-300 hover:border-white/25 hover:text-white"
                    }`}
                  >
                    <span className="font-mono text-[10px] text-accent-300">
                      0{index + 1}
                    </span>
                    {item.title}
                  </button>
                );
              })}
            </div>

            <div
              className="mt-5 min-h-[110px] border-l-2 border-accent-400 pl-4"
              aria-live="polite"
              aria-atomic="true"
            >
              <h3 className="font-display text-base font-semibold text-white">
                {stage.title}
              </h3>
              <p className="mt-2 max-w-prose text-sm leading-6 text-ink-300">
                {stage.description}
              </p>
            </div>
          </FadeIn>

          <FadeIn delay={0.08} className="xl:col-span-8">
            <div className="border-y border-white/15 bg-ink-900/35 px-3 py-5 sm:px-5 sm:py-7">
              <PipelineVisual
                activeStage={activeStage}
                className="h-auto w-full"
              />
              <p className="mt-4 border-t border-white/10 pt-3 text-xs leading-5 text-ink-400">
                The implementation follows coarse attention with importance
                estimation, configured parent routing, child recomputation, and
                correction.{" "}
                <a
                  href="/avqa/docs/#architecture"
                  className="inline-flex min-h-8 items-center gap-1 text-ink-100 underline decoration-white/30 underline-offset-4 hover:decoration-accent-300"
                >
                  Read the algorithm notes
                  <ArrowUpRight aria-hidden="true" className="h-3 w-3" />
                </a>
              </p>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
