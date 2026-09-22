import { ArrowUpRight, Download } from "lucide-react";
import { FadeIn } from "./FadeIn";
import { PAPER_URL } from "../lib/links";

export function CitationSection() {
  return (
    <section id="paper" className="relative py-16 sm:py-20">
      <div className="container-edge">
        <FadeIn>
          <div className="grid gap-8 border-t border-white/15 pt-6 lg:grid-cols-12 lg:gap-12">
            <div className="lg:col-span-3">
              <span className="eyebrow">Paper and provenance</span>
            </div>
            <div className="lg:col-span-9">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-500">
                Accepted at ECCV 2026 · arXiv:2607.12789
              </p>
              <h2 className="mt-3 max-w-3xl font-display text-2xl font-semibold leading-snug tracking-[-0.03em] text-white sm:text-3xl">
                AVQ-Attention: Adaptive Vector-Quantized Attention
              </h2>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-ink-300">
                Winfried van den Dool, Patrick Forré, Amir Habibian, Yuki M.
                Asano, and Max Welling.
              </p>
              <div className="mt-5 flex flex-wrap gap-x-6 gap-y-2">
                <a
                  href={PAPER_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex min-h-10 items-center gap-1.5 text-sm text-ink-100 underline decoration-white/30 underline-offset-4 hover:decoration-accent-300"
                >
                  Read the paper
                  <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
                </a>
                <a
                  href={`data:text/plain;charset=utf-8,${encodeURIComponent(CITATION_BIBTEX)}`}
                  download="avqa-citation.bib"
                  className="inline-flex min-h-10 items-center gap-1.5 text-sm text-ink-100 underline decoration-white/30 underline-offset-4 hover:decoration-accent-300"
                >
                  <Download aria-hidden="true" className="h-3.5 w-3.5" />
                  BibTeX
                </a>
              </div>

              <aside className="mt-7 border-l-2 border-signal/70 pl-4">
                <p className="text-sm font-semibold text-ink-100">
                  Independent implementation
                </p>
                <p className="mt-1 max-w-3xl text-xs leading-5 text-ink-400">
                  This repository is not affiliated with the paper's authors or
                  their institutions. The paper reports custom Triton kernels;
                  CUDA execution, CUDA/Triton equivalence, and GPU performance
                  for this repository have not been tested in a CUDA
                  environment.
                </p>
              </aside>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

const CITATION_BIBTEX = `@inproceedings{van2026avq,
  title     = {AVQ-Attention: Adaptive Vector-Quantized Attention},
  author    = {van den Dool, Winfried and Forr{\\'e}, Patrick and Habibian, Amir and Asano, Yuki M. and Welling, Max},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  eprint    = {2607.12789},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2607.12789},
}`;
