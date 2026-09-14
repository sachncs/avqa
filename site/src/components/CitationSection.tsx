import { FadeIn } from "./FadeIn";
import { Quote } from "lucide-react";
import { PAPER_URL } from "../lib/links";

export function CitationSection() {
  return (
    <section className="relative py-28 sm:py-36">
      <div className="container-edge">
        <FadeIn className="max-w-4xl">
          <span className="eyebrow">Citation</span>
          <div className="relative mt-8 overflow-hidden rounded-3xl border border-white/[0.06] bg-gradient-to-b from-white/[0.03] to-white/[0.01] p-8 sm:p-12">
            <Quote className="absolute -left-2 -top-2 h-20 w-20 text-accent-300/10" />
            <h3 className="font-display text-[22px] font-semibold leading-snug text-white sm:text-[26px]">
              AVQ-Attention: Adaptive Vector-Quantized Attention
            </h3>
            <p className="mt-4 text-[14px] leading-relaxed text-ink-300">
              van den Dool, Winfried · Forré, Patrick · Habibian, Amir ·
              Asano, Yuki M. · Welling, Max. European Conference on Computer
              Vision (ECCV), 2026. arXiv:2607.12789.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <a
                href={PAPER_URL}
                target="_blank"
                rel="noreferrer"
                className="btn-primary"
              >
                Read on arXiv
              </a>
              <a
                href={`data:text/plain;charset=utf-8,${encodeURIComponent(CITATION_BIBTEX)}`}
                download="avqa-citation.bib"
                className="btn-ghost"
              >
                Download .bib
              </a>
            </div>
          </div>
          <p className="mt-6 text-[12px] text-ink-500">
            AVQA is an independent, community-driven implementation. The
            maintainer is not affiliated with the paper's authors or their
            institutions.
          </p>
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