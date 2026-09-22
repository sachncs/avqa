import { ArrowUpRight, Github } from "lucide-react";
import { DOCS_URL, GITHUB_URL, RELEASE_LABEL } from "../lib/links";

const COLS = [
  {
    title: "Product",
    links: [
      { label: "Method", href: "#method" },
      { label: "Implementation", href: "#implementation" },
      { label: "Architecture", href: "#architecture" },
      { label: "Evidence", href: "#evidence" },
      { label: "Paper", href: "#paper" },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "Documentation", href: DOCS_URL },
      { label: "GitHub repo", href: GITHUB_URL },
      { label: "arXiv paper", href: "https://arxiv.org/abs/2607.12789" },
      {
        label: "Benchmark protocol",
        href: `${GITHUB_URL}/blob/main/BENCHMARKS.md`,
      },
      { label: "Specification", href: `${GITHUB_URL}/blob/main/SPEC.md` },
    ],
  },
  {
    title: "Project",
    links: [
      {
        label: "Contributing",
        href: `${GITHUB_URL}/blob/main/CONTRIBUTING.md`,
      },
      {
        label: "Code of Conduct",
        href: `${GITHUB_URL}/blob/main/CODE_OF_CONDUCT.md`,
      },
      { label: "Security", href: `${GITHUB_URL}/blob/main/SECURITY.md` },
      { label: "Support", href: `${GITHUB_URL}/blob/main/SUPPORT.md` },
    ],
  },
];

export function Footer() {
  return (
    <footer className="relative border-t border-white/5 bg-ink-950/60">
      <div className="container-edge py-16 sm:py-20">
        <div className="grid gap-10 lg:grid-cols-12">
          <div className="lg:col-span-4">
            <a href="#top" className="inline-flex items-center gap-2">
              <img
                src="/avqa/avqa-wordmark.svg"
                alt="AVQA"
                className="h-[34px] w-auto"
              />
            </a>
            <p className="mt-4 max-w-xs text-[13px] leading-relaxed text-ink-400">
              Adaptive Vector Quantized Attention for PyTorch. Public-alpha
              reference implementation, Apache 2.0.
            </p>
            <div className="mt-6 flex items-center gap-2">
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                aria-label="AVQA on GitHub"
                className="inline-flex h-10 w-10 items-center justify-center border border-white/15 text-ink-300 transition hover:border-accent-300 hover:text-white"
              >
                <Github className="h-4 w-4" />
              </a>
              <span className="font-mono text-[11px] uppercase tracking-widest text-ink-500">
                {RELEASE_LABEL}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-8 sm:grid-cols-3 lg:col-span-8">
            {COLS.map((col) => (
              <div key={col.title}>
                <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-ink-500">
                  {col.title}
                </div>
                <ul className="mt-4 space-y-3">
                  {col.links.map((l) => (
                    <li key={l.label}>
                      <a
                        href={l.href}
                        target={
                          l.href.startsWith("http") ? "_blank" : undefined
                        }
                        rel={
                          l.href.startsWith("http") ? "noreferrer" : undefined
                        }
                        className="inline-flex min-h-9 items-center text-[13px] text-ink-300 transition hover:text-white"
                      >
                        {l.label}
                        {l.href.startsWith("http") && (
                          <ArrowUpRight
                            aria-hidden="true"
                            className="ml-1 h-3 w-3 text-ink-500"
                          />
                        )}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-14 flex flex-col items-start justify-between gap-3 border-t border-white/5 pt-8 text-[12px] text-ink-500 sm:flex-row sm:items-center">
          <div>© {new Date().getFullYear()} sachncs. Apache License 2.0.</div>
          <div className="font-mono">
            Built with care · Independent implementation · No affiliation.
          </div>
        </div>
      </div>
    </footer>
  );
}
