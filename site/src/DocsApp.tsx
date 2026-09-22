import { ArrowLeft, BookOpen, Github, Menu, Network, Sigma } from "lucide-react";
import { useState } from "react";
import type { ReactNode } from "react";
import { DOCS_URL, GITHUB_URL, RELEASE_LABEL } from "./lib/links";

type Page = "overview" | "quickstart" | "api" | "architecture" | "research" | "reproducibility";

const PAGES: { id: Page; label: string; group: string }[] = [
  { id: "overview", label: "Overview", group: "Build with AVQA" },
  { id: "quickstart", label: "Quick start", group: "Build with AVQA" },
  { id: "api", label: "API and configuration", group: "Build with AVQA" },
  { id: "architecture", label: "Architecture", group: "Understand AVQA" },
  { id: "research", label: "Math and research", group: "Understand AVQA" },
  { id: "reproducibility", label: "Benchmarks and reproducibility", group: "Understand AVQA" },
];

export default function DocsApp() {
  const [page, setPage] = useState<Page>(pageFromPath());
  const [menuOpen, setMenuOpen] = useState(false);
  const selectPage = (next: Page) => {
    setPage(next);
    setMenuOpen(false);
    window.history.replaceState({}, "", `${DOCS_URL}${next === "overview" ? "" : `#${next}`}`);
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  return (
    <div className="min-h-screen bg-ink-950 text-ink-100">
      <header className="sticky top-0 z-30 border-b border-white/10 bg-ink-950/90 backdrop-blur-xl">
        <div className="container-edge flex h-16 items-center justify-between gap-5">
          <a href="/avqa/" className="inline-flex items-center gap-3" aria-label="AVQA home">
            <img src="/avqa/avqa-mark.svg" alt="" className="h-8 w-8" />
            <span className="font-semibold tracking-[0.16em] text-white">AVQA <span className="font-normal tracking-normal text-ink-400">docs</span></span>
          </a>
          <nav className="hidden items-center gap-4 md:flex">
            <a href="/avqa/" className="nav-link inline-flex items-center gap-2"><ArrowLeft className="h-4 w-4" /> Product</a>
            <a href={GITHUB_URL} className="nav-link inline-flex items-center gap-2" target="_blank" rel="noreferrer"><Github className="h-4 w-4" /> GitHub</a>
          </nav>
          <button className="btn-ghost px-3 py-2 md:hidden" onClick={() => setMenuOpen((open) => !open)} aria-expanded={menuOpen} aria-controls="docs-navigation" aria-label={menuOpen ? "Close documentation navigation" : "Open documentation navigation"}><Menu className="h-4 w-4" /></button>
        </div>
      </header>
      <div className="border-b border-accent-300/15 bg-accent-400/[0.06]">
        <div className="container-edge flex flex-col gap-1 py-3 text-sm sm:flex-row sm:items-center sm:justify-between">
          <span><strong className="text-white">{RELEASE_LABEL}.</strong> APIs and performance claims may change.</span>
          <a className="text-accent-200 hover:text-white" href={`${GITHUB_URL}/blob/main/RELEASE.md`} target="_blank" rel="noreferrer">Read release boundaries →</a>
        </div>
      </div>
      <div className="container-edge grid gap-10 py-10 lg:grid-cols-[220px_minmax(0,760px)] lg:gap-16">
        <aside id="docs-navigation" className={`${menuOpen ? "block" : "hidden"} lg:block`}>
          <div className="sticky top-24 space-y-7">
            {(["Build with AVQA", "Understand AVQA"] as const).map((group) => (
              <div key={group}>
                <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-ink-500">{group}</div>
                <div className="space-y-1">
                  {PAGES.filter((item) => item.group === group).map((item) => (
                    <button key={item.id} onClick={() => selectPage(item.id)} aria-current={page === item.id ? "page" : undefined} className={`block w-full rounded-lg px-3 py-2 text-left text-sm transition ${page === item.id ? "bg-accent-400/10 text-white" : "text-ink-400 hover:bg-white/[0.04] hover:text-white"}`}>{item.label}</button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </aside>
        <main className="min-w-0"><DocsPage page={page} /></main>
      </div>
    </div>
  );
}

function DocsPage({ page }: { page: Page }) {
  if (page === "quickstart") return <QuickStart />;
  if (page === "api") return <ApiPage />;
  if (page === "architecture") return <ArchitecturePage />;
  if (page === "research") return <ResearchPage />;
  if (page === "reproducibility") return <ReproducibilityPage />;
  return <Overview />;
}

function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return <div className="mb-12"><div className="eyebrow">{eyebrow}</div><h1 className="mt-5 max-w-3xl font-display text-4xl font-semibold leading-tight tracking-tightest text-white sm:text-6xl">{title}</h1><p className="mt-6 max-w-2xl text-lg leading-relaxed text-ink-300">{children}</p></div>;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="mt-12 border-t border-white/10 pt-8"><h2 className="font-display text-2xl font-semibold text-white">{title}</h2><div className="mt-4 space-y-4 text-[15px] leading-7 text-ink-300">{children}</div></section>;
}

function Code({ children }: { children: string }) { return <pre className="overflow-x-auto rounded-2xl border border-white/10 bg-ink-900/70 p-5 font-mono text-[13px] leading-6 text-ink-100"><code>{children}</code></pre>; }

function Overview() {
  return <><PageHeader eyebrow="Documentation · public alpha" title="Routed attention, explained clearly.">AVQA is an independent Apache-2.0 PyTorch reference implementation of Adaptive Vector Quantized Attention. Use these docs to build a working prototype, understand the algorithm, and reproduce the repository’s evidence.</PageHeader><div className="grid gap-4 sm:grid-cols-2"><DocCard icon={<BookOpen />} title="Build with AVQA" text="Install from source, run the first forward pass, configure routing, and understand supported boundaries." /><DocCard icon={<Sigma />} title="Understand AVQA" text="Follow the architecture, equations, benchmark protocol, and relationship to the reference paper." /></div><Section title="Status and support"><p>AVQA is alpha software. The shipped backend is pure PyTorch; framework adapters and vendor kernels are not included in the core distribution. Open a focused issue with a minimal reproduction, or use GitHub Discussions for usage questions.</p><p>Supported baseline: Python 3.10–3.12 and PyTorch 2.1+. Validate your own workload before relying on performance or numerical behavior in production.</p></Section></>;
}

function DocCard({ icon, title, text }: { icon: ReactNode; title: string; text: string }) { return <div className="surface rounded-2xl p-6"><div className="mb-5 inline-flex h-10 w-10 items-center justify-center rounded-xl border border-accent-300/20 bg-accent-400/10 text-accent-200">{icon}</div><h2 className="font-display text-xl font-semibold text-white">{title}</h2><p className="mt-2 text-sm leading-6 text-ink-300">{text}</p></div>; }

function QuickStart() {
  return <><PageHeader eyebrow="Build with AVQA" title="From clone to first tensor in minutes.">The alpha distribution is currently installed from source. Keep the first run small, validate shapes and devices, then tune the routing budget for your workload.</PageHeader><Section title="Install"><Code>{`git clone https://github.com/sachncs/avqa.git\ncd avqa\npython -m pip install -e ".[dev]"`}</Code></Section><Section title="Run the module API"><Code>{`import torch\nfrom avqa import AVQAttention, AVQConfig\nfrom avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig\n\nconfig = AVQConfig(\n    attention=AttentionShapeConfig(embed_dim=64, num_heads=4, head_dim=16),\n    codebook=CodebookConfig(num_codewords=8, children_per_codeword=2),\n    routing=RoutingConfig(refinement_budget=3),\n)\nattention = AVQAttention(config, in_proj=False, out_proj=False)\nq = torch.randn(2, 8, 64)\nk = torch.randn(2, 16, 64)\nv = torch.randn(2, 16, 64)\nout = attention(q, k, v)  # [2, 8, 64]`}</Code></Section><Section title="What to verify"><ul className="list-disc space-y-2 pl-5"><li>Query, key, and value tensors use the documented rank-3 layout `[batch, tokens, embed_dim]`.</li><li>All tensors share device and dtype constraints accepted by the selected backend.</li><li>Start with the default Torch backend; compare against SDPA before tuning refinement.</li></ul></Section></>;
}

function ApiPage() { return <><PageHeader eyebrow="Build with AVQA" title="A small public surface with explicit tradeoffs.">The primary entry points are `AVQAttention`, the functional `attention` helper, and immutable nested configuration objects.</PageHeader><Section title="Public entry points"><div className="grid gap-3 sm:grid-cols-2"><DocCard icon={<Network />} title="AVQAttention" text="nn.Module wrapper for the reference attention pipeline." /><DocCard icon={<Sigma />} title="AVQConfig" text="Immutable configuration for attention shape, codebook, routing, refinement, cache, and backend behavior." /></div></Section><Section title="Configuration guidance"><p>Use the smallest codebook and refinement budget that preserves quality for your task. The backend is selected from configuration, while KV caches and HVAQ schedules add workload-specific behavior.</p><p>Serialization is JSON-oriented for configuration and state-dict based for learned/runtime tensors. Unknown configuration fields should be rejected rather than silently ignored.</p></Section><Section title="Integration boundary"><p>Hugging Face, vLLM, FlashAttention, and xFormers adapters are intentionally not bundled in the core alpha package. See the integration scaffold in the repository before writing a framework-specific wrapper.</p></Section></>; }

function ArchitecturePage() { return <><PageHeader eyebrow="Understand AVQA" title="A layered stack with replaceable boundaries.">The pipeline quantizes keys into a hierarchical codebook, computes parent-level attention, routes the most important regions, and corrects those regions with child-level attention.</PageHeader><Section title="Data flow"><ol className="list-decimal space-y-2 pl-5"><li>Validate tensor contracts and configuration.</li><li>Assign keys to the hierarchical codebook and accumulate counts/values.</li><li>Compute parent-level online-softmax attention.</li><li>Select parents with the configured router and refinement budget.</li><li>Recompute selected children and replace parent contributions while preserving normalization.</li></ol></Section><Section title="Dependency rule"><p>The core algorithm does not import profiling or visualization modules. Backends, routers, merge strategies, and schedulers expose extension points without changing the mathematical pipeline.</p><p>See the repository’s <a className="text-accent-200 hover:text-white" href="https://github.com/sachncs/avqa/blob/main/docs/architecture.md" target="_blank" rel="noreferrer">architecture reference</a> for the source-to-spec mapping.</p></Section></>; }

function ResearchPage() { return <><PageHeader eyebrow="Understand AVQA" title="Research context without overstating evidence.">AVQA is an independent implementation of the AVQ-Attention ideas described in the linked paper. The repository distinguishes reference behavior, extensions, experiments, and unsupported integrations.</PageHeader><Section title="Core ideas"><p>Hierarchical vector quantization reduces the number of key representatives considered initially. Adaptive refinement spends additional work where parent attention indicates signal. BCAR updates codebooks online, while HVAQ changes temperature based on routed attention entropy.</p></Section><Section title="Limitations"><p>The pure-PyTorch reference path may be slower than optimized vendor kernels for short sequences. Performance is workload-dependent, and the alpha release does not promise drop-in compatibility with external serving frameworks.</p></Section><Section title="Read the equations"><p><a className="text-accent-200 hover:text-white" href="https://github.com/sachncs/avqa/blob/main/docs/math.md" target="_blank" rel="noreferrer">Mathematical formulation</a>, <a className="text-accent-200 hover:text-white" href="https://arxiv.org/abs/2607.12789" target="_blank" rel="noreferrer">reference paper</a>, and <a className="text-accent-200 hover:text-white" href="https://github.com/sachncs/avqa/blob/main/SPEC_COMPLIANCE.md" target="_blank" rel="noreferrer">compliance matrix</a>.</p></Section></>; }

function ReproducibilityPage() { return <><PageHeader eyebrow="Understand AVQA" title="Treat benchmarks as evidence, not decoration.">Every reported result should include the implementation revision, environment, tensor shape, precision, seed, warm-up policy, repetitions, raw output, and interpretation.</PageHeader><Section title="Minimum benchmark record"><ul className="list-disc space-y-2 pl-5"><li>Python, PyTorch, OS, CPU/GPU, CUDA, and optional dependency versions.</li><li>Batch size, sequence lengths, heads, head dimension, codebook size, routing budget, and precision.</li><li>Correctness comparison against the selected baseline before timing.</li><li>Raw JSON plus a human-readable summary checked into the benchmark record.</li></ul></Section><Section title="Repository protocol"><p>Use <a className="text-accent-200 hover:text-white" href="https://github.com/sachncs/avqa/blob/main/BENCHMARKS.md" target="_blank" rel="noreferrer">BENCHMARKS.md</a> as the canonical protocol. Existing numbers are historical reference results, not universal performance guarantees.</p></Section></>; }

function pageFromPath(): Page { const hash = window.location.hash.slice(1) as Page; return PAGES.some((page) => page.id === hash) ? hash : "overview"; }
