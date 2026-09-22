import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { WhySection } from "./components/WhySection";
import { SolutionSection } from "./components/SolutionSection";
import { FeaturesSection } from "./components/FeaturesSection";
import { ArchitectureSection } from "./components/ArchitectureSection";
import { CodeSection } from "./components/CodeSection";
import { BenchmarksSection } from "./components/BenchmarksSection";
import { CitationSection } from "./components/CitationSection";
import { CtaSection } from "./components/CtaSection";
import { Footer } from "./components/Footer";
import { DOCS_URL, RELEASE_LABEL } from "./lib/links";

export default function App() {
  return (
    <div className="relative min-h-screen bg-ink-950 text-ink-100">
      <div className="border-b border-white/10 bg-ink-950 px-5 py-2.5">
        <div className="container-edge flex flex-wrap items-center justify-between gap-x-5 gap-y-1 !px-0 text-xs">
          <p className="text-ink-200">
            <span className="font-semibold text-white">{RELEASE_LABEL}</span>
            <span aria-hidden="true" className="px-2 text-ink-600">
              /
            </span>
            Research reference · CPU baseline
          </p>
          <a
            className="min-h-7 font-medium text-glow underline-offset-4 hover:underline"
            href={DOCS_URL}
          >
            Compatibility and limits <span aria-hidden="true">→</span>
          </a>
        </div>
      </div>
      <Nav />
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-lg focus:bg-white focus:px-4 focus:py-3 focus:text-ink-950"
      >
        Skip to content
      </a>
      <main id="main-content">
        <Hero />
        <WhySection />
        <SolutionSection />
        <FeaturesSection />
        <ArchitectureSection />
        <CodeSection />
        <BenchmarksSection />
        <CitationSection />
        <CtaSection />
      </main>
      <Footer />
    </div>
  );
}
