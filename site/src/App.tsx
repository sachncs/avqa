import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { TrustStrip } from "./components/TrustStrip";
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
      <div className="pointer-events-none fixed inset-0 -z-20 bg-noise opacity-50" />
      <div className="border-b border-accent-300/15 bg-accent-400/[0.06] px-6 py-2.5 text-center text-xs text-ink-300 sm:text-sm">
        <span className="font-semibold text-white">{RELEASE_LABEL}.</span> Validate performance and integrations against your own workload. <a className="ml-1 text-accent-200 underline-offset-4 hover:underline" href={DOCS_URL}>Read the docs →</a>
      </div>
      <Nav />
      <main>
        <Hero />
        <TrustStrip />
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
