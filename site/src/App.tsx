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
import { MotionConfig } from "framer-motion";

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <div className="relative min-h-screen bg-ink-950 text-ink-100">
        <div className="pointer-events-none fixed inset-0 -z-20 bg-noise opacity-50" />
        <div className="border-b border-accent-300/20 bg-ink-900 px-4 py-2.5 text-center text-xs text-ink-200 sm:text-sm">
          <span className="font-semibold text-white">{RELEASE_LABEL}</span>
          <span aria-hidden="true"> · </span>CPU reference implementation; CUDA
          has not been tested.{" "}
          <a
            className="ml-1 font-medium text-glow underline-offset-4 hover:underline"
            href={DOCS_URL}
          >
            Read support notes →
          </a>
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
    </MotionConfig>
  );
}
