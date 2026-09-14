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

export default function App() {
  return (
    <div className="relative min-h-screen bg-ink-950 text-ink-100">
      <div className="pointer-events-none fixed inset-0 -z-20 bg-noise opacity-50" />
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