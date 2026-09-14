import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Github, BookOpen, Command, Menu, X } from "lucide-react";
import { GITHUB_URL } from "../lib/links";

const links = [
  { href: "#why", label: "Why AVQA" },
  { href: "#features", label: "Features" },
  { href: "#architecture", label: "Architecture" },
  { href: "#benchmarks", label: "Benchmarks" },
];

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="fixed inset-x-0 top-0 z-50 flex justify-center pt-4"
    >
      <div
        className={`pointer-events-auto flex w-[min(96%,1100px)] items-center justify-between rounded-full px-3 py-2 transition-all duration-500 ${
          scrolled
            ? "border border-white/10 bg-ink-950/70 backdrop-blur-xl shadow-[0_10px_40px_-10px_rgba(0,0,0,0.6)]"
            : "border border-transparent"
        }`}
      >
        <a href="#top" className="flex items-center gap-2 pl-2 pr-3 py-1">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-accent-300 to-accent-600 shadow-[0_0_24px_-4px_rgba(124,140,255,0.6)]">
            <span className="block h-2.5 w-2.5 rounded-sm bg-white" />
          </span>
          <span className="text-sm font-semibold tracking-tight text-white">AVQA</span>
        </a>

        <nav className="hidden md:flex items-center gap-1">
          {links.map((l) => (
            <a key={l.href} href={l.href} className="nav-link rounded-full px-3 py-1.5">
              {l.label}
            </a>
          ))}
        </nav>

        <div className="hidden md:flex items-center gap-2">
          <a
            href={`${GITHUB_URL}#installation`}
            className="nav-link rounded-full px-3 py-1.5 inline-flex items-center gap-1.5"
          >
            <BookOpen className="h-3.5 w-3.5" /> Install
          </a>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-full bg-white px-3.5 py-1.5 text-sm font-medium text-ink-950 transition hover:shadow-glow"
          >
            <Github className="h-4 w-4" /> GitHub
          </a>
        </div>

        <button
          aria-label="Toggle menu"
          onClick={() => setOpen((s) => !s)}
          className="md:hidden inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/5"
        >
          {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
            className="absolute top-[78px] w-[92%] rounded-2xl border border-white/10 bg-ink-950/95 p-3 backdrop-blur-xl md:hidden"
          >
            <div className="flex flex-col">
              {links.map((l) => (
                <a
                  key={l.href}
                  href={l.href}
                  onClick={() => setOpen(false)}
                  className="rounded-lg px-3 py-2 text-sm text-ink-200 hover:bg-white/5"
                >
                  {l.label}
                </a>
              ))}
              <div className="my-2 h-px bg-white/5" />
              <a
                href={`${GITHUB_URL}#installation`}
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2 text-sm text-ink-200 hover:bg-white/5 inline-flex items-center gap-2"
              >
                <Command className="h-4 w-4" /> Install
              </a>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                className="rounded-lg bg-white px-3 py-2 text-sm font-medium text-ink-950 inline-flex items-center gap-2"
              >
                <Github className="h-4 w-4" /> GitHub
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.header>
  );
}