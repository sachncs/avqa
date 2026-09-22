import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, BookOpen, Github, Menu, X } from "lucide-react";
import { DOCS_URL, GITHUB_URL } from "../lib/links";

const LINKS = [
  { href: "#method", label: "Method" },
  { href: "#implementation", label: "Implementation" },
  { href: "#architecture", label: "Architecture" },
  { href: "#evidence", label: "Evidence" },
];

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState("");
  const menuButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const sections = LINKS.map(({ href }) =>
      document.querySelector(href),
    ).filter((section): section is Element => section !== null);
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActive(`#${visible.target.id}`);
      },
      { rootMargin: "-20% 0px -65% 0px", threshold: [0, 0.2, 0.5] },
    );
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        menuButton.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  const closeMenu = () => setOpen(false);

  return (
    <header
      className={`sticky top-0 z-50 border-b transition-colors duration-200 ${
        scrolled
          ? "border-white/10 bg-ink-950/95 backdrop-blur"
          : "border-transparent bg-ink-950/75 backdrop-blur-sm"
      }`}
    >
      <div className="container-edge flex min-h-[68px] items-center justify-between gap-6">
        <a
          href="#top"
          onClick={closeMenu}
          className="inline-flex min-h-11 items-center"
          aria-label="AVQA home"
        >
          <img
            src="/avqa/avqa-wordmark.svg"
            alt=""
            className="h-[34px] w-auto"
            width="210"
            height="56"
          />
        </a>

        <nav
          aria-label="Primary navigation"
          className="hidden items-center gap-7 md:flex"
        >
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              aria-current={active === link.href ? "location" : undefined}
              className="nav-link inline-flex min-h-11 items-center border-b border-transparent hover:border-accent-400"
            >
              {link.label}
            </a>
          ))}
          <a
            href={DOCS_URL}
            className="nav-link inline-flex min-h-11 items-center gap-2"
          >
            <BookOpen aria-hidden="true" className="h-4 w-4" /> Docs
          </a>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex min-h-10 items-center gap-2 border border-white/20 px-3 text-sm font-medium text-ink-50 transition-colors hover:border-accent-300 hover:text-accent-200"
          >
            GitHub <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
          </a>
        </nav>

        <button
          ref={menuButton}
          type="button"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          aria-controls="mobile-navigation"
          onClick={() => setOpen((wasOpen) => !wasOpen)}
          className="inline-flex h-11 w-11 items-center justify-center border border-white/20 text-ink-100 transition-colors hover:border-accent-300 md:hidden"
        >
          {open ? (
            <X aria-hidden="true" className="h-5 w-5" />
          ) : (
            <Menu aria-hidden="true" className="h-5 w-5" />
          )}
        </button>
      </div>

      <div
        id="mobile-navigation"
        hidden={!open}
        className="border-t border-white/10 bg-ink-950 px-5 pb-4 pt-2 md:hidden"
      >
        <nav
          aria-label="Mobile primary navigation"
          className="container-edge !px-0"
        >
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              onClick={closeMenu}
              aria-current={active === link.href ? "location" : undefined}
              className="flex min-h-12 items-center border-b border-white/10 text-sm text-ink-100 hover:text-accent-200"
            >
              {link.label}
            </a>
          ))}
          <div className="flex gap-5 pt-3">
            <a
              href={DOCS_URL}
              onClick={closeMenu}
              className="inline-flex min-h-11 items-center gap-2 text-sm text-ink-200 hover:text-white"
            >
              <BookOpen aria-hidden="true" className="h-4 w-4" /> Documentation
            </a>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex min-h-11 items-center gap-2 text-sm text-ink-200 hover:text-white"
            >
              <Github aria-hidden="true" className="h-4 w-4" /> GitHub
            </a>
          </div>
        </nav>
      </div>
    </header>
  );
}
