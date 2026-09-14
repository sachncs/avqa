const ITEMS = [
  "PyTorch",
  "FlashAttention-2 online softmax",
  "torch.compile ready",
  "Apache 2.0",
  "Strict typing · mypy clean",
  "≥90% test coverage",
];

export function TrustStrip() {
  return (
    <section className="relative border-y border-white/5 bg-ink-950/40 py-8 backdrop-blur">
      <div className="container-edge">
        <div className="flex flex-col items-center gap-6 lg:flex-row lg:gap-10">
          <div className="shrink-0 text-[11px] font-medium uppercase tracking-[0.22em] text-ink-500">
            Built on
          </div>
          <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-3 lg:flex-1">
            {ITEMS.map((item) => (
              <span
                key={item}
                className="font-display text-sm font-medium text-ink-300/80 transition hover:text-white"
              >
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}