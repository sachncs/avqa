type Props = {
  className?: string;
  size?: number;
};

const NODES = [
  { cx: 50, cy: 50, r: 4.5, level: 0 },
  { cx: 22, cy: 38, r: 3.5, level: 1 },
  { cx: 38, cy: 28, r: 3.5, level: 1 },
  { cx: 62, cy: 36, r: 3.5, level: 1 },
  { cx: 78, cy: 46, r: 3.5, level: 1 },
  { cx: 12, cy: 64, r: 3, level: 2 },
  { cx: 28, cy: 56, r: 3, level: 2 },
  { cx: 36, cy: 70, r: 3, level: 2 },
  { cx: 50, cy: 60, r: 3, level: 2 },
  { cx: 64, cy: 68, r: 3, level: 2 },
  { cx: 78, cy: 76, r: 3, level: 2 },
  { cx: 88, cy: 62, r: 3, level: 2 },
];

const EDGES: [number, number][] = [
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [1, 5],
  [1, 6],
  [2, 6],
  [2, 7],
  [3, 8],
  [3, 9],
  [4, 10],
  [4, 11],
];

const ACTIVE_PATH = [0, 3, 8, 9];

export function HeroVisual({ className = "", size = 420 }: Props) {
  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <div className="absolute inset-0 rounded-full bg-[radial-gradient(circle_at_center,rgba(124,140,255,0.18),transparent_60%)] blur-2xl" />

      <svg
        viewBox="0 0 100 100"
        className="relative h-full w-full overflow-visible"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="edge" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#7C8CFF" stopOpacity="0.6" />
            <stop offset="1" stopColor="#7DF9FF" stopOpacity="0.05" />
          </linearGradient>
          <linearGradient id="edge-active" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#9AA8FF" />
            <stop offset="1" stopColor="#7DF9FF" />
          </linearGradient>
          <radialGradient id="node" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor="#C8D0FF" />
            <stop offset="1" stopColor="#5F6EF0" />
          </radialGradient>
          <radialGradient id="node-active" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor="#FFFFFF" />
            <stop offset="0.5" stopColor="#9AA8FF" />
            <stop offset="1" stopColor="#7DF9FF" />
          </radialGradient>
          <filter id="soft">
            <feGaussianBlur stdDeviation="1.2" />
          </filter>
        </defs>

        {EDGES.map(([a, b], i) => {
          const isActive = ACTIVE_PATH.includes(a) && ACTIVE_PATH.includes(b);
          const ax = NODES[a].cx;
          const ay = NODES[a].cy;
          const bx = NODES[b].cx;
          const by = NODES[b].cy;
          return (
            <line
              key={i}
              x1={ax}
              y1={ay}
              x2={bx}
              y2={by}
              stroke={isActive ? "url(#edge-active)" : "url(#edge)"}
              strokeWidth={isActive ? 1.1 : 0.5}
              strokeLinecap="round"
              className={isActive ? "animate-pulse-soft" : ""}
            />
          );
        })}

        {NODES.map((n, i) => {
          const isActive = ACTIVE_PATH.includes(i);
          return (
            <g key={i}>
              {isActive && (
                <circle
                  cx={n.cx}
                  cy={n.cy}
                  r={n.r + 2.5}
                  fill="url(#node-active)"
                  opacity="0.25"
                  filter="url(#soft)"
                />
              )}
              <circle
                cx={n.cx}
                cy={n.cy}
                r={n.r}
                fill={isActive ? "url(#node-active)" : "url(#node)"}
                opacity={isActive ? 1 : 0.85}
              />
            </g>
          );
        })}

        <g>
          <text x="6" y="14" fill="#8088A1" fontSize="3" fontFamily="ui-sans-serif, sans-serif" letterSpacing="0.6">
            ROOT
          </text>
          <text x="6" y="34" fill="#8088A1" fontSize="3" fontFamily="ui-sans-serif, sans-serif" letterSpacing="0.6">
            PARENTS
          </text>
          <text x="6" y="68" fill="#8088A1" fontSize="3" fontFamily="ui-sans-serif, sans-serif" letterSpacing="0.6">
            CHILDREN
          </text>
        </g>
      </svg>

      <div className="pointer-events-none absolute -top-2 -right-2 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-ink-950/80 px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-ink-300 backdrop-blur">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse-soft" />
        Hierarchical routing
      </div>
      <div className="pointer-events-none absolute -bottom-3 -left-2 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-ink-950/80 px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-ink-300 backdrop-blur">
        <span className="h-1.5 w-1.5 rounded-full bg-accent-300 animate-pulse-soft" />
        n=4096
      </div>
    </div>
  );
}
