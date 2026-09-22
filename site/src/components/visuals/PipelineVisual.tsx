type Props = {
  className?: string;
};

export function PipelineVisual({ className = "" }: Props) {
  return (
    <svg
      viewBox="0 0 600 320"
      className={className}
      role="img"
      aria-label="AVQA pipeline: keys → codebook → parent attention → adaptive refinement"
    >
      <defs>
        <linearGradient id="pl-line" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#7C8CFF" stopOpacity="0.0" />
          <stop offset="0.5" stopColor="#7C8CFF" stopOpacity="0.8" />
          <stop offset="1" stopColor="#7DF9FF" stopOpacity="0.0" />
        </linearGradient>
        <linearGradient id="pl-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#9AA8FF" stopOpacity="0.5" />
          <stop offset="1" stopColor="#5F6EF0" stopOpacity="0.0" />
        </linearGradient>
        <linearGradient id="pl-fill-2" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#7DF9FF" stopOpacity="0.4" />
          <stop offset="1" stopColor="#5F6EF0" stopOpacity="0.0" />
        </linearGradient>
        <radialGradient id="pl-node" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#FFFFFF" />
          <stop offset="1" stopColor="#5F6EF0" />
        </radialGradient>
        <radialGradient id="pl-node-active" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#FFFFFF" />
          <stop offset="0.5" stopColor="#7DF9FF" />
          <stop offset="1" stopColor="#7C8CFF" />
        </radialGradient>
      </defs>

      <rect x="0" y="0" width="600" height="320" fill="transparent" />

      <g transform="translate(20, 60)">
        <text
          fill="#8088A1"
          fontSize="10"
          fontFamily="ui-sans-serif, sans-serif"
          letterSpacing="2"
        >
          KEYS
        </text>
        {Array.from({ length: 24 }).map((_, i) => {
          const x = (i % 6) * 14;
          const y = Math.floor(i / 6) * 14;
          return (
            <rect
              key={i}
              x={x}
              y={y + 14}
              width={10}
              height={10}
              rx={2}
              fill="url(#pl-fill)"
              opacity={0.7}
            />
          );
        })}
      </g>

      <line
        x1="120"
        y1="160"
        x2="170"
        y2="160"
        stroke="url(#pl-line)"
        strokeWidth="2"
      />

      <g transform="translate(170, 90)">
        <text
          fill="#8088A1"
          fontSize="10"
          fontFamily="ui-sans-serif, sans-serif"
          letterSpacing="2"
        >
          HIERARCHICAL CODEBOOK
        </text>
        <circle
          cx="60"
          cy="60"
          r="9"
          fill="url(#pl-node-active)"
          opacity="0.4"
        ></circle>
        <circle cx="60" cy="60" r="5" fill="url(#pl-node)" />
        {[
          [20, 30],
          [50, 20],
          [95, 35],
          [110, 70],
          [85, 100],
          [35, 95],
        ].map(([cx, cy], i) => (
          <g key={i}>
            <line
              x1="60"
              y1="60"
              x2={cx}
              y2={cy}
              stroke="#7C8CFF"
              strokeOpacity="0.35"
              strokeWidth="0.8"
            />
            <circle cx={cx} cy={cy} r={3.5} fill="#7C8CFF" opacity="0.85" />
            {[
              [cx - 8, cy - 6],
              [cx + 8, cy - 6],
              [cx - 8, cy + 6],
              [cx + 8, cy + 6],
            ].map(([x, y], j) => (
              <circle
                key={j}
                cx={x}
                cy={y}
                r={2}
                fill="#5F6EF0"
                opacity="0.55"
              />
            ))}
          </g>
        ))}
      </g>

      <line
        x1="320"
        y1="160"
        x2="370"
        y2="160"
        stroke="url(#pl-line)"
        strokeWidth="2"
      />

      <g transform="translate(370, 60)">
        <text
          fill="#8088A1"
          fontSize="10"
          fontFamily="ui-sans-serif, sans-serif"
          letterSpacing="2"
        >
          PARENT ATTENTION
        </text>
        <rect
          x="0"
          y="14"
          width="100"
          height="100"
          rx="6"
          fill="url(#pl-fill)"
          opacity="0.5"
        />
        {[
          [12, 22],
          [40, 28],
          [70, 38],
          [85, 65],
          [45, 80],
          [22, 70],
        ].map(([cx, cy], i) => (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r="4"
            fill="url(#pl-node-active)"
            opacity={0.9}
          />
        ))}
      </g>

      <line
        x1="475"
        y1="160"
        x2="525"
        y2="160"
        stroke="url(#pl-line)"
        strokeWidth="2"
      />

      <g transform="translate(525, 60)">
        <text
          fill="#8088A1"
          fontSize="9"
          fontFamily="ui-sans-serif, sans-serif"
          letterSpacing="2"
        >
          REFINE
        </text>
        <rect
          x="0"
          y="14"
          width="60"
          height="100"
          rx="6"
          fill="url(#pl-fill-2)"
          opacity="0.5"
        />
        {Array.from({ length: 6 }).map((_, i) => (
          <circle
            key={i}
            cx={30}
            cy={28 + i * 14}
            r={i === 1 || i === 4 ? 4 : 2.5}
            fill="url(#pl-node-active)"
            opacity={i === 1 || i === 4 ? 0.95 : 0.6}
          />
        ))}
      </g>

      <g transform="translate(20, 240)">
        <text
          fill="#8088A1"
          fontSize="9"
          fontFamily="ui-sans-serif, sans-serif"
          letterSpacing="2"
        >
          CORRECTION
        </text>
        <line
          x1="0"
          y1="20"
          x2="120"
          y2="20"
          stroke="#7DF9FF"
          strokeOpacity="0.6"
          strokeWidth="1.2"
        />
        <text
          x="0"
          y="44"
          fill="#7DF9FF"
          fontSize="9"
          fontFamily="ui-monospace, monospace"
        >
          replace · preserve norm
        </text>
      </g>

      <g transform="translate(180, 240)">
        <text
          fill="#8088A1"
          fontSize="9"
          fontFamily="ui-sans-serif, sans-serif"
          letterSpacing="2"
        >
          BCAR
        </text>
        <line
          x1="0"
          y1="20"
          x2="120"
          y2="20"
          stroke="#7DF9FF"
          strokeOpacity="0.6"
          strokeWidth="1.2"
        />
        <text
          x="0"
          y="44"
          fill="#7DF9FF"
          fontSize="9"
          fontFamily="ui-monospace, monospace"
        >
          online codebook adaptation
        </text>
      </g>

      <g transform="translate(360, 240)">
        <text
          fill="#8088A1"
          fontSize="9"
          fontFamily="ui-sans-serif, sans-serif"
          letterSpacing="2"
        >
          HVAQ
        </text>
        <line
          x1="0"
          y1="20"
          x2="120"
          y2="20"
          stroke="#7DF9FF"
          strokeOpacity="0.6"
          strokeWidth="1.2"
        />
        <text
          x="0"
          y="44"
          fill="#7DF9FF"
          fontSize="9"
          fontFamily="ui-monospace, monospace"
        >
          per-query temperature
        </text>
      </g>
    </svg>
  );
}
