export type PipelineStage = 0 | 1 | 2 | 3;

type Props = {
  className?: string;
  activeStage: PipelineStage;
};

const STAGES = [
  "Group keys",
  "Score parents",
  "Route groups",
  "Refine children",
];

function stageOpacity(activeStage: PipelineStage, index: PipelineStage) {
  return activeStage === index ? 1 : 0.48;
}

/** A conceptual four-stage schematic. Highlighted state is selected by the reader. */
export function PipelineVisual({ className = "", activeStage }: Props) {
  return (
    <>
      <svg
        viewBox="0 0 800 330"
        className={`hidden h-auto w-full min-[900px]:block ${className}`}
        role="img"
        aria-labelledby="pipeline-title pipeline-description"
      >
        <title id="pipeline-title">Hierarchical attention pipeline</title>
        <desc id="pipeline-description">
          Conceptual diagram of keys grouped under parent codewords, coarse
          parent scoring, configured parent selection, and child-level
          refinement. The drawing is explanatory, not a data visualization or
          model output.
        </desc>
        <g stroke="#536351" fill="none" strokeWidth="1">
          <path d="M181 168h20m-7-7 7 7-7 7M381 168h20m-7-7 7 7-7 7M581 168h20m-7-7 7 7-7 7" />
        </g>
        <g
          fontFamily="'IBM Plex Mono', monospace"
          fontSize="12"
          letterSpacing="1"
        >
          {STAGES.map((label, index) => {
            const stage = index as PipelineStage;
            const x = 20 + index * 200;
            const selected = stage === activeStage;
            return (
              <g key={label} opacity={stageOpacity(activeStage, stage)}>
                <rect
                  x={x}
                  y="26"
                  width="174"
                  height="270"
                  fill={selected ? "#17221a" : "#101911"}
                  stroke={selected ? "#b9d692" : "#354237"}
                  strokeWidth={selected ? 1.5 : 1}
                />
                <text x={x + 14} y="50" fill="#b9d692">
                  0{index + 1}
                </text>
                <text x={x + 14} y="72" fill="#f5f3e9" fontSize="13">
                  {label.toUpperCase()}
                </text>
                <path d={`M${x + 14} 86h146`} stroke="#354237" />
              </g>
            );
          })}
        </g>

        <g opacity={stageOpacity(activeStage, 0)}>
          {[0, 1, 2, 3].map((row) =>
            [0, 1, 2, 3, 4].map((column) => (
              <rect
                key={`${row}-${column}`}
                x={38 + column * 26}
                y={111 + row * 27}
                width="16"
                height="16"
                fill={column === 1 || column === 4 ? "#e9784f" : "#536351"}
              />
            )),
          )}
          <text
            x="38"
            y="245"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            Assign keys to codewords
          </text>
        </g>

        <g opacity={stageOpacity(activeStage, 1)}>
          {[0, 1, 2, 3].map((parent) => {
            const x = 236 + (parent % 2) * 68;
            const y = 112 + Math.floor(parent / 2) * 84;
            return (
              <g key={parent}>
                <path
                  d={`M${x + 21} ${y + 6}v12m0 0-12 10m12-10 12 10`}
                  fill="none"
                  stroke="#83907E"
                />
                <circle cx={x + 21} cy={y + 3} r="5" fill="#b9d692" />
                <circle cx={x + 9} cy={y + 31} r="4" fill="#536351" />
                <circle cx={x + 33} cy={y + 31} r="4" fill="#536351" />
                <text x={x + 44} y={y + 8} fill="#b3bba9" fontSize="11">
                  P{parent + 1}
                </text>
              </g>
            );
          })}
          <text
            x="236"
            y="275"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            Parent / child hierarchy
          </text>
        </g>

        <g opacity={stageOpacity(activeStage, 2)}>
          {[74, 54, 35, 20].map((height, index) => (
            <g key={index}>
              <rect
                x={432 + index * 31}
                y={208 - height}
                width="17"
                height={height}
                fill={index < 2 ? "#e9784f" : "#536351"}
              />
              <text
                x={440 + index * 31}
                y="228"
                textAnchor="middle"
                fill="#939e8c"
                fontSize="10"
              >
                {index + 1}
              </text>
            </g>
          ))}
          <path d="M426 209h135" stroke="#536351" />
          <text
            x="430"
            y="275"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            Importance ranks groups
          </text>
          <text x="430" y="258" fill="#939e8c" fontSize="11">
            SCHEMATIC SCORES
          </text>
        </g>

        <g opacity={stageOpacity(activeStage, 3)}>
          {[0, 1].map((group) => (
            <g key={group}>
              <text x="632" y={119 + group * 82} fill="#f5a17c" fontSize="11">
                ROUTED PARENT {group + 1}
              </text>
              {[0, 1, 2, 3].map((child) => (
                <rect
                  key={child}
                  x={632 + child * 31}
                  y={132 + group * 82}
                  width="23"
                  height="28"
                  fill="#e9784f"
                  stroke="#f5a17c"
                />
              ))}
            </g>
          ))}
          <text
            x="632"
            y="275"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            Replace parent contribution
          </text>
        </g>

        <text
          x="780"
          y="318"
          textAnchor="end"
          fill="#83907E"
          fontFamily="'IBM Plex Mono', monospace"
          fontSize="11"
          letterSpacing="0.6"
        >
          CONCEPTUAL · NOT MODEL OUTPUT
        </text>
      </svg>
      <ol className="space-y-0 border-y border-white/10 min-[900px]:hidden">
        {[
          "Keys are assigned to child codewords beneath parent groups.",
          "A coarse attention pass estimates parent importance.",
          "A configured router selects parents within the refinement budget.",
          "Selected parent contributions are replaced with child-level results.",
        ].map((description, index) => (
          <li
            key={STAGES[index]}
            className={`grid grid-cols-[2.5rem_1fr] gap-2 border-b border-white/10 py-3 last:border-b-0 ${index === activeStage ? "opacity-100" : "opacity-60"}`}
          >
            <span className="font-mono text-[10px] text-accent-300">
              0{index + 1}
            </span>
            <div>
              <p className="text-xs font-semibold text-ink-100">
                {STAGES[index]}
              </p>
              <p className="mt-1 text-xs leading-5 text-ink-400">
                {description}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </>
  );
}
