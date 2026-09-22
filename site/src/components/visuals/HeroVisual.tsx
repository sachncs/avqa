type Props = {
  className?: string;
};

const KEY_ROWS = [0, 1, 2, 3];
const KEY_COLUMNS = [0, 1, 2, 3, 4];
const CHILDREN = [0, 1, 2, 3];
const SCORE_HEIGHTS = [76, 52, 33, 20];

/** A static-first, illustrative view of coarse-to-fine key refinement. */
export function HeroVisual({ className = "" }: Props) {
  return (
    <figure className={className}>
      <svg
        viewBox="0 0 800 400"
        className="hidden h-auto w-full min-[900px]:block"
        role="img"
        aria-labelledby="allocation-title allocation-description"
      >
        <title id="allocation-title">Coarse-to-fine attention routing</title>
        <desc id="allocation-description">
          An illustrative four-stage diagram: key vectors are grouped into a
          hierarchy, coarse parent scores rank groups, selected groups are
          routed, and their child keys receive finer attention computation. It
          is a conceptual diagram, not a model output.
        </desc>
        <defs>
          <marker
            id="allocation-arrow"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="4"
            orient="auto"
          >
            <path d="M1 1 6 4 1 7" fill="none" stroke="#8f9a8d" />
          </marker>
        </defs>

        <g
          fill="#939e8c"
          fontFamily="'IBM Plex Mono', monospace"
          fontSize="13"
          letterSpacing="1.2"
        >
          <text x="28" y="30">
            01 / KEY VECTORS
          </text>
          <text x="224" y="30">
            02 / HIERARCHY
          </text>
          <text x="425" y="30">
            03 / COARSE SCORE
          </text>
          <text x="626" y="30">
            04 / REFINE
          </text>
        </g>

        <g stroke="#465247" strokeWidth="1">
          <path d="M28 44h158M224 44h158M425 44h158M626 44h146" />
          <path
            d="M202 203h13m-5-5 5 5-5 5M403 203h13m-5-5 5 5-5 5M604 203h13m-5-5 5 5-5 5"
            fill="none"
            markerEnd="url(#allocation-arrow)"
          />
        </g>

        <g>
          {KEY_ROWS.map((row) =>
            KEY_COLUMNS.map((column) => {
              const selected = column === 1 || column === 4;
              return (
                <rect
                  key={`${row}-${column}`}
                  x={39 + column * 27}
                  y={107 + row * 27}
                  width="17"
                  height="17"
                  fill={selected ? "#e9784f" : "#354237"}
                  stroke={selected ? "#f5a17c" : "#536351"}
                  strokeWidth="1"
                />
              );
            }),
          )}
          <text
            x="38"
            y="248"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            key positions
          </text>
          <text
            x="38"
            y="273"
            fill="#939e8c"
            fontFamily="'IBM Plex Mono', monospace"
            fontSize="11"
          >
            grouped by codebook
          </text>
        </g>

        <g>
          {[0, 1, 2, 3].map((parent) => {
            const selected = parent === 0 || parent === 2;
            const x = 235 + (parent % 2) * 66;
            const y = 120 + Math.floor(parent / 2) * 102;
            return (
              <g key={parent}>
                <rect
                  x={x}
                  y={y}
                  width="50"
                  height="70"
                  fill={selected ? "#2e3027" : "#19221b"}
                  stroke={selected ? "#e9784f" : "#465247"}
                  strokeWidth={selected ? "2" : "1"}
                />
                <text
                  x={x + 7}
                  y={y + 15}
                  fill={selected ? "#f5a17c" : "#939e8c"}
                  fontFamily="'IBM Plex Mono', monospace"
                  fontSize="11"
                >
                  P{parent + 1}
                </text>
                {CHILDREN.map((child) => (
                  <rect
                    key={child}
                    x={x + 8 + (child % 2) * 17}
                    y={y + 26 + Math.floor(child / 2) * 17}
                    width="10"
                    height="10"
                    fill={selected ? "#e9784f" : "#536351"}
                  />
                ))}
              </g>
            );
          })}
          <text
            x="235"
            y="348"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            parent → child keys
          </text>
        </g>

        <g>
          <line x1="437" y1="294" x2="572" y2="294" stroke="#566351" />
          {SCORE_HEIGHTS.map((height, index) => {
            const selected = index < 2;
            const x = 447 + index * 32;
            return (
              <g key={index}>
                <rect
                  x={x}
                  y={294 - height}
                  width="18"
                  height={height}
                  fill={selected ? "#e9784f" : "#536351"}
                />
                <text
                  x={x + 9}
                  y="314"
                  textAnchor="middle"
                  fill="#939e8c"
                  fontFamily="'IBM Plex Mono', monospace"
                  fontSize="11"
                >
                  {index + 1}
                </text>
              </g>
            );
          })}
          <text
            x="437"
            y="348"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            illustrative parent scores
          </text>
        </g>

        <g>
          {[0, 1].map((group) => (
            <g key={group} transform={`translate(635 ${126 + group * 106})`}>
              <text
                x="0"
                y="0"
                fill="#f5a17c"
                fontFamily="'IBM Plex Mono', monospace"
                fontSize="12"
              >
                P{group === 0 ? "1" : "3"} SELECTED
              </text>
              {CHILDREN.map((child) => (
                <rect
                  key={child}
                  x={child * 29}
                  y="17"
                  width="21"
                  height="35"
                  fill="#e9784f"
                  stroke="#f5a17c"
                />
              ))}
            </g>
          ))}
          <text
            x="635"
            y="348"
            fill="#b3bba9"
            fontFamily="'Manrope', sans-serif"
            fontSize="14"
          >
            finer work on routed groups
          </text>
        </g>

        <g fontFamily="'IBM Plex Mono', monospace" fontSize="11">
          <rect x="30" y="373" width="9" height="9" fill="#e9784f" />
          <text x="46" y="382" fill="#b3bba9">
            ROUTED / REFINED
          </text>
          <rect x="204" y="373" width="9" height="9" fill="#536351" />
          <text x="220" y="382" fill="#b3bba9">
            NOT SELECTED AT THIS STAGE
          </text>
          <text x="770" y="382" textAnchor="end" fill="#83907E">
            CONCEPTUAL · NOT MODEL OUTPUT
          </text>
        </g>
      </svg>
      <ol className="space-y-0 border-y border-white/10 min-[900px]:hidden">
        {[
          [
            "01",
            "Group keys",
            "Assign key vectors to hierarchical codebook groups.",
          ],
          [
            "02",
            "Score parents",
            "Estimate coarse importance for each parent group.",
          ],
          [
            "03",
            "Route groups",
            "Select candidate parents under the configured budget.",
          ],
          [
            "04",
            "Refine children",
            "Recompute selected groups at finer resolution.",
          ],
        ].map(([number, title, description]) => (
          <li
            key={number}
            className="grid grid-cols-[2.5rem_1fr] gap-2 border-b border-white/10 py-3 last:border-b-0"
          >
            <span className="font-mono text-[10px] text-accent-300">
              {number}
            </span>
            <div>
              <p className="text-xs font-semibold text-ink-100">{title}</p>
              <p className="mt-1 text-xs leading-5 text-ink-400">
                {description}
              </p>
            </div>
          </li>
        ))}
      </ol>
      <figcaption className="mt-3 flex flex-wrap items-center justify-between gap-x-5 gap-y-2 border-t border-white/10 pt-3 font-mono text-[11px] uppercase tracking-[0.08em] text-ink-400">
        <span>
          Coarse scores route groups; refinement spends work within selected
          groups.
        </span>
        <span className="text-ink-500">Illustrative, not measured</span>
      </figcaption>
    </figure>
  );
}
