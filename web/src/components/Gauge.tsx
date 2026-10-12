interface Props {
  /** 0..100, or null when there is no verdict. */
  value: number | null;
  color: string;
}

const W = 300;
const H = 160;
const CX = W / 2;
const CY = 142;
const R = 120; // outer radius of the band
const r = 84; // inner radius of the band
const BAR_THICKNESS = 0.28;
const TICKS = [0, 20, 40, 60, 80, 100];
const STEPS = [
  { from: 0, to: 40, fill: "rgba(94, 242, 160, 0.12)" },
  { from: 40, to: 70, fill: "rgba(255, 189, 46, 0.12)" },
  { from: 70, to: 100, fill: "rgba(255, 107, 107, 0.12)" },
];

const point = (v: number, radius: number): [number, number] => {
  const a = Math.PI * (1 - v / 100);
  return [CX + radius * Math.cos(a), CY - radius * Math.sin(a)];
};

/** Annular sector between values v0..v1 (0..100) of the half-circle. */
function band(v0: number, v1: number, outer: number, inner: number): string {
  const [ox0, oy0] = point(v0, outer);
  const [ox1, oy1] = point(v1, outer);
  const [ix1, iy1] = point(v1, inner);
  const [ix0, iy0] = point(v0, inner);
  return (
    `M${ox0} ${oy0} A${outer} ${outer} 0 0 1 ${ox1} ${oy1} ` +
    `L${ix1} ${iy1} A${inner} ${inner} 0 0 0 ${ix0} ${iy0} Z`
  );
}

export default function Gauge({ value, color }: Props) {
  const mid = (R + r) / 2;
  const half = ((R - r) * BAR_THICKNESS) / 2;
  const shown = value === null ? null : Number(value.toFixed(1));

  return (
    <svg className="gauge" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label={shown === null ? "AI probability unavailable" : `AI probability ${shown}%`}>
      {STEPS.map((s) => (
        <path key={s.from} d={band(s.from, s.to, R, r)} fill={s.fill} />
      ))}
      <path d={band(0, 100, R, r)} fill="none" stroke="#2c313a" strokeWidth={1} />
      {shown !== null && shown > 0 && (
        <path d={band(0, shown, mid + half, mid - half)} fill={color} />
      )}
      {TICKS.map((t) => {
        const [x1, y1] = point(t, R);
        const [x2, y2] = point(t, R + 5);
        const [lx, ly] = point(t, R + 10);
        const c = Math.cos(Math.PI * (1 - t / 100));
        return (
          <g key={t}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#4b5263" />
            <text x={lx} y={ly + 3} fontSize={11} fill="#98a0ac"
                  textAnchor={c > 0.2 ? "start" : c < -0.2 ? "end" : "middle"}>
              {t}
            </text>
          </g>
        );
      })}
      <text x={CX} y={CY - 6} textAnchor="middle" fontSize={36} fontWeight={600} fill={color}>
        {shown === null ? "—" : `${shown}%`}
      </text>
    </svg>
  );
}
