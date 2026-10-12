import type { StageOutput } from "../types";
import { COLOR } from "../theme";

interface Row {
  name: string;
  value: number;
  color: string;
  text: string;
  ran: boolean;
  ai: boolean;
}

function row(name: string, s: StageOutput): Row {
  const value = s.ran && s.p_ai !== null ? s.p_ai * 100 : 0;
  const ai = s.ran && value >= 50;
  return {
    name,
    value,
    ran: s.ran,
    ai,
    color: !s.ran ? COLOR.Idle : ai ? COLOR.AI : COLOR.Human,
    text: s.ran ? `${value.toFixed(1)}% (${s.label})` : `not run (${s.note})`,
  };
}

const TICKS = [0, 25, 50, 75, 100];

export default function StageBars({ stage1, stage2 }: { stage1: StageOutput; stage2: StageOutput }) {
  const rows = [row("check 1: raw-ai", stage1), row("check 2: reworded-ai", stage2)];

  return (
    <div className="bars" role="group" aria-label="cascade checks, probability of AI">
      {rows.map((r) => {
        // Label sits inside the bar when it is wide enough, otherwise beside it.
        const inside = r.value >= 45;
        return (
          <div className="bars-row" key={r.name}>
            <span className="bars-name">{r.name}</span>
            <div className="bars-track">
              <span className="bars-mid" aria-hidden="true" />
              <div className={`bars-fill${r.ai ? " ai" : ""}`}
                   style={{ width: `${r.value}%`, background: r.color }}>
                {inside && <span className="bars-text in">{r.text}</span>}
              </div>
              {!inside && (
                <span className="bars-text out" style={{ left: `calc(${r.value}% + 8px)` }}>
                  {r.text}
                </span>
              )}
            </div>
          </div>
        );
      })}
      <div className="bars-row">
        <span className="bars-name muted">AI at ≥ 50%</span>
        <div className="bars-axis">
          {TICKS.map((t) => (
            <span key={t} className={t === 50 ? "mark" : ""} style={{ left: `${t}%` }}>{t}%</span>
          ))}
        </div>
      </div>
    </div>
  );
}
