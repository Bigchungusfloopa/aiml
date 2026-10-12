import type { ReactNode } from "react";
import { clamp01 } from "../theme";

interface Props {
  label: string;
  value: number;
  /** Value that fills the bar completely. */
  max: number;
  /** 0..1, how AI-like this signal is; drives colour and hatching. */
  lean: number;
  /** Text form of the value (also the accessible value). */
  display: string;
  note?: ReactNode;
  /** Reference range, e.g. "human-typical > 60". */
  hint?: string;
}

export const leanOf = (x: number) => (x >= 0.55 ? "ai" : x <= 0.35 ? "human" : "mixed");

/** One evidence signal: label, thin bar, value, interpretation, reference range. */
export default function Meter({ label, value, max, lean, display, note, hint }: Props) {
  const pct = clamp01(value / max) * 100;
  return (
    <div className="meter">
      <span className="muted meter-label">&middot; {label}</span>
      <span
        className={`meter-track ${leanOf(lean)}`}
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-valuenow={Math.min(value, max)}
        aria-valuetext={`${display}${note ? `, ${typeof note === "string" ? note : ""}` : ""}`}
      >
        <span className="meter-fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="meter-text">
        <span className="strong">{display}</span>
        {note && <span className="muted"> {note}</span>}
        {hint && <span className="hint"> [{hint}]</span>}
      </span>
    </div>
  );
}
