import { useEffect, useState } from "react";
import type { Turn as TurnData } from "../types";
import { COLOR, DECIDED_BY } from "../theme";
import { turnReport } from "../lib/report";
import CheckedText from "./CheckedText";
import Evidence from "./Evidence";
import Gauge from "./Gauge";
import StageBars from "./StageBars";

function Elapsed() {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const id = setInterval(() => setSecs((performance.now() - start) / 1000), 100);
    return () => clearInterval(id);
  }, []);
  return (
    <span className="muted">
      analyzing… {secs.toFixed(1)}s
      {secs > 6 && " · long texts, and the first check after startup, take longer"}
    </span>
  );
}

function CopyButton({ turn }: { turn: TurnData }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(turnReport(turn));
      setDone(true);
      setTimeout(() => setDone(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context); nothing sensible to do */
    }
  };
  return (
    <button type="button" className="link copy" onClick={copy}>
      {done ? "copied" : "copy"}
    </button>
  );
}

interface Props {
  turn: TurnData;
  /** Open the file viewer for this turn's source file. */
  onView: (turn: TurnData) => void;
}

export default function Turn({ turn, onView }: Props) {
  const r = turn.result;
  const color = r ? COLOR[r.label] ?? COLOR.Uncertain : COLOR.Uncertain;
  const phrases = r?.style.cliches_found ?? [];

  return (
    <article aria-label="check result">
      <div className="term-block cmd-line">
        <span className="cmd">
          <span className="p">&gt;</span> check{" "}
          {turn.pending ? (
            <Elapsed />
          ) : (
            <span className="muted">
              ({turn.words ?? 0} words
              {turn.src && (
                <>
                  {" < "}
                  <button type="button" className="link file" onClick={() => onView(turn)}
                          title="view file">
                    {turn.src}
                  </button>
                </>
              )}
              )
            </span>
          )}
        </span>
        {!turn.pending && (r || turn.error) && <CopyButton turn={turn} />}
      </div>

      {!turn.pending && turn.text && (
        <CheckedText text={turn.text} phrases={phrases} truncated={turn.textTruncated} />
      )}

      {turn.error && <div className="term-block error" role="alert">{turn.error}</div>}

      {r && (
        <>
          <div className="verdict-row">
            <div className="gauge-col">
              <div className="gauge-title">P(AI) PROBABILITY</div>
              <Gauge value={r.ai_pct} color={color} />
            </div>
            <div className="term-block verdict-block">
              <div className="k">verdict</div>
              <div className="verdict" style={{ color }}>{r.confidence_band}</div>
              <dl className="facts">
                <dt className="muted">how sure</dt>
                <dd>{(r.confidence * 100).toFixed(0)}%</dd>
                <dt className="muted">decided by</dt>
                <dd>{DECIDED_BY[r.decisive_stage] ?? r.decisive_stage}</dd>
              </dl>
            </div>
          </div>
          <StageBars stage1={r.stage1} stage2={r.stage2} />
          <Evidence result={r} />
        </>
      )}
    </article>
  );
}
