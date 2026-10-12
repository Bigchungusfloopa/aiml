import type { DetectionResult } from "../types";
import { burstinessWord, clamp01, perplexityWord, py, rhythmWord, scoreWord } from "../theme";
import Meter from "./Meter";

const Head = ({ label, children }: { label: string; children: string }) => (
  <div className="ev-head">
    <span className="k">{label.padEnd(11)}</span>
    {children}
  </div>
);

function Style({ r }: { r: DetectionResult }) {
  const s = r.style;
  const found = s.cliches_found ?? [];
  const n = s.cliche_count ?? 0;
  const em = s.em_dash_per_1k_words ?? 0;
  const u = s.sentence_length_uniformity ?? 0;
  const score = s.style_ai_score ?? 0;
  const shown = found.slice(0, 6).join(", ") + (found.length > 6 ? "…" : "");

  return (
    <section className="ev-group" aria-label="writing style evidence">
      <Head label="style">AI stylistic habits</Head>
      <Meter label="phrases" value={n} max={3} lean={n / 3} display={`${n} found`}
             note={found.length ? shown : undefined} hint="AI-typical ≥ 3" />
      <Meter label="dashes" value={em} max={12} lean={em / 12}
             display={`${py(em)} per 1k words`} hint="AI-typical ≥ 8" />
      <Meter label="rhythm" value={u} max={1} lean={clamp01((u - 0.6) / 0.4)}
             display={`${(u * 100).toFixed(0)}% uniform`} note={`(${rhythmWord(u)})`}
             hint="robotic ≥ 80%" />
      <Meter label="score" value={score} max={1} lean={score}
             display={`${score.toFixed(2)} / 1.00`} note={scoreWord(score)} />
    </section>
  );
}

function Fluency({ r }: { r: DetectionResult }) {
  const p = r.perplexity;
  const ppl = p.perplexity ?? 0;
  const burst = p.burstiness ?? 0;
  const score = p.perplexity_ai_score ?? 0;

  return (
    <section className="ev-group" aria-label="fluency evidence">
      <Head label="fluency">GPT-2 predictability</Head>
      {p.available ? (
        <>
          <Meter label="perplexity" value={ppl} max={150} lean={clamp01((80 - ppl) / 45)}
                 display={py(p.perplexity)} note={`(${perplexityWord(p.perplexity)})`}
                 hint="AI < 35 · human > 80" />
          <Meter label="burstiness" value={burst} max={100} lean={clamp01((60 - burst) / 35)}
                 display={py(p.burstiness)} note={`(${burstinessWord(p.burstiness)})`}
                 hint="human-typical > 60" />
          <Meter label="score" value={score} max={1} lean={score}
                 display={`${score.toFixed(2)} / 1.00`}
                 note={`${scoreWord(score)} · ${p.note ?? ""}`} />
        </>
      ) : (
        <div className="ev-line muted">&middot; unavailable &mdash; {p.note ?? ""}</div>
      )}
    </section>
  );
}

export default function Evidence({ result }: { result: DetectionResult }) {
  const hasStyle = Object.keys(result.style).length > 0;
  const hasPerplexity = Object.keys(result.perplexity).length > 0;

  return (
    <div className="term-block evidence">
      {hasStyle && <Style r={result} />}
      {hasPerplexity && <Fluency r={result} />}
      <section className="ev-group" aria-label="summary">
        <div className="ev-head">
          <span className="k">{"summary".padEnd(11)}</span>
          <span className="strong">{result.summary}</span>
        </div>
        {result.messages.map((m, i) => (
          <div key={i} className="ev-line muted">note: {m}</div>
        ))}
      </section>
    </div>
  );
}
