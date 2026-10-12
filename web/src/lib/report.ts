import type { Turn } from "../types";
import { DECIDED_BY, burstinessWord, perplexityWord, py, rhythmWord, scoreWord } from "../theme";

/** Plain-text version of one turn, for the clipboard. */
export function turnReport(t: Turn): string {
  const src = t.src ? ` < ${t.src}` : "";
  const lines = [`> check (${t.words ?? 0} words${src})`];
  if (t.error) return [...lines, t.error].join("\n");
  const r = t.result;
  if (!r) return lines.join("\n");

  lines.push(
    `verdict:     ${r.confidence_band}`,
    `how sure:    ${(r.confidence * 100).toFixed(0)}%`,
    `decided by:  ${DECIDED_BY[r.decisive_stage] ?? r.decisive_stage}`,
    `P(AI):       ${r.ai_pct === null ? "n/a" : `${Number(r.ai_pct.toFixed(1))}%`}`,
  );
  for (const [name, s] of [["check 1: raw-ai", r.stage1], ["check 2: reworded-ai", r.stage2]] as const) {
    lines.push(
      s.ran ? `${name}: ${((s.p_ai ?? 0) * 100).toFixed(1)}% (${s.label})` : `${name}: not run (${s.note})`,
    );
  }

  const st = r.style;
  if (Object.keys(st).length) {
    const u = st.sentence_length_uniformity ?? 0;
    lines.push(
      "",
      "style:",
      `  phrases: ${st.cliche_count ?? 0}${st.cliches_found?.length ? ` — ${st.cliches_found.join(", ")}` : ""}`,
      `  dashes:  ${py(st.em_dash_per_1k_words)} per 1000 words`,
      `  rhythm:  ${(u * 100).toFixed(0)}% uniform (${rhythmWord(u)})`,
      `  score:   ${(st.style_ai_score ?? 0).toFixed(2)} / 1.00 (${scoreWord(st.style_ai_score ?? 0)})`,
    );
  }
  const p = r.perplexity;
  if (Object.keys(p).length) {
    lines.push("", "fluency (GPT-2):");
    if (p.available) {
      lines.push(
        `  perplexity: ${py(p.perplexity)} (${perplexityWord(p.perplexity)})`,
        `  burstiness: ${py(p.burstiness)} (${burstinessWord(p.burstiness)})`,
        `  score:      ${(p.perplexity_ai_score ?? 0).toFixed(2)} / 1.00 (${scoreWord(p.perplexity_ai_score ?? 0)}, ${p.note ?? ""})`,
      );
    } else {
      lines.push(`  unavailable — ${p.note ?? ""}`);
    }
  }
  lines.push("", `summary: ${r.summary}`);
  for (const m of r.messages) lines.push(`note: ${m}`);
  return lines.join("\n");
}
