export interface Piece {
  text: string;
  hit: boolean;
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Split `text` into pieces, marking case-insensitive occurrences of `phrases`. */
export function splitHighlights(text: string, phrases: string[]): Piece[] {
  const uniq = [...new Set(phrases.map((p) => p.trim()).filter(Boolean))].sort(
    (a, b) => b.length - a.length, // prefer the longest phrase at a position
  );
  if (!text || uniq.length === 0) return [{ text, hit: false }];

  const re = new RegExp(uniq.map(escapeRe).join("|"), "gi");
  const out: Piece[] = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m[0] === "") continue;
    if (m.index > last) out.push({ text: text.slice(last, m.index), hit: false });
    out.push({ text: m[0], hit: true });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), hit: false });
  return out;
}
