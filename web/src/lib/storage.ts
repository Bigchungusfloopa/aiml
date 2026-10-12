import type { Turn } from "../types";

const KEY = "czechtext.history.v1";
const MAX_TURNS = 50;
const MAX_TEXT = 20_000;

function shrink(t: Turn): Turn {
  if (t.text && t.text.length > MAX_TEXT) {
    return { ...t, text: t.text.slice(0, MAX_TEXT), textTruncated: true };
  }
  return t;
}

export function loadHistory(): Turn[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (t): t is Turn => !!t && typeof t === "object" && typeof (t as Turn).id === "number" && !(t as Turn).pending,
    );
  } catch {
    return []; // storage blocked or corrupt — start fresh
  }
}

export function saveHistory(turns: Turn[]): void {
  const settled = turns.filter((t) => !t.pending).map(shrink);
  // Quota errors are possible with large documents; retry with fewer turns.
  for (const keep of [MAX_TURNS, 10, 3, 0]) {
    try {
      if (keep === 0 || settled.length === 0) localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, JSON.stringify(settled.slice(-keep)));
      return;
    } catch {
      /* try a smaller slice */
    }
  }
}
