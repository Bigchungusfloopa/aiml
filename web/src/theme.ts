export const COLOR = {
  AI: "#ff6b6b",
  Human: "#5ef2a0",
  Uncertain: "#c9c9c9",
  Idle: "#4b5263",
} as const;

export const DECIDED_BY: Record<string, string> = {
  stage1: "the raw-AI check",
  stage2: "the reworded-AI check",
  none: "not enough text",
};

/** Python-style float: whole numbers keep one decimal (0 -> "0.0"). */
export function py(n: number | null | undefined): string {
  if (n === null || n === undefined) return "None";
  return Number.isInteger(n) ? n.toFixed(1) : String(n);
}

export function scoreWord(s: number): string {
  if (s >= 0.66) return "very AI-like";
  if (s >= 0.45) return "somewhat AI-like";
  if (s >= 0.3) return "mixed";
  return "human-like";
}

export function rhythmWord(u: number): string {
  return u >= 0.8 ? "very even — robotic" : u >= 0.6 ? "fairly even" : "varied — human-like";
}

export function perplexityWord(ppl: number | null | undefined): string {
  if (ppl && ppl < 35) return "low — predictable, typical of AI";
  if (ppl && ppl < 80) return "medium";
  return "high — surprising, typical of humans";
}

export function burstinessWord(b: number | null | undefined): string {
  if (b === null || b === undefined) return "";
  if (b < 25) return "low — even pacing";
  if (b >= 60) return "high — varied pacing";
  return "medium";
}

export const clamp01 = (x: number): number => Math.max(0, Math.min(1, x));
