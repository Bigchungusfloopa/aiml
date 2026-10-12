export type Label = "AI" | "Human" | "Uncertain";
export type DecisiveStage = "stage1" | "stage2" | "none";

export interface StageOutput {
  ran: boolean;
  label: "AI" | "Human" | null;
  p_ai: number | null;
  confidence: number | null;
  note: string;
}

export interface StyleReport {
  word_count: number;
  sentence_count: number;
  em_dash_count: number;
  em_dash_per_1k_words: number;
  cliches_found: string[];
  cliche_count: number;
  mean_sentence_length: number;
  sentence_length_variance: number;
  sentence_length_uniformity: number;
  style_ai_score: number;
}

export interface PerplexityReport {
  perplexity: number | null;
  burstiness: number | null;
  mean_sentence_perplexity: number | null;
  n_sentences_scored: number;
  perplexity_ai_score: number;
  model: string;
  available: boolean;
  note: string;
}

export interface DetectionResult {
  label: Label;
  confidence: number;
  confidence_band: string;
  decisive_stage: DecisiveStage;
  stage1: StageOutput;
  stage2: StageOutput;
  /** Empty object when the input was too short to analyse. */
  style: Partial<StyleReport>;
  perplexity: Partial<PerplexityReport>;
  messages: string[];
  /** P(AI) as 0..100, or null when there is no verdict to plot. */
  ai_pct: number | null;
  summary: string;
}

export interface CheckResponse {
  words: number;
  src: string | null;
  error: string | null;
  /** The exact text that was analysed (typed text + extracted file text). */
  text: string;
  result: DetectionResult | null;
}

/** One entry in the terminal scrollback. `pending` while the request is in flight. */
export interface Turn extends Partial<CheckResponse> {
  id: number;
  pending?: boolean;
  /** Set when `text` was shortened to fit browser storage. */
  textTruncated?: boolean;
}
