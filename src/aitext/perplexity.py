"""GPT-2 perplexity and burstiness analysis (FR-14 .. FR-16).

Uses a *pre-trained* GPT-2 small (~124M) purely for statistical scoring — it is
never trained by this project (SRS constraint 2.4). CPU-only; model is loaded
lazily and cached for the process lifetime.

- perplexity  : exp(mean token NLL) over the whole text — lower ⇒ more predictable
                ⇒ more AI-like.
- burstiness  : population variance of per-sentence perplexity — humans vary more,
                so higher burstiness ⇒ more human-like.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .preprocess import split_sentences

_MODEL_NAME = "gpt2"
_MAX_TOKENS = 1024  # GPT-2 context window


@dataclass
class PerplexityReport:
    perplexity: float
    burstiness: float
    mean_sentence_perplexity: float
    n_sentences_scored: int
    # 0..1 nudge toward "AI-like" (low perplexity + low burstiness). Supporting
    # evidence only (FR-16, FR-20).
    perplexity_ai_score: float
    model: str = _MODEL_NAME
    available: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(_MODEL_NAME)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model


def _text_perplexity(text: str, tok, model) -> float | None:
    import torch

    enc = tok(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return None
    out = model(input_ids, labels=input_ids)
    # out.loss is mean NLL per token (natural log).
    return float(torch.exp(out.loss))


def _unavailable(reason: str) -> PerplexityReport:
    return PerplexityReport(
        perplexity=float("nan"),
        burstiness=float("nan"),
        mean_sentence_perplexity=float("nan"),
        n_sentences_scored=0,
        perplexity_ai_score=0.0,
        available=False,
        note=reason,
    )


def analyze(text: str) -> PerplexityReport:
    text = (text or "").strip()
    if len(text.split()) < 5:
        return _unavailable("text too short for perplexity analysis")

    try:
        tok, model = _load_model()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully (NFR-3)
        return _unavailable(f"GPT-2 unavailable: {exc}")

    try:
        overall = _text_perplexity(text, tok, model)
        sent_ppls = []
        for s in split_sentences(text):
            if len(s.split()) >= 4:
                p = _text_perplexity(s, tok, model)
                if p is not None and math.isfinite(p):
                    sent_ppls.append(p)

        if overall is None or not math.isfinite(overall):
            return _unavailable("could not compute perplexity")

        if len(sent_ppls) >= 2:
            mean_sp = sum(sent_ppls) / len(sent_ppls)
            burstiness = sum((p - mean_sp) ** 2 for p in sent_ppls) / len(sent_ppls)
        else:
            mean_sp = overall
            burstiness = 0.0

        # Map to a soft AI-likeness score.
        # Low perplexity (<~30) and low burstiness (low vari:mean ratio) => AI-ish.
        ppl_component = max(0.0, min(1.0, (60.0 - overall) / 50.0))
        rel_burst = (burstiness ** 0.5) / mean_sp if mean_sp else 0.0
        burst_component = max(0.0, min(1.0, (0.5 - rel_burst) / 0.5))
        ai_score = round(0.6 * ppl_component + 0.4 * burst_component, 4)

        return PerplexityReport(
            perplexity=round(overall, 3),
            burstiness=round(burstiness, 3),
            mean_sentence_perplexity=round(mean_sp, 3),
            n_sentences_scored=len(sent_ppls),
            perplexity_ai_score=ai_score,
        )
    except Exception as exc:  # noqa: BLE001
        return _unavailable(f"perplexity computation failed: {exc}")
