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

from .paths import MODELS_DIR
from .preprocess import split_sentences

_MODEL_NAME = "gpt2"
_MAX_TOKENS = 1024  # GPT-2 context window
_CALIB_PATH = MODELS_DIR / "perplexity_calib.pkl"


@lru_cache(maxsize=1)
def _load_calibrator():
    """Optional logistic calibrator from ``aitext.calibrate_perplexity``."""
    try:
        import joblib
        return joblib.load(_CALIB_PATH)
    except Exception:  # noqa: BLE001 - heuristic fallback is fine
        return None


def _ai_score(overall: float, burstiness: float, mean_sp: float) -> float:
    """P(AI) from the perplexity features — calibrated model if available,
    else the hand-tuned heuristic."""
    calib = _load_calibrator()
    if calib is not None:
        feats = [[math.log(overall), math.log1p(burstiness),
                  math.log(max(mean_sp, 1e-6))]]
        return round(float(calib["model"].predict_proba(feats)[0, 1]), 4)
    ppl_component = max(0.0, min(1.0, (60.0 - overall) / 50.0))
    rel_burst = (burstiness ** 0.5) / mean_sp if mean_sp else 0.0
    burst_component = max(0.0, min(1.0, (0.5 - rel_burst) / 0.5))
    return round(0.6 * ppl_component + 0.4 * burst_component, 4)


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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME).to(device).eval()
    torch.set_grad_enabled(False)
    return tok, model, device


def _text_perplexity(text: str, tok, model, device) -> float | None:
    import torch

    enc = tok(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS)
    input_ids = enc["input_ids"].to(device)
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
        tok, model, device = _load_model()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully (NFR-3)
        return _unavailable(f"GPT-2 unavailable: {exc}")

    try:
        overall = _text_perplexity(text, tok, model, device)
        sent_ppls = []
        for s in split_sentences(text):
            if len(s.split()) >= 4:
                p = _text_perplexity(s, tok, model, device)
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

        ai_score = _ai_score(overall, burstiness, mean_sp)

        return PerplexityReport(
            perplexity=round(overall, 3),
            burstiness=round(burstiness, 3),
            mean_sentence_perplexity=round(mean_sp, 3),
            n_sentences_scored=len(sent_ppls),
            perplexity_ai_score=ai_score,
            note="calibrated" if _load_calibrator() is not None else "heuristic",
        )
    except Exception as exc:  # noqa: BLE001
        return _unavailable(f"perplexity computation failed: {exc}")
