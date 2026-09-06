"""Detection engine — cascade orchestration (FR-17 .. FR-25, SRS 6).

Flow:
  1. Always run the rule-based checker and GPT-2 perplexity analyzer (FR-17).
  2. Run Stage 1 (raw AI vs human). If it confidently says "AI", stop — that is
     the final label (FR-18).
  3. Otherwise run Stage 2 (humanized AI vs human); its verdict becomes final
     (FR-19).
  4. Combine the decisive stage's label + confidence with the always-on
     evidence (FR-20) and derive a confidence band (FR-25).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import joblib

from . import perplexity as ppl_mod
from . import rules as rules_mod
from .paths import (
    MODEL_HUMANIZED,
    MODEL_RAW,
    VECTORIZER_HUMANIZED,
    VECTORIZER_RAW,
)
from .preprocess import normalize

# Stage 1 needs at least this P(AI) to short-circuit and skip Stage 2 (FR-18).
# Tuned on the held-out split: 0.85 maximises 3-way (human/raw-AI/humanized-AI)
# accuracy (~0.78) while keeping humanized-AI recall ~0.82. Lower values let
# Stage 1 grab too many humanized samples and label them "raw AI"; the overall
# "is this AI?" rate (~0.97) is flat across the range.
STAGE1_AI_CONFIDENCE = 0.85
# Below this, a "human" verdict is reported as tentative (FR-25).
CONFIDENT_THRESHOLD = 0.75
MIN_WORDS = 10  # NFR-3


@dataclass
class StageOutput:
    ran: bool
    label: str | None = None          # "AI" | "Human"
    p_ai: float | None = None         # P(class == ai)
    confidence: float | None = None   # P(predicted label)
    note: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DetectionResult:
    label: str                      # "AI" | "Human" | "Uncertain"
    confidence: float               # 0..1 for the reported label
    confidence_band: str            # e.g. "Human (confident)"
    decisive_stage: str             # "stage1" | "stage2" | "none"
    stage1: StageOutput
    stage2: StageOutput
    style: dict = field(default_factory=dict)
    perplexity: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["stage1"] = self.stage1.as_dict()
        d["stage2"] = self.stage2.as_dict()
        return d


@lru_cache(maxsize=1)
def _load_stage(model_path_str: str, vec_path_str: str):
    return joblib.load(model_path_str), joblib.load(vec_path_str)


class Detector:
    """Loads persisted artifacts once and serves detection calls."""

    def __init__(self) -> None:
        self._stage1 = None
        self._stage2 = None
        self._load_errors: list[str] = []
        self._try_load()

    def _try_load(self) -> None:
        try:
            self._stage1 = _load_stage(str(MODEL_RAW), str(VECTORIZER_RAW))
        except Exception as exc:  # noqa: BLE001
            self._load_errors.append(f"Stage 1 model not loaded: {exc}")
        try:
            self._stage2 = _load_stage(
                str(MODEL_HUMANIZED), str(VECTORIZER_HUMANIZED))
        except Exception as exc:  # noqa: BLE001
            self._load_errors.append(f"Stage 2 model not loaded: {exc}")

    @property
    def ready(self) -> bool:
        return self._stage1 is not None

    # ------------------------------------------------------------------ #
    def _predict(self, stage, text_norm: str) -> tuple[float, str, float]:
        clf, vec = stage
        X = vec.transform([text_norm])
        classes = list(clf.classes_)
        proba = clf.predict_proba(X)[0]
        # positive class is 1 == "ai" (see train.py)
        p_ai = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
        label = "AI" if p_ai >= 0.5 else "Human"
        confidence = p_ai if label == "AI" else 1.0 - p_ai
        return p_ai, label, confidence

    # ------------------------------------------------------------------ #
    def classify_cascade(self, text: str) -> tuple[str, str]:
        """Two-stage cascade only — no rule/perplexity evidence.

        Returns ``(final_label, decisive_stage)`` where final_label is
        "AI" | "Human" | "Uncertain". Used by the evaluation harness so it
        doesn't pay the GPT-2 cost per sample.
        """
        if not self.ready or len(text.split()) < MIN_WORDS:
            return "Uncertain", "none"
        tn = normalize(text)
        p_ai1, label1, _ = self._predict(self._stage1, tn)

        # FR-18: Stage 1 confidently AI -> done (raw AI).
        if label1 == "AI" and p_ai1 >= STAGE1_AI_CONFIDENCE:
            return "AI", "stage1"
        if self._stage2 is None:
            return label1, "stage1"

        _, label2, _ = self._predict(self._stage2, tn)
        if label1 == "Human":
            # FR-19: Stage 2 is decisive.
            return ("AI", "stage2") if label2 == "AI" else ("Human", "stage1")
        # Stage 1 said AI but not confidently: Stage 2 tells us if it's the
        # humanized kind; if Stage 2 disagrees we still trust Stage 1's "AI".
        return ("AI", "stage2") if label2 == "AI" else ("AI", "stage1")

    # ------------------------------------------------------------------ #
    def detect(self, text: str) -> DetectionResult:
        messages: list[str] = []
        text = text or ""
        n_words = len(text.split())

        # NFR-3: graceful handling of empty / very short input.
        if n_words < MIN_WORDS:
            return DetectionResult(
                label="Uncertain",
                confidence=0.0,
                confidence_band="Not enough text to analyze",
                decisive_stage="none",
                stage1=StageOutput(ran=False, note="input too short"),
                stage2=StageOutput(ran=False, note="input too short"),
                messages=[f"Please provide at least {MIN_WORDS} words "
                          f"(got {n_words})."],
            )

        # FR-17: always-on evidence.
        style = rules_mod.analyze(text)
        perp = ppl_mod.analyze(text)
        if not perp.available:
            messages.append(f"Perplexity analysis unavailable: {perp.note}")

        if not self.ready:
            messages.extend(self._load_errors)
            messages.append(
                "Trained models are missing — run `python -m aitext.datasets_build` "
                "then `python -m aitext.train`. Showing supporting evidence only.")
            band, lbl, conf = self._evidence_only_band(style, perp)
            return DetectionResult(
                label=lbl, confidence=conf, confidence_band=band,
                decisive_stage="none",
                stage1=StageOutput(ran=False, note="model unavailable"),
                stage2=StageOutput(ran=False, note="model unavailable"),
                style=style.as_dict(), perplexity=perp.as_dict(),
                messages=messages,
            )

        text_norm = normalize(text)

        # FR-18: Stage 1 first.
        p_ai1, label1, conf1 = self._predict(self._stage1, text_norm)
        s1 = StageOutput(ran=True, label=label1, p_ai=round(p_ai1, 4),
                         confidence=round(conf1, 4))

        if label1 == "AI" and p_ai1 >= STAGE1_AI_CONFIDENCE:
            s2 = StageOutput(ran=False, note="skipped — Stage 1 confidently AI")
            return self._finalize("stage1", label1, conf1, s1, s2,
                                  style, perp, messages)

        # FR-19: Stage 1 said Human (or low-confidence AI) → run Stage 2.
        if self._stage2 is None:
            messages.append("Stage 2 model unavailable; using Stage 1 result.")
            s2 = StageOutput(ran=False, note="model unavailable")
            return self._finalize("stage1", label1, conf1, s1, s2,
                                  style, perp, messages)

        p_ai2, label2, conf2 = self._predict(self._stage2, text_norm)
        s2 = StageOutput(ran=True, label=label2, p_ai=round(p_ai2, 4),
                         confidence=round(conf2, 4))
        return self._finalize("stage2", label2, conf2, s1, s2,
                              style, perp, messages)

    # ------------------------------------------------------------------ #
    def _finalize(self, decisive, label, confidence, s1, s2,
                  style, perp, messages) -> DetectionResult:
        band = self._confidence_band(label, confidence, decisive, style, perp)
        return DetectionResult(
            label=label,
            confidence=round(float(confidence), 4),
            confidence_band=band,
            decisive_stage=decisive,
            stage1=s1,
            stage2=s2,
            style=style.as_dict(),
            perplexity=perp.as_dict(),
            messages=messages,
        )

    @staticmethod
    def _evidence_agrees_ai(style, perp) -> bool:
        signals = [style.style_ai_score >= 0.5]
        if perp.available:
            signals.append(perp.perplexity_ai_score >= 0.5)
        return any(signals)

    def _confidence_band(self, label, confidence, decisive, style, perp) -> str:
        confident = confidence >= CONFIDENT_THRESHOLD
        if label == "AI":
            if decisive == "stage2":
                return ("Likely humanized AI text"
                        if confident else
                        "Possibly humanized AI text (low confidence)")
            return "AI-generated (confident)" if confident else \
                   "Likely AI-generated"
        # label == Human
        if decisive == "stage2" and self._evidence_agrees_ai(style, perp):
            return "Likely human, but shows signs of humanized AI"
        if confident:
            return "Human (confident)"
        return "Likely human, but not certain"

    @staticmethod
    def _evidence_only_band(style, perp) -> tuple[str, str, float]:
        parts = [style.style_ai_score]
        if perp.available:
            parts.append(perp.perplexity_ai_score)
        score = sum(parts) / len(parts) if parts else 0.0
        if score >= 0.6:
            return "Likely AI-generated (evidence only, no model)", "AI", score
        if score <= 0.35:
            return "Likely human (evidence only, no model)", "Human", 1 - score
        return "Uncertain (evidence only, no model)", "Uncertain", 0.5
