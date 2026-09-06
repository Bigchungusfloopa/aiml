"""Rule-based stylistic checker (FR-10 .. FR-13).

Always-on supporting evidence. Runs independently of the cascade; its output is
attached to every result regardless of which stage produced the label.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

from .paths import CLICHES_FILE
from .preprocess import split_sentences

_EM_DASH_RE = re.compile(r"[—–]|(?<!-)--(?!-)")  # — – or ASCII --
_WORD_RE = re.compile(r"\b[\w']+\b")


@lru_cache(maxsize=1)
def _load_cliches() -> tuple[str, ...]:
    """Load the maintained cliché list (NFR-6: editable without code changes)."""
    try:
        lines = CLICHES_FILE.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ()
    out = []
    for ln in lines:
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln.lower())
    return tuple(out)


@dataclass
class StyleReport:
    word_count: int
    sentence_count: int
    em_dash_count: int
    em_dash_per_1k_words: float
    cliches_found: list[str] = field(default_factory=list)
    cliche_count: int = 0
    mean_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    sentence_length_uniformity: float = 0.0  # 0..1, higher = more uniform
    # Heuristic 0..1 nudge toward "AI-like". Supporting evidence only — never
    # the decisive label (FR-13, FR-20).
    style_ai_score: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def analyze(text: str) -> StyleReport:
    text = text or ""
    words = _WORD_RE.findall(text)
    wc = len(words)
    sentences = split_sentences(text)
    sc = len(sentences)

    em = len(_EM_DASH_RE.findall(text))
    em_per_1k = (em / wc * 1000) if wc else 0.0

    low = text.lower()
    found = [c for c in _load_cliches() if c in low]

    sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences] or [0]
    mean_len = statistics.fmean(sent_lengths)
    var = statistics.pvariance(sent_lengths) if len(sent_lengths) > 1 else 0.0
    # Uniformity: 1 - coefficient of variation, clamped to [0, 1].
    cv = (var ** 0.5 / mean_len) if mean_len else 0.0
    uniformity = max(0.0, min(1.0, 1.0 - cv))

    # Blend into a soft supporting score.
    score = 0.0
    score += min(0.35, em_per_1k / 15 * 0.35)          # em-dash density
    score += min(0.40, len(found) / 4 * 0.40)          # cliché hits
    if sc >= 4:
        score += max(0.0, (uniformity - 0.6)) / 0.4 * 0.25  # too-even rhythm
    score = round(min(1.0, score), 4)

    return StyleReport(
        word_count=wc,
        sentence_count=sc,
        em_dash_count=em,
        em_dash_per_1k_words=round(em_per_1k, 3),
        cliches_found=found,
        cliche_count=len(found),
        mean_sentence_length=round(mean_len, 2),
        sentence_length_variance=round(var, 3),
        sentence_length_uniformity=round(uniformity, 4),
        style_ai_score=score,
    )
