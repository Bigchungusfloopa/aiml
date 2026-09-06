"""Text preprocessing (FR-2).

Light-touch normalisation only: lowercasing and whitespace collapsing. Punctuation
is intentionally kept — em-dash frequency and sentence segmentation depend on it
downstream (FR-10, FR-12, FR-15).
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse runs of whitespace to single spaces."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text.lower()).strip()


def normalize_series(texts):
    """Vectorised :func:`normalize` for a pandas Series / iterable of strings."""
    return [normalize(t if isinstance(t, str) else "") for t in texts]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter used by burstiness / uniformity stats.

    Not linguistically perfect, but dependency-free and good enough for variance
    signals. Operates on the ORIGINAL (un-normalised) text so casing/punctuation
    cues survive.
    """
    if not text or not text.strip():
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]
