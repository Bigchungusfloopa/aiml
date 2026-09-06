"""Dataset acquisition and preparation (FR-1).

Pulls source corpora via the Hugging Face ``datasets`` library and writes two
prepared CSVs, each with ``text`` and ``label`` columns (label in {human, ai}):

    data/stage1_raw.csv        human  vs  raw (unedited) AI text
    data/stage2_humanized.csv  human  vs  humanized (paraphrased) AI text

The human portion is kept identical across both files so Stage 1 / Stage 2
accuracy differences isolate the effect of humanization (SRS 7.2).

Run:
    python -m aitext.datasets_build --max-per-class 8000
"""

from __future__ import annotations

import argparse
import random
import re

import pandas as pd

from .paths import STAGE1_CSV, STAGE2_CSV, ensure_dirs

HUMAN = "human"
AI = "ai"


# --------------------------------------------------------------------------- #
# Source loaders
# --------------------------------------------------------------------------- #
def _load_hc3(max_per_class: int) -> tuple[list[str], list[str]]:
    """HC3 (Hello-SimpleAI/HC3): human answers vs ChatGPT answers."""
    from datasets import load_dataset

    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    human, ai = [], []
    for row in ds:
        for h in row.get("human_answers") or []:
            if h and len(h.split()) >= 20:
                human.append(h.strip())
        for a in row.get("chatgpt_answers") or []:
            if a and len(a.split()) >= 20:
                ai.append(a.strip())
    random.shuffle(human)
    random.shuffle(ai)
    return human[:max_per_class], ai[:max_per_class]


def _load_mage_supplement(max_items: int) -> list[str]:
    """Optional extra raw-AI samples from MAGE (yaful/MAGE). Best effort."""
    try:
        from datasets import load_dataset

        ds = load_dataset("yaful/MAGE", split="train", streaming=True)
        out = []
        for row in ds:
            # MAGE: label 0 = machine, 1 = human (per dataset card)
            if row.get("label") == 0 and row.get("text"):
                out.append(row["text"].strip())
                if len(out) >= max_items:
                    break
        return out
    except Exception as exc:  # noqa: BLE001 - supplement is optional
        print(f"[datasets_build] MAGE supplement skipped: {exc}")
        return []


# --------------------------------------------------------------------------- #
# Humanization fallback
# --------------------------------------------------------------------------- #
_CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "cannot": "can't", "will not": "won't", "is not": "isn't",
    "are not": "aren't", "it is": "it's", "that is": "that's",
    "there is": "there's", "we are": "we're", "they are": "they're",
}
_SWAPS = {
    "utilize": "use", "commence": "start", "numerous": "many",
    "additionally": "also", "however": "but", "therefore": "so",
    "furthermore": "plus", "approximately": "about", "demonstrate": "show",
    "individuals": "people", "prior to": "before", "in order to": "to",
}


def _pseudo_humanize(text: str, rng: random.Random) -> str:
    """Cheap, dependency-free stand-in for a paraphrasing tool.

    This is a FALLBACK ONLY. For a meaningful Stage 2 model, replace
    data/stage2_humanized.csv with real Quillbot / Undetectable.ai / LLM-rewritten
    output (SRS 7.1). A banner is printed when this path is used.
    """
    t = text
    for a, b in _CONTRACTIONS.items():
        t = re.sub(rf"\b{a}\b", b, t, flags=re.IGNORECASE)
    for a, b in _SWAPS.items():
        t = re.sub(rf"\b{a}\b", b, t, flags=re.IGNORECASE)
    # Perturb sentence order slightly and drop the occasional filler opener.
    sents = re.split(r"(?<=[.!?])\s+", t)
    sents = [s for s in sents if s.strip()]
    if len(sents) > 3 and rng.random() < 0.5:
        i = rng.randrange(len(sents) - 1)
        sents[i], sents[i + 1] = sents[i + 1], sents[i]
    return " ".join(sents)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _write(path, texts_labels: list[tuple[str, str]]) -> None:
    df = pd.DataFrame(texts_labels, columns=["text", "label"])
    df = df.dropna().drop_duplicates(subset="text")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[datasets_build] wrote {path}  ({len(df)} rows, "
          f"{(df.label == HUMAN).sum()} human / {(df.label == AI).sum()} ai)")


def build(max_per_class: int = 8000, seed: int = 42,
          humanized_csv: str | None = None) -> None:
    ensure_dirs()
    random.seed(seed)
    rng = random.Random(seed)

    human, raw_ai = _load_hc3(max_per_class)
    if len(raw_ai) < max_per_class:
        raw_ai += _load_mage_supplement(max_per_class - len(raw_ai))

    n = min(len(human), len(raw_ai))
    human, raw_ai = human[:n], raw_ai[:n]

    # Stage 1: human vs raw AI
    _write(STAGE1_CSV,
           [(t, HUMAN) for t in human] + [(t, AI) for t in raw_ai])

    # Stage 2: same human pool vs humanized AI
    if humanized_csv:
        hdf = pd.read_csv(humanized_csv)
        humanized = hdf.loc[hdf["label"].str.lower() == AI, "text"].tolist()
        print(f"[datasets_build] using real humanized text from {humanized_csv}")
    else:
        print("[datasets_build] WARNING: no --humanized-csv given; generating a "
              "pseudo-humanized Stage 2 set. Replace with real paraphrased data "
              "for a trustworthy Stage 2 model (SRS 9).")
        humanized = [_pseudo_humanize(t, rng) for t in raw_ai]

    m = min(len(human), len(humanized))
    _write(STAGE2_CSV,
           [(t, HUMAN) for t in human[:m]] + [(t, AI) for t in humanized[:m]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 1 / Stage 2 training CSVs")
    ap.add_argument("--max-per-class", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--humanized-csv", default=None,
                    help="CSV of real humanized AI text (text,label) for Stage 2")
    args = ap.parse_args()
    build(args.max_per_class, args.seed, args.humanized_csv)


if __name__ == "__main__":
    main()
