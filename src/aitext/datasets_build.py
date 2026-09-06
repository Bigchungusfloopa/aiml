"""Dataset acquisition and preparation (FR-1).

Writes two prepared CSVs, each with ``text`` and ``label`` columns
(label in {human, ai}):

    data/stage1_raw.csv        human  vs  raw (unedited) AI text
    data/stage2_humanized.csv  human  vs  humanized (paraphrased) AI text

Source: HC3 (Hello-SimpleAI/HC3) — human answers vs ChatGPT answers.
The human pool is identical across both files, and the Stage 2 AI side is the
*same* ChatGPT answers passed through a real paraphrasing model
(``humarin/chatgpt_paraphraser_on_T5_base``), so Stage 1 vs Stage 2 accuracy
differences isolate the effect of humanization (SRS 7.2).

Run:
    python -m aitext.datasets_build --max-per-class 6000
    python -m aitext.datasets_build --humanizer pseudo         # fast, no model
    python -m aitext.datasets_build --humanized-csv mine.csv   # bring your own
"""

from __future__ import annotations

import argparse
import json
import random
import re

import pandas as pd

from .paths import STAGE1_CSV, STAGE2_CSV, ensure_dirs

HUMAN = "human"
AI = "ai"
MIN_WORDS = 20


# --------------------------------------------------------------------------- #
# Source loader — HC3
# --------------------------------------------------------------------------- #
def _load_hc3(max_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    """HC3 human answers vs ChatGPT answers.

    ``datasets`` 3+ dropped script-based loaders, so we pull the raw JSONL
    straight from the Hub instead of ``load_dataset``.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    human: list[str] = []
    ai: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            for h in row.get("human_answers") or []:
                if h and len(h.split()) >= MIN_WORDS:
                    human.append(_clean(h))
            for a in row.get("chatgpt_answers") or []:
                if a and len(a.split()) >= MIN_WORDS:
                    ai.append(_clean(a))

    rng = random.Random(seed)
    rng.shuffle(human)
    rng.shuffle(ai)
    return human[:max_per_class], ai[:max_per_class]


_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Humanizers
# --------------------------------------------------------------------------- #
def _humanize_t5(texts: list[str], batch_size: int = 24,
                 sents_per_chunk: int = 3, max_tokens: int = 200) -> list[str]:
    """Paraphrase texts with a real T5 paraphraser (humarin/...T5_base).

    Works in chunks of a few sentences at a time (not the whole document, which
    T5's 512-token window would truncate, and not sentence-by-sentence, which is
    ~5x more generation calls). Greedy decoding; uses MPS/CUDA when available.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from .preprocess import split_sentences

    name = "humarin/chatgpt_paraphraser_on_T5_base"
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[humanize] loading {name} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()

    # Break each text into ~sents_per_chunk-sentence chunks; remember ownership.
    flat: list[str] = []
    owner: list[int] = []
    for i, text in enumerate(texts):
        sents = split_sentences(text) or [text]
        for c in range(0, len(sents), sents_per_chunk):
            chunk = " ".join(sents[c:c + sents_per_chunk])
            flat.append(f"paraphrase: {chunk}")
            owner.append(i)

    out: list[str] = [""] * len(flat)
    total = len(flat)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = flat[start:start + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_tokens).to(device)
            gen = model.generate(**enc, num_beams=1, do_sample=False,
                                 max_new_tokens=max_tokens,
                                 no_repeat_ngram_size=3, repetition_penalty=1.2)
            for j, dec in enumerate(tok.batch_decode(gen,
                                                     skip_special_tokens=True)):
                out[start + j] = dec.strip()
            if start % (batch_size * 10) == 0:
                print(f"[humanize] {start}/{total} chunks", flush=True)

    buf: dict[int, list[str]] = {}
    for idx, piece in zip(owner, out):
        buf.setdefault(idx, []).append(piece)
    return [_clean(" ".join(buf.get(i, [texts[i]]))) for i in range(len(texts))]


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


def _humanize_pseudo(texts: list[str], seed: int) -> list[str]:
    """Cheap dependency-free stand-in. Lower quality than --humanizer t5."""
    rng = random.Random(seed)
    out = []
    for text in texts:
        t = text
        for a, b in {**_CONTRACTIONS, **_SWAPS}.items():
            t = re.sub(rf"\b{a}\b", b, t, flags=re.IGNORECASE)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        if len(sents) > 3 and rng.random() < 0.5:
            i = rng.randrange(len(sents) - 1)
            sents[i], sents[i + 1] = sents[i + 1], sents[i]
        out.append(_clean(" ".join(sents)))
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _write(path, rows: list[tuple[str, str]]) -> None:
    df = pd.DataFrame(rows, columns=["text", "label"])
    df = df.dropna()
    df = df[df["text"].str.split().str.len() >= 5]
    df = df.drop_duplicates(subset="text")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[datasets_build] wrote {path}  ({len(df)} rows, "
          f"{(df.label == HUMAN).sum()} human / {(df.label == AI).sum()} ai)")


def build(max_per_class: int = 6000, seed: int = 42,
          humanizer: str = "t5", humanized_csv: str | None = None) -> None:
    ensure_dirs()
    human, raw_ai = _load_hc3(max_per_class, seed)
    n = min(len(human), len(raw_ai))
    human, raw_ai = human[:n], raw_ai[:n]
    print(f"[datasets_build] HC3: {n} human / {n} raw-AI samples")

    # Stage 1
    _write(STAGE1_CSV,
           [(t, HUMAN) for t in human] + [(t, AI) for t in raw_ai])

    # Stage 2 — same human pool, humanized version of the same AI answers
    if humanized_csv:
        hdf = pd.read_csv(humanized_csv)
        humanized = hdf.loc[hdf["label"].astype(str).str.lower() == AI,
                            "text"].tolist()
        print(f"[datasets_build] Stage 2 AI side from {humanized_csv}")
    elif humanizer == "pseudo":
        print("[datasets_build] Stage 2: pseudo-humanizer (fast, low fidelity)")
        humanized = _humanize_pseudo(raw_ai, seed)
    else:
        print("[datasets_build] Stage 2: T5 paraphraser")
        humanized = _humanize_t5(raw_ai)

    m = min(len(human), len(humanized))
    _write(STAGE2_CSV,
           [(t, HUMAN) for t in human[:m]] + [(t, AI) for t in humanized[:m]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 1 / Stage 2 training CSVs")
    ap.add_argument("--max-per-class", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--humanizer", choices=["t5", "pseudo"], default="t5",
                    help="how to generate Stage 2 humanized AI text")
    ap.add_argument("--humanized-csv", default=None,
                    help="CSV (text,label) of real humanized AI text for Stage 2")
    args = ap.parse_args()
    build(args.max_per_class, args.seed, args.humanizer, args.humanized_csv)


if __name__ == "__main__":
    main()
