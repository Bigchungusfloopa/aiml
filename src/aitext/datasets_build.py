"""Dataset acquisition and preparation (FR-1).

Writes two prepared CSVs, each with ``text`` and ``label`` columns
(label in {human, ai}):

    data/stage1_raw.csv        human  vs  raw (unedited) AI text
    data/stage2_humanized.csv  human  vs  humanized (paraphrased) AI text

Stage 1 source: MAGE (yaful/MAGE) — 300k+ samples across 10 domains (reddit CMV,
ELI5, TL;DR, XSum, WritingPrompts, ROCStories, HellaSwag, SQuAD, SciGen, Yelp)
and 300+ generators. Broad domain coverage keeps the model from equating
"formal / encyclopaedic" with "AI" (SRS §9). HC3 is still available via
``--source hc3``.

Stage 2 humanized AI text is produced from the *same* Stage-1 machine texts
through a MIX of paraphrasers so the model can't just learn one tool's
fingerprint (SRS §9):
  * ``t5``       humarin/chatgpt_paraphraser_on_T5_base (neural, MPS/CUDA)
  * ``pseudo``   regex contraction / synonym / clause-shuffle rewrite
  * ``realpara`` genuine GPT-4-paraphrased text from MAGE's OOD para set
Default mix ≈ 55 % t5 / 20 % pseudo / 25 % realpara; override with --mix.
The human pool is identical across both stages (SRS §7.2).

Run:
    python -m aitext.datasets_build --max-per-class 8000
    python -m aitext.datasets_build --source hc3 --humanizer t5
    python -m aitext.datasets_build --humanized-csv mine.csv    # bring your own
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
MIN_WORDS = 25
MAX_WORDS = 450  # keep paraphrasing tractable; MAGE has very long docs


_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def _ok_len(text: str) -> bool:
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS


# --------------------------------------------------------------------------- #
# Stage-1 sources
# --------------------------------------------------------------------------- #
def _load_mage(max_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    """MAGE train.csv: label 1 = human, label 0 = machine. Domain-stratified."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("yaful/MAGE", "train.csv", repo_type="dataset")
    df = pd.read_csv(path)
    df["text"] = df["text"].map(_clean)
    df = df[df["text"].map(_ok_len)]
    df["domain"] = df["src"].str.split("_").str[0]

    def _sample(sub: pd.DataFrame) -> list[str]:
        # even quota per domain, top-up randomly if a domain is short
        per = max(1, max_per_class // sub["domain"].nunique())
        picks = (sub.groupby("domain", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), per), random_state=seed)))
        if len(picks) < max_per_class:
            extra = sub.drop(picks.index).sample(
                min(len(sub) - len(picks), max_per_class - len(picks)),
                random_state=seed)
            picks = pd.concat([picks, extra])
        return picks.sample(min(len(picks), max_per_class),
                            random_state=seed)["text"].tolist()

    human = _sample(df[df["label"] == 1])
    machine = _sample(df[df["label"] == 0])
    n = min(len(human), len(machine))
    return human[:n], machine[:n]


def _load_hc3(max_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    """HC3 human answers vs ChatGPT answers (raw JSONL from the Hub)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    human: list[str] = []
    ai: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            human += [_clean(h) for h in row.get("human_answers") or []]
            ai += [_clean(a) for a in row.get("chatgpt_answers") or []]
    human = [t for t in human if _ok_len(t)]
    ai = [t for t in ai if _ok_len(t)]
    rng = random.Random(seed)
    rng.shuffle(human)
    rng.shuffle(ai)
    n = min(len(human), len(ai), max_per_class)
    return human[:n], ai[:n]


def _load_mage_realpara(seed: int) -> list[str]:
    """Genuine GPT-4-paraphrased machine text from MAGE's OOD para test set."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download("yaful/MAGE", "test_ood_set_gpt_para.csv",
                               repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"[datasets_build] realpara pool unavailable: {exc}")
        return []
    df = pd.read_csv(path)
    df = df[df["src"].str.contains("gpt", case=False, na=False)]  # machine-para
    texts = [t for t in df["text"].map(_clean) if _ok_len(t)]
    random.Random(seed).shuffle(texts)
    return texts


# --------------------------------------------------------------------------- #
# Humanizers
# --------------------------------------------------------------------------- #
def _humanize_t5(texts: list[str], batch_size: int = 24,
                 sents_per_chunk: int = 3, max_tokens: int = 200) -> list[str]:
    """Paraphrase texts with humarin/chatgpt_paraphraser_on_T5_base.

    Chunks of a few sentences at a time; greedy decoding; MPS/CUDA when present.
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

    flat: list[str] = []
    owner: list[int] = []
    for i, text in enumerate(texts):
        sents = split_sentences(text) or [text]
        for c in range(0, len(sents), sents_per_chunk):
            flat.append("paraphrase: " + " ".join(sents[c:c + sents_per_chunk]))
            owner.append(i)

    out: list[str] = [""] * len(flat)
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            enc = tok(flat[start:start + batch_size], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=max_tokens).to(device)
            gen = model.generate(**enc, num_beams=1, do_sample=False,
                                 max_new_tokens=max_tokens,
                                 no_repeat_ngram_size=3, repetition_penalty=1.2)
            for j, dec in enumerate(tok.batch_decode(gen,
                                                     skip_special_tokens=True)):
                out[start + j] = dec.strip()
            if start % (batch_size * 10) == 0:
                print(f"[humanize] t5 {start}/{len(flat)} chunks", flush=True)

    buf: dict[int, list[str]] = {}
    for idx, piece in zip(owner, out):
        buf.setdefault(idx, []).append(piece)
    return [_clean(" ".join(buf.get(i, [texts[i]]))) for i in range(len(texts))]


_CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "cannot": "can't", "will not": "won't", "is not": "isn't",
    "are not": "aren't", "it is": "it's", "that is": "that's",
    "there is": "there's", "we are": "we're", "they are": "they're",
    "you are": "you're", "i am": "i'm", "would not": "wouldn't",
}
_SWAPS = {
    "utilize": "use", "commence": "start", "numerous": "many",
    "additionally": "also", "however": "but", "therefore": "so",
    "furthermore": "plus", "approximately": "about", "demonstrate": "show",
    "individuals": "people", "prior to": "before", "in order to": "to",
    "subsequently": "then", "nevertheless": "still", "obtain": "get",
    "purchase": "buy", "assist": "help", "regarding": "about",
}


def _humanize_pseudo(texts: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for text in texts:
        t = text
        for a, b in {**_CONTRACTIONS, **_SWAPS}.items():
            t = re.sub(rf"\b{a}\b", b, t, flags=re.IGNORECASE)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        if len(sents) > 3 and rng.random() < 0.6:
            i = rng.randrange(len(sents) - 1)
            sents[i], sents[i + 1] = sents[i + 1], sents[i]
        out.append(_clean(" ".join(sents)))
    return out


def _humanize_mixed(texts: list[str], realpara_pool: list[str],
                    mix: dict[str, float], seed: int) -> list[str]:
    """Split ``texts`` by the requested ratio and humanize each part its own way."""
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)

    total = sum(mix.values()) or 1.0
    n_t5 = int(len(texts) * mix.get("t5", 0) / total)
    n_pseudo = int(len(texts) * mix.get("pseudo", 0) / total)
    t5_idx = idx[:n_t5]
    pseudo_idx = idx[n_t5:n_t5 + n_pseudo]
    real_idx = idx[n_t5 + n_pseudo:]

    out: list[str] = [""] * len(texts)

    if t5_idx:
        print(f"[datasets_build] t5-paraphrasing {len(t5_idx)} texts")
        for k, v in zip(t5_idx, _humanize_t5([texts[i] for i in t5_idx])):
            out[k] = v
    if pseudo_idx:
        print(f"[datasets_build] pseudo-humanizing {len(pseudo_idx)} texts")
        for k, v in zip(pseudo_idx,
                        _humanize_pseudo([texts[i] for i in pseudo_idx], seed)):
            out[k] = v
    if real_idx:
        if realpara_pool:
            print(f"[datasets_build] {len(real_idx)} texts from real GPT-4 "
                  f"paraphrases (MAGE)")
            pool = (realpara_pool * (len(real_idx) // len(realpara_pool) + 1))
            rng.shuffle(pool)
            for k, v in zip(real_idx, pool):
                out[k] = _clean(v)
        else:  # fall back to t5 if the real pool is empty
            print(f"[datasets_build] realpara pool empty -> t5 for {len(real_idx)}")
            for k, v in zip(real_idx, _humanize_t5([texts[i] for i in real_idx])):
                out[k] = v
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


def _parse_mix(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def build(max_per_class: int = 8000, seed: int = 42, source: str = "mage",
          humanizer: str = "mix", mix: str = "t5=0.55,pseudo=0.20,realpara=0.25",
          humanized_csv: str | None = None) -> None:
    ensure_dirs()
    loader = _load_mage if source == "mage" else _load_hc3
    human, raw_ai = loader(max_per_class, seed)
    print(f"[datasets_build] {source}: {len(human)} human / {len(raw_ai)} raw-AI")

    _write(STAGE1_CSV,
           [(t, HUMAN) for t in human] + [(t, AI) for t in raw_ai])

    # Stage 2 — same human pool, humanized version of the same AI answers
    if humanized_csv:
        hdf = pd.read_csv(humanized_csv)
        humanized = hdf.loc[hdf["label"].astype(str).str.lower() == AI,
                            "text"].map(_clean).tolist()
        print(f"[datasets_build] Stage 2 AI side from {humanized_csv}")
    elif humanizer == "pseudo":
        humanized = _humanize_pseudo(raw_ai, seed)
    elif humanizer == "t5":
        humanized = _humanize_t5(raw_ai)
    else:  # mix
        realpara = _load_mage_realpara(seed) if source == "mage" else []
        humanized = _humanize_mixed(raw_ai, realpara, _parse_mix(mix), seed)

    humanized = [t for t in humanized if len(t.split()) >= 5]
    m = min(len(human), len(humanized))
    _write(STAGE2_CSV,
           [(t, HUMAN) for t in human[:m]] + [(t, AI) for t in humanized[:m]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 1 / Stage 2 training CSVs")
    ap.add_argument("--max-per-class", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", choices=["mage", "hc3"], default="mage")
    ap.add_argument("--humanizer", choices=["mix", "t5", "pseudo"], default="mix")
    ap.add_argument("--mix", default="t5=0.55,pseudo=0.20,realpara=0.25",
                    help="humanizer ratio for --humanizer mix")
    ap.add_argument("--humanized-csv", default=None,
                    help="CSV (text,label) of real humanized AI text for Stage 2")
    args = ap.parse_args()
    build(args.max_per_class, args.seed, args.source, args.humanizer,
          args.mix, args.humanized_csv)


if __name__ == "__main__":
    main()
