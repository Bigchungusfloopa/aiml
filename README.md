# AI-Generated Text Detection System

Detects whether text is **human-written**, **raw AI-generated**, or **humanized
(paraphrased) AI**. Web app + classical-ML backend. Implements
[SRS_AI_Text_Detection.md](SRS_AI_Text_Detection.md).

## Architecture (SRS §6)

```
                 ┌───────────────── always-on evidence (FR-17) ─────────────────┐
input text ──┬──▶ rule-based style checker  (rules.py)      ─┐
             ├──▶ GPT-2 perplexity/burstiness (perplexity.py) ─┤
             │                                                 ├──▶ combined result
             └──▶ Stage 1: raw AI vs human  (model_raw.pkl) ───┤        (detector.py)
                        │ confident "AI"? ──▶ done (FR-18)      │
                        │ else "Human"  ──▶ Stage 2 (FR-19) ────┘
                          Stage 2: humanized AI vs human (model_humanized.pkl)
```

- **Trained components are classical only** — TF-IDF + Logistic Regression
  (SRS constraint §2.4). GPT-2 small is used *pre-trained*, for perplexity
  scoring only — never trained here.

## Layout

| Path | Role |
|---|---|
| `src/aitext/preprocess.py` | text normalization, sentence splitting (FR-2) |
| `src/aitext/datasets_build.py` | pull HC3 / MAGE via 🤗 `datasets`, write training CSVs (FR-1) |
| `src/aitext/train.py` | train + persist Stage 1 / Stage 2 (FR-3…FR-9) |
| `src/aitext/rules.py` | em-dash, cliché, sentence-uniformity checker (FR-10…FR-13) |
| `src/aitext/perplexity.py` | GPT-2 perplexity + burstiness (FR-14…FR-16) |
| `src/aitext/detector.py` | cascade orchestration + confidence bands (FR-17…FR-25) |
| `src/aitext/evaluate.py` | held-out Stage 1 / Stage 2 / 3-way cascade metrics |
| `app.py` | Streamlit UI (FR-21…FR-25) |
| `config/cliches.txt` | maintained cliché list — edit without touching code (NFR-6) |
| `models/` | persisted `.pkl` artifacts (Appendix A) — gitignored |
| `data/` | prepared training CSVs — gitignored |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train the models

```bash
# 1. Build datasets (downloads HC3 + the T5 paraphraser the first time; needs internet)
PYTHONPATH=src python -m aitext.datasets_build --max-per-class 6000

#    Stage 2 "humanized" text is produced by running the SAME HC3 ChatGPT answers
#    through humarin/chatgpt_paraphraser_on_T5_base (a real paraphraser), so the
#    human pool is identical across stages (SRS 7.2). Alternatives:
#      --humanizer pseudo              fast regex stand-in, no model download
#      --humanized-csv data/mine.csv   your own Quillbot / Undetectable.ai output

# 2. Train + persist both stages (writes models/*.pkl and data/*_test.csv)
PYTHONPATH=src python -m aitext.train

# 3. Evaluate the cascade on the held-out split (Stage 1, Stage 2, 3-way)
PYTHONPATH=src python -m aitext.evaluate --sample 1500
```

Datasets ≥3 dropped script-based loaders, so HC3 is pulled as raw JSONL via
`huggingface_hub`. GPU/MPS is used automatically for the T5 paraphraser and
GPT-2 perplexity when available.

Run the module commands from `src/` on the path — either
`pip install -e .`-style, or:

```bash
PYTHONPATH=src python -m aitext.datasets_build
PYTHONPATH=src python -m aitext.train
```

## Results (HC3, 4,000 samples/class, held-out 20%)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Stage 1 — raw AI vs human | 95.9% | 0.993 |
| Stage 2 — humanized AI vs human | 95.5% | 0.991 |

End-to-end cascade, 3-way (`human` / `raw-ai` / `humanized-ai`): **77.9%** accuracy.
The `raw-ai` ↔ `humanized-ai` split is fuzzy (T5-paraphrased text still reads as
"AI"), but **97% of humanized-AI text is still caught as AI-generated**, and the
human false-positive rate is ~6%.

> Stage 2's headline accuracy is high because the T5 paraphraser leaves its own
> stylistic fingerprint — the model partly learns "T5 paraphrase" rather than
> "humanized in general" (SRS §9). Swap in varied humanizer output via
> `--humanized-csv` for a more honest number.

## Run the app

```bash
streamlit run app.py
```

The app degrades gracefully (NFR-3) if models are missing — it shows the
rule-based + perplexity evidence and tells you to train.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

## Deployment (NFR-7)

Streamlit Community Cloud or Hugging Face Spaces. Commit `models/*.pkl` (or
retrain in a build step) since `data/` and `models/` are gitignored by default.
Needs ≥1 GB RAM to hold GPT-2 + the classical models (SRS §2.5).

## Known limitations (SRS §9)

- Well-humanized AI text frequently evades detection — industry-wide.
- False positives on unusually formal/simple text and non-native English.
- English only. No cryptographic watermark verification.
